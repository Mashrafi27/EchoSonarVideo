"""Fast, batched vLLM-based evaluation of a trained GRPO checkpoint for the
frozen-EchoPrime + Qwen3-8B-text track.

`eval/echoprime/eval_checkpoint.py` (the original eval script) generates one example at a
time via plain HF `.generate()` -- correct, but far too slow to ever scale to a real
comparison against EchoSonar-R's own reported protocol (macro over 12 disease
categories, on all 1,215 private test studies -- see docs/OPEN_ISSUES.md #1c). This
script batches many requests through vLLM instead, the same serving path GRPO
training itself uses, so a few hundred (or eventually all 1,215) studies finish in
minutes, not hours/days.

Approach: merge the trained LoRA adapter into the base weights with peft's
`merge_and_unload()`, save that as a standalone HF checkpoint (base weights only, no
adapter files -- vLLM serves it directly, no LoRA-serving support needed for this
custom model class), then run real batched generation through our already-registered
`EchoPrimeQwen3ForCausalLMVLLM` class.

Usage:
    python -m eval.echoprime.eval_checkpoint_vllm \
        --checkpoint-dir /path/to/global_step_N \
        --val-parquet build/echoprime_grpo_val.parquet \
        --gpu-memory-utilization 0.46 --max-model-len 2560
"""
import argparse
import json
import os
import shutil
import statistics

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from echoprime_track.modeling import VIEW_TOKEN
from data_core.data.answers import parse_yes_no
from data_core.reward.score import extract_answer
from verl_bridge.reward import compute_score

CACHE_DIR = os.environ.get("ECHOPRIME_VIDEO_CACHE_DIR", "build/echoprime_video_cache")


def merge_checkpoint(base_checkpoint: str, checkpoint_dir: str, merged_out_dir: str) -> None:
    """Same load path as eval.echoprime.eval_checkpoint (full FSDP state dict, not the
    empty standalone lora_adapter/ export -- see CLAUDE.md), then bake the LoRA
    delta into the base weights and save a plain checkpoint vLLM can load with no
    further LoRA-serving support needed."""
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_checkpoint, trust_remote_code=True, dtype=torch.bfloat16)

    adapter_config_path = os.path.join(checkpoint_dir, "actor", "lora_adapter", "adapter_config.json")
    with open(adapter_config_path) as f:
        adapter_cfg = json.load(f)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["lora_alpha"],
        target_modules=adapter_cfg["target_modules"],
        lora_dropout=adapter_cfg.get("lora_dropout", 0.0),
        bias=adapter_cfg.get("bias", "none"),
    )
    peft_model = get_peft_model(model, lora_config)

    state_dict_path = os.path.join(checkpoint_dir, "actor", "model_world_size_1_rank_0.pt")
    state_dict = torch.load(state_dict_path, map_location="cpu")
    missing, _ = peft_model.load_state_dict(state_dict, strict=False)
    missing_lora = [m for m in missing if "lora_" in m]
    if missing_lora:
        raise RuntimeError(f"LoRA weights missing after load: {missing_lora[:5]}")

    merged = peft_model.merge_and_unload()

    os.makedirs(merged_out_dir, exist_ok=True)
    merged.config.auto_map = merged.get_auto_map()
    merged.save_pretrained(merged_out_dir)
    tokenizer.save_pretrained(merged_out_dir)
    import echoprime_track
    shutil.copy(
        os.path.join(os.path.dirname(echoprime_track.__file__), "modeling.py"),
        os.path.join(merged_out_dir, "modeling_echoprime_qwen3.py"))


def run_eval(args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from echoprime_track.register_vllm_model import register_echoprime_vllm_model
    register_echoprime_vllm_model()

    merged_dir = args.merged_dir or (args.checkpoint_dir.rstrip("/") + "_merged")
    if not os.path.exists(os.path.join(merged_dir, "config.json")) or args.force_remerge:
        print(f"merging LoRA into base weights -> {merged_dir}", flush=True)
        merge_checkpoint(args.base_checkpoint, args.checkpoint_dir, merged_dir)
    else:
        print(f"reusing already-merged checkpoint at {merged_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
    view_token_id = tokenizer.convert_tokens_to_ids(VIEW_TOKEN)

    llm = LLM(
        model=merged_dir, trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, enforce_eager=True, enable_mm_embeds=True,
    )

    val = pd.read_parquet(args.val_parquet)
    prompts = []
    meta = []
    for _, row in val.iterrows():
        messages = list(row["prompt"])
        study_uuid = row["extra_info"]["study_uuid"]
        cache = torch.load(os.path.join(CACHE_DIR, f"{study_uuid}.pt"))
        view_embeddings = cache["embeddings"][: args.max_views]
        n_views = view_embeddings.shape[0]

        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        actual_views = sum(1 for t in prompt_ids if t == view_token_id)
        assert actual_views == n_views, (
            f"study {study_uuid}: prompt has {actual_views} view placeholders but cache has {n_views}")

        image_data = view_embeddings.unsqueeze(1)
        prompts.append(TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data={"image": image_data}))
        meta.append({
            "study_uuid": study_uuid,
            "question_type": row["extra_info"].get("question_type"),
            "ground_truth": row["reward_model"]["ground_truth"],
        })

    sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0,
                         stop=["</answer>"], include_stop_str_in_output=True)
    print(f"generating {len(prompts)} completions in a real batch...", flush=True)
    outputs = llm.generate(prompts, sampling_params=sp)

    results = []
    for m, out in zip(meta, outputs):
        completion = out.outputs[0].text
        reward = compute_score("echoprime_grpo", completion, m["ground_truth"], m)
        entry = {"study_uuid": m["study_uuid"], "question_type": m["question_type"],
                 "reward": reward, "completion": completion}
        gt_obj = json.loads(m["ground_truth"])
        if gt_obj.get("kind") == "yesno":
            pred_answer = extract_answer(completion)
            entry["pred_label"] = parse_yes_no(pred_answer) if pred_answer else None
            entry["gold_label"] = gt_obj["target"]
        results.append(entry)

    print("\n=== Summary ===")
    print(f"mean reward (all {len(results)}): {statistics.mean(r['reward'] for r in results):.3f}")
    by_qtype: dict = {}
    for r in results:
        by_qtype.setdefault(r["question_type"], []).append(r["reward"])
    for qt, rewards in by_qtype.items():
        print(f"  {qt}: n={len(rewards)} mean_reward={statistics.mean(rewards):.3f}")

    yesno = [r for r in results if "gold_label" in r]
    if yesno:
        classes = sorted({r["gold_label"] for r in yesno})
        recalls = []
        for c in classes:
            subset = [r for r in yesno if r["gold_label"] == c]
            correct = sum(1 for r in subset if r["pred_label"] == c)
            recalls.append(correct / len(subset))
            print(f"  class {c!r}: n={len(subset)} recall={correct / len(subset):.3f}")
        print(f"  yes/no balanced accuracy (n={len(yesno)}): {statistics.mean(recalls):.3f}")

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(results)} predictions to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--base-checkpoint", default="build/echoprime_cold_start_checkpoint")
    ap.add_argument("--merged-dir", default=None,
                     help="where to save/reuse the merged (base+LoRA) checkpoint "
                          "(default: <checkpoint-dir>_merged)")
    ap.add_argument("--force-remerge", action="store_true")
    ap.add_argument("--val-parquet", default="build/echoprime_grpo_val.parquet")
    ap.add_argument("--max-views", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2560)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.46)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
