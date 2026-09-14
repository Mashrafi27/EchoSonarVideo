"""Evaluate a trained GRPO checkpoint for the frozen-EchoPrime + Qwen3-8B-text track.

Loads the base cold-start architecture, wraps it in the same LoRA config used for
training, then loads the checkpoint's FULL FSDP state dict directly -- NOT via the
standalone `lora_adapter/adapter_model.safetensors` export, which is empty for this
architecture: verl's `layered_summon_lora_params()` (external/verl/verl/utils/
fsdp_utils.py) hardcodes decoder-layer prefixes assuming the wrapped LM's decoder
sits at `.model.layers`/`.language_model.layers`; ours sits one level deeper at
`.lm.model.layers` (see docs/OPEN_ISSUES.md), so it silently finds zero layers and
writes an empty adapter. The full state dict (`model_world_size_1_rank_0.pt`) has no
such filtering and holds the real trained weights -- confirmed live weight-sync to
vLLM during training used the unaffected `get_peft_model_state_dict` path instead
(rollout.layered_summon defaults False), so training itself was unaffected.

Usage:
    python -m eval.echoprime.eval_checkpoint \
        --checkpoint-dir /data/.../checkpoints/<exp>/global_step_150 \
        --val-parquet build/echoprime_grpo_val.parquet
"""
import argparse
import json
import os
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


def load_model(base_checkpoint: str, checkpoint_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16)

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
    missing, unexpected = peft_model.load_state_dict(state_dict, strict=False)
    missing_lora = [m for m in missing if "lora_" in m]
    if missing_lora:
        raise RuntimeError(
            f"{len(missing_lora)} trained LoRA weights missing after load "
            f"(e.g. {missing_lora[:5]}) -- checkpoint/config mismatch, do not trust this eval")
    print(f"Loaded {len(state_dict)} tensors from checkpoint "
          f"({len(missing)} missing keys expected to be non-LoRA buffers, {len(unexpected)} unexpected)")

    peft_model.to(device).eval()
    return tokenizer, peft_model


def run_eval(args):
    device = "cuda"
    tokenizer, model = load_model(args.base_checkpoint, args.checkpoint_dir, device)
    view_token_id = tokenizer.convert_tokens_to_ids(VIEW_TOKEN)

    val = pd.read_parquet(args.val_parquet)
    results = []
    for i, row in val.iterrows():
        messages = list(row["prompt"])
        study_uuid = row["extra_info"]["study_uuid"]
        cache = torch.load(os.path.join(CACHE_DIR, f"{study_uuid}.pt"))
        view_embeddings = cache["embeddings"][: args.max_views]
        n_views = view_embeddings.shape[0]

        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)

        actual_views = (enc["input_ids"][0] == view_token_id).sum().item()
        assert actual_views == n_views, (
            f"study {study_uuid}: prompt has {actual_views} view placeholders but cache has {n_views}")

        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                view_embeddings=view_embeddings.to(device),
                view_counts=[n_views],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        completion = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

        gt = row["reward_model"]["ground_truth"]
        extra_info = dict(row["extra_info"])
        reward = compute_score("echoprime_grpo", completion, gt, extra_info)

        entry = {
            "study_uuid": study_uuid,
            "question_type": extra_info.get("question_type"),
            "reward": reward,
            "completion": completion,
        }
        gt_obj = json.loads(gt)
        if gt_obj.get("kind") == "yesno":
            pred_answer = extract_answer(completion)
            entry["pred_label"] = parse_yes_no(pred_answer) if pred_answer else None
            entry["gold_label"] = gt_obj["target"]

        results.append(entry)
        print(f"[{i + 1}/{len(val)}] {study_uuid} ({entry['question_type']}) reward={reward:.3f}")

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
    else:
        print("  no yes/no questions in this val set -- balanced accuracy not computable")

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(results)} predictions to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--base-checkpoint", default="build/echoprime_cold_start_checkpoint")
    ap.add_argument("--val-parquet", default="build/echoprime_grpo_val.parquet")
    ap.add_argument("--max-views", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
