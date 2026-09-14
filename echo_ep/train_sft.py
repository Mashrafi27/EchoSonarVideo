"""Cold-start SFT: frozen EchoPrime embeddings + trainable projector + Qwen3-8B (text), full
fine-tune. Matches EchoSonar-R's own recipe (3 epochs, LR 2e-5, AdamW, batch 64).

Uses HF Trainer + DeepSpeed ZeRO-3 (CPU offload of both params and optimizer state) across
the available GPUs -- full AdamW fine-tuning of an 8B model needs ~2x its size in optimizer
state alone (fp32 momentum + variance), which does not fit on one 48GB GPU alongside the
model itself.

FSDP was tried first (matching the FSDP+offload shape already used for the GRPO actor
elsewhere in this project) and hit a real, unresolved issue for THIS model specifically: HF
Trainer's `transformer_layer_cls_to_wrap` auto-wrap policy never found `Qwen3DecoderLayer`
boundaries through this custom composed architecture (the real LM lives nested one level
inside `EchoPrimeQwen3ForCausalLM.lm`, not at the top level Trainer sees), so FSDP treated
the entire 8B-parameter model as ONE flatten/shard unit instead of one per decoder layer --
confirmed by the OOM happening at `torch.cat` over every parameter at once (~30GB) rather
than a per-layer amount. DeepSpeed ZeRO-3 partitions parameter-by-parameter instead of by
auto-detected module boundaries, so it doesn't depend on that class-name matching at all --
see docs/OPEN_ISSUES.md for the full FSDP debugging trail if that gets revisited.

Usage (2 GPUs):
    accelerate launch --num_processes 2 --mixed_precision bf16 \\
        -m echo_ep.train_sft --epochs 3 --lr 2e-5 --micro-batch 1 --grad-accum 32

Smoke test first (small subset, few steps) before committing to a real run -- see the plan.
"""
import argparse
import os

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from echo_ep.dataset import EchoPrimeCollator, EchoPrimeSFTDataset
from echo_ep.modeling import EchoPrimeQwen3Config, EchoPrimeQwen3ForCausalLM, VIEW_TOKEN

DS_CONFIG_DIR = os.path.dirname(__file__)


def build_model_and_tokenizer(text_model_name: str):
    tok = AutoTokenizer.from_pretrained(text_model_name)
    tok.add_special_tokens({"additional_special_tokens": [VIEW_TOKEN]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = EchoPrimeQwen3Config(text_model_name=text_model_name,
                                view_token_id=tok.convert_tokens_to_ids(VIEW_TOKEN))
    # Unlike FSDP (see module docstring), DeepSpeed ZeRO-3 doesn't upcast bf16 weights to
    # fp32 on us -- the ds_zero3.json config's own "bf16": {"enabled": "auto"} handles
    # precision directly, so loading in bf16 here is safe for both the single-GPU eager
    # path and the multi-GPU ZeRO-3 path.
    model = EchoPrimeQwen3ForCausalLM.from_cold_start(cfg, dtype=torch.bfloat16)
    model.lm.resize_token_embeddings(len(tok))
    model.lm.config.use_cache = False  # incompatible with gradient checkpointing
    return model, tok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-jsonl", default="Archive 2 (1)/train_vqa_with_thinking.jsonl")
    ap.add_argument("--cache-dir", default="build/echoprime_video_cache")
    ap.add_argument("--text-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=64,
                     help="micro_batch * grad_accum * num_gpus should land near the paper's "
                          "effective batch size of 64")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=-1,
                     help="cap total steps -- for a smoke test, not the real run")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--no-deepspeed", action="store_true",
                     help="disable DeepSpeed entirely -- only viable with LoRA/frozen-base "
                          "setups, a full 8B fine-tune needs the offload to fit any single GPU")
    ap.add_argument("--zero-stage", choices=["2", "3"], default="3",
                     help="ds_zero2.json (no param sharding, faster, more per-GPU memory) "
                          "vs ds_zero3.json (param-sharded, slower, less per-GPU memory)")
    args = ap.parse_args(argv)

    # DeepSpeed's CPU offload is what makes an 8B full fine-tune fit at all -- needed on a
    # single GPU just as much as across several, so this does NOT gate on WORLD_SIZE > 1.
    use_deepspeed = not args.no_deepspeed

    # TrainingArguments must be constructed BEFORE the model when using DeepSpeed ZeRO-3:
    # `deepspeed=` activates transformers' HfDeepSpeedConfig context, which is what makes
    # the (nested, inside from_cold_start) AutoModelForCausalLM.from_pretrained call below
    # construct the model already-partitioned instead of fully materializing it first on
    # every rank -- this check is global (is_deepspeed_zero3_enabled()), not tied to being
    # literally the top-level model Trainer sees, which is exactly what FSDP's layer-name
    # auto-wrap policy needed and didn't get for this nested custom architecture.
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=False,  # enabled directly on model.lm below instead
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],  # wandb wiring added once past the smoke-test stage
        remove_unused_columns=False,  # our collator's dict keys aren't HF-standard column names
        deepspeed=(os.path.join(DS_CONFIG_DIR, f"ds_zero{args.zero_stage}.json")
                   if use_deepspeed else None),
    )

    model, tok = build_model_and_tokenizer(args.text_model)
    model.lm.gradient_checkpointing_enable()

    dataset = EchoPrimeSFTDataset(args.train_jsonl, args.cache_dir)
    collator = EchoPrimeCollator(tok, max_length=args.max_length)

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                       data_collator=collator)
    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[echo_ep] SFT complete -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
