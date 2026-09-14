"""Create an EchoPrimeQwen3ForCausalLM checkpoint initialised from Darya's SFT
Qwen3-8B-text weights.

The projector starts fresh (randomly initialised). The frozen EchoPrime encoder is
NOT part of this model -- it runs offline via build_video_cache and its output reaches
the model as precomputed view embeddings at rollout time.

Usage (on AMD, after pulling the latest code):

    python scripts/save_sft_init_checkpoint.py \\
        --sft-tokenizer /vast/users/mohammad.yaqub/report_generation/checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503 \\
        --sft-llm      /vast/users/mohammad.yaqub/report_generation/checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503/llm \\
        --out-dir      /vast/users/mohammad.yaqub/project/EchoSonarVideo/build/echoprime_sft_init
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

# Allow running as `python scripts/save_sft_init_checkpoint.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from transformers import AutoTokenizer

from echoprime_track.modeling import (
    EchoPrimeQwen3Config,
    EchoPrimeQwen3ForCausalLM,
    VIEW_TOKEN,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--sft-tokenizer", required=True,
        help="Dir containing Darya's tokenizer files (checkpoint-1503/, not checkpoint-1503/llm/).")
    ap.add_argument(
        "--sft-llm", required=True,
        help="Dir with Darya's SFT Qwen3ForCausalLM weights (checkpoint-1503/llm/).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    print("[sft_init] Loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.sft_tokenizer)
    # VIEW_TOKEN is not in Darya's tokenizer (vocab_size=151936, no custom view token).
    # add_special_tokens is idempotent if already present.
    tok.add_special_tokens({"additional_special_tokens": [VIEW_TOKEN]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    view_token_id = tok.convert_tokens_to_ids(VIEW_TOKEN)
    print(f"[sft_init] VIEW_TOKEN id={view_token_id}, total vocab={len(tok)}", flush=True)

    print(f"[sft_init] Building model from SFT LLM at {args.sft_llm} ...", flush=True)
    # from_cold_start calls AutoModelForCausalLM.from_pretrained(text_model_name) which
    # loads Darya's SFT weights directly from the local path.
    cfg = EchoPrimeQwen3Config(text_model_name=args.sft_llm, view_token_id=view_token_id)
    model = EchoPrimeQwen3ForCausalLM.from_cold_start(cfg, dtype=torch.bfloat16)

    # Darya's LLM has vocab_size=151936 (no VIEW_TOKEN).  Resize to include our one
    # added token.  The new VIEW_TOKEN embedding row is random -- it's always overwritten
    # in the forward pass by the projected EchoPrime embedding anyway.
    current_vocab = model.lm.get_input_embeddings().weight.shape[0]
    if current_vocab != len(tok):
        print(f"[sft_init] Resizing embeddings {current_vocab} -> {len(tok)}", flush=True)
        model.lm.resize_token_embeddings(len(tok))
    model.lm.config.use_cache = False

    # Serialise with the required auto_map so trust_remote_code=True loading works.
    model.config.auto_map = model.get_auto_map()
    # Propagate eos/pad/bos to generation_config so vLLM reads the right stop token.
    for _field in ("eos_token_id", "pad_token_id", "bos_token_id"):
        val = getattr(model.config, _field, None)
        if val is not None:
            setattr(model.generation_config, _field, val)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[sft_init] Saving to {args.out_dir} ...", flush=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tok.save_pretrained(args.out_dir)

    # trust_remote_code loading needs the actual modeling file next to config.json.
    shutil.copy(
        str(Path(__file__).resolve().parent.parent / "packages" / "echoprime_track" / "modeling.py"),
        os.path.join(args.out_dir, "modeling_echoprime_qwen3.py"),
    )
    print(f"[sft_init] Done -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
