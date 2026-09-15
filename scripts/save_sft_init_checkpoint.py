"""Create an EchoPrimeQwen3ForCausalLM checkpoint initialised from Darya's SFT
Qwen3-8B-text weights, with the clip/detr projectors initialised from her real
trained projector weights too (not randomly).

The frozen EchoPrime encoder is NOT part of this model -- it runs offline via
build_video_cache and its output reaches the model as precomputed view embeddings
at rollout time.

Usage (on AMD, after pulling the latest code):

    python scripts/save_sft_init_checkpoint.py \\
        --sft-tokenizer /vast/users/mohammad.yaqub/report_generation/checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503 \\
        --sft-llm      /vast/users/mohammad.yaqub/report_generation/checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503/llm \\
        --darya-projectors /vast/users/mohammad.yaqub/report_generation/checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503 \\
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
    CLIP_TOKEN,
    DETR_TOKEN,
)


def load_darya_projectors(clip_projector: torch.nn.Module, detr_projector: torch.nn.Module,
                           darya_checkpoint_dir: str) -> None:
    """Loads report_generation/sft_thinking/model.py::EchoVLM.save_projectors's saved format
    (clip_projector.pt / detr_projector.pt) into our own projector modules, in place. Raises
    FileNotFoundError if either file is missing -- a silent skip here would leave the
    projector randomly initialized with no signal that Darya's real weights never loaded (the
    exact bug this plan's Task 3 exists to fix).

    mmap=True is REQUIRED, not optional -- verified directly this session: these files are
    ~16GB on disk (`ls -la` on the real checkpoint dir) despite the actual projector being a
    few tens of MB (confirmed via `torch.load(..., mmap=True)`: state_dict keys are
    `0.weight`/`0.bias` (LayerNorm), `1.weight`/`1.bias` (Linear 768->4096),
    `3.weight`/`3.bias` (Linear 4096->4096), totaling ~40MB). Darya's `state_dict()` was
    called directly on a DeepSpeed-wrapped model without detaching the returned tensors from
    DeepSpeed's flat parameter buffer, so a plain `torch.save` serialized the WHOLE buffer
    (matching an ~8B-parameter model in bf16) even though the logical tensors are tiny. Her
    own `report_generation/sft_thinking/convert_checkpoint.py` (producing the `_clean.pt`
    siblings) attempted to fix this by reloading and re-saving, but doesn't actually shrink
    it (`.clone()` never gets called on the loaded tensors) -- the `_clean.pt` files are the
    same size, not smaller. Without `mmap=True`, `torch.load(clip_path, map_location="cpu")`
    tries to materialize the full ~16GB storage in RAM and reliably OOM-kills on a
    memory-constrained machine (confirmed twice on this project's login node this session,
    `exit code 137`) -- `mmap=True` avoids this by only paging in the small byte ranges the
    projector's actual tensor views touch. Use the plain (non-`_clean`) filenames -- the
    `_clean` variant fixes nothing and is misleadingly named."""
    clip_path = os.path.join(darya_checkpoint_dir, "clip_projector.pt")
    detr_path = os.path.join(darya_checkpoint_dir, "detr_projector.pt")
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"no clip_projector.pt at {darya_checkpoint_dir}")
    if not os.path.exists(detr_path):
        raise FileNotFoundError(f"no detr_projector.pt at {darya_checkpoint_dir}")
    clip_projector.load_state_dict(torch.load(clip_path, map_location="cpu", mmap=True))
    detr_projector.load_state_dict(torch.load(detr_path, map_location="cpu", mmap=True))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--sft-tokenizer", required=True,
        help="Dir containing Darya's tokenizer files (checkpoint-1503/, not checkpoint-1503/llm/).")
    ap.add_argument(
        "--sft-llm", required=True,
        help="Dir with Darya's SFT Qwen3ForCausalLM weights (checkpoint-1503/llm/).")
    ap.add_argument(
        "--darya-projectors", required=True,
        help="Dir containing Darya's clip_projector.pt/detr_projector.pt "
             "(same checkpoint-1503 dir as --sft-tokenizer, per EchoVLM.save_projectors).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    print("[sft_init] Loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.sft_tokenizer)
    # CLIP_TOKEN/DETR_TOKEN are not in Darya's tokenizer (vocab_size=151936, no custom
    # observation tokens). add_special_tokens is idempotent if already present.
    tok.add_special_tokens({"additional_special_tokens": [CLIP_TOKEN, DETR_TOKEN]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    clip_token_id = tok.convert_tokens_to_ids(CLIP_TOKEN)
    detr_token_id = tok.convert_tokens_to_ids(DETR_TOKEN)
    print(f"[sft_init] CLIP_TOKEN id={clip_token_id}, DETR_TOKEN id={detr_token_id}, "
          f"total vocab={len(tok)}", flush=True)

    print(f"[sft_init] Building model from SFT LLM at {args.sft_llm} ...", flush=True)
    # from_cold_start calls AutoModelForCausalLM.from_pretrained(text_model_name) which
    # loads Darya's SFT weights directly from the local path.
    cfg = EchoPrimeQwen3Config(text_model_name=args.sft_llm,
                                clip_token_id=clip_token_id, detr_token_id=detr_token_id)
    model = EchoPrimeQwen3ForCausalLM.from_cold_start(cfg, dtype=torch.bfloat16)

    load_darya_projectors(model.clip_projector, model.detr_projector, args.darya_projectors)
    print(f"[sft_init] loaded real clip/detr projector weights from {args.darya_projectors}",
          flush=True)

    # Darya's LLM has vocab_size=151936 (no CLIP_TOKEN/DETR_TOKEN).  Resize to include our
    # two added tokens.  The new embedding rows are random -- they're always overwritten
    # in the forward pass by the projected EchoPrime embeddings anyway.
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
