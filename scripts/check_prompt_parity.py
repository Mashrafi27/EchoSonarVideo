#!/usr/bin/env python3
"""Assert the local eval client renders prompts the way the model expects.

    python scripts/check_prompt_parity.py [--model PATH]

Processor-only: loads no weights, so it runs on the login node in seconds.

The failure this is here to catch is silent. If the eval client's rendering drifts
from what a served engine (or SFT training) produced, every number we report
measures the harness instead of the model, and nothing crashes to tell us. The two
assertions that actually bite:

  1. IMAGE PLACEHOLDER ACCOUNTING. The number of image-pad tokens in input_ids must
     equal the tokens the vision tower will emit, i.e. sum(t*h*w)/merge^2 over
     image_grid_thw. A mismatch means the model attends to the wrong positions --
     the classic silently-wrong-grounding failure.
  2. NO VIDEO PATH. Nothing may populate `videos`/`video_grid_thw`. Shipping the
     view menu as a video is a bug we already hit once: Qwen3VLVideoProcessor
     resamples it (do_sample_frames=True, fps=2) and 19 view previews arrive as
     4 frames.
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from echo_verl.eval.agentic_loop import SYSTEM_PROMPT, _image_part  # noqa: E402
from echo_verl.eval.local_client import to_qwen_messages            # noqa: E402

DEFAULT_MODEL = str(REPO / "checkpoints" / "echo-sft" / "merged" / "step100")
_results: list[tuple[str, bool, str]] = []


def check(name, fn):
    try:
        _results.append((name, True, str(fn() or "")))
    except Exception as e:
        _results.append((name, False, f"{type(e).__name__}: {e}"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-images", type=int, default=3)
    args = ap.parse_args(argv)

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)

    # A stand-in view menu: distinct sizes so a silent resize shows up in the grid.
    images = [Image.new("RGB", (224 + 32 * i, 224), (i * 40, 80, 160))
              for i in range(args.n_images)]
    content = [_image_part(im) for im in images]
    content.append({"type": "text", "text": "Is there any abnormality?"})
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]

    qwen_messages, decoded = to_qwen_messages(messages)
    text = proc.apply_chat_template(qwen_messages, tokenize=False,
                                    add_generation_prompt=True)
    batch = proc(text=[text], images=decoded or None, return_tensors="pt")

    def _roundtrip():
        assert len(decoded) == args.n_images, f"{len(decoded)} images survived, want {args.n_images}"
        for got, want in zip(decoded, images):
            assert got.size == want.size, f"data-URI roundtrip resized {want.size} -> {got.size}"
        return f"{args.n_images} images, sizes preserved"

    def _placeholders():
        pad_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        assert pad_id is not None and pad_id >= 0, "no <|image_pad|> token in this tokenizer"
        n_pad = int((batch["input_ids"] == pad_id).sum())
        grid = batch["image_grid_thw"]
        merge = getattr(proc.image_processor, "merge_size", 2)
        expected = int(grid.prod(dim=-1).sum()) // (merge ** 2)
        assert n_pad == expected, (
            f"{n_pad} image-pad tokens but vision tower emits {expected} "
            f"(grid={grid.tolist()}, merge={merge}) -- grounding would be misaligned")
        return f"{n_pad} pad tokens == {expected} vision tokens (merge={merge})"

    def _no_video():
        bad = [k for k in batch if "video" in k]
        assert not bad, f"video tensors present: {bad} -- the view menu must be IMAGES"
        return "no video tensors"

    def _generation_prompt():
        assert text.rstrip().endswith("<|im_start|>assistant"), repr(text[-60:])
        return "ends with assistant generation prompt"

    def _roles():
        assert text.count("<|im_start|>system") == 1, "expected exactly one system turn"
        assert text.count("<|im_start|>user") == 1, "expected exactly one user turn"
        return "system + user turns render once each"

    check("data-URI roundtrip preserves images", _roundtrip)
    check("image placeholders match vision-token count", _placeholders)
    check("no video path is taken", _no_video)
    check("add_generation_prompt applied", _generation_prompt)
    check("role structure", _roles)

    width = max(len(n) for n, _, _ in _results)
    failed = sum(1 for _, ok, _ in _results if not ok)
    for name, ok, detail in _results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    print(f"\n{len(_results) - failed}/{len(_results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
