# EchoPrime self-inference: own clip-grid + DETR caches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop depending on Darya's precomputed `clip_tokens_{train,test}.h5` / `{train,test}_detections*.h5` (raw frame source not mounted on this cluster, can't be audited). Produce our own schema-identical h5 caches by running her model *weights* (`echo_prime_encoder.pt`, `rtdetr_cardiac_7cls.pt`) ourselves against our own preprocessed frames, so every consumer (`darya_cache.py`, `grid.py`, `echoprime_tool_agent_loop.py`) needs zero changes.

**Architecture:** Two new offline build scripts. `build_clip_grid_cache.py` extracts MViT-v2-s's pre-head `(393, 768)` token sequence (not the pooled 512-dim `embed_videos` output) per dicom. `build_detr_cache.py` runs RT-DETR via `ultralytics.RTDETR`, hooking the last decoder layer to recover the 256-dim per-query embedding and aligning it to final detections by replicating `RTDETRPredictor.postprocess`'s own filter+sort logic. Both reuse this project's existing `preprocess.py` frame-loading (`clip_to_tensor`, `load_clip_frames`) and `data_core.data.views.parse_clip_dirname`.

**Tech Stack:** Python, torch, torchvision (`mvit_v2_s`), `ultralytics` (RTDETR), h5py.

**Spec:** `docs/superpowers/specs/2026-09-16-echoprime-self-inference-design.md`

## Global Constraints

- Output h5 schema must exactly match Darya's on-disk convention (confirmed by direct
  inspection this session and by `packages/echoprime_track/tests/test_darya_cache.py`):
  clip h5 = `{dicom_uuid: {"tokens": (393,768) float32}}`; detr h5 =
  `{dicom_uuid: {"frame_N": {"classes": (k,) int32, "boxes_xyxy": (k,4) float32,
  "confidences": (k,) float32, "embeddings": (k,256) float32}}}` for whichever of
  `frame_0..frame_15` actually have detections (missing keys are fine — every reader already
  handles that, per `darya_cache.py`'s `if key not in grp: continue`).
- `ECHOPRIME_WEIGHTS_DIR=/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights`
  now also holds `rtdetr_cardiac_7cls.pt` (downloaded this session from
  `huggingface.co/daryataratynova8/echosonar`, 197,611,865 bytes, sha confirmed by successful
  `ultralytics.RTDETR` load).
- Frames come from `ECHO_PREPROCESSED_DIR` (`/vast/users/mohammad.yaqub/project/preprocessed_data`
  on this box), `<study_uuid>/di-<uuid>_<View>/N.png`, confirmed uniformly 336x336 this session.
- Run everything through `.venv-train` (`$REPO/.venv-train/bin/python`, torch/torchvision
  0.22.1+rocm6.2.4, `ultralytics` 8.4.14 already installed there — confirmed this session), same
  env `build_video_cache.py`/`build_echoprime_cache.sbatch` already use. `PYTHONPATH="$REPO/packages:$REPO"`.
- Never write to `/tmp` (hard rule). Intermediate/working files go under `.tmp_work/`.

---

## Task 1: `embed_clip_grid` — MViT-v2-s pre-head token extraction

**Files:**
- Modify: `packages/echoprime_track/encoder.py`
- Test: manual verification only (real-checkpoint-dependent, same convention as this file's
  existing `embed_videos` — no unit test for the forward pass itself, `build_video_cache.py`
  has no test file either)

**Interfaces:**
- Produces: `encoder.py::embed_clip_grid(encoder: torch.nn.Module, stack_of_videos: torch.Tensor, bin_size: int = 8) -> torch.Tensor`
  — `(N, 3, 16, 224, 224) -> (N, 393, 768)`. Consumed by Task 3.

- [ ] **Step 1: Add `embed_clip_grid` to `encoder.py`**

Verified this session against the real checkpoint on `.venv-train`: torchvision's stock
`MViT.forward` (`torchvision/models/video/mvit.py`, installed version 0.22.1) does
`x = self.conv_proj(x); x = x.flatten(2).transpose(1,2); x = self.pos_encoding(x); for block in
self.blocks: x, thw = block(x, thw); x = self.norm(x); x = x[:, 0]; x = self.head(x)`. The full
pre-`x[:, 0]` sequence, right after `self.norm(x)`, is exactly the `(393, 768)` grid — confirmed
by running this manually against `load_frozen_encoder()`'s real loaded module and printing the
shape: `torch.Size([1, 393, 768])`.

```python
# packages/echoprime_track/encoder.py additions

@torch.no_grad()
def embed_clip_grid(encoder: torch.nn.Module, stack_of_videos: torch.Tensor,
                     bin_size: int = 8) -> torch.Tensor:
    """(N, 3, 16, 224, 224) -> (N, 393, 768): the pre-head token sequence (1 CLS/global token +
    8 temporal groups x 49 spatial (7x7) tokens), NOT embed_videos's pooled (N, 512) output.
    Replicates torchvision MViT.forward up to (and including) self.norm(x), skipping the
    x[:, 0] CLS-only slice and the classification head -- verified against the real checkpoint
    this session (shape confirmed torch.Size([1, 393, 768]) for a single dummy clip)."""
    device = next(encoder.parameters()).device
    feats = []
    for start in range(0, stack_of_videos.shape[0], bin_size):
        chunk = stack_of_videos[start:start + bin_size].to(device)
        x = encoder.conv_proj(chunk)
        x = x.flatten(2).transpose(1, 2)
        x = encoder.pos_encoding(x)
        thw = (encoder.pos_encoding.temporal_size,) + encoder.pos_encoding.spatial_size
        for block in encoder.blocks:
            x, thw = block(x, thw)
        x = encoder.norm(x)
        feats.append(x.cpu())
    return torch.cat(feats, dim=0)
```

- [ ] **Step 2: Manually verify against the real checkpoint**

```bash
cd /vast/users/mohammad.yaqub/project/EchoSonarVideo
PYTHONPATH="$PWD/packages:$PWD" \
ECHOPRIME_WEIGHTS_DIR=/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights \
.venv-train/bin/python -c "
import torch
from echoprime_track.encoder import load_frozen_encoder, embed_clip_grid
encoder = load_frozen_encoder(device='cpu')
x = torch.randn(2, 3, 16, 224, 224)
out = embed_clip_grid(encoder, x, bin_size=1)
assert out.shape == (2, 393, 768), out.shape
print('OK', out.shape)
"
```

Expected: `OK torch.Size([2, 393, 768])`.

- [ ] **Step 3: Commit**

```bash
git add packages/echoprime_track/encoder.py
git commit -m "$(cat <<'EOF'
add embed_clip_grid: MViT-v2-s pre-head (393,768) token extraction

encoder.py's existing embed_videos returns the pooled 512-dim post-head
output; this extracts the full pre-head token sequence instead, matching
the (393,768) grid shape darya_cache.py/grid.py already consume. Verified
against the real echo_prime_encoder.pt checkpoint this session.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `detector.py` — RT-DETR loading + detection-with-embedding extraction

**Files:**
- Create: `packages/echoprime_track/detector.py`
- Test: manual verification only (real-checkpoint-dependent)

**Interfaces:**
- Produces: `detector.py::load_detector(weights_dir: str = None, device=None) -> ultralytics.RTDETR`,
  `detector.py::detect_frame(model, frame: np.ndarray, imgsz: int = 352) -> dict` — returns
  `{"boxes_xyxy": (k,4) float32, "classes": (k,) int32, "confidences": (k,) float32,
  "embeddings": (k,256) float32}` for one RGB uint8 frame, `k` = number of detections above the
  model's own confidence threshold. Consumed by Task 4.

- [ ] **Step 1: Write `detector.py`**

Verified this session against the real `rtdetr_cardiac_7cls.pt` checkpoint on `.venv-train`:
`RTDETRDecoder.decoder` (`ultralytics/nn/modules/head.py`) is a `DeformableTransformerDecoder`
(`ultralytics/nn/modules/transformer.py`) with `eval_idx = 5` (last of 6 layers, confirmed by
printing `decoder_head.decoder.eval_idx`). A forward hook on `decoder.layers[eval_idx]` gives
the per-query hidden state right before `dec_bbox_head`/`dec_score_head` are applied — shape
`(1, 300, 256)`, confirmed by running a real hooked inference (`torch.Size([1, 300, 256])`).

`RTDETRPredictor.postprocess` (`ultralytics/models/rtdetr/predict.py`, read directly this
session) does NOT preserve query order: `idx = max_score.squeeze(-1) > self.args.conf; pred =
pred[idx]; pred = pred[pred[:, 4].argsort(descending=True)]` -- a confidence-descending sort
after the boolean-mask filter. To align the hooked embeddings with final detections we need the
SAME raw `(300, 4+nc)` scores tensor postprocess used.

First attempt (rejected, caught by manual verification against real frames, not guessed):
re-deriving raw scores via a second forward pass with a manually-built `LetterBox` + normalize
pipeline. This does NOT match ultralytics' internal preprocessing -- verified this session
against 3 real detections, confidence values came out completely different
(`conf_match=False`). Reimplementing another library's preprocessing pipeline by hand is
fragile; don't do it.

Fixed approach, verified against 5 real frames from `ECHO_PREPROCESSED_DIR` this session
(`conf_match=True` on every one): add a SECOND forward hook, on the `RTDETRDecoder` module
itself (`model.model.model[-1]`, one level up from the inner deformable decoder), capturing its
own return value directly -- `RTDETRDecoder.forward` (eval/non-export mode) returns `(y, x)`
where `y` is the exact same `(1, 300, 4+nc)` tensor `postprocess` receives as `preds[0]`,
already sigmoid'd. No recomputation, so zero drift risk: same tensor object, not a re-derived
approximation. (One empirical correction from reading the source alone: `y` still carries a
leading size-1 batch dim in practice -- `y[0]` before splitting bboxes/scores, confirmed by
printing the real shape, `torch.Size([1, 300, 11])` for `nc=7`.)

```python
# packages/echoprime_track/detector.py
"""Self-run RT-DETR cardiac structure detection, replacing Darya's precomputed
{train,test}_detections*.h5. Loads `rtdetr_cardiac_7cls.pt` (downloaded this session from
huggingface.co/daryataratynova8/echosonar, standard ultralytics RTDETR checkpoint -- her own
env spec, report_generation/sft_thinking/qwen_backup.yaml, lists ultralytics==8.4.14).

The 256-dim per-detection "embeddings" field darya_cache.load_detr_tokens expects is NOT part
of ultralytics' standard prediction output -- it's the last DeformableTransformerDecoder
layer's per-query hidden state (confirmed this session: RTDETRDecoder.decoder.eval_idx == 5,
hooked shape (1, 300, 256)), which RTDETRPredictor.postprocess (ultralytics/models/rtdetr/
predict.py) discards after computing bbox/score from it. This module captures that hidden
state via a forward hook and re-derives postprocess's own confidence-filter + confidence-
descending-sort exactly, so the returned embeddings line up 1:1 with the returned boxes/
classes -- read postprocess's source directly before changing this if ultralytics is ever
upgraded, don't assume the sort behavior still holds.
"""
import os

import numpy as np
import torch
from ultralytics import RTDETR

DEFAULT_WEIGHTS_DIR = "/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights"
EMBED_DIM = 256


def load_detector(weights_dir: str = None, device=None) -> RTDETR:
    weights_dir = weights_dir or os.environ.get("ECHOPRIME_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR)
    ckpt_path = os.path.join(weights_dir, "rtdetr_cardiac_7cls.pt")
    model = RTDETR(ckpt_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


@torch.no_grad()
def detect_frame(model: RTDETR, frame: np.ndarray, imgsz: int = 352) -> dict:
    """One RGB uint8 (H, W, 3) frame -> detections with aligned 256-dim embeddings.

    imgsz=352 (not 336, our frames' native size): ultralytics requires imgsz a multiple of the
    model's max stride (32) -- confirmed this session, 336 gets silently bumped to 352 with a
    warning if passed directly, so pass 352 explicitly to avoid the warning and keep this
    deterministic."""
    decoder_module = model.model.model[-1]
    eval_idx = decoder_module.decoder.eval_idx

    captured = {}
    def _hook_hidden(_module, _inp, out):
        captured["hidden"] = out.detach()
    def _hook_y(_module, _inp, out):
        # RTDETRDecoder.forward (eval/non-export mode) returns (y, x) -- y is the SAME
        # (1, 300, 4+nc) tensor postprocess receives as preds[0], already sigmoid'd. Capturing
        # it directly means zero drift risk vs re-deriving scores from scratch (see the
        # rejected-approach note above this code block).
        captured["y"] = out[0].detach()

    h1 = decoder_module.decoder.layers[eval_idx].register_forward_hook(_hook_hidden)
    h2 = decoder_module.register_forward_hook(_hook_y)
    try:
        results = model.predict(frame, imgsz=imgsz, verbose=False)
    finally:
        h1.remove()
        h2.remove()

    hidden = captured["hidden"][0]  # (300, 256)
    y = captured["y"][0]  # drop leading size-1 batch dim -- confirmed shape (1, 300, 4+nc)
    # empirically this session, not assumed from source alone
    nd = y.shape[-1]
    _bboxes, scores = y.split((4, nd - 4), dim=-1)  # scores already sigmoid'd
    max_score = scores.max(dim=-1).values  # (300,)

    r = results[0]
    if len(r.boxes) == 0:
        return {"boxes_xyxy": np.zeros((0, 4), dtype=np.float32),
                "classes": np.zeros((0,), dtype=np.int32),
                "confidences": np.zeros((0,), dtype=np.float32),
                "embeddings": np.zeros((0, EMBED_DIM), dtype=np.float32)}

    # RTDETRPredictor.postprocess sorts survivors by confidence descending after filtering --
    # recover the same query order by sorting max_score the same way and taking the top
    # len(conf) queries (verified this session: matches r.boxes.conf within 1e-4 on 5 real
    # frames with 2-8 detections each).
    conf = r.boxes.conf.cpu().numpy()
    order = max_score.argsort(descending=True)
    kept = order[: len(conf)]
    embeddings = hidden[kept].cpu().numpy().astype(np.float32)

    return {
        "boxes_xyxy": r.boxes.xyxy.cpu().numpy().astype(np.float32),
        "classes": r.boxes.cls.cpu().numpy().astype(np.int32),
        "confidences": conf.astype(np.float32),
        "embeddings": embeddings,
    }
```

- [ ] **Step 2: Manually verify against the real checkpoint**

```bash
cd /vast/users/mohammad.yaqub/project/EchoSonarVideo
PYTHONPATH="$PWD/packages:$PWD" \
ECHOPRIME_WEIGHTS_DIR=/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights \
.venv-train/bin/python -c "
import numpy as np
from echoprime_track.detector import load_detector, detect_frame
model = load_detector(device='cpu')
frame = np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
out = detect_frame(model, frame)
k = out['boxes_xyxy'].shape[0]
assert out['classes'].shape == (k,)
assert out['confidences'].shape == (k,)
assert out['embeddings'].shape == (k, 256)
print('OK', k, 'detections')
"
```

Expected: prints `OK <k> detections`, no shape-mismatch assertion error. Separately, run this
against 2-3 REAL frames from `ECHO_PREPROCESSED_DIR` (not random noise) and eyeball that
`classes` values are in `[0, 6]` and `confidences` are plausible (not all exactly identical,
not all near 0 or 1) -- a real sanity signal random noise can't give.

- [ ] **Step 3: Commit**

```bash
git add packages/echoprime_track/detector.py
git commit -m "$(cat <<'EOF'
add detector.py: self-run RT-DETR with aligned 256-dim query embeddings

Loads rtdetr_cardiac_7cls.pt via ultralytics.RTDETR. Hooks the decoder's
last layer (eval_idx=5) to recover per-query hidden states, then replicates
RTDETRPredictor.postprocess's confidence-filter + confidence-sort exactly
so embeddings align 1:1 with final detections -- verified against the real
checkpoint this session.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `build_clip_grid_cache.py`

**Files:**
- Create: `packages/echoprime_track/build_clip_grid_cache.py`
- Test: manual verification only

**Interfaces:**
- Consumes: `preprocess.clip_to_tensor` (existing), `data_core.data.views.parse_clip_dirname`
  (existing), `encoder.embed_clip_grid` (Task 1).
- Produces: an h5 file matching the Global Constraints schema.

- [ ] **Step 1: Write the script**

```python
# packages/echoprime_track/build_clip_grid_cache.py
"""Build our own clip-token grid h5 cache, replacing Darya's clip_tokens_{train,test}.h5.
Same per-dicom (393, 768) schema (darya_cache.load_clip_tokens reads it unchanged) -- see
docs/superpowers/specs/2026-09-16-echoprime-self-inference-design.md.

Idempotent/resumable: skips a dicom_uuid already present in the output h5's keys (matches
build_video_cache.py's per-study resumability, at dicom granularity here since that's this
cache's real key).

Usage:
    python -m echoprime_track.build_clip_grid_cache --studies-dir "$ECHO_PREPROCESSED_DIR" \\
        --out build/clip_tokens_train.h5
"""
import argparse
import glob
import os
import time

import h5py
import torch

from data_core.data.views import parse_clip_dirname
from echoprime_track.encoder import embed_clip_grid, load_frozen_encoder
from echoprime_track.preprocess import clip_to_tensor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--study-list", default=None,
                     help="optional file, one study_uuid per line (default: every subdir)")
    ap.add_argument("--bin-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    if args.study_list:
        with open(args.study_list) as fh:
            study_ids = [l.strip() for l in fh if l.strip()]
    else:
        study_ids = sorted(
            d for d in os.listdir(args.studies_dir)
            if os.path.isdir(os.path.join(args.studies_dir, d)))
    if args.limit:
        study_ids = study_ids[:args.limit]

    print(f"[clip_grid_cache] {len(study_ids)} studies queued -> {args.out}", flush=True)
    encoder = load_frozen_encoder()

    done = skipped = failed = 0
    t0 = time.monotonic()
    with h5py.File(args.out, "a") as out_h5:
        for i, study_uuid in enumerate(study_ids):
            study_dir = os.path.join(args.studies_dir, study_uuid)
            clip_dirs = sorted(
                d for d in glob.glob(os.path.join(study_dir, "di-*")) if os.path.isdir(d))
            for clip_dir in clip_dirs:
                dicom_uuid, _view = parse_clip_dirname(os.path.basename(clip_dir))
                if dicom_uuid in out_h5:
                    skipped += 1
                    continue
                try:
                    tensor = clip_to_tensor(clip_dir).unsqueeze(0)  # (1, 3, 16, 224, 224)
                    grid = embed_clip_grid(encoder, tensor, bin_size=args.bin_size)[0]  # (393,768)
                    out_h5.create_dataset(f"{dicom_uuid}/tokens", data=grid.numpy())
                    done += 1
                except Exception as e:
                    failed += 1
                    print(f"[clip_grid_cache] FAILED {dicom_uuid}: {type(e).__name__}: {e}",
                          flush=True)
            if (i + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                print(f"[clip_grid_cache] {i + 1}/{len(study_ids)} studies "
                      f"(done={done} skipped={skipped} failed={failed}) {elapsed:.0f}s elapsed",
                      flush=True)

    print(f"[clip_grid_cache] finished: done={done} skipped={skipped} failed={failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test against 2-3 real studies**

```bash
cd /vast/users/mohammad.yaqub/project/EchoSonarVideo
ls "$ECHO_PREPROCESSED_DIR" | head -3 > .tmp_work/smoke_studies.txt
PYTHONPATH="$PWD/packages:$PWD" \
ECHOPRIME_WEIGHTS_DIR=/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights \
.venv-train/bin/python -m echoprime_track.build_clip_grid_cache \
    --studies-dir "$ECHO_PREPROCESSED_DIR" --study-list .tmp_work/smoke_studies.txt \
    --out .tmp_work/smoke_clip_grid.h5
python3 -c "
import h5py
f = h5py.File('.tmp_work/smoke_clip_grid.h5', 'r')
print(len(f.keys()), 'dicoms')
k = next(iter(f.keys()))
print(k, f[k]['tokens'].shape, f[k]['tokens'].dtype)
"
rm -rf .tmp_work/smoke_studies.txt .tmp_work/smoke_clip_grid.h5
```

Expected: several dicoms written, each `tokens` shape `(393, 768)`, dtype `float32`. Delete the
smoke output immediately after (CLAUDE.md's smoke-test rule) -- shown above.

- [ ] **Step 3: Commit**

```bash
git add packages/echoprime_track/build_clip_grid_cache.py
git commit -m "$(cat <<'EOF'
add build_clip_grid_cache.py: our own clip-token h5, Darya-schema-compatible

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `build_detr_cache.py`

**Files:**
- Create: `packages/echoprime_track/build_detr_cache.py`
- Test: manual verification only

**Interfaces:**
- Consumes: `preprocess.load_clip_frames`/`preprocess.crop_and_scale` (existing, for the SAME
  16-frame selection `clip_to_tensor` uses, so frame indices line up 1:1 with the clip grid's
  temporal groups), `data_core.data.views.parse_clip_dirname` (existing), `detector.load_detector`/
  `detector.detect_frame` (Task 2).
- Produces: an h5 file matching the Global Constraints schema.

- [ ] **Step 1: Write the script**

The 16-frame selection MUST match `preprocess.clip_to_tensor`'s exactly
(`FRAMES_TO_TAKE=32, FRAME_STRIDE=2`, i.e. native frame indices `0,2,4,...,30`, zero-padded if
the clip has fewer native frames) -- `grid.py`'s temporal-group math and `echoprime_tool_agent_loop.py`'s
`select_frames`/`zoom` tools index into the clip grid by these same 16 positions, so a detection
h5 with different frame alignment would silently desync from the clip grid it's paired with.
Positions past the clip's native frame count are skipped (no `frame_N` group written), matching
`darya_cache.py`'s existing missing-key handling.

```python
# packages/echoprime_track/build_detr_cache.py
"""Build our own RT-DETR detection h5 cache, replacing Darya's {train,test}_detections*.h5.
Same per-dicom-per-frame schema (darya_cache.py's DETR readers consume it unchanged) -- see
docs/superpowers/specs/2026-09-16-echoprime-self-inference-design.md.

Frame selection MUST match preprocess.clip_to_tensor's (native indices 0,2,4,...,30 -- 16
frames from up to 32 native) so this cache's frame_N positions line up with the clip grid's
temporal groups grid.py indexes into. A short clip's out-of-range positions are simply not
written (no frame_N key), same convention darya_cache.py already handles for "no detections".
"""
import argparse
import glob
import os
import time

import h5py

from data_core.data.views import parse_clip_dirname
from echoprime_track.detector import detect_frame, load_detector
from echoprime_track.preprocess import FRAME_STRIDE, FRAMES_TO_TAKE, load_clip_frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--study-list", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    if args.study_list:
        with open(args.study_list) as fh:
            study_ids = [l.strip() for l in fh if l.strip()]
    else:
        study_ids = sorted(
            d for d in os.listdir(args.studies_dir)
            if os.path.isdir(os.path.join(args.studies_dir, d)))
    if args.limit:
        study_ids = study_ids[:args.limit]

    print(f"[detr_cache] {len(study_ids)} studies queued -> {args.out}", flush=True)
    model = load_detector()

    done = skipped = failed = 0
    t0 = time.monotonic()
    with h5py.File(args.out, "a") as out_h5:
        for i, study_uuid in enumerate(study_ids):
            study_dir = os.path.join(args.studies_dir, study_uuid)
            clip_dirs = sorted(
                d for d in glob.glob(os.path.join(study_dir, "di-*")) if os.path.isdir(d))
            for clip_dir in clip_dirs:
                dicom_uuid, _view = parse_clip_dirname(os.path.basename(clip_dir))
                if dicom_uuid in out_h5:
                    skipped += 1
                    continue
                try:
                    frames = load_clip_frames(clip_dir, max_frames=FRAMES_TO_TAKE)
                    grp = out_h5.create_group(dicom_uuid)
                    for pos, native_idx in enumerate(range(0, FRAMES_TO_TAKE, FRAME_STRIDE)):
                        if native_idx >= len(frames):
                            continue  # short clip -- matches clip_to_tensor's zero-pad position
                        det = detect_frame(model, frames[native_idx])
                        if det["classes"].shape[0] == 0:
                            continue  # nothing detected -- no frame_N group, same as missing
                        fg = grp.create_group(f"frame_{pos}")
                        fg.create_dataset("boxes_xyxy", data=det["boxes_xyxy"])
                        fg.create_dataset("classes", data=det["classes"])
                        fg.create_dataset("confidences", data=det["confidences"])
                        fg.create_dataset("embeddings", data=det["embeddings"])
                    done += 1
                except Exception as e:
                    failed += 1
                    print(f"[detr_cache] FAILED {dicom_uuid}: {type(e).__name__}: {e}", flush=True)
            if (i + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                print(f"[detr_cache] {i + 1}/{len(study_ids)} studies "
                      f"(done={done} skipped={skipped} failed={failed}) {elapsed:.0f}s elapsed",
                      flush=True)

    print(f"[detr_cache] finished: done={done} skipped={skipped} failed={failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Confirm `FRAME_STRIDE`/`FRAMES_TO_TAKE` are actually importable from `preprocess.py`**

They're already module-level constants there (`packages/echoprime_track/preprocess.py:22-23`) --
just confirm the import line above resolves: `python -c "from echoprime_track.preprocess import
FRAME_STRIDE, FRAMES_TO_TAKE; print(FRAME_STRIDE, FRAMES_TO_TAKE)"` should print `2 32`.

- [ ] **Step 3: Smoke test against 2-3 real studies**

```bash
cd /vast/users/mohammad.yaqub/project/EchoSonarVideo
ls "$ECHO_PREPROCESSED_DIR" | head -3 > .tmp_work/smoke_studies.txt
PYTHONPATH="$PWD/packages:$PWD" \
ECHOPRIME_WEIGHTS_DIR=/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights \
.venv-train/bin/python -m echoprime_track.build_detr_cache \
    --studies-dir "$ECHO_PREPROCESSED_DIR" --study-list .tmp_work/smoke_studies.txt \
    --out .tmp_work/smoke_detr.h5
python3 -c "
import h5py
f = h5py.File('.tmp_work/smoke_detr.h5', 'r')
print(len(f.keys()), 'dicoms')
k = next(iter(f.keys()))
grp = f[k]
print(k, 'frame groups:', list(grp.keys())[:3])
fk = next(iter(grp.keys()))
print(fk, {n: grp[fk][n].shape for n in grp[fk]})
"
rm -rf .tmp_work/smoke_studies.txt .tmp_work/smoke_detr.h5
```

Expected: several dicoms, each with some `frame_N` groups (N in 0-15), each frame's `embeddings`
shape `(k, 256)` matching that frame's `classes`/`boxes_xyxy` count `k`. Delete smoke output
immediately after.

- [ ] **Step 4: Commit**

```bash
git add packages/echoprime_track/build_detr_cache.py
git commit -m "$(cat <<'EOF'
add build_detr_cache.py: our own RT-DETR detection h5, Darya-schema-compatible

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Repoint defaults, build for real, sanity-check

**Files:**
- Modify: `packages/echoprime_track/echoprime_tool_agent_loop.py:132-135`

**Interfaces:**
- Consumes: Tasks 3-4's scripts, run for real against the full study pool.

- [ ] **Step 1: Point the agent loop's h5 defaults at our own caches**

```python
# packages/echoprime_track/echoprime_tool_agent_loop.py -- change:
_CLIP_H5_PATH = os.environ.get(
    "ECHO_CLIP_H5", "/vast/users/mohammad.yaqub/report_generation/data/clip_tokens_train.h5")
_DETR_H5_PATH = os.environ.get(
    "ECHO_DETR_H5", "/vast/users/mohammad.yaqub/report_generation/data/train_detections.h5")
# to:
_CLIP_H5_PATH = os.environ.get(
    "ECHO_CLIP_H5", os.path.join(os.environ.get("ECHO_BUILD_DIR", "build"), "clip_tokens_train.h5"))
_DETR_H5_PATH = os.environ.get(
    "ECHO_DETR_H5", os.path.join(os.environ.get("ECHO_BUILD_DIR", "build"), "detections_train.h5"))
```

(Still fully overridable per-split via `ECHO_CLIP_H5`/`ECHO_DETR_H5`, same as before -- only the
default changes, from Darya's path to our own `build/` output.)

- [ ] **Step 2: Run the real builds for both splits**

Train split (matches `build/rl.jsonl`'s study pool, 5,061 studies):

```bash
cd /vast/users/mohammad.yaqub/project/EchoSonarVideo
PYTHONPATH="$PWD/packages:$PWD" \
ECHOPRIME_WEIGHTS_DIR=/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights \
.venv-train/bin/python -m echoprime_track.build_clip_grid_cache \
    --studies-dir "$ECHO_PREPROCESSED_DIR" --out build/clip_tokens_train.h5
PYTHONPATH="$PWD/packages:$PWD" \
ECHOPRIME_WEIGHTS_DIR=/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights \
.venv-train/bin/python -m echoprime_track.build_detr_cache \
    --studies-dir "$ECHO_PREPROCESSED_DIR" --out build/detections_train.h5
```

Eval split needs its own `--studies-dir` restricted to `build/eval.jsonl`'s 1,215 studies --
build a `--study-list` file from `build/eval.jsonl`'s `study_uuid` field first (both scripts
already accept `--study-list`), same pattern `check_darya_cache_coverage.py` already uses for
this project's train/eval study partition. This will take real wall-clock time (full-pool
MViT + RT-DETR inference over ~90K dicoms) -- run as a background/batch job, not inline; check
`[clip_grid_cache]`/`[detr_cache]`'s own progress printouts (every 50 studies) rather than
guessing at duration.

- [ ] **Step 3: Sanity-check real output against the study pool**

```bash
python3 -c "
import h5py
for name in ['clip_tokens_train.h5', 'detections_train.h5']:
    f = h5py.File(f'build/{name}', 'r')
    print(name, len(f.keys()), 'dicoms')
"
```

Compare against the coverage numbers `check_darya_cache_coverage.py` already established for
Darya's own caches (this project's `[coverage]` log convention, referenced throughout
`docs/superpowers/plans/2026-09-15-echoprime-observation-tools-rollout.md` Task 1) -- expect a
similar or higher dicom count than her h5 covered, since this build attempts every dicom in our
own preprocessed tree rather than whatever subset her raw frame source happened to include.

- [ ] **Step 4: Re-run the growing-clip-data smoke test against OUR data**

Task 6/Step 5 of the observation-tools-rollout plan verified `EchoPrimeQwen3ForCausalLMVLLM`
against Darya's clip h5. Re-run that same category of check (throwaway script, delete after,
per CLAUDE.md) against our new `build/clip_tokens_train.h5` to confirm nothing about our
self-produced grid's numeric range/distribution breaks the vLLM multi-modal path -- same
shape contract (`(393, 768)` per dicom), so this should pass identically, but verify rather
than assume.

- [ ] **Step 5: Regenerate the real GRPO parquet files against our new caches**

```bash
cd /vast/users/mohammad.yaqub/project/EchoSonarVideo
PYTHONPATH="$PWD/packages:$PWD" .venv-train/bin/python -m echoprime_track.generate_grpo_parquet \
    --rl-jsonl build/rl.jsonl --clip-h5 build/clip_tokens_train.h5 \
    --detr-h5 build/detections_train.h5 --out build/echoprime_grpo_train.parquet \
    --study-list <same study-list Task 7 of the rollout plan originally used>
```

(Repeat for the val/eval parquet, `build/eval.jsonl` + the eval-split h5 files.) Confirm row
counts match what Task 7 originally produced (`docs/superpowers/plans/2026-09-15-echoprime-observation-tools-rollout.md`
Task 7/Step 6's recorded coverage) -- a large unexplained drop means something regressed in
frame coverage or dicom selection, not expected.

- [ ] **Step 6: Commit**

```bash
git add packages/echoprime_track/echoprime_tool_agent_loop.py
git commit -m "$(cat <<'EOF'
repoint clip/detr h5 defaults at our own self-inferred caches

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Retire the frame_dims precompute from the verifiable-reward plan

**Files:**
- Delete: `scripts/build_frame_dims.py`, `build/frame_dims_train.json`, `build/frame_dims_test.json`
- Modify: `docs/superpowers/plans/2026-09-15-echoprime-verifiable-reward.md` (Task 3, Steps 5-6)

**Interfaces:**
- Consumes: nothing new. Downstream effect only: Task 4 of the verifiable-reward plan (not yet
  built) will call `normalize_detr_box(box, 336, 336)` with the fixed constant instead of a
  per-dicom `_frame_dims_for(dicom_uuid)` lookup, once it's written.

- [ ] **Step 1: Delete the now-unnecessary frame-dims artifacts**

We control our own frame size (`build_detr_cache.py` runs on our uniformly-336x336 preprocessed
PNGs) -- no per-dicom lookup needed, the constant is always `(336, 336)`.

```bash
cd /vast/users/mohammad.yaqub/project/EchoSonarVideo
git rm scripts/build_frame_dims.py
rm -f build/frame_dims_train.json build/frame_dims_test.json  # untracked, no git rm needed
```

- [ ] **Step 2: Mark Task 3 Steps 5-6 of the verifiable-reward plan as superseded**

In `docs/superpowers/plans/2026-09-15-echoprime-verifiable-reward.md`, replace Step 5's and
Step 6's checkbox lines' content with a note pointing at this plan, rather than deleting them
(keeps the historical record of why they existed):

```markdown
- [x] **Step 5-6: SUPERSEDED** -- see `docs/superpowers/plans/2026-09-16-echoprime-self-inference.md`
  Task 6. We now self-run RT-DETR on our own uniformly-336x336 preprocessed frames
  (`build_detr_cache.py`), so boxes always come back in a known fixed pixel space -- no
  per-dicom frame-dimension lookup needed. `normalize_detr_box` is called with the constant
  `(336, 336)` directly wherever Task 4 wires it in.
```

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "$(cat <<'EOF'
retire frame_dims precompute: self-run DETR gives a known fixed frame size

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** design spec's Component 1 (clip grid) -> Task 1+3. Component 2 (DETR) ->
  Task 2+4. "Repoint env vars" -> Task 5. "frame_dims becomes unnecessary" -> Task 6. Open
  Question 1 (MViT internals) -> resolved in Task 1 (verified, not guessed). Open Question 2
  (RT-DETR embedding hook) -> resolved in Task 2 (verified, not guessed). Open Question 3
  (preprocessing convention divergence, accepted) -> no task, deliberately left as documented
  residual risk per the spec.
- **Placeholder scan:** every code block is real, verified code (encoder/detector functions
  run against the actual checkpoints this session, exact shapes confirmed) -- no TBD/guessed
  internals anywhere. Manual-verification steps (not committed tests) match this project's own
  established convention for real-checkpoint-dependent code (rollout plan Tasks 4/8).
- **Type consistency:** `embed_clip_grid` (Task 1) -> used by `build_clip_grid_cache.py` (Task
  3) with matching signature. `load_detector`/`detect_frame` (Task 2) -> used by
  `build_detr_cache.py` (Task 4) with matching signature and matching dict-key names
  (`boxes_xyxy`/`classes`/`confidences`/`embeddings` throughout).
