# EchoPrime self-inference: replace Darya's h5 caches with our own clip/DETR pipeline

Status: draft, approved in chat, pending spec review.

## Motivation

The frozen-EchoPrime + Qwen3-8B-text tool track (`packages/echoprime_track/`) currently reads
its per-dicom `(393, 768)` clip-token grid and RT-DETR structure detections directly from
Darya Taratynova's precomputed h5 caches (`report_generation/data/clip_tokens_{train,test}.h5`,
`{train,test}_detections*.h5`), via `darya_cache.py`. This was a deliberate simplification
(Task 7 of `docs/superpowers/plans/2026-09-15-echoprime-observation-tools-rollout.md`) over an
earlier from-scratch design.

Two problems surfaced while building the verifiable-reward plan's DETR-grounding task
(`docs/superpowers/plans/2026-09-15-echoprime-verifiable-reward.md`, Task 3):

- The raw frame source that fed her h5 caches, `/data/daryataratynova/icardio/...`, is not
  mounted on this cluster at all. We can never re-derive or audit what she computed.
- User decision (this session): stop depending on her precomputed caches entirely. Use only
  her model **weights** (`echo_prime_encoder.pt`, and the newly-found
  `rtdetr_cardiac_7cls.pt` on `huggingface.co/daryataratynova8/echosonar`); run all inference
  ourselves, on our own preprocessed frames
  (`/vast/users/mohammad.yaqub/project/preprocessed_data/<study_uuid>/<dicom_uuid>_<view>/N.png`,
  confirmed this session: uniformly 336x336 PNGs).

## Scope

In scope: two new offline scripts under `packages/echoprime_track/` that produce our own
`clip_tokens_{train,test}.h5` and `detections_{train,test}.h5`, in Darya's exact schema, plus
whatever encoder/RT-DETR extraction code they need. Repointing `ECHO_CLIP_H5`/`ECHO_DETR_H5`
(and any other hardcoded default paths) at the new files.

Out of scope, unchanged: `darya_cache.py`'s read functions, `grid.py`'s index math,
`echoprime_tool_agent_loop.py`, `generate_grpo_parquet.py`, `verl_bridge/reward.py`'s h5-reading
helpers. All of these consume an h5 file by schema, not by provenance — matching the schema is
what keeps them untouched.

## Architecture

Same-schema replacement, not a redesign of the consumer side:

```
our 336x336 PNGs -> [self-run MViT-v2-s, pre-head] -> our clip_tokens_{train,test}.h5
our 336x336 PNGs -> [self-run RT-DETR]              -> our detections_{train,test}.h5
                                                              |
                                            (schema-identical to Darya's files)
                                                              v
                              darya_cache.py / grid.py / echoprime_tool_agent_loop.py
                                        (unchanged -- reads by schema)
```

Rejected alternatives:
- **Live on-demand inference** (no cache, run both models per rollout request): would add a
  full MViT + RT-DETR forward pass to every GRPO rollout step's latency. Darya's own SFT
  training used precomputed caches for the same reason; no case for diverging here.
- **`.pt`-per-study cache** instead of h5: this is exactly the old pooled-embedding cache
  format Task 7 already moved away from, specifically for per-dicom lazy/concurrent read
  access at this scale (5,061 + 1,215 studies, up to 19 views each). No new reason to revisit.

## Component 1: clip-token grid (`build_clip_grid_cache.py`)

Extracts the `(393, 768)` pre-head token grid from `echo_prime_encoder.pt`
(`torchvision.models.video.mvit_v2_s`), instead of `encoder.py`'s current pooled `(1, 512)`
`embed_videos` output. Confirmed this session (user correction + spec cross-check against
`docs/superpowers/specs/2026-09-15-echoprime-tool-grpo-design.md` section 1): 393 = 1 CLS/global
token + 8 temporal groups (16 input frames -> 8 groups of 2) x 49 spatial tokens (7x7), all at
768-dim -- the network's own last-stage width, one layer before the head's `Linear(768, 512)`
projects it down.

Torchvision's stock `MViT.forward` only returns the pooled, post-head result
(`x = self.norm(x); x = x[:, 0]; x = self.head(x)`, per the architecture, pending confirmation
against the exact installed torchvision version's source -- see Open Questions). Getting the
full 393-token sequence means calling the encoder's own submodules directly
(`conv_proj`/`pos_encoding`/`blocks`/`norm`) up to `self.norm(x)`, skipping the `x[:, 0]` slice
and the `head` call entirely, rather than calling `encoder(x)`.

Input: our own 16-frame clips per dicom, resized 336x336 -> 224x224 (`encoder.py`'s documented
expected input shape `(N, 3, 16, 224, 224)`) -- same 16-frame sampling this project's
preprocessing already produces per dicom (matches Darya's own `n_original_frames`/
`sampling_method` convention referenced in her `train_frames.jsonl`, though we don't have or
need that file -- we use our own preprocessed frame directories directly).

Output: h5 file, `{dicom_uuid: {"tokens": (393, 768) float32}}` -- exact key shape
`darya_cache.load_clip_tokens` already expects.

## Component 2: DETR detections (`build_detr_cache.py`)

Loads `rtdetr_cardiac_7cls.pt` via `ultralytics.RTDETR` (confirmed loadable: her own env spec,
`report_generation/sft_thinking/qwen_backup.yaml`, lists `ultralytics==8.4.14`; three other
files in her repo reference this exact checkpoint by path). Standard `.predict()` gives
`boxes_xyxy`/`classes`/`confidences` directly -- everything the grounding reward (verifiable-
reward plan Task 3/4) needs.

What it does NOT give directly: Darya's h5 also carries a 256-dim `embeddings` field per
detection (consumed by `darya_cache.load_detr_tokens`, mean-pooled per class, spliced as
DETR_TOKEN placeholders in the turn-0 prompt). This is some internal RT-DETR decoder query
hidden state, not part of ultralytics' standard prediction output -- needs a hook into the
loaded model's internals, same category of work as Component 1's pre-head extraction. Flagged
as an open question below, not guessed here.

Since we run RT-DETR ourselves against our own uniformly-336x336 frames, boxes come back
directly in that known, fixed pixel space -- no per-dicom frame-dimension lookup needed at all.
This makes the verifiable-reward plan's Task 3 Steps 5-6
(`scripts/build_frame_dims.py`, `build/frame_dims_{train,test}.json`) unnecessary; that script
and its output are deleted as part of this work, and `normalize_detr_box` is called with the
fixed constant `(336, 336)` instead of a per-dicom lookup.

Output: h5 file, `{dicom_uuid: {"frame_0": {"boxes_xyxy", "classes", "confidences",
"embeddings"}, ..., "frame_15": {...}}}` -- exact shape `darya_cache.py`'s DETR readers expect.

## Open questions (resolve during implementation, don't guess now)

1. Exact torchvision `MViT` submodule names/forward internals for the installed version
   (`encoder.py`'s docstring says torchvision 0.25.0) -- confirm by reading the installed
   package source directly before writing the extraction code.
2. Where RT-DETR's 256-dim per-detection embedding actually lives inside `ultralytics.RTDETR`'s
   model graph (a decoder query hidden state, almost certainly, but the exact module/hook point
   needs reading `ultralytics`'s real RT-DETR head implementation, not assumed).
3. Whether our 336x336 preprocessing convention (resize/crop) materially differs from whatever
   convention Darya's `icardio` frame source used -- we can't compare directly (her source is
   inaccessible), so this is an accepted, undocumented divergence; not blocking, but worth one
   line in `CLAUDE.md` once implemented so nobody re-derives this same confusion later.

## Testing

Pure-logic pieces (grid math, embedding-extraction plumbing) get unit tests against synthetic
tensors, same pattern as `grid.py`'s existing tests. The actual encoder/RT-DETR forward passes
need a real checkpoint and real frames -- manual verification, not committed as a test (same
convention Task 4/Task 8 of the rollout plan used for real-checkpoint-dependent code). Sanity
check: run both new scripts against a handful of real studies, confirm shapes match
(`(393, 768)` per dicom for clip; per-frame `boxes_xyxy`/`classes`/`confidences`/`embeddings`
with `embeddings` at `(n, 256)` for detr) before running the real full-pool build.

## Downstream effect on the verifiable-reward plan

Task 3 (`docs/superpowers/plans/2026-09-15-echoprime-verifiable-reward.md`) needs updating:
Steps 5-6 (`build_frame_dims.py`, real script run) are dropped, replaced by the fixed-336
constant this design produces for free. Tasks 1-2 (canonical taxonomy, finding->DETR-class
map) and the pure `bbox_iou`/`normalize_detr_box` math (Task 3 Steps 1-4, already committed)
are unaffected.
