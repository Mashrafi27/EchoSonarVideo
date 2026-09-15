# EchoPrime + Qwen3-8B-text: adding select_frames/zoom tools, real SFT-matched observations, and verifiable-only reward

Status: draft, pending user review. Implements PLAN.md's "Next Steps" item 2.

## Motivation

`packages/echoprime_track/` (frozen EchoPrime mvit_v2_s encoder + Qwen3-8B-text,
GRPO) currently:

- Feeds the LLM **one pooled 512-dim vector per view** (`encoder.py`'s
  `embed_videos`, post-head), spliced via a single `VIEW_TOKEN` per view.
- Initializes its projector **randomly** (`save_sft_init_checkpoint.py` only
  copies Darya's SFT *LLM* weights, not her projector weights).
- Has **no tools** — `EchoPrimeAgentLoop` is explicitly single-turn.

But Darya's actual SFT (`report_generation/sft_thinking/`, the checkpoint this
track claims to initialize from) was trained on a completely different
observation: per view, the **full 393-token clip grid** (768-dim, pre-head,
via `clip_projector`) plus **RT-DETR structure tokens** (256-dim, via
`detr_projector`), interleaved with `"{view}:\n"` / `"Structure features:\n"`
text. The LLM's SFT training barely transfers to the pooled-512 format it's
never seen — this is likely a real contributor to why SPEC.md calls this
track a "no-SFT-cold-start ablation" despite claiming an SFT init.

This spec: (1) fixes that mismatch by adopting Darya's real observation format
verbatim (same h5 caches, same sequencing, same projector weights), (2) adds
`select_frames`/`zoom` as tools that index into the same precomputed
393-token grid (no new encoder inference), and (3) redesigns the reward so
tool-use is only rewarded when it's tied to something verifiable.

## Scope

In scope: `packages/echoprime_track/` (modeling, agent loop, SFT-init
script), `packages/data_core/reward/score.py`, `packages/verl_bridge/reward.py`,
this track's parquet generation / data scope.

Out of scope: the Qwen3-VL image-based `echo_grpo` track (untouched), the
real LLM-judge swap for text-type reward (already tracked as P3e — this spec
routes around it by excluding unverifiable question types instead of trying
to fix judging).

## 1. Observation format — adopt Darya's pipeline verbatim

Reuse, don't recompute:

- `report_generation/data/clip_tokens_{train,test}.h5` — per-`dicom_uuid`
  `(393, 768)` float32 tensor. 393 = 1 global token + 8 temporal groups (16
  frames → 8 groups of 2) × 49 spatial tokens (7×7 grid over the frame).
  Confirmed by direct inspection this session.
- `report_generation/data/{train,test}_detections.h5` — per-`dicom_uuid` →
  `frame_{0..15}` → `classes` (int32, RT-DETR class id), `boxes_xyxy`
  (float32, `(n,4)`), `confidences`, `embeddings` (256-dim). 7 classes: Left
  Ventricle, Left Atrium, Right Atrium, Right Ventricle, Mitral Valve,
  Tricuspid Valve, LVOT Area (`sft_thinking/dataset.py::RT_DETR_CLASSES`).
- Sequencing per view (`sft_thinking/dataset.py::__getitem__`, reused as-is):
  `"{view_name}:\n"` → 393 clip-token placeholders → (if any detections)
  `"Structure features:\n"` → per-class mean-pooled (over 16 frames) DETR
  tokens.

**Model changes (`echoprime_track/modeling.py`):**
- Replace the single `nn.Sequential` `projector` (512→hidden) with
  `clip_projector` (768→hidden) and `detr_projector` (256→hidden), matching
  `report_generation/sft_thinking/model.py::EchoVLM` layer-for-layer
  (`LayerNorm` + `Linear`, not the current `Linear/GELU/Linear`).
- Generalize `_splice_view_embeddings` from "one VIEW_TOKEN per view" to
  "arbitrary count of placeholder positions per example, each independently
  tagged clip-vs-detr" — i.e. adopt `EchoVLM.forward`'s `vision_mask`
  approach (boolean mask over the sequence, `torch.where(vision_mask,
  vision_embeds, text_embeds)`) instead of scanning for a single special
  token id. This is required anyway once tool calls append *new* placeholder
  runs mid-generation.

**`save_sft_init_checkpoint.py` change:** after `from_cold_start`, load
`clip_projector.pt` / `detr_projector.pt` from Darya's checkpoint dir
(`EchoVLM.load_projectors`'s save format) into the new projector attributes,
instead of leaving them randomly initialized.

**Risk verified this session, resolved — partial coverage is EXPECTED, not a
blocker:** Darya's h5 files are keyed by `dicom_uuid` from her own data
split. A real coverage check (implementation plan Task 1) found only
~40% of this project's `rl.jsonl`/`eval.jsonl` per-(study, view) pool lands
in her clip-token h5, on both splits, uniformly (every study partially
covered, 0% studies fully covered, 0% studies with zero coverage). Initial
read: this looked like the already-documented EchoPrime video-cache
partial-coverage gotcha in CLAUDE.md (a real blocker). It is not — verified
directly against `report_generation/sft_thinking/dataset.py`'s
`__getitem__` (lines ~195-207): it writes `f"{view_name}:\n"`
unconditionally but only appends the 393 clip-token placeholders
`if dicom_uuid in self.clip_h5`, silently omitting the block (view label
stays, tokens don't) otherwise. **Darya's own SFT checkpoint was trained on
exactly this partial-coverage pattern** — it is the cache's normal, expected
shape, not a pipeline bug or an incomplete build. §1's view-block
construction (and Task 7 of the implementation plan) must replicate this
guard exactly: skip clip/DETR placeholders for any view whose dicom isn't
in Darya's h5, keep the view-name text line regardless. Do NOT require or
gate on full coverage.

## 2. Tools: `select_frames`, `zoom` — no `select_view`

Unlike the Qwen3-VL `echo` tool, there is no `select_view`: all views are
already given in full at turn 0 (per §1), so there's no "pick a view" step.
Both new tools are **stateless** — `view` is a required argument on every
call, not implicit state from a prior `select_view`.

- `select_frames(view, frame_indices)`: map requested frame indices (0-15) to
  their temporal group `g = frame_idx // 2`. **Capped at 1 group per call**
  (49 tokens) — if requested indices span multiple groups, use the group of
  the first valid index and say so in the tool response text, mirroring how
  `tool_env/tools.py::select_frames` already truncates to `cfg.n_highres_frames`
  today.
- `zoom(view, bbox, frame_indices)`: resolve the temporal group as above (1
  group), then subselect the spatial cells of that group's 7×7 grid whose
  center falls inside `bbox` (bbox in the same normalized-`[0,1]` convention
  `tool_env/bbox.py::normalize_bbox` already uses elsewhere in this repo, for
  consistency — reuse/adapt that module rather than inventing a new
  convention).
- Both return **existing, precomputed** tokens (indexing into the study's
  already-loaded `(393, 768)` grid) — no live EchoPrime encoder inference at
  rollout time. Projection through `clip_projector` still happens per call
  (it's the trainable half), but that's cheap relative to encoder inference.

## 3. Agent loop: new multi-turn loop

`EchoPrimeAgentLoop` is single-turn by explicit design today. This needs a
new loop (or an extension) that:
1. Builds the turn-0 prompt per §1 (all views, clip + DETR tokens) and
   generates, same as today.
2. Detects a `<tool_call>{"name": "select_frames"|"zoom", "arguments": {...}}
   </tool_call>` in the generated text (Hermes format, matching the
   Qwen3-VL track's convention for consistency, even though this loop is
   fully custom).
3. Resolves the tool call against the study's cached grid (§2), projects the
   selected tokens, and continues generation with the new placeholder run
   appended — same `image_data`-as-precomputed-embeddings mechanism already
   proven in `echoprime_agent_loop.py`'s single call, invoked again for the
   continuation.
4. Bounds total turns (exact budget TBD in the implementation plan — mirror
   whatever cap `tool_env`/`budget.py` uses for the image track unless there's
   a reason to diverge).
5. Bans `VIEW_TOKEN`/clip-placeholder/detr-placeholder ids from being sampled
   as free text, same reasoning as the existing `logit_bias` ban in
   `echoprime_agent_loop.py` (untrained cold-start policy can sample special
   tokens as ordinary output).

## 4. Data scope: verifiable question types only

**Decision (explicit user requirement):** a tool-use bonus tied to "the
answer was correct" is only meaningful if "correct" is itself verifiable.
Today, `score_outcome`'s `text` branch (`structure_description`, `conclusion`,
`full_report` — 44.9% + 3.9% + 3.7% = 52.5% of the current pool) falls back to
`_score_text` = pure keyword-stem `entity_F1` whenever there's no structured
`gold` dict, since `NullJudge` always returns `None`. That's the exact
mechanism already implicated in the documented reward-divergence bug
(SPEC.md's `grpo-2b-video-0910-1319` run) — it's gameable, not a real
correctness check.

**This track's training/eval data is therefore restricted to `yesno`
(`abnormality_classification`) and `set` (`abnormality_list`) question types
only** — the two kinds with deterministic, non-gameable outcome scoring.
`structure_description`/`conclusion`/`full_report` are excluded from this
track's parquet generation until a real judge (P3e) makes their outcome
scoring verifiable, at which point they can be reconsidered. This roughly
halves the usable pool (need to recompute the actual study/row counts from
`build/rl.jsonl` once filtered — don't assume the SPEC.md question-type-mix
percentages hold exactly at the study level).

This is a data-generation-time filter (`generate_grpo_parquet.py` /
`build_grpo_parquet.sh` equivalent for this track), not a reward-time
discard — don't generate rows for excluded question types at all.

## 5. Reward

Cross-checked against Darya's own GRPO reward (`report_generation/grpo/rewards.py`)
this session. Her composite is `r_cor * r_fmt + [nli terms] + r_len`, and none of
her variants (Exp A-D) ground against DETR or a real judge either for
full_report/structure_description/conclusion (ROUGE-L or EchoPrime-cosine —
still text-similarity proxies, consistent with why her own GRPO step barely
moved GREEN over her SFT: 0.795 -> 0.799, macro over 13 sections, her own
GREEN prompt). Decisions below, confirmed one-by-one with the user:

1. **Outcome — `yesno`**: exact match, unchanged from current `score_yesno`.
2. **Outcome — `set`**: switch from the current open free-text
   `data_core.data.answers.finding_set` (arbitrary bullet-line substrings) to
   Darya's **closed 11-canonical-finding taxonomy + alias table + IoU**
   (`report_generation/grpo/rewards.py::CANONICAL_FINDINGS`, `_ALIASES`,
   `_extract_finding_set`, IoU not F1) — port these into `data_core`. Harder
   to game: bounded vocabulary match, not raw substring presence. This is
   the SAME 11 categories `packages/eval/diseases.py::disease_of` already
   uses at question level; now also used at per-finding level inside list answers.
3. **Format**: keep ours (`<think>` present + valid `<answer>` or
   `<tool_call>`) — not Darya's simpler "`</think>` appears once", since ours
   already accounts for tool-call turns and hers has none.
4. **Length penalty**: not adopting Darya's overshort penalty. No observed
   too-short-response problem in our own runs to justify it; revisit if one
   shows up.
5. **NLI self-consistency (her Exp C/D)**: not adopting. It checks whether
   `<think>` entails the answer — self-consistency, not correctness against
   ground truth or real evidence. Doesn't clear the "must be verifiable" bar.
6. **`text` kind**: moot — excluded from this track's data per §4.

Replace the current flat `tool_bonus = coef if tool_calls >= 1 else 0` with
kind-specific treatment. `total_reward`'s signature grows to accept whatever
the `set`-kind grounding term needs (see below); exact parameter shape is an
implementation-plan detail, but the three behaviors are fixed by this spec:

- **`yesno`**: `tool_bonus = tool_bonus_coef * outcome`. Since `outcome` is
  already binary (0/1) for this kind, this literally implements the
  DeepEyes-style rule: tool use is only rewarded when it went with a correct
  answer. No new plumbing — `outcome` is already computed.
- **`set`** (`abnormality_list`): `tool_bonus = tool_bonus_coef *
  mean(IoU over findings with a mapped RT-DETR class)`. For each line in the
  predicted finding list, map its structure to an RT-DETR class via a new
  keyword-rule table (`finding_to_detr_class`, same pattern as
  `packages/eval/diseases.py::disease_of` — ordered regex rules, first match
  wins), covering the 7 RT-DETR classes. Findings with no mapped class (e.g.
  anything aortic-valve — RT-DETR has no aortic valve class) contribute
  nothing to the average rather than being penalized. For each mapped
  finding, take the model's `zoom`/`select_frames` tool-call args referencing
  that finding's view (parsed from `<tool_call>` blocks in the completion,
  same regex `verl_bridge/reward.py::_TOOLCALL_RE` already uses for counting)
  and compute bbox IoU against the DETR-detected box for that class in the
  chosen frame(s). No match / no tool call touching that view → IoU 0 for
  that finding.
  - **This crosses `data_core.reward.score`'s current "pure, model-free, no
    I/O" contract.** The reward path needs the study/dicom's DETR boxes,
    which live in an h5 file, not in `reward_key`. Thread it through
    `extra_info` (already plumbed end-to-end per `verl_bridge/reward.py`,
    currently only carrying `tool_bonus_coef`) — add `dicom_uuid`-per-view
    (or equivalent) at parquet-generation time, and have `compute_score`
    (not `data_core.reward.score`, to keep that module's no-I/O property
    for its existing callers) open the relevant `_detections.h5` group and
    pass resolved boxes into `total_reward`.
- **`text`**: unchanged from today — moot in practice once §4's data-scope
  restriction lands, since no text-kind rows should reach this track's
  reward function at all. Keep the flat-bonus code path only as the
  pre-existing behavior for any other caller of `total_reward`, not as an
  active choice for this track.

**Also:** per-question-type reward normalization (z-score or rank-normalize
within `yesno` vs `set` before pooling into one GRPO batch) — cheap, targets
the exact symptom already measured (one channel's reward trend swamping
another's in the pooled number). Where exactly this lives (inside
`total_reward`, or in the batch-level code that calls it per-trajectory) is
an implementation-plan detail.

## Open questions for the implementation plan (not blocking this spec)

- Exact per-episode turn/token budget for the new multi-turn loop.
- Exact `extra_info` schema addition for per-view `dicom_uuid` (parquet
  column shape).
- Whether `finding_to_detr_class`'s keyword rules need clinical review before
  trusting the grounding reward, given `diseases.py`'s existing rules are
  question-text rules, not finding-text rules (different vocabulary distribution).
