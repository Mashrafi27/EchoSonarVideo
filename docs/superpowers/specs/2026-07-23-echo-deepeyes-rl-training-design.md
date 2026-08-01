# EchoSonarVideo — DeepEyes-style "Think-with-Echo" Training — Design

- **Date:** 2026-07-23
- **Status:** Draft for review
- **Base method:** [DeepEyes](https://github.com/Visual-Agent/DeepEyes) (agentic "think with images" RL on a VeRL fork)
- **Owner:** mohammad.yaqub / anaatef9@gmail.com

---

## 1. Goal & Context

Train a vision-language model that **actively looks at echocardiography video** while reasoning —
selecting views, picking informative frames, and cropping regions as tool calls interleaved into its
chain-of-thought — and produces echo VQA answers (structure descriptions, abnormality
classification, reports). We adapt DeepEyes (which crops *static high-res photos*) to **multi-view
cardiac ultrasound video**, whose signal lives in **view choice, motion, and Doppler**, not fine
static crops.

### Base model & method (decided)
- **Model:** **Qwen3-VL-8B-Instruct** (user decision 2026-07-24, superseding the original Qwen2.5-VL-7B
  DeepEyes default). Chosen because Qwen3-VL is **video-native** (timestamp-aware video, interleaved-MRoPE)
  — a direct fit for echo cine, and the reason the image-only video patch (§8) may *shrink*. **Cost:** the
  pinned DeepEyes/VeRL fork is Qwen2.5-VL-only (vendored VeRL `0.2.0.dev`, `transformers==4.51.3`, zero
  `qwen3` refs), so the base-model swap is a **transformers bump (≥~4.57) + a VeRL model-support port**,
  not a config flag — see Risk #9. This grows the model-support surface even as the video patch shrinks.
- **Frame geometry (decided):** Qwen3-VL merged visual patch = patch 16 × spatial_merge 2 = **32px** (vs
  Qwen2.5-VL's 28px). Native frame side must be a 32-multiple → **320×320** (`min_crop_side=32`,
  `highres_max_side=320`, `preview_max_side=160`). Phase-1 PNGs stay 336×336; the processor smart-resizes
  to the nearest 32-multiple at load. Re-preprocessing to native 320 is a Phase-3 data decision.
- **Recipe:** **light cold-start SFT → outcome-reward RL (GRPO)** on DeepEyes/VeRL, integrated as a
  **pinned git submodule** (`external/DeepEyes`) — pristine upstream + a versioned patch set, not a fork.
- **Compute:** large multi-node (32+ GPUs) — DeepEyes-scale RL is feasible as-is.

---

## 2. Verified Data Facts (ground truth for this design)

**Text / QA** (`Archive 2 (1)/`):
- `train_vqa_with_thinking.jsonl` — 128,215 QA records over **5,061 unique studies** (~25 QA/study).
- `test_vqa.jsonl` — 31,209 QA records (held-out).
- Each train record: `study_uuid`, `question_type`, `thinking` (a long **per-view** structured
  report), `messages` (system/user/assistant chat; last assistant turn = **target answer**).
- Question types: `structure_description` (~45%), `abnormality_classification` (~43%),
  `abnormality_list`, `conclusion`, `full_report`.

**Pixels** (`../preprocessed_data/`, i.e. `/vast/users/mohammad.yaqub/project/preprocessed_data/`):
- 75,356 per-study folders keyed by `study_uuid`; **100% of the 5,061 train studies join.**
- Each study → **~19 clips** (min 12, max 31), one subfolder per clip named
  `di-XXXX_<ViewName>` — **view label is baked into the folder name** (PLAX Standard, A4C,
  A3C, A2C, PSAX Mitral/Papillary/Apex, RVIT, A5C, Subcostal, SUB IVC, Suprasternal…).
- Each clip = **cine loop pre-extracted to PNG frames** `0.png … N.png`, **~42 frames/clip**
  (min 11, max 140). No video decoding needed.
- Studies mix **standard B-mode and color-Doppler** clips.
- **No bbox/segmentation metadata and no ES/ED phase labels** exist — only PNG frames.

**Structured gold labels** (`../output_with_labels/output/*.csv`): study-level ground truth —
`ejection_fraction_regression`, `aortic_regurgitation_classification`, `e_a_ratio_regression`,
`heart_failure_classification`, `ivc_classification`, etc. Usable as **reliable RL reward targets**.

### Two data risks this design must handle
1. **Per-view findings are partly replicated study-wide.** Over 2,000 multi-view studies, only 6%
   had identical findings across *all* views, but adjacent similar views (e.g. "PLAX Standard" vs
   "PLAX Mitral Cusps") are often byte-identical, and a view's "Clinical Findings" reads like a
   study-level diagnosis copied into each section. ⇒ **Synthetic SFT trajectories teach *behavior/format*,
   not ground-truth visual grounding.** View-selection supervision is only clean for
   `structure_description` (question targets a specific structure → specific view).
2. **Low answer entropy / prior-answerable.** `abnormality_classification` is 77% "No" / 23% "Yes"
   with only ~22 unique templated answers; everything is normal-heavy. ⇒ Naïve outcome-reward RL
   collapses to a **prior-parroting policy that never looks at pixels**. The RL split must be
   **class-balanced / abnormal-enriched** (§6).

---

## 3. System Architecture

```
preprocessed_data/st-*/di-*_<View>/N.png   ┐
train_vqa_with_thinking.jsonl (128k QA)    ┼─► Data builder (join on study_uuid)
output_with_labels/*.csv (gold)            ┘        │
                                                     ▼
   Stage 1: Cold-start SFT (Qwen3-VL-8B)       synthetic agentic trajectories (§5.1)
   teach tool format + look-then-reason habit + echo vocabulary   [LIGHT, ~1 epoch]
                                                     │ checkpoint
                                                     ▼
   Stage 2: GRPO RL on VeRL (DeepEyes submodule + echo overlay)
     ├─ Echo Agent Environment (new files, registered by import)  ← serves PNG frames on tool calls (§4)
     ├─ Reward: rule + LLM-judge + gold-CSV + format + annealed tool bonus (§5.2)
     └─ vLLM LLM-judge (Qwen) for free-text rewards
                                                     │
                                                     ▼
   Eval harness on held-out studies (§7): answer quality + tool-use + view-selection
```

- **DeepEyes as a pinned submodule** (`external/DeepEyes`), not a fork; reuse Ray multi-node + vLLM
  rollout infra nearly unchanged. Upstream stays pristine; our changes live in two layers:
  (a) **net-new files** in our own package (the 3 tools, reward scorer, data-gen, launch/config) that
  DeepEyes picks up by import-triggered registration; (b) a small **versioned patch set** for the one
  irreducible in-tree change — video observations aren't plumbed through DeepEyes today (image-only),
  so `verl/workers/agent/parallel_env.py` (obs round-trip) and `verl/utils/dataset/rl_dataset.py`
  (`origin_multi_modal_data["video"]`) need edits, applied to the submodule at setup.
- **New component:** an **Echo Agent Environment** replacing DeepEyes' single-image crop env; it
  hosts the three tools and returns PNG frames/crops from `preprocessed_data`.
- **Splits are at the study level** (all QA of a study stay on one side) to prevent leakage.

---

## 4. Echo Visual Environment & Tools

**Initial observation (cheap):** one mid-cycle thumbnail per available labeled view (~19), each tagged
`view_name` + `frame_count` (e.g. `A4C: 42 frames`). Unlabeled clips are discarded.

**Three free-form, model-driven tools** (space × time):

1. **`select_view(view_name)`** → returns a **sparse temporal preview**: ~4–6 evenly-spaced,
   low-res frames spanning one cardiac cycle, so the model sees motion and judges which phase matters.
   *(the "which clip / rough look" action)*
2. **`select_frames(view_name, [i, j, k, …])`** → returns those **specific whole frames at higher
   resolution** — the temporal analog of zoom; the model picks the informative moments.
3. **`zoom(view_name, bbox, frame_indices=[…])`** → one bbox applied across a model-chosen set of
   frames — a **spatiotemporal crop**. 1 index = static high-res crop (dimension/morphology);
   N indices = region-over-time (leaflet motion, segmental wall motion, jet dynamics).

**Design rationale:** DeepEyes' core is "the model decides where to look." Echo adds a **time** axis,
so the model must also decide *when*. All three actions are free-form and taught purely by outcome
reward — **no segmentation and no ES/ED labels required.** Constrained/segmentation-based zoom is a
documented *fallback* only if free-form grounding fails on grayscale echo (coarse per-view region
priors — no segmentation model); a real echo segmenter is **not** to be built preemptively (caps the
ceiling behind an imperfect model).

**Guardrails:** ≤ ~6 tool calls/episode; ≤ ~4 frames per `zoom`; a global frames-in-context cap so
the policy cannot pull the full ~800-frame study.

---

## 5. Training Recipe

### 5.1 Stage 1 — Cold-start SFT by "trace breakdown"

We have **outcome labels** (target answers + gold CSVs) but **no gold tool-use trajectories**. So SFT
is *synthesized*, not real imitation, and is kept deliberately light.

**Trajectory synthesis:** the `thinking` report is already structured per view
(`#### <View> → Detected Structures → Clinical Findings → Implications`). Programmatically rewrite each
into a tool-using trajectory:

```
<think>To answer this I should examine the relevant views.</think>
<tool>select_view("PLAX Standard")</tool> <obs>[frames]</obs>
<think>PLAX shows LA, LV, LVOT. LV normal dimensions and systolic function…</think>
<tool>select_view("A4C")</tool> <obs>[frames]</obs>
<think>A4C confirms RV normal, mild TR…</think>
<answer>…study target answer for this question_type…</answer>
```

Teaches: (a) tool-call **format**, (b) **look-then-reason** habit, (c) echo **vocabulary**,
(d) grounding reasoning to a **specific view**.

**Kept light (~1 epoch, format-focused)** because per-view findings are partly replicated study-wide
(§2 risk 1): SFT is a **behavioral prior**, not ground truth. RL does the real optimization against
verifiable rewards. *(Alternatives considered: pure RL (risk: cold-collapse on OOD, low-entropy data);
no-tool SFT (weaker — RL must find tools cold). Both rejected in favor of light synthetic-trajectory SFT.)*

### 5.2 Stage 2 — GRPO RL (outcome reward)

**Algorithm:** GRPO (critic-free; matches DeepEyes; stable for 7B multi-node).

**Reward = outcome + format + annealed tool bonus:**

| `question_type` | Outcome reward |
|---|---|
| `abnormality_classification` (yes/no) | rule-based exact match *(+ balancing, §6)* |
| `structure_description`, `conclusion`, `full_report` (free text) | **LLM-judge** (vLLM Qwen) semantic match to reference **+ clinical-entity F1** co-signal (extract findings, compare sets) |
| `abnormality_list` (set) | **set F1** over normalized finding names |
| where a gold CSV exists (EF, regurg grade, E/A…) | reward against the **structured gold value** (most reliable) |

- **Format reward:** valid tool syntax + well-formed `<think>/<answer>`.
- **Annealed tool-use bonus:** small early reward for ≥1 view tool call, decayed over training —
  prevents the no-look collapse while letting outcome reward take over. Directly counters §2 risk 2.

---

## 6. Data Building, Splits, and the Pixel-Necessity Fix

- **Builder** joins `study_uuid` → clip folders → view list + frame counts; emits (a) SFT trajectories
  (§5.1) and (b) RL prompts (question + initial observation, reward key).
- **Study-level split — use the canonical file, do NOT invent one.** The authoritative split is
  `../pretraining/data/echojepa_study_split_full.csv` (`study_uuid,split` → TRAIN/VAL/TEST; covers
  100% of our study_uuids). **Leakage trap:** `train_vqa_with_thinking.jsonl` is NOT pre-filtered —
  of its 5,061 studies, only 4,028 are canonically TRAIN; **529 are TEST and 504 are VAL**
  (`test_vqa.jsonl` is cleanly all-TEST). So **both SFT and RL must join to the canonical split and
  keep only TRAIN studies**, or ~1,033 held-out studies leak into training. (Same split mechanism the
  sibling EchoJEPAv2 / LeWorldModel projects use via a holdout-uuids list.)
- **Pixel-necessity / balancing (non-negotiable):** the RL sampling pool is **class-balanced and
  abnormal-enriched** — oversample "Yes"/abnormal cases and rare findings so a prior-parroting policy
  scores *badly*. Keep the **full** set for SFT; curate a **harder, pixel-necessary subset for RL**.
  Optionally filter to questions where the answer is not derivable from study-wide base rates.

---

## 7. Evaluation

- **Held-out studies** (study-disjoint from train).
- **Answer quality per `question_type`:** accuracy + balanced accuracy / F1 for classification (report
  both, since "always No" ≈ 77%); LLM-judge score + clinical-entity F1 for free text; MAE vs gold CSVs
  for EF / E/A.
- **Agentic behavior:** tool-call rate, tools-per-episode, and **view-selection accuracy** on
  `structure_description` (question's target structure → expected view — the one clean supervisable case).
- **Ablations:** (i) no-tool baseline (SFT-only, Option 3), (ii) pure-RL (Option 1), (iii) balanced vs
  unbalanced RL pool — to confirm tools + balancing actually improve over prior-parroting.

---

## 8. Repo / Infrastructure Change Map (DeepEyes submodule + echo overlay)

DeepEyes lives at `external/DeepEyes` (pinned submodule, upstream pristine). Changes are grouped by
**how they live**: **[new]** = net-new file in our own package, picked up by import-triggered
registration (no upstream edit); **[patch]** = part of the versioned patch set applied to the
submodule at setup (the irreducible in-tree edits). Verified extension points from the architecture
map (commit `11d20c6`) noted inline.

| Area | Change | Lives as |
|---|---|---|
| Echo Agent Environment | **New tool env** `envs/echo/echo_env.py`: `class EchoEnv(ToolBase)`, `name="echo"`, `reset()`/`execute()` dispatching `select_view`/`select_frames`/`zoom` from `<tool_call>{"name",...}</tool_call>` JSON (mirrors `visual_toolbox_v5.py`). Initial-observation thumbnail builder, PNG frame/crop server over `preprocessed_data`, budget guardrails. **Built & unit-tested offline first** (integration-agnostic). | **[new]** |
| Tool registration | One `import` line triggers metaclass registration (`ToolBase.registry`). Prefer importing `echo_env` from our launch/entry module so upstream `verl/workers/agent/__init__.py` stays untouched; fall back to a `__init__.py` patch only if import ordering forces it. | **[new]**, patch only if forced |
| **Video observation round-trip** | `parallel_env.py`: `_preprocess_multi_modal_inputs` + obs merge-back are **image-only** today; add a `<video>` → `<\|vision_start\|><\|video_pad\|><\|vision_end\|>` branch calling `processor(videos=...)` and merge `video_grid_thw`/`second_per_grid_ts`. **The one irreducible in-tree edit.** | **[patch]** |
| **Origin video frames** | `rl_dataset.py`: populates `multi_modal_data["video"]` but **not** `origin_multi_modal_data["video"]`; the echo env's `reset()` needs source-resolution frames to crop/select. Add it. | **[patch]** |
| Reward scorer | New `verl/utils/reward_score/echo.py` (judge-client pattern from `vl_agent.py`, but tool-use detector counts `<\|video_pad\|>`/tool-calls, not `<\|image_pad\|>`). Dispatch via `data_source="echo"`; prefer a wrapper/registration over editing `reward_score/__init__.py`. | **[new]**, thin patch if dispatch needs it |
| Data generation | New `envs/echo/generate_trainset.py` → parquet with `data_source="echo"`, `env_name="echo"`, `videos` column (Qwen3-VL video-dict format), `reward_model.ground_truth`, `extra_info`. Reuses the `echo_rl` builders from Phase 1. | **[new]** |
| Configs / launch | Qwen3-VL-8B SFT+GRPO configs (tool/obs limits, `agent.max_turns`, `agent.max_vllm_videos`, `tool_name_key=env_name`, judge endpoint); launch scripts from `examples/agent/train_grpo_vlagent_v3.sh` template, `export LLM_AS_A_JUDGE_BASE`. | **[new]** |
| Eval | Echo eval harness (§7); replace DeepEyes' bbox eval. | **[new]** |
| LLM-judge | vLLM-served Qwen judge with an **echo-specific** rubric/prompt. | **[new]** |

**Patch-set mechanics:** the `[patch]` edits (video round-trip in `parallel_env.py` + origin-video in
`rl_dataset.py`) are held as a versioned patch in our repo (e.g. `external/patches/echo-video-*.patch`)
and applied to the pinned submodule at setup. Keeps upstream trackable; re-pinning the submodule = re-applying/rebasing the patch.

---

## 9. Risks & Open Questions

1. **Grounding on grayscale echo may be hard for free-form `zoom`.** Mitigation: coarse region-prior
   menu fallback (no segmentation model). *Do not* build a real echo segmenter preemptively.
2. **Prior-parroting collapse** (low entropy). Mitigation: balanced/abnormal-enriched RL pool +
   annealed tool bonus + balanced-accuracy reporting. This is the single biggest project risk.
3. **Synthetic-trajectory artifact** (study-wide replicated findings). Mitigation: light SFT only;
   RL against verifiable outcomes.
4. **Free-text reward reliability** (LLM-judge variance on medical text). Mitigation: entity-F1
   co-signal + prefer gold-CSV reward wherever available.
5. *(Resolved)* **Frame resolution** confirmed: Phase-1 PNGs are 336×336. Qwen3-VL geometry sets the
   env tiers — 32px floor, 320 native target (§1). Processor smart-resizes 336→320 at load.
6. **Context budget** with multi-frame observations + spatiotemporal zoom — tune caps empirically.
7. *(Resolved)* Repo initialized; DeepEyes integrated as pinned submodule `external/DeepEyes` (commit
   `11d20c6`) + patch set, not a fork (per user decision 2026-07-24). Upstream stays pristine.
8. **Video is not plumbed through DeepEyes' rollout** (image-only obs round-trip) — an in-tree edit
   (§8 `[patch]`). Qwen3-VL is video-native, so this patch may *shrink* vs the Qwen2.5-VL plan, but the
   principle holds: mRoPE (`get_rope_index`) and `video_grid_thw`/`second_per_grid_ts` must come from the
   real HF processor, never hand-rolled, or temporal position ids break silently. Validate end-to-end
   before RL.
9. **⚠️ #1 Phase-3 risk — VeRL-fork Qwen3-VL support. *(RESOLVED to a decision 2026-08-01; execution still HEAVY.)*** The vendored VeRL (`0.2.0.dev`, `transformers==4.51.3`)
   has **zero** Qwen3-VL support. **Decision: (A) VeRL version-bump — project-owned rebase.** Overlay upstream
   VeRL **v0.6.0** (Qwen3-VL = PR #3681, Oct 2025; earliest tag v0.6.0) + pin **transformers≥4.57.0**, then
   re-apply DeepEyes' patch set on top. NOT a submodule-pin advance: DeepEyes upstream never rebased (its
   `main` *is* our pin `11d20c6`, still 0.2.0.dev), so EchoSonarVideo owns the overlay. Hand-backport rejected
   (reimplements PR #3681 on a dead base + tree-wide transformers jump). **Conflict map:** model mRoPE seam
   CLEAN (`qwen2_vl.py` survives, same signature), `monkey_patch.py` MODERATE, the `workers/agent` orchestration
   layer HEAVY (re-base onto v0.6.0's async-rollout internals). Consider v0.7.x before pinning. Full detail +
   the video-token seam (`second_per_grid_ts` removed → `qwen3_vl.get_rope_index`; `video_metadata` timestamps)
   in `echo_env/INTEGRATION.md`.

---

## 10. Suggested Phasing

1. **Data plumbing** — builder + trajectory synthesizer + splits + balancing; sanity-check a few
   studies end-to-end.
2. **Echo Agent Environment** — 3 tools + frame server + guardrails, unit-tested offline.
3. **Cold-start SFT** — light, verify format + look-then-reason emerge.
4. **GRPO RL** — reward stack + judge; watch for prior-parroting; tune annealed bonus + balancing.
5. **Eval + ablations** — confirm tools/balancing beat baselines.

---

## 11. Phase 1 Closeout (data plumbing) & carry-forward to later phases

**Status:** ✅ Complete. `echo_rl` package on branch `phase1-data-plumbing`, 41 tests pass, final
whole-branch review = **READY AS-IS**. Deliverables: `study_uuid`→frame join, per-view `thinking`
parser, synthetic agentic SFT trajectory builder (§5.1), reward-target + gold extraction, canonical
study split, class-balanced RL pool, and a `build-sft`/`build-rl`/`stats` CLI.

**Measured invariants (on real data, not just inspected):**
- **No leakage** — canonical `echojepa_study_split_full.csv` gives 4028 TRAIN / 504 VAL / 529 TEST / 0
  missing over train_vqa's 5,061 studies; default `--split train` excludes the 1,033 held-out studies.
- **Balancing works** — resampling lifts the `abnormality_classification` yes-fraction 23.2% → 38.9%,
  dropping an "always-No" policy from ~77% to ~61% accuracy.

**Carry-forward for Phase 4 (RL) — read before building the reward stack:**
1. **Balancing is GLOBAL, not per-type.** The pool balances a single binary abnormal/normal label pooled
   across *all five* question types, so classification lands at ~39% yes (not 50%). For a per-type
   "always-No ≈ 50%" guarantee, add **stratified (per-question-type / per-reward-kind) balancing** in
   Phase 4. (Phase-1's non-negotiable — "parroting scores badly" — is met; tighter balance is a Phase-4 knob.)
2. **`reward_key.target` can be `None`** for a malformed `yesno` answer that doesn't lead with yes/no —
   the Phase-4 reward scorer must handle `None` targets gracefully.
3. **Deferred Phase-1 test hardening (non-blocking, logged in `.superpowers/sdd/progress.md`):**
   persisted `parse_yes_no` boundary tests; a `max_views>4` trajectory test; `builders` coverage for the
   `set`/`text` reward-kinds + `iter_jsonl` + the gold-"designation"-strip assert.
4. **Vestigial code:** `split.load_test_studies` is unused after the canonical-split adoption; `stats`
   accepts but ignores `--split`. Safe to remove in a later cleanup.

**Note on plan vs implementation:** the plan's Task 8 documents the original gold-designation + md5-hash
`assign_split`; the shipped code correctly supersedes it with the canonical-CSV lookup per §6 (the
authoritative split). The plan markdown was not back-edited beyond the earlier `test_study_set`→
`load_test_studies` rename.
