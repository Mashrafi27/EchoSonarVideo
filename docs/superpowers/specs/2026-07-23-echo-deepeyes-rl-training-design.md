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
- **Model:** Qwen2.5-VL-7B (DeepEyes default; native multi-image/video; single-node inference).
- **Recipe:** **light cold-start SFT → outcome-reward RL (GRPO)** on a VeRL/DeepEyes fork.
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
   Stage 1: Cold-start SFT (Qwen2.5-VL-7B)     synthetic agentic trajectories (§5.1)
   teach tool format + look-then-reason habit + echo vocabulary   [LIGHT, ~1 epoch]
                                                     │ checkpoint
                                                     ▼
   Stage 2: GRPO RL on VeRL (DeepEyes fork)
     ├─ Echo Agent Environment (verl/workers/agent/)  ← serves PNG frames on tool calls (§4)
     ├─ Reward: rule + LLM-judge + gold-CSV + format + annealed tool bonus (§5.2)
     └─ vLLM LLM-judge (Qwen) for free-text rewards
                                                     │
                                                     ▼
   Eval harness on held-out studies (§7): answer quality + tool-use + view-selection
```

- **Fork DeepEyes/VeRL**; reuse Ray multi-node + vLLM rollout infra nearly unchanged.
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
- **Study-level split** train/val/test; reuse provided `test_vqa.jsonl` where study-disjoint.
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

## 8. Repo / Infrastructure Change Map (DeepEyes fork)

| Area | Change |
|---|---|
| `verl/workers/agent/` | **New Echo Agent Environment**: initial-observation builder (thumbnails), the 3 tools, PNG frame/crop server reading `preprocessed_data`, budget guardrails |
| `verl/trainer/config/` | Qwen2.5-VL-7B configs for SFT and GRPO; tool/observation limits; judge endpoint |
| `examples/agent/` | Launch scripts (SFT then RL), multi-node GPU + wandb + vLLM-judge wiring |
| `eval/` | Echo eval harness (§7); replace DeepEyes' bbox eval |
| new `data/` tooling | Trace-breakdown trajectory synthesizer; RL prompt builder; study-level split + balancing |
| LLM-judge | vLLM-served Qwen judge with an **echo-specific** rubric/prompt |

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
5. **Frame resolution** not yet confirmed (PIL unavailable in check env) — verify native PNG size to
   set thumbnail/high-res tiers.
6. **Context budget** with multi-frame observations + spatiotemporal zoom — tune caps empirically.
7. **Environment is not a git repo** — initialize before committing this spec / the fork.

---

## 10. Suggested Phasing

1. **Data plumbing** — builder + trajectory synthesizer + splits + balancing; sanity-check a few
   studies end-to-end.
2. **Echo Agent Environment** — 3 tools + frame server + guardrails, unit-tested offline.
3. **Cold-start SFT** — light, verify format + look-then-reason emerge.
4. **GRPO RL** — reward stack + judge; watch for prior-parroting; tune annealed bonus + balancing.
5. **Eval + ablations** — confirm tools/balancing beat baselines.
