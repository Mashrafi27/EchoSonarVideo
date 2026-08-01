# EchoSonarVideo — Phase 3 Training Integration — Execution Design

**Status:** approved decomposition (2026-08-01). This is an *execution* design for an
already-approved system design — the architecture lives in
`docs/superpowers/specs/2026-07-23-echo-deepeyes-rl-training-design.md` (§5–§8) and
`echo_env/INTEGRATION.md`. This doc fixes **decomposition, sequencing, and the
verification policy** for the work that turns those contracts into training-ready code.

Branch: `phase3-training` (base `main` @ `f288e93`).

---

## 0. Framing decisions (locked)

- **Base model:** Qwen3-VL-8B-Instruct (32px merged visual patch; native 320 target).
- **Frame path:** **TRUE VIDEO** (user, 2026-08-01) — the `<video>` observation seam is on
  the critical path, not image-multi-frame.
- **VeRL Qwen3-VL support:** **(A) version-bump = project-owned rebase** onto upstream
  `volcengine/verl` (target v0.6.0 *or* v0.7.x — pinned by gate G2 below) + `transformers>=4.57.0`,
  re-applying DeepEyes' patch set. NOT a submodule-pin advance (DeepEyes upstream never rebased).
  Rationale + conflict map + video-token seam: `echo_env/INTEGRATION.md` §0/§2, spec Risk 9.
- **Scale reality:** nothing in Phase 3 runs offline. Deliverables are labeled by verification
  tier (§2); the GPU run is the real gate. A green reviewer verdict confirms *contract alignment*,
  never *working integration*.

## 1. Decomposition — five sub-projects, each its own spec→plan→build

| # | Sub-project | Depends on | Dominant tier |
|---|---|---|---|
| **P3a** | **Offline echo-training core** — SFT trajectory synthesizer + reward pure-scoring functions + RL/SFT data-gen (parquet) | Phase-1 `echo_rl` builders only | 🟢 offline-testable |
| **P3b** | **VeRL Qwen3-VL enablement** — target verl tree overlay + `transformers>=4.57` + re-applied model-mRoPE patches + `Qwen3VLImageProcessor` branching | gates G1/G2 | 🟡 / 🔴 |
| **P3c** | **Echo integration wiring** — echo tool-env adapter (native or ToolBase) + video patch (`qwen3_vl.get_rope_index`/`video_metadata`) + reward-scorer registration | P3a, P3b | 🟡 (🔴 if port) |
| **P3d** | **Cold-start SFT** — SFT config + launch + run | P3a, P3c | authored 🟡, run = GPU |
| **P3e** | **GRPO RL + eval** — reward stack wiring + LLM-judge + GRPO config + eval harness + ablations | P3d | authored 🟡, run = GPU |

P3a is unblocked and built **first**. P3b/P3c are gated on the two research questions below.

## 2. Verification tiers (every deliverable carries its tier)

- **🟢 Offline-testable → genuinely DONE.** Same bar as Phases 1–2: real unit tests, real green.
  Covers all of P3a (pure-Python, reuses `echo_rl`, no model/runtime).
- **🟡 Authored-against-real-tree → "applies-clean, imports-resolve, UNRUN."** Done =
  applies to the *fetched* target verl tree + `py_compile`/imports resolve + token/field names
  match the research-pinned facts (`second_per_grid_ts` removed; `<|video_pad|>`,`<|vision_start|>`,
  `<|vision_end|>`,`video_grid_thw` unchanged) + reviewer confirms contract alignment. **Never "done."**
- **🔴 Not honestly authorable blind → runbook, not code.** The agent-layer port *if* gate G1
  = PORT. Deliverable is a **rebase runbook + conflict inventory + integration points** grounded
  against the real tree — NOT fabricated agent-loop code (which reads as delivered but sends a GPU
  operator debugging fiction; negative value).

## 3. Gates before P3b/P3c authoring (research resolving now)

- **G1 — native-vs-port.** Does the target verl's *native* agent/tool-env framework expose a
  ToolBase-equivalent registration + reset/execute seam? If **native**, P3c's adapter is a thin
  🟡 registration and the 🔴 port evaporates. If **port**, the agent layer is 🔴 (runbook).
- **G2 — target version.** v0.6.0 (earliest Qwen3-VL tag) vs latest v0.7.x (more qwen3vl fixes +
  agent-framework maturity). Pin **before** any rebase authoring — wrong pin = heavy work twice.
- **Enabler:** fetch the real target verl tree onto disk (scratchpad) so 🟡 gates are mechanical
  (`patch applies`, `imports resolve`) rather than inferred.

## 4. P3a interfaces (the buildable-now slice)

All in the `echo_rl` package (stdlib-only) or a new `echo_train` package; no DeepEyes/torch import.

1. **Trajectory synthesizer** (`echo_rl/sft/synthesize.py`) — rewrites each per-view `thinking`
   report (`#### <View> → Detected Structures → Clinical Findings → Implications`) into a
   tool-using SFT trajectory: `<think>…</think><tool>select_view("…")</tool><obs>…</obs>…<answer>…</answer>`
   (spec §5.1). Pure text transform over the Phase-1 records. Light (format-focused). Fully 🟢.
2. **Reward pure-scoring** (`echo_rl/reward/score.py`) — `question_type`-dispatched outcome scorers:
   yes/no exact-match (+ balancing hook), set-F1 over normalized finding names, clinical-entity-F1,
   gold-CSV value match; plus a format-reward check for `<think>/<answer>` + tool syntax (spec §5.2).
   Model-free (LLM-judge is a later 🟡 client, stubbed behind an interface here). Fully 🟢.
3. **Data-gen** (`echo_rl/build/trainset.py`) — emits SFT trajectories + RL prompts joined to the
   **canonical split** (train-only; the leakage trap, spec §6) with class-balanced abnormal-enriched
   RL pool. Reuses Phase-1 builders. Parquet layout (`data_source="echo"`, `env_name="echo"`,
   `videos` column, `reward_model.ground_truth`, `extra_info` incl. `study_uuid`) authored to the
   §8 contract but the parquet-schema-vs-verl-loader match is 🟡 (verified against fetched tree).

## 5. Out of scope for Phase 3

Real GPU training/eval runs (flagged, not executed); a real echo segmentation model (spec Risk 1 —
coarse region-prior fallback only); re-preprocessing PNGs 336→320 (processor smart-resizes at load).

---

## Change log
- 2026-08-01: created. Decomposition + verification tiers approved; P3a first; G1/G2 gating P3b/P3c.
