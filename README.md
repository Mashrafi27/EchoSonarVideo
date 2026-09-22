# EchoSonarVideo

Current direction (2026-09-22): basic inference with the released DeepEyes
checkpoint on a few held-out echo QA pairs. All earlier training/evaluation
tracks are past work retained for reference. See [PLAN.md](PLAN.md).

Agentic RL on multi-view cardiac ultrasound video: cold-start SFT -> GRPO, on
upstream verl (pinned submodule `external/verl` @ v0.7.1, NOT forked). Two
tracks share one repo — see `SPEC.md` for which config/checkpoint goes with
which.

Start with `CLAUDE.md` (operational rules and known pitfalls), `SPEC.md`
(data/architecture/results facts), and `PLAN.md` (current next steps).
`AGENTS.md` governs where files belong.

## Layout

| Path | Contents |
| --- | --- |
| `packages/` | The five in-repo Python libraries below, grouped under one folder. |
| `packages/tool_env/` | Tool environment: observations, frame/view selection, zoom, budgets, parsing. See `packages/tool_env/INTEGRATION.md`. |
| `packages/data_core/` | Data preparation, reward logic, SFT serialization (`data/`, `reward/`, `sft/` subpackages). |
| `packages/verl_bridge/` | verl integration: tool/session adapters, native-vision training/rollout wiring, `packages/verl_bridge/configs/` YAMLs for both tracks. |
| `packages/echoprime_track/` | Frozen-EchoPrime + Qwen3-8B-text track: encoder, projector, dataset, SFT/GRPO training, rollout. |
| `packages/eval/` | Unified evaluation for both tracks: metrics, NLG/BERTScore/GREEN, EchoSonar-R Table 1-3 comparisons, run/score CLI drivers. `packages/eval/echoprime/` holds the EchoPrime track's checkpoint-eval scripts. |
| `scripts/` | Launchers and operator commands (`build_*`, `run_*`, `check_*`). |
| `docs/` | Human documentation: `OPEN_ISSUES.md`, `meetings/`, `references/` (as needed), `experiments/` (as needed). |
| `external/` | Pinned submodules (`verl`, `DeepEyes`) and their patch files — clone with `--recurse-submodules`. |
| `build/`, `runs/`, `data/raw_vqa/` | Generated/local artifacts, gitignored. Not backed up by this repo. |
| `.claude/skills/` | Project skills, including `repo-cleanup` (this layout's own maintenance skill). |

## Entry commands

```bash
# Validate the training environment before any SFT/GRPO launch
python scripts/check_train_env.py

# Build the RL/eval parquet (echo_grpo track)
bash scripts/build_grpo_parquet.sh

# Launch GRPO — echo_grpo track (Qwen3-VL, tool-based)
bash scripts/run_grpo.sh

# Launch GRPO — echoprime_track track (frozen EchoPrime + Qwen3-8B-text, no tools)
bash scripts/run_echoprime_grpo.sh

# Tests (packages/tool_env, packages/data_core, packages/verl_bridge, packages/eval; echoprime_track has no test suite yet)
python -m pytest
```

## Where should I put a new file?

| I'm adding... | It goes in |
| --- | --- |
| Tool-environment logic (view/frame/zoom, observation packing) | `packages/tool_env/` |
| Data prep, reward scoring, or SFT serialization shared by both tracks | `packages/data_core/` (its `data/`, `reward/`, or `sft/` subpackage) |
| verl adapters, tool/session wiring, native-vision training/rollout code | `packages/verl_bridge/` |
| A verl YAML config | `packages/verl_bridge/configs/` |
| Frozen-EchoPrime track code (encoder, projector, dataset, training, rollout) | `packages/echoprime_track/` |
| Evaluation metrics, NLG/BERTScore/GREEN, EchoSonar-R comparisons, eval CLI drivers | `packages/eval/` (or `packages/eval/echoprime/` for EchoPrime-track checkpoint eval) |
| A launcher or one-off operator command | `scripts/`, named `build_*`/`run_*`/`check_*`/`score_*`/`eval_*` |
| A test | `<owning package>/tests/test_<behavior>.py` |
| A meeting note or slide deck | `docs/meetings/YYYY-MM-DD_<topic>.md` |
| A curated experiment summary | `docs/experiments/YYYY-MM-DD_<experiment>.md` |
| A durable operational rule / pitfall | `CLAUDE.md` |
| A research fact (data, architecture, results) | `SPEC.md` |
| A current open question / next step | `PLAN.md` or `docs/OPEN_ISSUES.md` |

Full placement and naming rules: `AGENTS.md`.
