# PLAN

## Current direction (2026-09-22)

Start again from a basic inference baseline, as requested by the user. All
previous SFT, GRPO, EchoPrime, tool-training and evaluation experiments are
**past work**, retained for reference. They are not the active research plan.
The [previous plan](docs/past_work/2026-09-22_previous_plan.md) is preserved;
existing results, code, checkpoints and operational lessons remain in place.

## First question

How does the released `ChenShawn/DeepEyes-7B` checkpoint answer a few of our
held-out echo QA pairs, before any task-specific training?

1. Use the original trained DeepEyes checkpoint, with its identity/revision
   recorded. Do not substitute our trained models or the untrained base model.
2. Start with 10 fixed QA examples from `build/eval.jsonl`, seed 0, covering
   the five question types (two per type), with distinct studies. Verify train
   study exclusion and all selected image paths before GPU inference.
3. Begin with direct answers from the existing preview frames unless the user
   chooses the original DeepEyes zoom tool. Record the exact prompt and input
   images; a preview-frame baseline does not evaluate full-video understanding.
4. Keep each question, reference answer, raw model output, extracted answer,
   generation settings and truncation/error status together for inspection.
   Log the experiment to W&B. Treat this as a small qualitative baseline;
   do not draw aggregate clinical-performance conclusions from 10 examples.
5. Inspect the answers together before deciding the next experiment.

## Execution status

Baseline preparation in progress; no DeepEyes inference results yet.
At the reset, the older `echo_full_test_step53` evaluation was still running
(Slurm checked 2026-09-22). It belongs to past work; the reset itself does not
cancel it or delete its outputs.
