# Open issues

Tracked across replies until resolved. Newest context at the top of each entry.

## 1. Base-model prompting for the EchoSonar-R comparison — DECIDED, RUNNING (2026-08-27)
Went with (a): a plain non-agentic prompt, one turn, no tools, same overview images and
the same 500-episode set (`--seed 0 --per-type 100`). Under our agentic prompt a base
model scores ~0 as a format artifact, not a capability measurement.
Implemented as `--prompt-mode plain` (`run_plain_episode` in `echo_verl/eval/agentic_loop.py`),
threaded through `run_eval.py` and `scripts/run_eval.sbatch` via `PROMPT_MODE`.
Their reported base row (F1 19.6 / BAcc 50.3 / BLEU-4 0.010) is a SANITY CHECK on our
metric code, not a number to reproduce: their exact base-eval protocol is not documented
anywhere we have, so our visual input (the view menu, median 18 previews) is an assumption.
Option (b) — the agentic prompt scored on `answer or final_text` — was deliberately NOT run.

## 2. SFT trained on 2.3% of the corpus — RESOLVED (2026-08-27)
Replaced by `s5-balanced-19k`: a seeded stratified sample of 19,734 of 102,098 records
across 4,028 studies, balanced by question type (manifest `build/sft_train_s5.manifest.json`).
Trained to step616, merged, evaluated (job 156202). The old step100 numbers are superseded.
Still open underneath this: whether to go past one balanced epoch.

## 3. No ROCm vLLM — blocks GRPO
Evaluation is unblocked (in-process client). GRPO still needs a real vLLM server. All
container routes are closed on this filesystem; the open one is read access to a
colleague's prebuilt `.sif`
(`/vast/users/praneeth.vepakomma/document/container/vllm-openai-rocm-nightly.sif`),
already proven serving on 8x MI210 here. Requires the user to ask.

## 4. Two pre-existing test failures
`echo_verl/tests/test_sft_dataset_rope.py`: `test_upstream_dataset_still_breaks_on_qwen3_vl`
and `test_echo_dataset_builds_qwen3_vl_position_ids`. Present on HEAD before recent work
(confirmed by stashing). The first failing may mean upstream no longer breaks, which would
make our shim redundant — worth checking before the next SFT run.

## 5. GREEN score not implemented
EchoSonar-R reports it; the formula is not in their paper. Needs the definition from
arXiv 2405.03595. Deliberately not approximated.

## 6. ROCm preamble duplicated across sbatch scripts
Four traps re-pasted into every `scripts/*.sbatch`. Should be one sourced file.

## 7. Annealed `tool_bonus_coef` unwired
Cannot pass through static `extra_info`; needs a custom reward manager.
