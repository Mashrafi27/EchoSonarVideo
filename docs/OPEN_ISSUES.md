# Open issues

Tracked across replies until resolved. Newest context at the top of each entry.

## 1. Base-model prompting for the EchoSonar-R comparison — RESOLVED (2026-08-27)
Went with (a): plain non-agentic prompt, one turn, no tools, same overview images and the
same 500-episode set (`--seed 0 --per-type 100`). Implemented as `--prompt-mode plain`
(`run_plain_episode`), threaded through `run_eval.py` and `run_eval.sbatch` via `PROMPT_MODE`.
Ran as job 160066, wandb run q5osho1z.

Metric-code check against their reported base row:
- BLEU-4: ours 0.013 (conclusion) / 0.015 (full_report) vs their 0.010. Agrees.
- Balanced accuracy: ours 0.631 vs their 0.503. Does NOT agree, and is not explained by
  a metric bug (0 unparsable, gold 80/20 no/yes, prediction 76/24). Most likely their base
  eval saw different visual input; their exact protocol is not documented anywhere we have,
  so our view-menu input (median 18 previews) remains an assumption.
Option (b) was deliberately not run.

## 1b. SFT made abnormality classification WORSE in balanced terms — DECIDED (2026-08-27)
Decision: leave it, and let GRPO fix it with a balanced reward. Not re-running SFT.
Action item carried into GRPO: the classification reward must be balanced, or the
policy will keep collecting 0.8 by answering "no".
Base plain: BAcc 0.631, macro-F1 0.621, predicts yes 24/100.
SFT step616: BAcc 0.569, macro-F1 0.604, predicts yes 5/100 with 6 unparsable.
SFT taught the majority class. Plain accuracy went up (0.74 -> 0.79) and balanced accuracy
went down, which is exactly the failure the balanced-accuracy rule exists to catch. Options:
reweight the classification slice of the SFT mixture, or leave it for GRPO to fix with a
balanced reward. Note the two runs are not a clean A/B: base is one plain turn, SFT is
agentic with tools.

## 1c. EchoSonar-R Table 1 — HAVE IT, and our metric was not comparable (2026-08-27)
The PDF is at `2606.28164v1.pdf` (arXiv is proxy-blocked from this cluster; the site
proxy 403s the CONNECT). Their numbers are now hard-coded in
`echo_verl/eval/diseases.py` so this never blocks again. Do not commit the PDF.

Their protocol: per-disease positive-class F1 and balanced accuracy over 12 abnormality
categories, macro-averaged, on all 1,215 private test studies.
  EchoSonar-R GRPO 49.4 / 67.4 | SFT-only 45.1 / 65.1 | Qwen3-VL base 19.6 / 50.3

CORRECTION to what this file said earlier: the base-model gap (our BAcc 0.631 vs their
50.3) is NOT explained by different visual input. It is the AGGREGATION. Ours was one
pooled yes/no score over a frequency-weighted question mix; theirs is a macro over 12
per-disease binary tasks, each of which the base model scores near 50 on. The two are
different quantities and neither can be read off the other.

Confirmed same test set: our per-disease prevalences match their Table 1 to within 0.1%
on all 11 diseases we ask about (54.5/54.6, 27.2/27.3, 2.6/2.5, ...). So an exact
apples-to-apples row IS buildable, which it was not clear it would be.
We have 11 of their 12 categories; there is no "Healthy" question in our test file, so
`their_macro()` recomputes THEIR macro over the same 11 rather than comparing to their
published macro-12.

## 1d. Where to look in wandb (2026-08-27)
Project `echo-eval` (https://wandb.ai/anaatef9-mbzuai/echo-eval).
- `comparison-*` runs: every cross-run table in one place, from
  `scripts/log_comparison_tables.py` over the `report.json` files. Cells are STRINGS
  on purpose: wandb rejects a mixed column, and a metric we do not compute must read
  "not computed", never 0.
- `*-perdisease` runs: `echo_verl/eval/score_per_disease.py --wandb`. Three tables:
  the Table 1 comparison with bootstrap CIs, the macro summary, and every
  classification episode with its parsed prediction for spot checks.
- Per-checkpoint eval runs keep logging themselves from `run_eval.sbatch`.

## 2. SFT trained on 2.3% of the corpus — RESOLVED (2026-08-27)
Replaced by `s5-balanced-19k`: a seeded stratified sample of 19,734 of 102,098 records
across 4,028 studies, balanced by question type (manifest `build/sft_train_s5.manifest.json`).
Trained to step616, merged, evaluated (job 156202). The old step100 numbers are superseded.
Still open underneath this: whether to go past one balanced epoch.

## 3. No ROCm vLLM — SOURCE BUILD RULED OUT (2026-08-27), back to needing the .sif
Two gate-1 attempts, both failed the same way:
  - torch 2.9.0+rocm6.4: every kernel launch (h2d copy, elementwise, matmul) failed
    with `hipErrorInvalidDeviceFunction`.
  - torch 2.9.1+rocm6.3, matching this cluster's ROCm 6.3.3 -- same failure, identical
    error, on every op including a bare `torch.ones(..., device='cuda')`.
CONTROL: torch 2.7.1+rocm6.2.4 (the existing qwen_backup build) ran h2d + bf16 matmul
cleanly on the SAME node, SAME GPU, in the SAME job step immediately after. So this is
not the node, not the GPU, not a version-pairing mistake on our part.

Root cause: `amdgpu-dkms 6.10.5.60303` is the kernel module installed on this cluster
(`dpkg -l amdgpu-dkms`), and it predates whatever KFD ioctl/queue ABI the 2.9-series
ROCm userspace requires. This is a KERNEL DRIVER ceiling, not a library version to swap.
No pip torch wheel changes it. vLLM 0.11.2 (first Qwen3-VL release) needs a torch new
enough that its ROCm build sits above that ceiling, so there is no combination of
"newer vLLM + old-enough torch" to reach for either.

This closes the source-build route per the two-attempt time-box set when the plan was
written. Two real options left, both outside what a session on this account can do:
  (a) a cluster admin updates amdgpu-dkms (or the kernel), which is a system change,
      not a project one, and may affect other users' jobs -- needs the admins, not us.
  (b) a prebuilt image built against THIS driver, i.e. the .sif at
      /vast/users/praneeth.vepakomma/document/container/vllm-openai-rocm-nightly.sif,
      already proven serving on 8x MI210 HERE (so it does not hit this ceiling). We
      cannot read it: that home directory is `drwx------`. Needs either that person's
      permission (not "someone I don't know" -- their user id is who owns the one file
      that already works on this exact hardware) or an admin to copy/re-share it.
`.venv-vllm` and `.tmp_work/vllm-src` are dead weight now; not deleted in case the
next session wants to re-verify the diagnosis before trying (a) or (b).

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
