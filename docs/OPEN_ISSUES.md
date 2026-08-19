# Open issues

Tracked across replies until resolved. Newest context at the top of each entry.

## 1. Base-model prompting for the EchoSonar-R comparison — NEEDS DECISION
Base Qwen3-VL under our agentic system prompt will mostly never emit `<answer>`, so it
scores ~0 as a format artifact, not a capability measurement.
- (a) Plain non-agentic prompt — matches how EchoSonar-R evaluated it, so their reported
  row (F1 19.6 / BAcc 50.3 / BLEU-4 0.010) becomes a real check on our metric code.
- (b) Same agentic prompt, score `answer or final_text` — measures base capability under
  our conditions, but is not comparable to their number.
Recommendation: (a).

## 2. SFT trained on 2.3% of the corpus
`checkpoints/echo-sft/merged/step100` saw 2,382 of 102,098 records — a `--limit 3000`
smoke default that went unrevisited. A full re-run needs a mixture decision first: one
uncapped epoch is ~41.7h against a 12h wall limit. Options were a per-study cap (~8 ->
13.1h) or balancing by question type.

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
