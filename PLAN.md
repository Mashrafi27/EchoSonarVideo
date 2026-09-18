# PLAN

Living document. Where things stand and what happens next, revisit and edit this rather than
letting it go stale — see SPEC.md for the reference facts this plan is built on.

# Current state (2026-09-18)

Smoke job 195792 PASSED but with zero reward. Root cause identified and fixed:
- Darya's SFT never trained the model to predict `<think>` (always masked in training)
- `EchoPrimeToolAgentLoop` now primes assistant turn with `<think>\n`
- System prompt updated to match Darya's format + add tool schemas
- Reward formula rewritten to multiplicative gating (`r_fmt × (r_cor + r_tool) + r_len`)
- SFT collator `<answer>` wrapping removed (format alignment)

# Immediate next steps

1. **Rebuild parquet** (system prompt changed):
   ```
   sbatch scripts/build_echoprime_parquet.sbatch
   ```
2. **Smoke test** (after parquet done, ~10 min):
   ```
   sbatch --dependency=afterok:<parquet_job> scripts/smoke_grpo_echoprime_amd.sbatch
   ```
   Check rollout output: should see `<think>` blocks and non-zero reward.
3. **Real run** (after smoke passes):
   ```
   sbatch --dependency=afterok:<smoke_job> scripts/run_grpo_echoprime_amd.sbatch
   ```

# Longer-term

- Validate eval pipeline against EchoSonar-R's reported Table 1 numbers (Qwen3-VL track).
- Once echoprime_track run has checkpoints: eval with `packages/eval/echoprime/eval_checkpoint_vllm.py`, compare against EchoSonar-R Table 3 (use `packages/eval/darya_report_bridge.py` for per-section scoring).



