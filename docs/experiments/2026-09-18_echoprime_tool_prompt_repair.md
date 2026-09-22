# EchoPrime GRPO with repaired tool prompts, 2026-09-18

Status: full restart queued as Slurm job `196288`, experiment
`grpo-echoprime-sft-init-196288`. No training metrics or W&B run URL exist yet.
It depends on GPU validation and checks its results before starting training.

From `/vast/users/mohammad.yaqub/project/EchoSonarVideo`:

```bash
sbatch --parsable --dependency=afterok:196280 --kill-on-invalid-dep=yes \
  --export=ALL,ECHO_GRPO_GATE_DIR=/vast/users/mohammad.yaqub/project/EchoSonarVideo/.tmp_work/smoke_echoprime_196280 \
  scripts/run_grpo_echoprime_amd.sbatch
```

## Code, checkpoint, data

Base commit is `903de20a9cf22e21966d7af1fcf97778640af4e6`, plus uncommitted
prompt repair/validation, tracing, token-budget, and Triton-cache changes.
Exact source files (including new untracked modules), patch files, code hashes,
dataset hashes, and launch command are retained on AMD under
`build/grpo_echoprime_sft_init_196288/{source.tar,source.patch,verl.patch,provenance.json}`.

Initialization is `build/echoprime_sft_init`, converted from Darya's
`/vast/users/mohammad.yaqub/report_generation/checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503`.
Resume is disabled: this run does not continue the stale-prompt run.

The training parquet contains 5,061 records from the `build/rl.jsonl` train
pool; validation contains 1,215 records from the `build/eval.jsonl` held-out
pool. All system messages were repaired; independent Arrow comparisons
confirmed unchanged schema, row order, all non-prompt columns, and every
non-system message. Original artifacts and hashes are documented in the
[prompt incident](../troubleshooting/echoprime_stale_prompts.md).

Configuration is [echoprime_grpo.yaml](../../packages/verl_bridge/configs/echoprime_grpo.yaml)
with [AMD launcher overrides](../../scripts/run_grpo_echoprime_amd.sbatch):
2 MI210 GPUs; batch 8, 2 rollouts per prompt; 3 epochs / 1,896 planned updates;
prompt 3,584, response 2,048, context 6,144 tokens; save/evaluate every 50
updates; no validation before training; 126-hour wall-time limit. No explicit
seed override is supplied; the previous resolved config had `data.seed=null`.
Other runtime seed defaults are not audited.

## Launch gate and limits

The dependency requires a successful diagnostic process, including targeted
tests. The full launcher then requires two completed updates and weight syncs,
at least one nonzero finite gradient, positive reward, reward variation within
paired responses, the canonical system message and priming in every diagnostic
rollout, at least half of responses closing thinking, and mean truncation no
greater than 25%. If tool calls occur, at least one tool result must be present.
Any failure stops the full launcher before model startup.

This gate does not require spontaneous policy tool use in the small sample;
the report explicitly distinguishes whether it was observed. The synthetic
continuation test checks loop mechanics, not policy competence or clinical
quality. Tool-use learning and held-out quality remain unverified.

W&B project: `anaatef9-mbzuai/echo-grpo`.
[Weave project](https://wandb.ai/anaatef9-mbzuai/echo-grpo/weave), filtered by
`experiment_name=grpo-echoprime-sft-init-196288` once this run starts.
