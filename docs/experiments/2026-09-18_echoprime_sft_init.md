# EchoPrime SFT-init GRPO, 2026-09-18

## Run

- Experiment: `grpo-echoprime-sft-init-196227`; Slurm job `196227`.
- Started: `2026-09-18T07:17:33` UTC, 2 AMD MI210 GPUs, 126-hour allocation.
- [W&B run](https://wandb.ai/anaatef9-mbzuai/echo-grpo/runs/005y1s7m).
- [Weave traces](https://wandb.ai/anaatef9-mbzuai/echo-grpo/weave), filtered by
  `experiment_name=grpo-echoprime-sft-init-196227`.
- Status: intentionally cancelled after 10 completed updates, elapsed 59m21s,
  after discovering stale system messages without tool instructions.

Launch command, from `/vast/users/mohammad.yaqub/project/EchoSonarVideo`:

```bash
sbatch --parsable scripts/run_grpo_echoprime_amd.sbatch
```

## Code and initialization

Base commit: `903de20a9cf22e21966d7af1fcf97778640af4e6`, plus uncommitted changes
for per-process Triton caches, Weave tracing, and the larger response/context
budget. This is not an unmodified run of the base commit.

Exact source provenance is retained on AMD in
`build/grpo_echoprime_sft_init_196227/{provenance.json,source.patch,verl.patch}`.
The source patch SHA-256 is
`daf0f4767614e01a8dddcd6f68741978070409d64f2761f592d4ee2af2f99711`.
`provenance.json` includes the verl revision, patch hash, and changed code hashes.

Initialization is `build/echoprime_sft_init`, converted from Darya's
`/vast/users/mohammad.yaqub/report_generation/checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503`.
The full run starts from that SFT initialization, with `resume_mode=disable`.

Configuration source: [echoprime_grpo.yaml](../../packages/verl_bridge/configs/echoprime_grpo.yaml)
plus [AMD launcher overrides](../../scripts/run_grpo_echoprime_amd.sbatch).
Captured launch configuration: `build/grpo_echoprime_sft_init_196227/resolved.yaml`
on AMD. Library warnings precede the YAML content in that capture.

## Data and settings

| Setting | Value |
| --- | --- |
| Train parquet | `build/echoprime_grpo_train.parquet`, 5,061 records |
| Held-out parquet | `build/echoprime_grpo_val.parquet`, 1,215 records |
| Source partitions | `build/rl.jsonl` train pool; `build/eval.jsonl` held-out pool |
| Batch and rollouts | 8 prompts per batch, 2 rollouts per prompt |
| Schedule | 3 epochs, 1,896 planned updates |
| Token budgets | Prompt 3,584; response 2,048; context 6,144 |
| Checkpoint and eval frequency | Every 50 updates; no validation before training |
| Seed | `data.seed` is null; no explicit seed override in launch command; other runtime defaults not audited |
| Weave | 0.53.9; decoded agent outputs; up to 1 sampled prompt per worker per step |

The parquet counts are the actual training/evaluation inputs, not the source
JSONL row counts. This run does not train on all 128,215 source QA rows per epoch.

## Validation and limitations

Subsequent prompt audit: all 5,061 training rows and all 1,215 validation rows
lack `select_frames` and `zoom` descriptions in their system messages. They retain
the old instruction to emit `<answer>` tags. The deployed `dataset.SYSTEM_PROMPT`
contains tool descriptions and requests a plain final answer, but the dataset and
agent use the messages already stored in parquet. Updating the source constant
did not update these artifacts. This run therefore does not receive the intended
tool prompt; nonzero gradients do not validate the intended tool-training setup.
After user approval, the job was stopped and both input artifacts were repaired.
The original parquet files are retained under this run's `input_artifacts`, in
subdirectories named by their SHA-256 hashes. See the
[prompt incident](../troubleshooting/echoprime_stale_prompts.md). A future full
run starts again from SFT, not from these updates.

Early full-run training metrics (not held-out evaluation):

| Update | Mean reward | Gradient norm | Response truncation | Step duration |
| --- | --- | --- | --- | --- |
| 1 | 0.625 | 0 | 18.75% | 369.13 s |
| 2 | 0.5625 | 0.06171 | 0% | 322.06 s |

Step 1 contained 16 responses across 8 prompts. All response pairs differed in
text but had equal rewards, explaining the zero GRPO advantage and gradient.
Thirteen responses closed their thinking block, ten earned positive reward,
and none called tools. Step 2 had nonzero positive and negative advantages.
These observations establish an initial learning signal, not sustained progress
or tool-use learning.

The targeted cache/tool tests and two-update infrastructure/learning-signal gate
passed before launch. The original missing-cache-file cause remains unconfirmed;
see [the incident notes](../troubleshooting/echoprime_triton_cache.md).

No clinical evaluation result is available at launch. Tool-use learning remains
unverified. Monitor truncation and within-group reward differences as well as
mean reward. The 126-hour allocation is a time limit, not a verified completion
estimate for all three epochs; checkpoints support a later continuation if needed.
