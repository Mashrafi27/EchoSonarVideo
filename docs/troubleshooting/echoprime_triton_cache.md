# EchoPrime Triton cache failure

## Observed failure

On 2026-09-18, AMD full run `196208` (`grpo-echoprime-sft-init-196208`)
failed after 5 minutes 27 seconds, before any completed training update.
The source commit was `903de20a9cf22e21966d7af1fcf97778640af4e6`.
[W&B run](https://wandb.ai/anaatef9-mbzuai/echo-grpo/runs/3qtlgaa6).

The first concrete exception was `FileNotFoundError` reading
`_lora_shrink_kernel.source` under the job's `scratch_196208_auh7-1b-gpu-198/triton/`
directory. The traceback goes through Triton's compiler and
`CompiledKernel.__init__`, invoked by vLLM's LoRA logits path. The subsequent
`EngineDeadError` is the consequence of that worker failure.

The missing file's cause is **not established**. All workers inherited one
Triton cache directory on the private FUSE-mounted ext4 scratch image.
Concurrent cache access is a hypothesis, not a proven root cause. The inspected
logs did not establish an out-of-memory or full-disk error.

## Mitigation

`scripts/rocm_python_startup/sitecustomize.py` now sets `TRITON_CACHE_DIR` to
`<ECHO_ROCM_SCRATCH_ROOT>/triton/<pid>` during Python startup. Spawned workers
derive their cache directly from the job scratch root, ignoring the cache
directory inherited from their parent. The vLLM launcher uses `spawn`.

Keep these directories on the existing ext4 scratch mount. Do not move ROCm
compiler temporary files onto VAST directly or system `/tmp`; retain the
existing COMGR and socket workarounds described in [ROCM_VALIDATION.md](../ROCM_VALIDATION.md).

Before a full relaunch, validate two actor updates plus rollout weight sync and
inspect response truncation, format completion, gradients, and within-group
reward variation. A successful process exit is not sufficient to establish
usable GRPO training. Do not alter the reward definition to pass this check.
The two-update gate passed with the per-process caches and larger response
budget on 2026-09-18. This validates the mitigation over that execution; it does
not prove a cache race caused the original failure or establish long-run stability.

## Tracing

Weave `0.53.9` is installed in `.tmp_work/rocm_validation_20260914/env` on AMD.
The launchers set `actor_rollout_ref.rollout.trace.backend=weave`,
`token2text=True`, and `max_samples_per_step_per_worker=1`. This is a per-worker
limit, not a global one. The custom agent's `run` method uses verl's trace
decorator so its response is decoded in the trace. `weave.init()` alone does
not activate verl's rollout tracing.

For a fresh environment, install with that environment's Python:
`python -m pip install weave==0.53.9`.
