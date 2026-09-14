# Qwen3-VL + vLLM + GRPO on MI210, 2026-09-14

Qwen3-VL-8B-Instruct, vLLM, ROCm, and GRPO work together on this cluster's
AMD MI210 GPUs. Inference passed in Slurm job **185987**. Full GRPO validation
passed **two consecutive steps** in job **186304** on two GPUs on
`auh7-1b-gpu-198`, using `faculty`, `faculty-acc`, and `myqos`.
Slurm reports **COMPLETED, exit 0:0**, elapsed **22m58s**. The result checker
reports `pass: true` with no issues.

## Fixed issues and verification

These fixes apply to the private environment and launchers recorded below.
The inference and two-step GRPO results predate the W&B shutdown change.

| Observed failure | Fix | Verification |
|---|---|---|
| COMGR could not unpack `gfx90a:sramecc+:xnack-` kernels on VAST; misleading `hipErrorInvalidDeviceFunction` | Private ext4 scratch inside the workspace via [`rocm_workspace_scratch.sh`](../scripts/rocm_workspace_scratch.sh); short socket aliases; no shared `/tmp` | 185957 reproduced the failure; 185962 passed with the original torch wheel and new scratch; 185987 passed full inference |
| Legacy CUDA vLLM wheel and missing native libraries | Isolated ROCm vLLM environment, private ROCm 7.0.2, MPI and C++ libraries | 185987 verified loaded library paths and Qwen text/image inference; 186304 completed GRPO |
| Missing ROCm device UUID API, inconsistent HIP/CUDA visibility, and hardcoded weight-transfer sockets | [`ROCm platform plugin`](../tools/rocm_runtime_plugin/) and allocation-bounded startup normalization; redirect weight-transfer sockets into workspace scratch | UUID identity/remapping and cross-process socket checks passed; 186304 completed both weight transfers |
| Missing `qwen-vl-utils` during the first rollout | Install version 0.0.14 and `av==18.1.0`; preflight real Echo tool images | 186104 checked the exact helper API; 186304 completed tool rollouts |
| Tool response had 21 images but 20 image placeholders | [`EchoToolAgentLoop`](../packages/verl_bridge/agent_loop.py) emits one placeholder per returned image; one tool call per turn | Real Qwen processor preflight passed; both rollouts completed in 186304 |
| Qwen vision position indices on CPU with embedding weights on GPU | [`FSDP extension`](../packages/verl_bridge/fsdp_compat.py) retains `visual.pos_embed` in the root FSDP group | Two-GPU forward/backward regression passed in 186271; full updates passed in 186304 |
| 2374 MiB FP32 embedding exceeded the 2048 MiB transfer bucket | Set `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072` | 186304 transferred initial and updated model weights |
| Second rollout exceeded the 24-image vLLM limit | Set the smoke limit to 35: 19 initial images plus two tool responses of at most eight frames | 186304 completed both steps; this bound depends on the current image and turn settings |
| W&B had no step metrics and remained `running` after successful training | Recover real metrics into the existing run; add explicit worker-side finish and local scalar logs for future launches | Recovery 186356 verified remote steps 1/2 and `finished`; offline CPU Ray regression 186378 passed success/failure cases; fresh full GPU validation of the logging change remains outstanding |

## Completed GRPO evidence

[Successful GRPO run 186304](https://wandb.ai/anaatef9-mbzuai/echo-runtime-validation/runs/mm0adoky)
completed multi-turn Echo tool rollouts, actor/reference log probabilities,
backward propagation, optimizer updates, and updated-weight transfers to vLLM
in both steps. Actual weight changes and nonzero gradients were measured.

| Metric | Step 1 | Step 2 |
|---|---:|---:|
| Actor gradient norm | 2.71518 | 5.09314 |
| Policy gradient loss | -7.45058e-09 | 1.86265e-08 |
| KL loss | 0.00110595 | 0.00141527 |
| Maximum measured vision-position weight change | 1.90735e-06 | 1.01328e-06 |
| Advantage minimum | -0.866024 | -1.5 |
| Advantage maximum | 0.866024 | 1.5 |
| Actor update time, seconds | 220.557 | 156.545 |
| Updated-weight transfer time, seconds | 4.51924 | 4.05432 |

All 32 sampled trajectories (16 per step) included tool calls. Exact prompts,
outputs, rewards, and the numerical result are preserved under
`.tmp_work/rocm_validation_20260914/grpo_186304/` (`rollouts/1.jsonl`,
`rollouts/2.jsonl`, `result.json`). Slurm peak batch RSS was about 230 GiB.
The exact submitted script, resolved configuration, and source SHA256 manifest
are `grpo_186304.sbatch`, `grpo_186304_resolved.yaml`, and
`grpo_186304_source_sha256.json` in the validation root.

This is a two-step infrastructure test using four real TRAIN studies, reduced
image sizes, base Qwen weights, and the existing Echo reward. It does not measure
clinical quality or production-scale reliability. Checkpoint saving was disabled.

## W&B recovery and future runs

The original W&B client lost its local Unix-socket connection before its final
flush (`Reached EOF` at 08:47:20 UTC, followed by `UnixTransport closed=True`).
VeRL relies on `Tracking.__del__` for W&B completion; here it ran after the
connection had closed. The run initially had no uploaded step history and was
still marked `running`, despite both training steps passing locally. The exact
reason the service connection closed is not established by the available logs.

After user authorization, CPU-only Slurm job **186356** completed in 14 seconds
on `auh7-1b-gpu-213` and restored both real steps from `result.json` to the same
W&B run. An API read verified **`finished`**, global steps **1 and 2**, and gradient
norms **2.7151827812194824** and **5.093136787414551**. The upload is explicitly
marked as recovered in the run summary, with the source SHA256. No training was
rerun and no images or trajectory text were uploaded. Evidence:
`grpo_186304/wandb_verification.json`, `wandb_recover_186356.out`, and
`grpo_186304/wandb_recovery_payload.json` in the validation root.

The validation launcher now uses `verl_bridge.main_ppo`, a local subclass of the
upstream Ray task runner. It calls W&B finish synchronously before the worker
returns, sets a failing exit code when training raises, and reports flush errors
on successful training. It also enables VeRL's unbuffered file logger at
`grpo_JOB_ID/metrics.jsonl`. Upstream VeRL remains unchanged. This addresses the
observed late finalization; it does not prove the service can never disconnect.
The CPU-only regression in `scripts/check_wandb_lifecycle.py` exercises the real
W&B client with uvloop in Ray actors, using offline synthetic diagnostics for
both successful and failing tasks. Slurm job **186378** passed both cases on
`auh7-1b-gpu-217` in 50 seconds (exit `0:0`); the new Hydra entry point also
resolved its configuration successfully. Evidence is `wandb_check_186378/result.json`
and `wandb_check_186378.out` in the validation root. This logging regression
did not rerun GPU training.

### Launch coverage and remaining validation

The change is active in [`scripts/validate_grpo.sbatch`](../scripts/validate_grpo.sbatch):

- It runs `python -m verl_bridge.main_ppo`. [`EchoTaskRunner`](../packages/verl_bridge/main_ppo.py)
  uses [`run_with_wandb_finish`](../packages/verl_bridge/wandb_lifecycle.py) before returning
  control to Ray. Success finishes with exit code 0; a training exception is
  preserved while W&B is finished with exit code 1. Flush failures on a successful
  task propagate instead of being hidden in a destructor.
- `trainer.logger=[console,file,wandb]` and `VERL_FILE_LOGGER_PATH` enable the
  unbuffered local backup at
  `.tmp_work/rocm_validation_20260914/grpo_JOB_ID/metrics.jsonl`.
  This file is enabled for future launches; the recovered run 186304 uses its
  original stdout and `result.json` as the metric source.
- Other launchers calling `verl.trainer.main_ppo` do not automatically inherit
  this change. They need the local entry point for explicit finalization and
  the file logger configuration for metric backups, alongside their usual
  Slurm, ROCm, and workspace scratch setup.

The CPU test verified finalization and retained metrics on successful and failing
Ray tasks with the real W&B SDK and uvloop. It did not reproduce the original
service disconnect or rerun the full Qwen/GRPO workload. The disconnect trigger,
full GPU verification of the new logger, and reliability over long training runs
remain open. For the next full run, check `ECHO_WANDB_FINISH_COMPLETE`, local
metric rows, and the remote W&B history and final state before declaring logging
validated end to end. The W&B Weave suggestion was informational.

### Logs and result files

These links resolve from this repository checkout. The artifacts are local and
are not included in a fresh Git clone.

| Artifact | Location |
|---|---|
| GRPO stdout | [grpo_186304.out](../.tmp_work/rocm_validation_20260914/grpo_186304.out) |
| GRPO stderr, including the original W&B exception | [grpo_186304.err](../.tmp_work/rocm_validation_20260914/grpo_186304.err) |
| Strict two-step result and metrics | [result.json](../.tmp_work/rocm_validation_20260914/grpo_186304/result.json) |
| Exact submitted configuration | [grpo_186304_resolved.yaml](../.tmp_work/rocm_validation_20260914/grpo_186304_resolved.yaml) |
| Actual generated trajectories | [step 1](../.tmp_work/rocm_validation_20260914/grpo_186304/rollouts/1.jsonl), [step 2](../.tmp_work/rocm_validation_20260914/grpo_186304/rollouts/2.jsonl) |
| W&B recovery log | [wandb_recover_186356.out](../.tmp_work/rocm_validation_20260914/wandb_recover_186356.out) |
| Verified remote W&B state and history | [wandb_verification.json](../.tmp_work/rocm_validation_20260914/grpo_186304/wandb_verification.json) |
| Offline logging regression | [result.json](../.tmp_work/rocm_validation_20260914/wandb_check_186378/result.json), [stdout](../.tmp_work/rocm_validation_20260914/wandb_check_186378.out), [stderr](../.tmp_work/rocm_validation_20260914/wandb_check_186378.err) |

## Completed inference evidence

[W&B run qwen3vl8b-rocm-inference-185987](https://wandb.ai/anaatef9-mbzuai/echo-runtime-validation/runs/y8sgd1bi)
contains four passing checks. Full exact prompts, responses, engine arguments,
versions, and loaded library paths are in
`.tmp_work/rocm_validation_20260914/inference_185987/result.json`.

| Check | Observed result |
|---|---|
| GPU arithmetic, BF16 matmul, BF16 vision convolution | Passed; loaded ROCm libraries all came from the private runtime |
| Text: "What is 2 + 2? Reply with only the number." | `4`, 2 generated tokens |
| Two synthetic solid-color images, red then blue | Correct colors in the correct order, 40 tokens |
| Prior tool-call conversation with a newly returned blue image | "The newly returned image is solid blue.", 9 tokens |

The tool-response test supplies a previous tool call and observation. It proves
multimodal conversation handling, not autonomous tool use or clinical accuracy.
The engine loaded 16.78 GiB of model weights; full engine initialization took
88.62 seconds. These are smoke-test observations, not throughput benchmarks.

## What was actually wrong

Job **185957** exposed COMGR's real failure on ordinary VAST scratch:

```text
Unbundle Objects Error: .../output/hipfatbin-hipv4-amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-.o: Invalid argument
```

VAST rejects colons in filenames. This caused `hipErrorInvalidDeviceFunction`
when the runtime could not unpack GPU kernels. Job **185962**, on
`auh7-1b-gpu-192`, ran the original `.venv-vllm` torch **2.9.1+rocm6.3** with private
ext4 scratch and passed copy, ones/reduction, and BF16 matmul. Therefore the
previous kernel-driver-ceiling diagnosis was not supported.

`scripts/rocm_workspace_scratch.sh` mounts a private ext4 image through `fuse2fs`.
The image and mountpoint are inside the workspace. Compiler caches and temporary
files live there. An open directory descriptor gives a short `/proc/<pid>/fd/<fd>`
alias for Unix sockets. **No shared `/tmp` files are used.** The Slurm shell must
stay alive while its workers use the alias. An exit trap unmounts the image.

The fresh environment also needed ROCm shared libraries (not bundled in the
vLLM wheel), MPI runtime libraries, and a C++ runtime exporting `GLIBCXX_3.4.30`.
The borrowed Python interpreter otherwise selected an older conda C++ library.
All additions are private; shared conda environments and system ROCm were untouched.

## Exact environment

Root: `.tmp_work/rocm_validation_20260914/env`.

| Component | Version/source |
|---|---|
| Python | 3.12.12 venv; base interpreter from `perception_models`, no writes to that env |
| vLLM | `0.17.0+rocm700` from the official ROCm wheel index; runtime reports `0.17.0` |
| PyTorch | `2.9.1+git8907517`, HIP `7.0.51831-a3e329ad8` |
| Transformers | `4.57.3` |
| Triton | `3.4.0` |
| ROCm userspace | AMD 7.0.2 packages extracted into `env/rocm-root/opt/rocm-7.0.2` |
| MPI and C++ libraries | Private `env/native-root/usr/lib/x86_64-linux-gnu` |
| VeRL source | `external/verl` at `bec9ef74768dd201881cd4e54cd0385e87caae27`, editable, no upstream edits |
| tensordict / numpy | `0.10.0` / `2.2.6` |
| Qwen vision helper | `qwen-vl-utils==0.0.14`, `av==18.1.0` |
| Model | Cached Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |

The exact package freeze is `requirements.verl.freeze.txt` under the validation
root. `rocm_runtime_manifest.json` records 22 AMD packages and their SHA256 checks;
`native_runtime.sha256` records the private Ubuntu packages and C++ library.
Installation scripts and logs are preserved in the same root: `install.sbatch`,
`runtime.sbatch`, `install_rocm_runtime.py`, `native_runtime.sbatch`, and
`verl_install.sbatch`. All setup and GPU validation ran through Slurm.

Installation sources:
[official vLLM ROCm wheels](https://wheels.vllm.ai/rocm/0.17.0/rocm700/),
[AMD ROCm 7.0.2 package index](https://repo.radeon.com/rocm/apt/7.0.2/dists/jammy/main/binary-amd64/Packages.gz).

The serving recipe uses BF16, eager execution, `TRITON_ATTN` for decoder attention,
`TORCH_SDPA` for vision attention, TP=1, and AITER disabled. The successful inference
run emitted a caught optional Quark plugin import warning; it did not prevent any
check. GRPO disables optional vLLM plugins because it uses unquantized BF16.

## Rerun

From the repository root, using the installed environment:

```bash
sbatch --exclude=auh7-1b-gpu-188 scripts/validate_qwen_vllm.sbatch
# After the inference result has pass=true:
sbatch --exclude=auh7-1b-gpu-188,auh7-1b-gpu-260 scripts/validate_grpo.sbatch
```

Both scripts request `faculty`, `faculty-acc`, and `myqos`. Nodes 188 and 260 were excluded because allocations there exposed no usable HIP
GPUs. On node 188, diagnostic jobs 186070 and 186071 tried the original Slurm
visibility masks, ROCr-only masks, HIP masks without GPU_DEVICE_ORDINAL, and the
allocated physical GPU IDs; all failed GPU initialization. Their underlying cause
is unresolved. This is not proof of a global ROCm or driver incompatibility.

The GRPO smoke configuration now requests two GPUs and uses four distinct TRAIN studies (two yes, two no),
four sampled responses per prompt, two steps, the existing Echo tool/reward, and
base Qwen weights. Images are reduced to 160x160 pixel budgets to keep this an
infrastructure test. It does not establish clinical quality or production scale.
Per-process MIOpen databases are configured by
`scripts/rocm_python_startup/sitecustomize.py` only when the launcher supplies
`ECHO_ROCM_SCRATCH_ROOT`.

## GRPO compatibility discovered by the first attempt

Job 185993 failed at VeRL's `ServerAdapter` construction when it called
`RocmPlatform.get_device_uuid`, which vLLM 0.17 leaves unimplemented. The
opt-in platform plugin in `tools/rocm_runtime_plugin` supplies this method through
HIP's device UUID API. The check in `scripts/check_rocm_compat.py` requires distinct
GPU UUIDs and verifies that selecting a single physical GPU in a child process
returns the same identity.

VeRL also hardcodes two weight-transfer endpoints under `/tmp`. The local socket
adapter redirects only `ipc:///tmp/rl-colocate-zmq-*` endpoints into the job's
private workspace filesystem, including bind/connect and cleanup operations. A
cross-process round-trip verifies this before model loading. The failure in job
185993 occurred before these socket endpoints were constructed. No upstream VeRL
files are edited. Other socket addresses are unchanged.

Install the local plugin into the isolated environment with `uv pip install
--python <env>/bin/python --no-deps -e tools/rocm_runtime_plugin` from a Slurm job.
The launcher's `VLLM_PLUGINS=echo_rocm_runtime` selects this plugin and excludes
unused optional quantization plugins.

Job 185999 exposed a further ROCm integration issue: VeRL narrows
`CUDA_VISIBLE_DEVICES` to the replica GPU but leaves `HIP_VISIBLE_DEVICES` with
the entire allocation. vLLM rejects that mismatch in its model-inspection child.
The startup hook now synchronizes HIP to VeRL's selection only when both lists
are subsets of the original Slurm allocation. The preflight reproduces this
conflicting-list launch and requires the same physical GPU UUID in the child.

Job 186004 reached `AsyncLLM` engine construction but failed at vLLM's parent-side
`_sync_visible_devices_env_vars()` hook. The opt-in platform plugin now wraps the
ROCm synchronization function to apply the same allocation-bounded normalization
before vLLM's own consistency check. The preflight tests narrowing CUDA visibility
both before process startup and after platform import.

The missing Qwen helper was an omission from this environment's initial install,
not a ROCm incompatibility. Job 186104 installed it and exercised the exact
`process_vision_info(..., image_patch_size=16, return_video_metadata=True)` call.
The smoke data preparer also now creates an Echo session for every selected study,
executes a valid `select_view`, and checks its returned images with that helper.

[Failed first-rollout run 186073](https://wandb.ai/anaatef9-mbzuai/echo-runtime-validation/runs/kd0bbfr6)
has no completed training steps. Its compute node reports kernel `5.15.0-177`,
loaded AMD driver `6.14.14`, and system ROCm `7.0.0` (recorded in
`hardware_186073.txt` under the validation root); the job uses the private 7.0.2
runtime described above.


Job **186113** generated tool calls and then failed with
`Failed to apply prompt replacement for mm_items['image'][20]`: VeRL appended all
two returned frames but emitted only one image placeholder. The resulting request
had 21 images and 20 placeholders. `packages/verl_bridge/agent_loop.py` supplies a local
`EchoToolAgentLoop` that emits one placeholder per returned frame. The smoke
configuration selects this agent and allows one tool call per turn, so the image
assignment is unambiguous. The preflight in `scripts/check_echo_tool_images.py`
checks the actual Qwen processor with both an existing image and two new frames.
Upstream VeRL remains unchanged.

[Failed multi-image run 186113](https://wandb.ai/anaatef9-mbzuai/echo-runtime-validation/runs/afyhn9hh)
has no completed optimizer steps.


Job **186132** passed the multi-turn rollout but failed during reference log
probabilities with CPU image-position indices and GPU embedding weights.
`packages/verl_bridge/fsdp_compat.py`, selected with `model.external_lib`, keeps Qwen3-VL's
small `visual.pos_embed` in the root FSDP group instead of sharding/offloading it
as a separate embedding unit. The root's pre-forward gather runs before Qwen
reads the weight device to construct those indices. No Transformers or VeRL
source files are edited. `scripts/check_qwen_fsdp.py` checks real Qwen vision
forward/backward and optimizer changes over two GPUs with FSDP2 CPU offload before
loading the full model.

The retry uses FP32 actor parameters, BF16 compute, and FSDP2 CPU offload, with
384 GiB of host RAM requested. `packages/verl_bridge/training_progress.py` reports the start
and end of actor/reference log probabilities and policy updates. It also records
actual changes to the vision position weights; the final smoke checker requires
nonzero gradients, within-group advantage variation, completed updates and weight
synchronization, and a measured parameter change.


Job **186271** passed the two-GPU FSDP regression (synthetic vision loss
0.125485 then 0.029049; actual position-weight changes checked on both ranks).
Its initial full-model transfer then rejected the FP32 token embedding, which
is 2374 MiB, because the default bucket was 2048 MiB. The smoke launcher now sets
`rollout.checkpoint_engine.update_weights_bucket_megabytes=3072`. Retry 186294 reuses the completed
FSDP regression with `SKIP_FSDP_REGRESSION=1`; default launches still run it.

Job 186292 stopped in configuration validation because the bucket override was
placed at the old path named in the error message. The correct path for this
VeRL revision is `rollout.checkpoint_engine.update_weights_bucket_megabytes`;
job 186294 uses it.


Job **186294** completed its first policy update and weight synchronization,
then its second rollout exceeded `limit_mm_per_prompt.image=24`. Echo's
`select_frames` and `zoom` can each return eight frames. The launcher's three
assistant turns permit two tool responses, so 19 + 2 * 8 = 35 covers the smoke
conversation bound. Job 186304 uses that limit and explicitly pins eight high
resolution frames per tool call. `ECHO_MAX_TOTAL_FRAMES` is an EchoEnv budget;
the standalone EchoSession adapter does not enforce it as a trajectory budget.
