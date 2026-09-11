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

## 8. GRPO on the CUDA box — full fine-tune still OOMs, unresolved (2026-09-02)
Target is full fine-tune (`lora_rank=0`), not LoRA. On 4x RTX A6000 48GB, no NVLink,
every full-FT attempt from 2026-09-01 08:50 through 11:07 OOM'd, 39-45 GiB in use
against the 48 GiB budget, across several mitigation sweeps (`use_dynamic_bsz` on/off,
offload combinations). This is still open — nothing below fixed it.

One of those attempts (10:31, `grpo-smoke-0901-1024`, actor
`enable_activation_offload=True`) hit a different, real bug instead of an OOM:
`RuntimeError: Output 0 of GroupCommitFunctionBackward is a view and is being modified
inplace`. Traced to `external/verl/verl/utils/activation_offload.py`'s
`GroupCommitFunction` colliding with transformers' Qwen3-VL `_deepstack_process`, which
does an in-place `hidden_states[visual_pos_masks, :] = local_this` on what offload hands
back as a view. verl's activation offload is incompatible with Qwen3-VL's deepstack path.
Do not re-enable `actor_rollout_ref.model.enable_activation_offload` for this model until
upstream fixes the view/inplace conflict — it is a real incompatibility, not something a
retry or a different batch size routes around.

As a DIAGNOSTIC ONLY, not the adopted approach, ran once with LoRA rank 32
(`.*visual.*` excluded from target modules, `actor.fsdp_config.param_offload`/
`optimizer_offload` both True, `rollout.tensor_model_parallel_size=2`) to check
whether the multi-turn tool-RL harness itself worked, decoupled from the full-FT
memory budget: `grpo-lora-r32-0901-1302`, wandb run `irkypwr0`. Ran clean for ~12h to
step 80, zero OOM, zero traceback anywhere in the log — so the harness (agent loop,
tool calls, reward, weight sync) is confirmed working. It then stopped with no error
and no python/ray process left alive. Likely cause: it was launched in a bare shell,
not tmux/nohup, so a dropped SSH session would silently kill the Ray driver — matches
the log exactly (clean stop mid checkpoint-dump, no exception). Not confirmed:
dmesg/journalctl weren't reachable from this session to rule out an OS-level OOM-kill,
but system RAM peaked at 137 GiB of 251 GiB total, nowhere near exhausted, so that's
unlikely to be it.

Tried `actor_rollout_ref.actor.ulysses_sequence_parallel_size=2` on GPU 0/1 only
(2026-09-02, `grpo-fullft-seq2-0902-1114`) to shard the long multi-turn/multi-image
activations across GPUs, since `GPU_MEM_UTIL` sweeps only touch vLLM's rollout pool
and had already been ruled out. RULED OUT INSTANTLY, never touched a GPU: verl
hard-requires `use_remove_padding=True` for ulysses SP (`verl/workers/config/
actor.py:319`), and `use_remove_padding` is exactly the Qwen2-VL-shaped packing path
`echo_grpo.yaml` already keeps off (its own comment says so; see CLAUDE.md's
Qwen2VLImageProcessorFast/rope note). Don't retry `SEQ_PARALLEL` without first
fixing `use_remove_padding` support for Qwen3-VL — that's a bigger, separate problem,
not a one-line unblock.

Still unresolved: full fine-tune on this box. 39-45 GiB against a 48 GiB budget with
no NVLink (all-gather over PCIe) is tight even before ref-model and vLLM rollout
memory are accounted for. Next steps to actually fix full FT, not routed around it:
try `GPU_MEM_UTIL` lower than 0.5, `TP_SIZE=4` instead of 2 (less rollout memory per
GPU, more communication), or accept the offload combination that was already at
True/True and look at whether `MAX_TOK_PER_GPU`/micro-batch sizing, or
`max_response_length`/`max_prompt_length` themselves, are the remaining slack. Always
launch multi-hour runs under tmux/nohup regardless, so a dropped connection doesn't
take a run down silently again.

2026-09-02, GPU 0/1 only, `ATTN_IMPL=sdpa` (flash_attn standalone package was never
installed in this venv -- only vLLM's bundled copy, which doesn't cover the ref
worker's `from_pretrained(attn_implementation=...)` check; it fails with `ImportError`
at `ref_init_model` specifically for full FT, since LoRA's ref reuses the actor with
adapters disabled and skips this path entirely) and `max_prompt_length`/
`max_response_length` halved to 2048/2048: got further than any prior full-FT
attempt -- past model load, into actual rollout (tool schema flowing to the model) --
then hit a NEW failure mode, HOST RAM, not GPU memory. Ray's own OOM killer fired:
`ray.exceptions.OutOfMemoryError: 2 worker(s) were killed due to the node running`
out of memory. System RAM: 173 GiB free / 251 GiB total after the crash, so it
recovered clean. Top memory users at kill time: the two FSDP actor WorkerDicts
(param_offload+optimizer_offload push the whole 8B model's shards into host RAM) at
~42 GiB and ~40 GiB each, the vLLM engine at ~22 GiB, and 8 `AgentLoopWorker`
processes (`AGENT_WORKERS=8` default) at 9-17 GiB each -- these add up past what 2
GPU-workers' worth of host RAM headroom can hold. wandb run `pyrtnkc9`
(https://wandb.ai/anaatef9-mbzuai/echo-grpo/runs/pyrtnkc9), no steps logged, died
before the first one. Next: cut `AGENT_WORKERS` (8 -> 4 or fewer) to shrink concurrent
rollout-worker host RAM before touching the actor's offload settings, since that's
the more direct lever for this specific failure. `run_grpo.sh`'s `tee -a` target
(`/hdd2/ahmedaly/echogrpo/logs/...`) doesn't exist -- the script only `mkdir -p`s a
`logs/` dir relative to the repo -- so use that path or fix the tee target before the
next attempt, this run's stdout only survived in the tmux scrollback.

`AGENT_WORKERS=4` tried next (2026-09-02, same GPU 0/1 + sdpa + short-seq config):
SAME failure, `240.92GB / 251.50GB` on the node. Halving worker count did NOT halve
memory -- each of the 4 `AgentLoopWorker`s just grew to 17.7-27 GiB instead of the
prior 8 workers' 9-17 GiB, because they're splitting the SAME total in-flight rollout
volume (`train_batch_size=32 * rollout.n=5` = 160 episodes), not reducing it.
`AGENT_WORKERS` only changes how that volume is chunked across processes, not its
total size -- it is not a real memory lever, don't retry it for this. Breakdown at
kill time: actor WorkerDicts ~85 GiB combined (fixed cost of full-FT sharded only 2
ways -- doesn't shrink without more GPUs or LoRA, both off the table), AgentLoopWorkers
~94 GiB combined, vLLM engine+workers ~33 GiB. The real lever is total rollout
volume: `ROLLOUT_N` (5) and/or `TRAIN_BATCH_SIZE` (32) directly set how much of that
94 GiB exists in the first place. Try `ROLLOUT_N=2` next, revert `AGENT_WORKERS` to
its default.

`ROLLOUT_N=2` (down from 5) tried next, `AGENT_WORKERS` back to default 8: SAME
failure again, 244.98GB/251.50GB, 3 workers killed this time (worse). Breakdown:
actor WorkerDicts unchanged at ~86 GiB combined (expected -- offloaded model+optimizer
state is static, doesn't scale with rollout.n at all), but AgentLoopWorkers ALSO
stayed at ~94 GiB combined despite halving samples-per-prompt. `ROLLOUT_N` is not a
real lever either then -- whatever is sizing these workers isn't proportional to
rollout.n. The one dimension not yet touched: `TRAIN_BATCH_SIZE` (32 prompts/step,
each carrying its own multi-turn image history -- view menu + every tool
observation). Trying `TRAIN_BATCH_SIZE=8` (+ matching `PPO_MINI_BATCH_SIZE=8`) next,
since prompt count, not sample count, looks like the actual driver of rollout-worker
memory.

`TRAIN_BATCH_SIZE=8` (down from 32, a 4x cut) tried: SAME failure again,
245.94GB/251.50GB. Breakdown essentially IDENTICAL to every prior attempt: actor
WorkerDicts ~85 GiB, AgentLoopWorkers ~97-110 GiB, vLLM ~14-33 GiB. Four attempts now
(`AGENT_WORKERS` 8->4, `ROLLOUT_N` 5->2, `TRAIN_BATCH_SIZE` 32->8, and combinations)
have moved the total by less than a few GiB despite each one cutting the nominal
rollout workload 2-4x. CONCLUSION: none of the volume-side knobs are the actual
lever. AgentLoopWorker memory looks like a roughly FIXED per-process cost (each of
the 8 workers imports the full torch/transformers/vllm stack plus whatever the echo
tool environment loads), not something proportional to batch size or sample count —
worth checking whether `echo_tool.py`/the env's dataset backing eagerly loads more
than the current episode needs, since that would explain memory that doesn't move
with workload. Checked: system-wide, other users' processes are NOT the cause (their
combined RSS was under 6 GiB during the crash) — this is entirely our job.

Stopping the pure config-knob sweep here; four consecutive near-identical ~246GB
failures despite touching every rollout-volume knob in `run_grpo.sh` means the next
useful step is reading the code (why is AgentLoopWorker memory workload-independent?),
not more CLI overrides. Left un-tried and likely genuinely out of reach on 2 GPUs:
going back to 4 GPUs (halves the fixed ~85 GiB actor cost via finer FSDP sharding,
but was ruled off by a resource-sharing constraint, not a technical one).

**RESOLVED (2026-09-09/10), see #9**: the mystery above (AgentLoopWorker RAM not
moving with any rollout-volume knob) was never a rollout-memory problem at all --
`trainer.val_before_train` (verl default `True`) runs a full agentic pass over the
entire val set before step 1 even starts, and this project's val parquet was every
QA pair from the full 1,215-study held-out set (31,209 rows). No CLI override in
this doc's sweep touches that path, which is why nothing moved. Fix:
`trainer.val_before_train=False` for every training-loop run; the 31k set is for a
standalone final eval (`echo_verl/eval/run_eval.py`), never verl's in-loop
validation.

## 9. GRPO + video: the NCCL hang, root-caused and fixed; training running (2026-09-10)

The video-tool NCCL hang blocked this project across multiple sessions (see
`echo_verl/echo_tool.py`'s `ECHO_SELECT_VIEW_VIDEO` gate history and the
`external/verl` patch to `tool_agent_loop.py` for the earlier "not forked" deviation
that made tool-returned video possible at all). Root cause found and fixed here.

**Root cause**: verl's default FSDP sharding (ZeRO-3, `reshard_after_forward=True`)
re-gathers each FSDP-wrapped submodule's parameters on every forward call it makes.
Qwen3-VL's forward invokes its vision/video encoder a DATA-DEPENDENT number of times
depending on whether a given micro-batch's samples carry image vs video multimodal
content. Once `select_view` returns real video, the two data-parallel ranks'
micro-batches can carry different image/video mixes (which episodes land on which
rank is not controlled), so the two ranks issue different NUMBERS of parameter
all-gather collectives and deadlock. Confirmed via NCCL's own watchdog dump: rank 0
stuck on `SeqNum=6794`, rank 1 on `SeqNum=6787` -- a 7-collective gap -- with
different `NumelIn`/`NumelOut` on the stuck collective on each rank, ruling out a
simple shape-mismatch (an earlier session's hypothesis, already fixed by normalizing
every view's video tensor to an identical shape -- that fix was necessary but not
sufficient).

**Fix**: `actor_rollout_ref.actor.fsdp_config.fsdp_size=1`. Switches the actor to
DDP-style replication -- full params resident on every GPU, no per-module shard
all-gather to desync, just a plain gradient all-reduce once per step. This hit one
verl bug: `compute_log_prob`/`compute_ref_log_prob` unconditionally call
`._handle.reshard(True)` assuming a sharded strategy, which asserts
(`Expects sharded strategy`) under `fsdp_size=1`'s `NO_SHARD`. Patched (guarded with
the same `.uses_sharded_strategy` check the assertion itself uses) --
`external/verl-video-nccl-fix.patch`, must be re-applied (`git apply
../verl-video-nccl-fix.patch` from `external/verl/`) after any fresh
`git submodule update --init`, since the submodule remote is the read-only upstream
`volcengine/verl` and this commit cannot live there. `scripts/check_train_env.py`
now asserts the patch is present rather than warning if verl still refuses video.

**Memory cost**: full (unsharded) params + fp32 Adam optimizer state resident per
GPU instead of split across 2 pushed a 2B model's actor to the edge of a 48GB A6000
alongside vLLM's rollout engine -- needed `GPU_MEM_UTIL` 0.5->0.3 and
`MICRO_BATCH_SIZE` 4->1 to get real headroom (the logits tensor in the loss
computation, `micro_batch x seqlen x vocab`, is the spike that OOM'd at the old
settings).

**Also fixed alongside this** (each cost real time chasing separately, noted so they
don't get re-discovered): `use_kl_loss=False` (the `kl_loss_coef=0.001` default
contributed ~2% of the gradient signal for an 89s/step reference-model pass -- off
for throughput runs, revisit if recipe-comparability to EchoSonar-R matters later);
Ray's `_temp_dir` must be a short LOCAL path (`/tmp/rayecho`, not `/data/...` --
GCS's port-file poll timed out on the latter) and `RAY_raylet_start_wait_time_s=180`
(the box can be under enough load from other users' jobs that the default 60s wait
races the raylet's actual startup).

**Confirmed**: 160+ real GRPO training steps clean as of this writing
(`grpo-2b-video-0910-1319`, wandb `1rn8g4bt` for steps 1-50 / a fresh run id after
the resume for 51+, checkpoints every 10 steps, one deliberate pause+resume at step
50 for a mid-run eval -- resume via `trainer.resume_mode=auto` finding
`latest_checkpointed_iteration.txt` worked cleanly, just requires pinning
`EXP_NAME`/`CKPT_HOME` to the original run's rather than the script's default
timestamped name). `ECHO_SELECT_VIEW_VIDEO` now defaults ON in `echo_tool.py`.

**Separately, not this project's fault**: `/hdd2/ahmedaly/echogrpo/` (the old venv +
checkpoints + preprocessed-data-adjacent location) threw real `Input/output error`s
and briefly remounted read-only during the same session a `flash-attn` source build
OOM'd the whole box (124 `cicc` compiler processes, ~242GB RAM + all swap). Came back
clean after a reboot with no further errors since, so most likely memory-pressure-
induced I/O starvation during that OOM, not physical disk failure -- but the working
env was moved to `/data/ahmedaly/mashrafi_echogrpo/` (conda prefix env, healthier
disk, 2.9TB free) regardless and stayed there. `HF_HOME`/`ECHO_PREPROCESSED_DIR`
still point at `/hdd2` (its `env.sh`, unmoved) since that data was never actually
lost.

## 10. Eval harness (`agentic_loop.py`) drifted from the training tool contract (2026-09-10)

`run_eval.py`'s agentic loop is a standalone reimplementation of verl's
ToolAgentLoop -- deliberately, per its own docstring (eval shouldn't need the whole
Ray/FSDP stack up, and wants the tool trace as a first-class output verl's loop
doesn't hand back). It fell out of sync with training in three ways: a different
hand-written system prompt instead of `generate_trainset._SYSTEM`; the tool
description passed as free text in the prompt instead of via the API's `tools=`
param + chat template (so the model saw a differently-rendered tool definition than
in training); `max_tokens=1024` vs training's `max_response_length=4096`.

**Symptom before the fix**: evaluating the step-50 checkpoint gave
`tool_call_rate=0.02` (training rollout does ~5 calls/episode) and 37% of episodes
truncated mid-`<think>`, never reaching an answer or a tool call.

**Fixed**: `SYSTEM_PROMPT` now imports `generate_trainset._SYSTEM` verbatim.
`ECHO_TOOL_SCHEMA` loads the same `echo_verl/configs/echo_tool_config.yaml` verl's
rollout uses and is passed via `tools=` (the vLLM eval server needs
`--enable-auto-tool-choice --tool-call-parser hermes` for this). `max_tokens`
1024->4096 in both `run_episode` and `run_plain_episode`. vLLM's
`--limit-mm-per-prompt` image cap raised 32->64 to match `--max-images`.

**After the fix**: `tool_call_rate` jumped to 0.495, all three ops used
(`select_view` 185, `select_frames` 229, `zoom` 36 across 200 episodes). This
surfaced a REAL finding, not a harness artifact: the step-50 model over-explores and
often never converges to an answer. Classification finish reasons: `max_turns`
23/40, `answered` 9/40, `no_answer_no_tool` 7/40; `balanced_accuracy` 0.048 (below
the 0.5 majority baseline; EchoSonar-R reports ~0.49 for their fully-trained GRPO
model). Consistent with training's own `num_turns/mean` sitting near the 6-turn cap
the whole time. Likely needs an explicit penalty for hitting `max_turns` without
answering, or a format reward that requires `<answer>`, before scores are
meaningful -- not yet done, revisit once the current epoch finishes (this was step
50 of 316).

**Not fixed, structural**: the eval loop being a hand-maintained copy of training's
protocol is itself the bug class here, and will drift again. It should either run
through verl's own generation path, or be a thin wrapper over verl's tool-calling
components (`echo_tool_config.yaml`, `EchoTool`), not an independent
reimplementation that has to be manually kept in lockstep.
