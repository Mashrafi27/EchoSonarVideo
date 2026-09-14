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

## 11. New track: `echo_ep` -- frozen EchoPrime + Qwen3-8B-text cold-start SFT, matching EchoSonar-R's actual architecture (2026-09-11/12)

**Why**: the running Qwen3-VL GRPO track (see #9, #10) never got a cold-start SFT --
the policy is learning the `<think>`/`<answer>` format, clinical report-writing
style, and the task itself all at once during GRPO, which is plausibly why
report-writing question types stay near-zero reward the whole run and why held-out
classification accuracy doesn't track the training reward (checked at steps
50/170/250). EchoSonar-R's own paper (`2606.28164v1.pdf`, "Implementation
Details", read directly) does a real cold-start first: a **frozen EchoPrime (mViT)
video encoder**, dv=512 in the actual checkpoint (the paper states 768; the real
weights produce 512 -- trust the checkpoint), feeding a **text-only Qwen3-8B**
(not Qwen3-VL) language model, **SFT for 3 epochs** (LR 2e-5, AdamW, batch 64),
*then* GRPO (LoRA r=64/a=128, frozen base, 5 epochs). User's decision: replicate
this architecture for real, and continue GRPO on the SAME architecture afterward
(the vLLM-rollout-serving problem for this custom architecture is a known, explicitly
deferred follow-up -- verl's GRPO rollout goes through vLLM's own model registry,
which has no path today for a frozen-encoder+custom-LM fusion; not solved here).

**Found on this box**: `/home/mashrafimonon/Mashrafi/EchoPrime/` is a complete,
already-working local EchoPrime install (video/text encoder weights, view
classifier, prior linear-probe run logs) -- no download needed. License is
Cedars-Sinai's Academic Software License (not MIT), fine for this academic
project, requires an acknowledgment on any publication
(Vukadinovic/He/Ouyang, Cedars-Sinai), no redistribution. Public upstream is
github.com/echonet/EchoPrime (arXiv:2410.09704), MIT-licensed on GitHub itself --
the discrepancy is real, defer to the LICENSE file actually shipped with the
weights we're using. `/hdd2/ahmedaly/echoprime_study_cache.pt` (owned by `darya`,
an EchoSonar-R co-author) is a *text*-encoder embedding cache for a different,
larger study set (79,584 studies) -- not reusable for our video embeddings, and
not investigated further (not our file).

**New package**: `echo_ep/` -- `preprocess.py` (ports EchoPrime's own frame
sampling/normalization: 32-native-frame window, stride 2 -> 16 frames, 224px,
EchoPrime's own per-channel mean/std on the 0-255 scale, NOT ImageNet-style;
`crop_and_scale`'s center-crop-to-aspect + 10% zoom, verbatim from their
`utils.py`), `encoder.py` (frozen mViT-v2-S wrapper), `build_video_cache.py` (see
below), `modeling.py` (`EchoPrimeQwen3ForCausalLM`: LLaVA-style splice of one
projected soft token per view into Qwen3-8B's input embeddings at `<|view_embed|>`
placeholder positions), `dataset.py` (collator + label-masking), `train_sft.py`.

**Caching, not per-QA-pair encoding**: the frozen encoder's output for a study
never changes regardless of which of that study's ~25 QA pairs (128,215 pairs /
5,061 train studies) is training, so `build_video_cache.py` runs it once per study
and caches `(N, 512)` embeddings to `build/echoprime_video_cache/<study_uuid>.pt`,
not once per QA pair (25x fewer encoder passes). Only the train split's 5,061
studies are cached for now -- the 1,215 test studies aren't needed until the later
eval/verification step, no reason to spend that compute now. Launched as 8
parallel shards (`build/train_study_ids_shard0{0..7}.txt`), logs at
`/data/ahmedaly/mashrafi_echogrpo/logs/epcache_shard*.log`. **Real, load-bearing
bug found and fixed during this build**: the original `load_clip_frames` read
every native frame in a clip's PNG directory (up to 90 seen) when EchoPrime's own
`[start:start+32:stride]` slice never looks past index 32 -- capped the read at
`FRAMES_TO_TAKE` frames, cutting wasted I/O by 2-3x for long clips. Even after
that fix this is genuinely slow (I/O-bound against `/hdd2`, not CPU-bound --
`user` CPU time was consistently about half of wall time): roughly 15-20+ hours
for all 5,061 studies at 8-way parallelism on this box's disk. One-time cost, not
repeated per training run.

**Architecture validated correct, three real config bugs found and fixed while
getting a training-loop smoke test working**:

1. Single-GPU eager-mode forward+backward (no FSDP/DeepSpeed) confirmed the model
   itself is correct: real loss (~0.7, sane), non-zero gradients into both the
   projector and the LM, 33.7GB peak with gradient checkpointing -- but
   `gradient_checkpointing_enable()` only actually activates in `model.train()`
   mode; leaving the model in eval() (the default after construction) makes HF
   silently skip checkpointing and store full activations instead, which looks
   identical to checkpointing "not helping" until you notice the model was never
   put in train mode. Easy to miss, real trap.

2. **FSDP was tried first** (matching the actor's own FSDP+offload shape) and hit
   a real, unresolved-for-this-architecture problem: `transformer_layer_cls_to_wrap:
   ["Qwen3DecoderLayer"]` never actually found per-layer wrap boundaries, because
   the real LM lives nested one level inside `EchoPrimeQwen3ForCausalLM.lm`, not at
   the top level Trainer sees -- confirmed by the OOM happening at
   `flatten_tensors_into_flat_param`'s `torch.cat` over EVERY parameter at once
   (~30GB attempted allocation) instead of one decoder layer's worth (~500MB).
   Along the way also hit and fixed: FSDP's mixed-precision wrapping silently
   upcasting bf16-loaded weights back to fp32 first ("Upcasted low precision
   parameters ... because mixed precision turned on in FSDP" -- load in fp32 for
   FSDP, let `bf16=True` + `--mixed_precision bf16` handle real casting, not
   manual bf16 loading), and the FSDP wrap-step's transient full-unsharded-model-
   per-rank memory spike (mitigated with `sync_module_states` +
   `cpu_ram_efficient_loading` + setting `ACCELERATE_USE_FSDP`/
   `FSDP_CPU_RAM_EFFICIENT_LOADING` env vars *before* the nested `from_pretrained`
   call, since Trainer's own FSDP setup runs too late to affect it). None of this
   fixed the core per-layer-wrap problem, though -- switched to DeepSpeed instead
   of chasing it further.

3. **DeepSpeed ZeRO-3 worked** (`echo_ep/ds_zero3.json`, CPU offload for both
   params and optimizer state) -- partitions parameter-by-parameter, so it never
   depended on the layer-class-name auto-wrap matching that broke FSDP for this
   nested architecture. Three environment issues surfaced getting `cpu_adam`'s
   JIT-compiled extension to actually build+load, all specific to this
   pip-wheel-based conda env (not present in a "normal" system CUDA install):
   a stale `TORCH_CUDA_ARCH_LIST` env var (left over from an earlier, unrelated
   flash-attn build attempt this session) included an arch string ("10.1") this
   torch version's `_get_cuda_arch_flags` doesn't recognize -- override to `8.6`
   (this box's real Ampere/A6000 arch) for any future JIT-compiled CUDA extension
   in this env; the pip `nvidia-curand-cu12` package only ships a versioned
   `libcurand.so.10`, no unversioned `libcurand.so` symlink, so `-lcurand` failed
   at link time -- fixed with a symlink in place (`ln -sf libcurand.so.10
   libcurand.so` in that package's `lib/` dir) plus `LIBRARY_PATH` pointing at it;
   and the conda env's own newer `libstdc++.so.6` (matching its gcc 14.3.0, needed
   for the `CXXABI_1.3.15` symbol the compiled `.so` requires) wasn't being found
   at import time in favor of the system's older one -- fixed with
   `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:...`. All three needed for ANY future
   DeepSpeed CPU-offload op JIT-compile in this exact env, not just this one.

**Result**: with all of the above, a 3-step smoke test (2 GPUs, DeepSpeed ZeRO-3,
`Qwen/Qwen3-8B` + frozen EchoPrime + projector, ~800 studies cached at the time)
completed all 3 steps with real, sane losses (0.79, 0.91, 0.91) and grad norms
(10-13) -- forward, backward, gradient accumulation, and the optimizer step are
confirmed working end to end. `trainer.save_model()` afterward hung on a
ZeRO-3 weight-gather broadcast and hit the 30-minute NCCL watchdog timeout --
almost certainly system-load-induced (this box was at load average 155-280 at the
time, from the video-cache-build job plus this session's own repeated
smoke-test iterations competing for the same disk/CPU) rather than a code bug;
worth retrying the save step in isolation once load is normal before assuming
otherwise.

**Not yet done**: the real 3-epoch training run (waiting on the video cache to
finish -- see above), a clean `save_model()` under normal load, and picking the
final `<think>`/`<answer>` assembly convention for the assistant turn (currently
`<think>{thinking field verbatim}</think><answer>{messages' assistant content}</answer>`,
matching `echo_verl/generate_trainset.py`'s existing convention -- not verified
against what EchoSonar-R's own paper actually does for their "reasoning chain",
which isn't detailed enough in the paper text pulled so far to confirm).

## 12. Real GRPO rollout via vLLM working end to end; a real data-scale/split mistake caught late (2026-09-13)

**Real vLLM serving for the `echo_ep` composite model (see #11) now works.** Standalone
smoke test confirmed vLLM custom-model registration + weight loading + generation
mechanics; wired into verl's `AgentLoopManager` via a custom `EchoPrimeAgentLoop`
(`echo_ep/echoprime_agent_loop.py`). A full training run (150 steps, single GPU) ran
to completion with checkpoints every 15 steps, wandb logging throughout. Getting there
took a long chain of real bugs -- see CLAUDE.md's "GRPO gotchas specific to the
frozen-EchoPrime + Qwen3-8B-text composite model" for the full technical detail
(`tie_word_embeddings` defaulting wrong on the composite config, `eos_token_id` not
serializing to the top-level `config.json`, `rollout.load_format: dummy` broken for
this model's initial weight sync, the standalone LoRA-adapter export saving empty,
untrained-policy `VIEW_TOKEN` self-sampling). The `tie_word_embeddings` bug in
particular cost the most time: it produces complete, temperature-independent garbage
output that looked identical to several unrelated hypotheses (response-length cap,
sampling temperature, `eos_token_id`) tried first, and standard diagnostics (weight
loading, config hyperparameters, forward-pass structure) all checked out fine while it
was still broken.

**Separately, and more seriously: an entire session's worth of GRPO training and eval
ran against a broken train/val split, caught only after the user asked "why 20
studies, not the ~31k I remember."** Two compounding mistakes in
`echo_ep/generate_grpo_parquet.py`'s original invocation:

1. `--limit 200` / `--limit 20` with no `--study-list`, silently capping the
   echoprime track to ~0.16% of the real train pool (`build/rl.jsonl`: 128,215 rows /
   5,061 studies) and an equally tiny val set, with no record left behind of why
   200/20 were chosen. This is the exact "Data-scale defaults are a known trap"
   failure mode this file already warned about, hit again.
2. The "val" set built this way had **100% study overlap with the "train" set** --
   `generate_grpo_parquet.py` filters by `--study-list`, not by the record's own
   `split` field, and no study-list was passed, so both parquets pulled from the same
   file-order prefix of `rl.jsonl`. Every eval number produced during this session
   (checkpoint-quality checks, "the model can produce a real answer" confirmations)
   was measuring train-set recall, not held-out generalization. The mechanical
   findings above (vLLM serving works, the bugs are real and fixed) are unaffected --
   they're about the *pipeline* running correctly, not about model quality -- but no
   quality/reward number from tonight should be quoted as a real result.

**Ground truth for the real split** (from `scripts/build_grpo_parquet.sh`'s own
header, previously mis-derived from the wrong field -- see CLAUDE.md's data-pipeline
section for the full explanation): `build/rl.jsonl` is the entire train_vqa pool, no
internal val split; `build/eval.jsonl` is the entire test_vqa pool, confirmed zero
study overlap with `rl.jsonl` by construction. The Qwen3-VL GRPO run's own val parquet
was built from the **full** `eval.jsonl` (~31k rows) -- matching the user's own memory,
which is what surfaced this. `generate_grpo_parquet.py` now supports `--shuffle-seed`
for a real random sample; the two study-uuid lists must come from `rl.jsonl`/
`eval.jsonl`'s file-level membership, never from a record's own `split` field.

**Also found while fixing this**: the frozen-EchoPrime video-embedding cache
(`build/echoprime_video_cache/`) had 0% coverage for `eval.jsonl`'s 1,215 real test
studies -- it had only ever been built against `rl.jsonl`'s 5,061 train studies.
Rebuilding it for the test studies was in progress at the time of this entry
(`echo_ep/build_video_cache.py --study-list build/echoprime_test_study_ids.txt`).

**Not yet done**: a real GRPO run on a correctly-scaled, correctly-split train set;
a real held-out eval once the test-study cache finishes building; a vLLM-batched
rewrite of `echo_ep/eval_checkpoint.py` (currently one-example-at-a-time HF
`.generate()`, which does not scale to a few-thousand-row eval in reasonable time);
reconciling this with a comparison against EchoSonar-R's own reported numbers and the
existing Qwen3-VL GRPO run's numbers on the same real eval set.
