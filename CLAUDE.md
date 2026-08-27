# EchoSonarVideo

Agentic RL on multi-view cardiac ultrasound video: cold-start SFT -> GRPO, on
upstream verl (pinned submodule `external/verl` @ v0.7.1, NOT forked). Base model
Qwen3-VL-8B-Instruct. One composite `echo` tool with three ops: `select_view`,
`select_frames`, `zoom`.

This file holds RULES that are expensive to rediscover. Architecture and rationale
live in `echo_env/INTEGRATION.md` and `docs/TRAINING_ENV.md`; the dated design docs
under `docs/superpowers/` are history, not current state.

## Substrate: this is AMD, not NVIDIA

MI210 / gfx90a / ROCm 6.3.3, partition `faculty`, `--account=faculty-acc --qos=gtqos`.
No FP8, no flash-attn (use `attn_implementation=sdpa`). Any CUDA pin is wrong here;
`requirements-train.txt` is kept only as the portable CUDA spec and does NOT describe
this cluster.

Python is `.venv-train/bin/python` — an overlay venv (`--system-site-packages`) on the
`qwen_backup` conda env. **Never install into `qwen_backup`**; it is load-bearing and
shared. verl is installed there with `--no-deps`.

`qwen_backup`'s **vLLM is a CUDA wheel and cannot run on this GPU** (`_C.abi3.so` links
`libcudart.so.12`, no `_rocm_C`). It imports fine and then dies when the engine starts.
Evaluation therefore runs the model **in-process** via `echo_verl/eval/local_client.py`.
GRPO will still need a real ROCm vLLM. Every container route is closed here — verified,
do not re-litigate: `docker://` unpack hits the colon rule below; no docker group; no
writable non-VAST filesystem to unpack on; this apptainer 1.4.2 has no OCI-SIF support.
The open route is a colleague's prebuilt `.sif` (one file, no colon in its name).

`scripts/check_train_env.py` gates all of this. It asserts **runnability, not
importability** — the earlier gate reported 12/12 green while vLLM could not serve.

## Substrate: the second machine (4x CUDA GPU, for GRPO)

The AMD cluster has no working ROCm vLLM and no route to one: two from-source build
attempts both hit a **kernel driver ceiling** (`amdgpu-dkms` too old for any modern
ROCm-built torch, confirmed against a control run of the working torch build on the
same node/GPU) and the one prebuilt `.sif` known to work here is unreadable
(`drwx------` on its owner's home). SFT and evaluation stay on the AMD cluster; GRPO
moves to this second machine because it needs a served vLLM engine.

**UNVERIFIED — nobody has run any of this here yet.** This section is a checklist for
first contact with the new machine, not a confirmed-working recipe like the AMD
section above. Update it once real numbers replace the guesses.

**Repo:** `git clone --recurse-submodules https://github.com/Mashrafi27/EchoSonarVideo.git`
(two submodules, `external/verl` and `external/DeepEyes` — the `--recurse-submodules`
flag is not optional, a plain clone leaves both empty).

**`requirements-train.txt` is the real spec here**, not a reference kept for someday.
Its header says so explicitly: CUDA wheels (vllm 0.17.0, flash-attn, torch cu129),
derived from pinned `external/verl@v0.7.1`'s own `setup.py`. It has never been
installed from — first install on this machine is also its first real test.

**Everything ROCm-specific goes away, and has to be found and removed, not just
ignored:**
  - `attn_implementation="sdpa"` → `"flash_attention_2"`, in `run_sft.sbatch`,
    `sft_smoke.sbatch`, and `echo_verl/eval/local_client.py`. sdpa was the ROCm
    workaround (CLAUDE.md's own AMD section: "No flash-attn"); a CUDA box has no
    reason to still take the slower path.
  - The four ROCm sbatch traps (`ROCR_VISIBLE_DEVICES`→`HIP_VISIBLE_DEVICES`,
    `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES`, the per-process MIOpen DB dirs,
    the `no_padding`/position-id nested-tensor sizing invariant) are ROCm-only. The
    fourth (`samples_per_rank * max_seq_len <= max_token_len_per_gpu`) is a real verl
    invariant and may still bite on CUDA; the other three simply do not apply and
    copying them into a fresh sbatch script would be dead code, not caution.
  - `echo_verl/eval/local_client.py` (in-process HF-`generate()` evaluation) exists
    **only** because the installed vLLM here is a CUDA wheel that cannot serve on
    ROCm. On a real CUDA machine, vLLM should serve normally — evaluation can
    probably go back to `run_eval.py --base-url` against a served engine instead of
    `--local-model`, which is the whole point of moving here for GRPO in the first
    place. Confirm this before assuming it; do not delete `local_client.py`, the AMD
    cluster still needs it for SFT-checkpoint eval.
  - No `qwen_backup` conda env, no `.venv-train` overlay on this machine — those
    names refer to the AMD cluster's specific environment layout. Build a fresh env
    from `requirements-train.txt` there is no equivalent to reuse.

**Before trusting any result:** write (or adapt) a `check_train_env.py` equivalent
for this machine before running real training on it — the ROCm one exists because an
earlier gate reported 12/12 green while vLLM silently could not serve. Assume the
same failure mode is possible here until something actually proves the served engine
answers a real multi-turn, multi-image tool-call request, not just that it imports.

**Data path:** `ECHO_PREPROCESSED_DIR` and any hardcoded `/vast/...` paths in configs
need to resolve on this machine too — check whether it can see the same VAST mount or
whether the preprocessed data needs copying over.

## Filesystem: VAST rejects characters in filenames

`*` and `:` both fail with Errno 22 / "No such file or directory". This is what killed
the container pull (`gcc-12-base:amd64.list`) and what disables the torch NVRTC kernel
cache (`gfx90a:sramecc+:xnack-` — a warning, not a failure). Fix the tool's flags; never
fall back to `/tmp`. Temporary work goes in `.tmp_work/`, and gets cleaned up.

## Every SLURM script needs the ROCm preamble

Four traps, each found by burning a job. All four are already handled in
`scripts/*.sbatch` — copy an existing script rather than writing one from scratch
(`merge_ckpt.sbatch` was written fresh and died in 32s on trap 1).

1. SLURM exports `ROCR_VISIBLE_DEVICES`; torch 2.7 ROCm hard-errors. Translate to
   `HIP_VISIBLE_DEVICES` and unset ROCR. Needed even in jobs that never touch the GPU.
2. `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES` (verl's AMD docs recommend it) breaks
   the **torchrun** SFT path. It is for the ray RLHF path only.
3. MIOpen's kernel DB under `~/.config/miopen` is read-only on VAST -> any Conv (the
   Qwen3-VL vision patch-embed) dies with `miopenStatusInternalError`. Set
   `MIOPEN_USER_DB_PATH`/`MIOPEN_CUSTOM_CACHE_DIR` to a writable per-job dir — and a
   **per-process** dir when sharding, or contention re-trips it.
4. `pad_mode=no_padding` collates via `torch.nested`; a micro-batch holding ONE sample
   makes torch pick the wrong ragged dim on `(4, seq_len)` VLM position ids. Invariant:
   `samples_per_rank * max_seq_len <= max_token_len_per_gpu`.

The QOS rejects GPU-less jobs (`QOSMinGRES`), so CPU-only work still requests one idle GPU.

## verl gotchas

- verl 0.7.1 applies the **Qwen2-VL** rope to Qwen3-VL — its `image_processor` is
  `Qwen2VLImageProcessorFast`, so verl's substring gate passes and it raises `IndexError`.
  Worked around by `echo_verl/sft_dataset.py` via `data.custom_cls`, not by patching verl.
- `WANDB_PROJECT`/`WANDB_NAME` are **ignored**: verl calls `wandb.init(project=..., name=...)`
  from `trainer.project_name`/`experiment_name` (`utils/tracking.py:80`), and explicit
  kwargs beat the environment. Those config keys are the real knobs.
- ToolAgentLoop refuses tool-returned **video**, so tool observations are IMAGES
  (the HYBRID frame path). Never populate `videos` anywhere: `Qwen3VLVideoProcessor`
  has `do_sample_frames=True, fps=2` and silently resampled a 19-image view menu to 4
  frames. `scripts/check_prompt_parity.py` guards this.

## Working rules

- Compute nodes DO have outbound HTTPS, so wandb logs online; credentials are in
  `~/.netrc` and no key belongs in the repo.
- Data-scale defaults are a known trap: a `--limit 3000` smoke default went unrevisited
  and one full SFT run trained on 2.3% of the corpus. State the record count you are
  actually training on.
- The comparison paper is **EchoSonar-R** (arXiv 2606.28164) — same private dataset,
  same SFT->GRPO recipe, no tools. CardioBench (arXiv 2510.00520) is only the source of
  metric definitions. Do not confuse them.
- `abnormality_classification` is 82% "no", so always report balanced accuracy; plain
  accuracy flatters an always-no model.
- Never silently approximate a metric we cannot compute (METEOR, BERTScore, GREEN are
  absent on purpose rather than reported as 0.0).

## More Background

- The project is mainly based on two papers:
  1. EchoSonar-R by Taratynova, D. et al
  2. DeepEyes by Zheng, Z. et al
  Cardiobench by Aly, A and Taratynova, D. et al is also a good background for context

## How to reply to me
- It is important to format the replies in a very concise way.
- Replies should only concern one issue and list other issues it wants to raise in the end, it should keep the other issues that were raised and not talked about in a sepearate file and keep listing them at the end of replies unless it is resolved.
- it should have a tldr before starting to talk about something
- in the end it should also list any experiments going on, it should not refer to it by job_id, rather a small descriptive name/phrase.

## Wandb
- It is very important that every experiment is being logged to wandb
- after starting an experiment you should check the wandb link for that experiment and add to the reply

## General Consideration
- Learn from my speaking style and start talking like that
- never use and em-dash, and "not just <>, it's <>" basically, any sort of dramatic way of saying things. Say things normally.