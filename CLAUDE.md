# EchoSonarVideo

## Active direction (2026-09-22)

The user has reset the research to basic inference with the released DeepEyes
checkpoint on a few held-out QA pairs. Previous training and evaluation work
is past work. Keep its artifacts and these operational lessons; follow
`PLAN.md` for current next steps rather than resuming older training plans.

Agentic RL on multi-view cardiac ultrasound video: cold-start SFT -> GRPO, on
upstream verl (pinned submodule `external/verl` @ v0.7.1, NOT forked). Base model
Qwen3-VL-8B-Instruct. One composite `echo` tool with three ops: `select_view`,
`select_frames`, `zoom`.

This file holds RULES that are expensive to rediscover. Architecture and rationale
live in `packages/tool_env/INTEGRATION.md`. See also `SPEC.md` (facts: data, architectures,
real experiment results) and `PLAN.md` (current open questions and next steps).

## Substrate: 4x CUDA GPU box, shared with other users

Real GRPO runs confirmed here on the frozen-EchoPrime + Qwen3-8B-text track
(separate from the Qwen3-VL tool-based track). That track started as a
cold-start-no-SFT ablation; since we got Darya's folder on the AMD machine it
initializes from her SFT checkpoint and runs there too (see the AMD section).

- **Clone with `--recurse-submodules`** (`external/verl`, `external/DeepEyes`) — a
  plain clone leaves both empty.
- `requirements-train.txt` is the real installed spec on this box (vllm 0.17.0,
  flash-attn, torch cu129, derived from `external/verl@v0.7.1`'s own setup.py).
  `attn_implementation=sdpa` in the echoprime config is intentional — flash-attn
  isn't installed in this conda env, sdpa is fine.
- vLLM serves normally here for both tracks. Small offline eval:
  `packages/eval/echoprime/eval_checkpoint.py` (HF `.generate()`, no vLLM) or
  `packages/eval/echoprime/eval_checkpoint_vllm.py` (LoRA merged into base, served via vLLM, faster
  for large evals). All evaluation code (both tracks + shared metrics) lives
  under the single `packages/eval/` package, not split per-track.
- `ECHO_PREPROCESSED_DIR` = `/hdd2/ahmedaly/echo_preprocessed_mmm` (6276 study
  subdirs = full train_vqa + test_vqa pool). `packages/echoprime_track/build_video_cache.py` builds
  the frozen-EchoPrime embedding cache, one file per study, idempotent/resumable,
  but only covers whatever `--study-list` you pointed it at.
- **Before trusting any eval number:** check `ls build/echoprime_video_cache | wc
  -l` against the study set you're actually evaluating. The cache was once built
  only against train studies and silently had zero coverage of the real held-out
  eval set for a full session. Zero coverage fails loud (`FileNotFoundError`); a
  *partial* cache would not — don't assume coverage, check it.

## Substrate: AMD MI210 cluster (ROCm), Qwen3-VL tool track

Partition `faculty`, `--account=faculty-acc --qos=myqos`. The login node has system
ROCm 6.3.3, which is the wrong runtime: the validated env is
`.tmp_work/rocm_validation_20260914/env` (vLLM 0.17.0 ROCm wheel, private ROCm 7.0.2
libs). Inference passed (job 185987) and two full GRPO updates plus vLLM weight sync
passed on 2x MI210 (job 186304), on the pre-`packages/` layout with image tool
observations. Details, versions and limits: `docs/ROCM_VALIDATION.md`.

- Launch through `scripts/validate_qwen_vllm.sbatch` / `scripts/validate_grpo.sbatch`.
  They source `scripts/rocm_workspace_scratch.sh`: VAST rejects `:` in filenames and
  COMGR's `gfx90a:sramecc+:xnack-` kernels then fail with a misleading
  `hipErrorInvalidDeviceFunction`. Scratch is a private ext4 image in `.tmp_work/`,
  never `/tmp`.
- Keep each spawned worker's `TRITON_CACHE_DIR` under
  `<ECHO_ROCM_SCRATCH_ROOT>/triton/<pid>` (set by `sitecustomize.py`). Do not restore
  a shared compiler cache without revalidating concurrent vLLM startup. See
  [the EchoPrime cache incident](docs/troubleshooting/echoprime_triton_cache.md).
- ROCm extensions: `tools/rocm_runtime_plugin` (vLLM device UUID + IPC socket
  redirect), `packages/verl_bridge/agent_loop.py` (one image placeholder per returned
  image), `packages/verl_bridge/fsdp_compat.py` (keep Qwen3-VL `visual.pos_embed` in
  the root FSDP unit under CPU offload), `packages/verl_bridge/main_ppo.py` (finish W&B
  inside the Ray worker).
- `validate_grpo.sbatch` sets `ECHO_SELECT_VIEW_VIDEO=0`: the video tool path has not
  been run on ROCm.
- SLURM exports `ROCR_VISIBLE_DEVICES`; translate to `HIP_VISIBLE_DEVICES` and unset
  it. MIOpen's DB under `~/.config/miopen` is read-only on VAST, so point
  `MIOPEN_USER_DB_PATH` at a per-process writable dir (done in
  `scripts/rocm_python_startup/sitecustomize.py`).
- The EchoPrime track also runs here now, off Darya's folder
  `/vast/users/mohammad.yaqub/report_generation/`: EchoPrime weights
  (`EchoPrime/model_data/weights`, exported as `ECHOPRIME_WEIGHTS_DIR`) and her SFT
  checkpoint (`checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503`),
  which `scripts/save_sft_init_checkpoint.py` converts to `build/echoprime_sft_init`.
  `ECHO_PREPROCESSED_DIR` on this machine is
  `/vast/users/mohammad.yaqub/project/preprocessed_data`. Launch via
  `scripts/smoke_grpo_echoprime_amd.sbatch` gating `scripts/run_grpo_echoprime_amd.sbatch`.

## Data pipeline: `rl.jsonl` / `eval.jsonl`

Ground truth is `scripts/build_grpo_parquet.sh`'s header, not the per-record
`split` field — that field is leftover lineage from an abandoned CSV split scheme
and does not mean "held out."

- `build/rl.jsonl` = all of `train_vqa_with_thinking.jsonl`. Full train pool, no
  further split inside it. 128,215 rows / 5,061 studies (count by `study_uuid`,
  not `split`).
- `build/eval.jsonl` = all of `test_vqa.jsonl`. Real held-out set. 31,209 rows /
  1,215 studies. Zero study overlap with `rl.jsonl` (5061+1215=6276, matches the
  frame tree exactly) — a real invariant of the source files, not re-checked
  downstream.
- Any new track's val parquet should source from the full `eval.jsonl` (no
  `--limit`), matching the tool-based track's `build/rl_val.parquet` convention —
  not an ad-hoc slice of `rl.jsonl`.
- **Always pass an explicit `--study-list`** built from the real partition above.
  Never rely on `--limit` alone: it caps by file order, not randomly, and doesn't
  filter by study — a real run once got `--limit 200`/`--limit 20` with no
  `--study-list`, capping train/val to ~0.2% of the pool with 100% study overlap
  between them. `generate_grpo_parquet.py` supports `--shuffle-seed` for a real
  random sample instead.

## GRPO gotchas: frozen-EchoPrime + Qwen3-8B-text (`packages/echoprime_track/`)

- **System prompts are stored in parquet.** Editing the source constant does not
  update existing training inputs. Before launch, run
  `python -m echoprime_track.check_grpo_prompts` on both parquet files with
  `--tokenizer <checkpoint> --max-prompt-length 3584`. Both AMD launchers enforce
  this, and the tool loop rejects stale messages. Repair with `--repair` and
  `--archive-dir <prior-run>/input_artifacts` to retain exact originals. See
  [the stale-prompt incident](docs/troubleshooting/echoprime_stale_prompts.md).

A from-scratch `PretrainedConfig` + custom vLLM model class hits sharp edges a
stock HF/vLLM model never does. All fixed in `packages/echoprime_track/modeling.py` unless noted.

- **`tie_word_embeddings` defaults to `True`** on `EchoPrimeQwen3Config` (HF's
  `PretrainedConfig.__init__` sets it before delegation to the nested real Qwen3
  config). Qwen3-8B is untied — wrong default routes output through the tied
  embedding matrix instead of the real `lm_head`, producing garbage output at any
  temperature. Fix: `from_cold_start` copies `eos_token_id`/`pad_token_id`/
  `bos_token_id`/`tie_word_embeddings` from the inner Qwen3 config explicitly.
- **`eos_token_id` has the same shadowing bug.** HF's live `__getattr__` masks it
  in-process, but it's missing from the raw `config.json`, which breaks vLLM
  specifically (100% of responses hit `max_tokens`, zero variance). Same fix.
- **`rollout.load_format: dummy` is broken for this model.** Per-step LoRA sync is
  fine, but the very first base-model sync doesn't work for this class (dummy →
  near-empty completions; `auto` → correct output, everything else identical).
  `packages/verl_bridge/configs/echoprime_grpo.yaml` defaults to `auto`. Root cause not fully
  isolated; suspect `collect_lora_params` in `verl/utils/fsdp_utils.py`.
- **Standalone `lora_adapter/adapter_model.safetensors` export is empty.**
  `layered_summon_lora_params` (`external/verl/verl/utils/fsdp_utils.py`) assumes
  the decoder sits at `.model.layers` or `.language_model.layers`; this model
  nests it at `.lm.model.layers`, so it matches zero layers. The full FSDP
  checkpoint (`model_world_size_*.pt`) is unaffected. Fix in
  `external/verl-lora-checkpoint-save-fix.patch`.
- **Untrained cold-start policy can sample its own `VIEW_TOKEN` as free text.**
  `_splice_view_embeddings` can't tell prompt from model output, so a stray token
  during generation trips an assertion later in training. Fixed by banning the
  token via vLLM `logit_bias` at generation time
  (`packages/echoprime_track/echoprime_agent_loop.py`), not by loosening the assertion.

## verl gotchas

- verl 0.7.1 applies Qwen2-VL rope to Qwen3-VL (its `image_processor` is
  `Qwen2VLImageProcessorFast`, so verl's substring gate passes wrongly and raises
  `IndexError`). Worked around via `data.custom_cls` in `packages/verl_bridge/sft_dataset.py`,
  not by patching verl.
- `WANDB_PROJECT`/`WANDB_NAME` are ignored — verl passes `project_name`/
  `experiment_name` as explicit kwargs to `wandb.init()` (`utils/tracking.py:80`),
  which beat the environment. Use those config keys instead.
- Stock ToolAgentLoop refuses tool-returned video — patched in
  `external/verl-video-nccl-fix.patch` (re-apply after any fresh `git submodule
  update --init`; `scripts/check_train_env.py` asserts it's present) so
  `select_view` can return the real clip. `select_frames`/`zoom` stay IMAGES —
  never pass the 19-view menu through the video path: `Qwen3VLVideoProcessor` has
  `do_sample_frames=True, fps=2` and silently resamples it to 4 frames.
  `scripts/check_prompt_parity.py` guards this. The NCCL hang this caused is fixed
  via `actor_rollout_ref.actor.fsdp_config.fsdp_size=1` — see
  `docs/OPEN_ISSUES.md` #9 before re-litigating.

## Working rules

- Compute nodes have outbound HTTPS, wandb logs online, credentials in
  `~/.netrc` — no key belongs in the repo.
- State the record count you're actually training on. A `--limit 3000` smoke
  default once went unrevisited and a full SFT run trained on 2.3% of the corpus.
- **Never save/commit a smoke run or anything it produced.** Real training uses
  the train set in `SPEC.md`, real eval uses the val/test set in `SPEC.md`. Delete
  smoke test output immediately after verifying the pipeline works.
- Comparison paper is **EchoSonar-R** (arXiv 2606.28164) — same private dataset,
  same SFT->GRPO recipe, no tools. CardioBench (arXiv 2510.00520) is only the
  source of metric definitions. Don't confuse them.
- `abnormality_classification` is 82% "no" — always report balanced accuracy,
  plain accuracy flatters an always-no model.
- Never silently approximate a metric. METEOR (real nltk metric, `packages/eval/nlg.py`;
  needs nltk data `wordnet`/`omw-1.4`/`punkt_tab`, downloaded once to `~/nltk_data`
  on the AMD login node) and BERTScore (PubMedBERT, `packages/eval/bertscore.py`)
  are now computed for real. GREEN (`packages/eval/green_approx.py`) is the original
  PUBLIC GREEN prompt — EchoSonar-R's echo-adapted prompt was never published, so
  their 0.800 is not reproducible; always label ours "GREEN (ours, public prompt)"
  and never place it unlabeled next to theirs.
- `eval.nlg`'s report-generation numbers are corpus-level over WHOLE reports.
  EchoSonar-R's Table 3 is per-SECTION (13 cardiac structures + conclusions),
  sentence-level — a different number on identical text, confirmed by reading
  Darya's `report_generation/evaluation/report_pipeline/evaluate_reports.py`
  directly. For a real Table 3 comparison use `packages/eval/darya_report_bridge.py`,
  which calls her scoring code unmodified against her real `test.json`. It only
  scores studies whose answer has her `**Header:**` markers (the base Qwen3-VL
  plain-prompt run got 808/1184 for free, without ever using her prompt template);
  unparsed studies are excluded, never zero-padded — report that count every time.

## Background

Based on two papers: EchoSonar-R (Taratynova, D. et al) and DeepEyes (Zheng, Z. et
al). CardioBench (Aly, A. and Taratynova, D. et al) is good background context.

## How to reply to me

- Be concise. One issue per reply; list other open issues at the end and keep
  carrying them forward until resolved.
- Start with a TLDR.
- List any experiments still running at the end, by a short descriptive
  name/phrase, not by job_id.

## Wandb

- Every experiment must log to wandb. After starting one, check its wandb link
  and include it in the reply.

## General

- Match my speaking style.
- No em-dashes, no "not just X, it's Y" or other dramatic phrasing. Say things
  normally.
- Avoid talking like an ADHD person: no scattershot bullet dumps, no jumping
  between unrelated points. Stay focused and linear.
