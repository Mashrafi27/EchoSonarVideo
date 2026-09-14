# EchoSonarVideo

Agentic RL on multi-view cardiac ultrasound video: cold-start SFT -> GRPO, on
upstream verl (pinned submodule `external/verl` @ v0.7.1, NOT forked). Base model
Qwen3-VL-8B-Instruct. One composite `echo` tool with three ops: `select_view`,
`select_frames`, `zoom`.

This file holds RULES that are expensive to rediscover. Architecture and rationale
live in `echo_env/INTEGRATION.md`. See also `SPEC.md` (facts: data, architectures,
real experiment results) and `PLAN.md` (current open questions and next steps).

## Substrate: 4x CUDA GPU box, shared with other users

**VERIFIED: real GRPO runs here**, on the frozen-EchoPrime + Qwen3-8B-text track
(a from-cold-start-no-SFT ablation of that architecture, not the Qwen3-VL tool-based
track -- see below). Confirmed via full training runs producing real, varying,
non-zero reward across many steps, not just a smoke test. `rollout.load_format: dummy`
was one wrong assumption that cost real time, see the gotchas below.

**Repo:** `git clone --recurse-submodules https://github.com/Mashrafi27/EchoSonarVideo.git`
(two submodules, `external/verl` and `external/DeepEyes` -- the `--recurse-submodules`
flag is not optional, a plain clone leaves both empty).

**`requirements-train.txt`** is a real, installed spec on this machine (CUDA wheels:
vllm 0.17.0, flash-attn, torch cu129, derived from pinned `external/verl@v0.7.1`'s own
`setup.py`). `attn_implementation=sdpa` is kept for the echoprime track's own config
(not switched to flash-attn) simply because flash-attn isn't installed in this conda
env -- sdpa is fine, it's torch's own fast path regardless.

vLLM serves normally on this box for both tracks (a real served engine, confirmed via
the `echo_grpo` track's `run_eval.py --base-url http://localhost:8000/v1` and the
`echo_ep` track's custom-registered vLLM model class). For a small offline eval, the
echoprime track also has a purpose-built `echo_ep/eval_checkpoint.py` (loads the
checkpoint + real LoRA state dict directly, HF `.generate()`, no vLLM needed) and
`echo_ep/eval_checkpoint_vllm.py` (merges LoRA into base weights, serves via vLLM,
real batched generation, much faster for a large eval).

**Data path:** `ECHO_PREPROCESSED_DIR` resolves fine on this box
(`/hdd2/ahmedaly/echo_preprocessed_mmm`, 6276 study subdirs = the full train_vqa +
test_vqa pool). `echo_ep/build_video_cache.py` builds the frozen-EchoPrime embedding
cache from this tree, one file per study, idempotent/resumable by design -- but it
only covers whatever `--study-list` (or full directory scan) you actually point it
at. **Confirmed gotcha:** the cache was built once against `build/rl.jsonl`'s 5061
train studies and just left there; nobody re-ran it against `build/eval.jsonl`'s 1215
test studies, so the real held-out eval set had **zero** cache coverage for a full
session's worth of "the eval script works" confidence. Check `ls
build/echoprime_video_cache | wc -l` against the actual study set you intend to use
BEFORE trusting an eval number -- 0% coverage fails loud (`FileNotFoundError` per
study) so this specific gap could not have silently produced wrong numbers, but a
*partial* cache could quietly bias small hand-picked eval samples. Don't assume
coverage, check it.

### Data pipeline: what `rl.jsonl` / `eval.jsonl` actually are

Ground truth, from `scripts/build_grpo_parquet.sh`'s own header and comments -- do
not re-derive this from the `split` field on individual records, that field is
leftover per-record lineage metadata from an abandoned CSV-based split scheme
("we do NOT use echojepa_study_split_full.csv") and does **not** mean "this record
is held out." A record's `split` value tells you nothing about which file it should
be in.

- `build/rl.jsonl` = **all** of `train_vqa_with_thinking.jsonl` (`ECHO_VQA_TRAIN`),
  joined to the preprocessed frame tree. This is the entire train pool -- there is no
  further train/val split inside it. 128,215 rows / 5,061 studies as of this
  writing (count studies directly by `study_uuid`, not via the per-record `split`
  field -- see above).
- `build/eval.jsonl` = **all** of `test_vqa.jsonl` (`ECHO_VQA_TEST`). This is the
  real held-out set. 31,209 rows / 1,215 studies. **Confirmed zero study overlap**
  with `rl.jsonl` (5061 + 1215 = 6276 distinct studies, matches the frame tree
  exactly) -- this is a real invariant of the source files, not something a
  downstream script re-checks, so an accidental re-partition of either raw file
  would silently break it.
- The tool-based Qwen3-VL GRPO run's val parquet (`build/rl_val.parquet`) was built
  from the **full** `build/eval.jsonl` (`echo_verl/generate_trainset.py --rl-jsonl
  build/eval.jsonl`, no `--limit`) -- i.e. ~31k rows really was the val set for that
  run. Any new track's val set should match this convention (source from
  `eval.jsonl`, not from carving an ad-hoc slice out of `rl.jsonl`) to stay
  comparable.

**A real, expensive mistake made on this data once already** (see "Data-scale
defaults are a known trap" above -- this is the same trap, hit again): an early
`echo_ep/generate_grpo_parquet.py` build used `--limit 200` / `--limit 20` with no
`--study-list`, which (a) silently capped the echoprime track's train/val parquets to
~0.2% of the real pool with no record of why, and (b) took its first-N-rows-in-file-
order "val" set from studies that were *also* in the "train" set (100% overlap) --
because the script filters by `--study-list`, not by the record's `split` field, and
none was passed. An entire session's worth of GRPO training and eval ran against this
before it was caught. `generate_grpo_parquet.py` now supports `--shuffle-seed` for a
real random sample instead of a file-order cutoff; always pass an explicit
`--study-list` built from the real file-level partition above, never rely on
`--limit` alone to produce a sane subset.

### GRPO gotchas specific to the frozen-EchoPrime + Qwen3-8B-text composite model

All confirmed this session, real bugs in `echo_ep/` -- a from-scratch custom
`PretrainedConfig` + custom vLLM model class hits sharp edges that a stock HF/vLLM
model never exercises.

- **`tie_word_embeddings` silently defaults to `True`** on `EchoPrimeQwen3Config`
  (HF's `PretrainedConfig.__init__` sets it as a real top-level attribute before
  `__getattr__` delegation to the nested real Qwen3 config ever gets a chance --
  same shadowing class as the `eos_token_id` bug below). Qwen3-8B is **untied**
  (`tie_word_embeddings: False`); training with the wrong default means vLLM
  projects every output token through the tied embedding matrix instead of the
  real, separately-trained `lm_head` -- this reliably produces complete, temperature-
  independent garbage output (confirmed: literal noise at temperature 0.0, 0.5, and
  1.0 alike) while every other diagnostic (weight loading, config hyperparameters,
  forward-pass structure) checks out fine. This was the single biggest time sink of
  the whole session because the symptom (garbage generation) looked identical to
  several unrelated hypotheses tried first (response length, sampling temperature,
  `eos_token_id`). Fixed in `echo_ep/modeling.py`'s `EchoPrimeQwen3Config.__init__`
  and `from_cold_start` by explicitly copying `eos_token_id`/`pad_token_id`/
  `bos_token_id`/`tie_word_embeddings` from the real inner Qwen3 config.
- **`eos_token_id` has the same shadowing bug**, independently: it saves as missing
  from the top-level `config.json` (HF's live `__getattr__` delegation still works
  in-process, so this only breaks consumers that read the raw JSON structurally,
  like vLLM). Symptom: 100% of generated responses hit the hard `max_tokens` ceiling,
  zero variance, regardless of temperature -- confirmed the model was capable of
  correctly predicting `<|im_end|>` under HF's own `.generate()`, just not through
  vLLM with the missing top-level field. Fix (same code, one loop) copies this
  alongside `tie_word_embeddings`.
- **`rollout.load_format: dummy` is broken for this composite model.** The
  documented reasoning ("weights come from FSDP's own weight-sync every step
  regardless") is correct for the per-step LoRA-only sync, but the very first
  base-model sync (before which vLLM's weights are literal dummy/random init) did
  not work correctly for this class -- confirmed by direct comparison: the exact
  same fixed checkpoint produced real, coherent, correctly-scored completions when
  loaded via `load_format: auto` (a direct file load at startup) and near-empty
  completions (0-11 tokens, immediate EOS) when loaded via `load_format: dummy`,
  with everything else identical. Root cause not fully isolated (suspect
  `collect_lora_params`'s "not base_sync_done" full-base-model sync path in
  `verl/utils/fsdp_utils.py`, worth revisiting). `echo_verl/configs/
  echoprime_grpo.yaml` now defaults to `auto`.
- **The standalone `lora_adapter/adapter_model.safetensors` checkpoint export is
  empty** (real bug, not a training-correctness issue): `layered_summon_lora_params`
  (`external/verl/verl/utils/fsdp_utils.py`) hardcodes decoder-layer path prefixes
  assuming the wrapped LM's decoder sits at `.model.layers` or
  `.language_model.layers`; this composite model nests it one level deeper at
  `.lm.model.layers`, so the function silently finds zero matching layers and saves
  an empty adapter. The full FSDP checkpoint (`model_world_size_*.pt`) is
  unaffected (no such prefix filtering) and has the real trained weights; see
  `external/verl-lora-checkpoint-save-fix.patch` for the fix (uses
  `peft_model.get_base_model()` before calling `custom_object_save`, so the bundled
  custom-code file is ours, not peft's own).
- **The untrained cold-start policy can sample its own `VIEW_TOKEN` as free text**
  (near-uniform sampling, ~8.9 nats entropy) -- `_splice_view_embeddings` scans the
  whole prompt+response sequence for that token id with no way to tell "prompt" from
  "model's own output," so a stray one during generation trips an assertion during
  the later training forward pass. Fixed by banning the token via vLLM's
  `logit_bias` at generation time (`echo_ep/echoprime_agent_loop.py`), not by
  loosening the assertion.

## verl gotchas

- verl 0.7.1 applies the **Qwen2-VL** rope to Qwen3-VL — its `image_processor` is
  `Qwen2VLImageProcessorFast`, so verl's substring gate passes and it raises `IndexError`.
  Worked around by `echo_verl/sft_dataset.py` via `data.custom_cls`, not by patching verl.
- `WANDB_PROJECT`/`WANDB_NAME` are **ignored**: verl calls `wandb.init(project=..., name=...)`
  from `trainer.project_name`/`experiment_name` (`utils/tracking.py:80`), and explicit
  kwargs beat the environment. Those config keys are the real knobs.
- Stock ToolAgentLoop refuses tool-returned **video** — patched in `external/verl`
  (`external/verl-video-nccl-fix.patch`, re-apply after any fresh
  `git submodule update --init`, `scripts/check_train_env.py` asserts it's present)
  so `select_view` can return the view's real clip. `select_frames`/`zoom` stay
  IMAGES (the HYBRID path) — never pass the 19-view menu itself through the video
  path: `Qwen3VLVideoProcessor` has `do_sample_frames=True, fps=2` and silently
  resamples a 19-image menu to 4 frames. `scripts/check_prompt_parity.py` guards this.
  The NCCL hang this used to cause (verl's ZeRO-3 param-shard all-gather count going
  data-dependent once video's involved) is fixed via
  `actor_rollout_ref.actor.fsdp_config.fsdp_size=1` — see `docs/OPEN_ISSUES.md` #9,
  do not re-litigate before reading it.

## Working rules

- Compute nodes DO have outbound HTTPS, so wandb logs online; credentials are in
  `~/.netrc` and no key belongs in the repo.
- Data-scale defaults are a known trap: a `--limit 3000` smoke default went unrevisited
  and one full SFT run trained on 2.3% of the corpus. State the record count you are
  actually training on.
- **Never save/commit a smoke run or anything it produced.** Real training always uses
  the train set in `SPEC.md`, real eval always uses the val/test set in `SPEC.md`. If a
  smoke test is needed to verify a pipeline, run it then delete everything it produced
  immediately after.
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