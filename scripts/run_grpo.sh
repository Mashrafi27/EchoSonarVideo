#!/usr/bin/env bash
# GRPO launch -- CUDA box (sn4622121129), 4x RTX A6000 48GB, NO SLURM.
#
#   bash scripts/run_grpo.sh
#
# Prereqs (see docs/MEETING_2026-08-31.md checklist):
#   1. training env built at $VENV (vllm 0.17 / torch 2.10 cu12x / verl v0.7.1)
#   2. base model in $HF_HOME  (Qwen/Qwen3-VL-8B-Instruct)
#   3. ECHO_PREPROCESSED_DIR populated (scripts/build_preprocessed_tree.py)
#   4. parquet built            (scripts/build_grpo_parquet.sh)
#
# This is NOT the ROCm SFT path: no ROCR/HIP translation, no MIOpen DB dirs, no
# RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES -- all ROCm-only. FP8 is still out
# (A6000 = Ampere). flash-attn IS available here, unlike the MI210 cluster.
set -xeuo pipefail

REPO=${REPO:-/home/mashrafimonon/EchoSonarVideo}
cd "$REPO"

# --- environment -------------------------------------------------------------
# env.sh (on /hdd2) still holds the valid HF_HOME / ECHO_PREPROCESSED_DIR / offline
# flags -- model cache + preprocessed frame tree were never moved. The old /hdd2
# venv is dead (replaced by a conda env on /data after the 2026-09-09 /hdd2 wobble
# + env rebuild); point PY at that instead.
source /hdd2/ahmedaly/echogrpo/env.sh
VENV=${VENV:-/data/ahmedaly/mashrafi_echogrpo/conda_env}
PY="$VENV/bin/python"

NGPUS=${NGPUS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
# NOTE: expandable_segments:True cuts fragmentation for the FSDP actor but vLLM's
# memory pool rejects it (pytorch#147851) -- do not set it globally here.
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-$REPO/build/rl_train.parquet}
VAL_FILES=${VAL_FILES:-$REPO/build/rl_val.parquet}
EXP_NAME=${EXP_NAME:-grpo-qwen3vl8b-$(date +%m%d-%H%M)}
CKPT_HOME=${CKPT_HOME:-/data/ahmedaly/mashrafi_echogrpo/checkpoints/$EXP_NAME}
# flash-attn is NOT installed in the conda env (its source build OOM'd the shared
# box -- 124 cicc procs, 2026-09-10). sdpa on torch 2.10 already dispatches to a
# bundled flash kernel, so this is not the slow O(n^2) path. Revisit flash-attn
# as its own task if the actor fwd/bwd proves too slow.
ATTN_IMPL=${ATTN_IMPL:-sdpa}
mkdir -p "$CKPT_HOME" logs

# --- wandb -----------------------------------------------------------------
# credentials in ~/.netrc; nothing written here. verl calls wandb.init from
# trainer.project_name / trainer.experiment_name (env WANDB_PROJECT is ignored).
export WANDB_DIR=${WANDB_DIR:-/hdd2/ahmedaly/echogrpo/wandb}
export WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-echo-grpo}
mkdir -p "$WANDB_DIR"

CONFIG_PATH="$REPO/echo_verl/configs"

"$PY" -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name=echo_grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-32} \
    data.max_prompt_length=${MAX_PROMPT_LEN:-4096} \
    data.max_response_length=${MAX_RESP_LEN:-4096} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN:-20480} \
    actor_rollout_ref.model.path="$MODEL_ID" \
    +actor_rollout_ref.model.override_config.attn_implementation="$ATTN_IMPL" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE:-1} \
    actor_rollout_ref.model.lora_rank=${LORA_RANK:-0} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA:-32} \
    actor_rollout_ref.model.target_modules=${LORA_TARGETS:-all-linear} \
    actor_rollout_ref.model.exclude_modules="${LORA_EXCLUDE:-.*visual.*}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-True} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD:-True} \
    actor_rollout_ref.model.enable_activation_offload=${ACT_OFFLOAD:-False} \
    actor_rollout_ref.actor.use_torch_compile=${TORCH_COMPILE:-False} \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${OFFLOAD_POLICY:-False} \
    actor_rollout_ref.actor.use_dynamic_bsz=${DYNAMIC_BSZ:-False} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOK_PER_GPU:-16384} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SEQ_PARALLEL:-1} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${DYNAMIC_BSZ:-False} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOK_PER_GPU:-16384} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${DYNAMIC_BSZ:-False} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOK_PER_GPU:-16384} \
    actor_rollout_ref.rollout.layered_summon=${LAYERED_SUMMON:-False} \
    actor_rollout_ref.rollout.load_format=${LOAD_FORMAT:-dummy} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE:-2} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.5} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N:-5} \
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS:-8} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS:-6} \
    custom_reward_function.path="$REPO/echo_verl/reward.py" \
    custom_reward_function.name=compute_score \
    trainer.logger="[${TRAINER_LOGGER:-'console','wandb'}]" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_HOME" \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=${EPOCHS:-1} \
    trainer.save_freq=${SAVE_FREQ:-20} \
    trainer.test_freq=${TEST_FREQ:-20} \
    "$@"

# Tuning, ONE change per run:
#   OOM (actor)        -> MICRO_BATCH_SIZE stays 1 (already floor); OPTIMIZER_OFFLOAD/
#                          PARAM_OFFLOAD already True. 2026-09-01 full-FT OOMs (44-45GB/48GB)
#                          held steady across GPU_MEM_UTIL 0.7->0.3, which only touches vLLM's
#                          rollout pool -- proves the actor's own fwd/bwd is the ceiling, not
#                          rollout. SEQ_PARALLEL (ulysses) is NOT a usable lever here: verl
#                          hard-requires use_remove_padding=True for it (actor.py:319), and
#                          use_remove_padding is the same Qwen2-VL-shaped packing path
#                          echo_grpo.yaml already keeps off (see its comment there, and
#                          CLAUDE.md's rope/position-id note). Confirmed 2026-09-02: instant
#                          ValueError at config validation, never reached the GPUs. Don't
#                          retry SEQ_PARALLEL without first fixing use_remove_padding for
#                          Qwen3-VL. See docs/OPEN_ISSUES.md #8.
#   OOM (vllm engine)  -> GPU_MEM_UTIL=0.4, then TP_SIZE=4 (already ruled out for the actor OOM
#                          above, still valid if vLLM itself OOMs after the actor fits)
#   flash-attn missing -> ATTN_IMPL=sdpa
#   rollout too slow   -> ROLLOUT_N=3, MAX_TURNS=4
#   no wandb           -> WANDB_MODE=disabled  (or TRAINER_LOGGER="'console'")
#
# Host-RAM OOM on full-FT (2026-09-02/03), GPU 0/1 only: see docs/OPEN_ISSUES.md #8
# for the full attempt log. Two extra Hydra overrides worth keeping in the launch
# command when memory is tight (not wired as env-var knobs above, pass them as
# extra CLI args after the script):
#     +ray_kwargs.ray_init.object_store_memory=15000000000
#     +ray_kwargs.ray_init._temp_dir=/hdd2/ahmedaly/echogrpo/ray_tmp
#   Ray's plasma object store defaults to living in /dev/shm (real host RAM) sized
#   off total node memory with no cap -- these force it to spill to /hdd2 disk
#   (4.6TB free, vs /tmp's 55GB) well before hitting the node OOM threshold.
#   MALLOC_ARENA_MAX=1 (env var) is also worth trying alongside these; unclear yet
#   which of the two actually matters, both were on for every successful-so-far run.
#
# GOTCHA: `tmux kill-session` on a launch does NOT reliably kill the vLLM worker
# subprocesses Ray spawns (VLLM::Worker_TP*) -- they can survive as orphans still
# holding full GPU memory. A relaunch right after killing a session can then fail
# at vLLM init with "Free memory on device cuda:N is less than desired GPU memory
# utilization" even though nvidia-smi looked idle moments before. Confirmed
# 2026-09-03. Always check `nvidia-smi --query-compute-apps=pid,used_memory,
# process_name --format=csv` after killing a session and `kill -9` anything still
# holding the GPUs before relaunching.
