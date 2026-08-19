#!/usr/bin/env bash
#SBATCH --job-name=echo_sft
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Cold-start SFT — the real run (2382 trajectories, build/sft_train.parquet).
#
#   mkdir -p logs && sbatch scripts/run_sft.sbatch
#
# MEASURED over all 2382 rows (full pass, 2026-08-17): min 1387, p50 4213,
# p90 4855, p99 5524, MAX 6088, mean 3804. Nothing exceeds 8192.
#
# INVARIANT: every rank's samples must fit in ONE dynamic micro-batch, i.e.
#   samples_per_rank * max_seq_len <= max_token_len_per_gpu
# rearrange_micro_batches splits by ceildiv(total_tokens, max_token_len) and
# all-reduces the count with MAX across DP ranks, so a rank can end up with a
# SINGLETON micro-batch -- and a lone (4, seq_len) position_ids tensor makes
# torch.nested pick the wrong ragged dim ("split_sizes=[4]", see sft_smoke.sbatch).
# 32 / 8 GPUs = 4 samples/rank; 4 * 6088 = 24352 <= 24576, so each rank needs
# exactly one micro-batch and no singleton can appear. Re-derive both numbers from
# the measured max above before changing either.
#
# Prerequisite: scripts/sft_smoke.sbatch must have gone green first. This script is
# deliberately the SAME configuration as the smoke run, only bigger — do not turn on
# use_remove_padding and ulysses SP in the same submission (see "Tuning" below).
#
# Substrate: docs/TRAINING_ENV.md §2a (ROCm/MI210 overlay venv).
set -xeuo pipefail

REPO=${REPO:-/vast/users/mohammad.yaqub/project/EchoSonarVideo}
cd "$REPO"

PY=${PY:-$REPO/.venv-train/bin/python}
NGPUS=${NGPUS:-8}
TRAIN_FILES=${TRAIN_FILES:-$REPO/build/sft_train.parquet}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}
EXP_NAME=${EXP_NAME:-coldstart-qwen3vl8b}
CKPT_HOME=${CKPT_HOME:-$REPO/checkpoints/echo-sft/$EXP_NAME}
EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-5}
mkdir -p "$CKPT_HOME" logs

export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx90a}
# See sft_smoke.sbatch: RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES breaks the torchrun
# SFT path (worker.py:279 IndexError); it is for the ray-based RLHF path only.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi
# MIOpen keeps a SQLite kernel DB under ~/.config/miopen, which is READ-ONLY here:
#   MIOpen Error: kern_db.hpp:183: ... attempt to write a readonly database
#   -> RuntimeError: miopenStatusInternalError in the Qwen3-VL vision Conv3d
# Probed on one MI210: fails with the default path, passes with a writable one.
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-$REPO/.tmp_work/miopen/${SLURM_JOB_ID:-local}}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export ECHO_PREPROCESSED_DIR=${ECHO_PREPROCESSED_DIR:-/vast/users/mohammad.yaqub/project/preprocessed_data}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# --- wandb ---------------------------------------------------------------
# Credentials come from ~/.netrc (already present); no API key is written here.
# Compute nodes DO have outbound HTTPS (verified against api.wandb.ai from
# auh7-1b-gpu-216), so online mode works and there is no offline-sync step.
# WANDB_DIR must stay inside the project tree -- run dirs are large and /tmp is
# forbidden on this cluster. Set WANDB_MODE=offline to log without the network,
# or WANDB_MODE=disabled to turn it off entirely.
export WANDB_DIR=${WANDB_DIR:-$REPO/.wandb}
export WANDB_MODE=${WANDB_MODE:-online}
# NOTE: WANDB_PROJECT/WANDB_NAME would be IGNORED here -- verl calls
# wandb.init(project=..., name=...) explicitly from trainer.project_name /
# trainer.experiment_name (utils/tracking.py:80), and explicit kwargs beat the
# environment. So those two config keys are the real knobs; these vars only
# feed them.
WANDB_PROJECT=${WANDB_PROJECT:-echo-sft}
# Tie the wandb run to the SLURM job so a run page always names the job whose log
# and checkpoints belong to it; matching a run to a checkpoint dir after the fact
# is otherwise guesswork. The CHECKPOINT dir deliberately stays on the bare
# EXP_NAME so resume paths do not move between jobs.
RUN_NAME="${EXP_NAME}-${SLURM_JOB_ID:-local}"
mkdir -p "$WANDB_DIR"

"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NGPUS" \
    -m verl.trainer.sft_trainer \
    data.train_files="$TRAIN_FILES" \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-32} \
    data.micro_batch_size_per_gpu=1 \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-24576} \
    data.max_length=8192 \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.num_workers=4 \
    data.custom_cls.path="$REPO/echo_verl/sft_dataset.py" \
    data.custom_cls.name=EchoMultiTurnSFTDataset \
    model.path="$MODEL_ID" \
    model.use_remove_padding=${USE_REMOVE_PADDING:-False} \
    model.enable_gradient_checkpointing=True \
    +model.override_config.attn_implementation=sdpa \
    engine=fsdp \
    optim=fsdp \
    optim.lr="$LR" \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    optim.warmup_style=cosine \
    engine.strategy=fsdp2 \
    engine.model_dtype=bf16 \
    engine.ulysses_sequence_parallel_size=${SP_SIZE:-1} \
    engine.optimizer_offload=${OPTIMIZER_OFFLOAD:-false} \
    trainer.device=cuda \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.total_epochs="$EPOCHS" \
    trainer.save_freq=${SAVE_FREQ:-100} \
    trainer.test_freq=-1 \
    trainer.max_ckpt_to_keep=3 \
    trainer.logger=[${TRAINER_LOGGER:-'console','wandb'}] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$RUN_NAME" \
    trainer.default_local_dir="$CKPT_HOME" \
    checkpoint.save_contents=[model,optimizer,extra]

# Tuning, in this order, ONE change per submission:
#   OOM              -> OPTIMIZER_OFFLOAD=true, then MAX_TOKEN_LEN_PER_GPU=8192
#   too slow         -> USE_REMOVE_PADDING=True     (UNTESTED with our (4, seq_len)
#                       Qwen3-VL position ids -- verl's packing path was written
#                       against Qwen2-VL; watch for a shape error on step 1)
#   still too slow   -> SP_SIZE=2                   (also untested with those position ids)
#   no wandb         -> WANDB_MODE=disabled  (or TRAINER_LOGGER="'console'")
