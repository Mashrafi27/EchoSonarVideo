#!/usr/bin/env bash
#SBATCH --job-name=echo_sft_smoke
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Cold-start SFT SMOKE TEST — smallest run that proves the whole path:
# tokenization, Qwen3-VL mRoPE position ids, forward, backward, optimizer step.
# 8 samples, 1 epoch, no checkpoint. NOT a training run.
#
#   sbatch scripts/sft_smoke.sbatch
#
# Substrate: docs/TRAINING_ENV.md §2a (ROCm/MI210 overlay venv).
#
# TRAP: keep train_batch_size >= 2 * NGPUS. pad_mode=no_padding collates through
# torch.nested jagged tensors, and VLM position_ids are (4, seq_len) per sample. With
# exactly ONE sample on a rank, torch cannot tell which dim is ragged and picks dim 1
# (the 4), so a later op explodes with
#   RuntimeError: split_with_sizes expects split_sizes to sum exactly to <seq_len> ...
#   but got split_sizes=[4]
# With >=2 samples the ragged dim resolves correctly (verified on CPU).
set -xeuo pipefail

REPO=${REPO:-/vast/users/mohammad.yaqub/project/EchoSonarVideo}
cd "$REPO"

PY=${PY:-$REPO/.venv-train/bin/python}
NGPUS=${NGPUS:-4}
TRAIN_FILES=${TRAIN_FILES:-$REPO/build/sft_smoke.parquet}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}
CKPT_HOME=${CKPT_HOME:-$REPO/.tmp_work/sft_smoke_ckpt}
mkdir -p "$CKPT_HOME"

# ROCm: no flash-attn on gfx90a -> sdpa. verl's default is flash_attention_2.
export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx90a}
# NOTE: do NOT set RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES here. verl's AMD docs
# recommend it for the ray-based RLHF path, but this SFT job runs under torchrun with a
# local ray instance that owns no accelerators, so the flag sends
# Worker._setup_env_cuda_visible_devices down `ray...get_accelerator_ids()["GPU"][0]`
# -> IndexError: list index out of range (worker.py:279).
# SLURM hands out GPUs via ROCR_VISIBLE_DEVICES; torch 2.7 ROCm hard-errors on it
# ("Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES"). Translate.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export ECHO_PREPROCESSED_DIR=${ECHO_PREPROCESSED_DIR:-/vast/users/mohammad.yaqub/project/preprocessed_data}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NGPUS" \
    -m verl.trainer.sft_trainer \
    data.train_files="$TRAIN_FILES" \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=16384 \
    data.max_length=16384 \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.train_max_samples=8 \
    data.num_workers=2 \
    data.custom_cls.path="$REPO/echo_verl/sft_dataset.py" \
    data.custom_cls.name=EchoMultiTurnSFTDataset \
    model.path="$MODEL_ID" \
    model.use_remove_padding=False \
    model.enable_gradient_checkpointing=True \
    +model.override_config.attn_implementation=sdpa \
    engine=fsdp \
    optim=fsdp \
    optim.lr=1e-5 \
    engine.strategy=fsdp2 \
    engine.model_dtype=bf16 \
    engine.ulysses_sequence_parallel_size=1 \
    trainer.device=cuda \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.logger=['console'] \
    trainer.project_name=echo-sft \
    trainer.experiment_name=smoke \
    trainer.default_local_dir="$CKPT_HOME"
