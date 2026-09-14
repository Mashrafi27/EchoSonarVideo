#!/bin/bash
#SBATCH --job-name=echo_qwen_vllm_check
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=myqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=.tmp_work/rocm_validation_20260914/inference_%j.out
#SBATCH --error=.tmp_work/rocm_validation_20260914/inference_%j.err
set -euo pipefail
cd /vast/users/mohammad.yaqub/project/EchoSonarVideo
RUN_DIR="$PWD/.tmp_work/rocm_validation_20260914"
source scripts/rocm_workspace_scratch.sh
rocm_scratch_setup "$RUN_DIR" 8G
trap rocm_scratch_cleanup EXIT
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; unset ROCR_VISIBLE_DEVICES; fi
unset LD_PRELOAD HSA_OVERRIDE_GFX_VERSION
export ROCM_PATH="$RUN_DIR/env/rocm-root/opt/rocm-7.0.2"
export LD_LIBRARY_PATH="$RUN_DIR/env/native-root/usr/lib/x86_64-linux-gnu:$ROCM_PATH/lib:$ROCM_PATH/lib64"
export LD_PRELOAD="$RUN_DIR/env/native-root/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export PATH="$ROCM_PATH/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VLLM_ROCM_USE_AITER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/vast/users/mohammad.yaqub/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DIR="$RUN_DIR" WANDB_CACHE_DIR="$RUN_DIR/wandb_cache"
export PYTHONPATH="$PWD/packages:$PWD"
CACHE="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct"
REVISION=$(cat "$CACHE/refs/main")
MODEL="$CACHE/snapshots/$REVISION"
"$RUN_DIR/env/bin/python" -u scripts/validate_qwen_vllm.py \
    --model "$MODEL" --output "$RUN_DIR/inference_$SLURM_JOB_ID"
