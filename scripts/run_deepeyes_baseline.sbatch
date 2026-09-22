#!/usr/bin/env bash
#SBATCH --job-name=deepeyes_basic_qa
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=myqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=build/deepeyes_baseline_20260922/%x_%j.out
#SBATCH --error=build/deepeyes_baseline_20260922/%x_%j.err
set -euo pipefail
REPO="${ECHO_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$REPO"
RUNTIME="$REPO/.tmp_work/rocm_validation_20260914/env"
OUT="$REPO/build/deepeyes_baseline_20260922/run_${SLURM_JOB_ID}"
mkdir -p "$OUT"
source scripts/rocm_workspace_scratch.sh
rocm_scratch_setup "$OUT" 8G
trap rocm_scratch_cleanup EXIT
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi
unset LD_PRELOAD HSA_OVERRIDE_GFX_VERSION
export ROCM_PATH="$RUNTIME/rocm-root/opt/rocm-7.0.2"
export LD_LIBRARY_PATH="$RUNTIME/native-root/usr/lib/x86_64-linux-gnu:$ROCM_PATH/lib:$ROCM_PATH/lib64"
export LD_PRELOAD="$RUNTIME/native-root/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export PATH="$ROCM_PATH/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$REPO/build/deepeyes_baseline_20260922/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/packages:$REPO"
export WANDB_DIR="$OUT" WANDB_CACHE_DIR="$OUT/wandb_cache" WANDB_MODE=online
"$RUNTIME/bin/python" -u -m eval.run_deepeyes_baseline --out-dir "$OUT" "$@"
