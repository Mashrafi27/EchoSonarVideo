#!/usr/bin/env bash
#SBATCH --job-name=deepeyes_original_tools
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
OUT="$REPO/build/deepeyes_baseline_20260922/tools_${SLURM_JOB_ID}"
mkdir -p "$OUT"
source scripts/rocm_workspace_scratch.sh
rocm_scratch_setup "$OUT" 8G
SERVER_PID=''
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rocm_scratch_cleanup
}
trap cleanup EXIT
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi
unset LD_PRELOAD HSA_OVERRIDE_GFX_VERSION ROCM_HOME
export ROCM_PATH="$RUNTIME/rocm-root/opt/rocm-7.0.2"
export LD_LIBRARY_PATH="$RUNTIME/lib/python3.12/site-packages/amdsmi:$RUNTIME/native-root/usr/lib/x86_64-linux-gnu:$ROCM_PATH/lib:$ROCM_PATH/lib64"
export LD_PRELOAD="$RUNTIME/native-root/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export PATH="$RUNTIME/bin:$ROCM_PATH/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export VLLM_ROCM_USE_AITER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export ECHO_ROCM_SCRATCH_ROOT="$ROCM_SCRATCH_MOUNT"
export ECHO_ROCM_ALLOWED_DEVICES="${HIP_VISIBLE_DEVICES:?Missing allocated GPUs}"
export HF_HOME="$REPO/build/deepeyes_baseline_20260922/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/scripts/rocm_python_startup:$REPO/packages:$REPO"
export WANDB_DIR="$OUT" WANDB_CACHE_DIR="$OUT/wandb_cache" WANDB_MODE=offline
PY="$RUNTIME/bin/python"
PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
SERVER_ARGS=(--model "$REPO/checkpoints/deepeyes_7b" --served-model-name deepeyes
    --host 127.0.0.1 --port "$PORT" --dtype bfloat16
    --gpu-memory-utilization 0.70 --max-num-seqs 1 --max-num-batched-tokens 16384
    --enforce-eager --no-enable-prefix-caching --skip-mm-profiling
    --limit-mm-per-prompt '{"image":12,"video":0}'
    --attention-backend TRITON_ATTN --mm-encoder-attn-backend TORCH_SDPA
    --disable-custom-all-reduce --seed 1)
printf '%q ' "${SERVER_ARGS[@]}" > "$OUT/server_command.txt"
"$PY" -m vllm.entrypoints.openai.api_server "${SERVER_ARGS[@]}" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
for attempt in $(seq 1 180); do
    kill -0 "$SERVER_PID" || { tail -60 "$OUT/server.log"; exit 1; }
    if curl --silent --fail "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
    sleep 2
done
curl --silent --fail "http://127.0.0.1:$PORT/health" > /dev/null
echo "VLLM_LOCAL_SERVER_READY"
"$PY" -u -m eval.run_deepeyes_tools \
    --api-url "http://127.0.0.1:$PORT/v1" --out-dir "$OUT" "$@"
