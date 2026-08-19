#!/usr/bin/env bash
#SBATCH --job-name=echo_evalsmoke
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Prove the in-process agentic eval loop runs END TO END before spending a node on
# 1,215 studies. This loop has never executed a single episode -- it was authored
# against a vLLM server we then discovered cannot run here.
#
#   MODEL=checkpoints/echo-sft/merged/step100 sbatch scripts/eval_smoke.sbatch
set -xeuo pipefail

REPO=${REPO:-/vast/users/mohammad.yaqub/project/EchoSonarVideo}
cd "$REPO"
PY=${PY:-$REPO/.venv-train/bin/python}
MODEL=${MODEL:-$REPO/checkpoints/echo-sft/merged/step100}
N=${N:-4}

# ROCm trap 1: SLURM exports ROCR_VISIBLE_DEVICES and torch 2.7 ROCm hard-errors on it.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi
# ROCm trap 4: MIOpen's kernel DB under ~/.config is read-only on VAST -> any Conv
# (the Qwen3-VL vision patch-embed) dies with miopenStatusInternalError.
export MIOPEN_USER_DB_PATH="$REPO/.tmp_work/miopen/$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH" logs build

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

"$PY" scripts/check_prompt_parity.py --model "$MODEL"

"$PY" -m echo_verl.eval.run_eval \
    --local-model "$MODEL" \
    --eval-jsonl build/eval.jsonl \
    --out build/eval_smoke.jsonl \
    --limit "$N" --per-type 1

echo "=== smoke episodes ==="
"$PY" - <<'PYEOF'
import json
for line in open("build/eval_smoke.jsonl"):
    r = json.loads(line)
    print(f"--- {r['question_type']} | finish={r['finish_reason']} "
          f"turns={r['turns']} tools={len(r['tool_calls'])} imgs={r['images_used']}")
    print("  Q:", r["question"][:120])
    print("  GOLD:", str(r["gold_answer"])[:160])
    print("  PRED:", str(r["answer"])[:400])
    if r["tool_calls"]:
        print("  TRACE:", json.dumps(r["tool_calls"])[:300])
PYEOF
