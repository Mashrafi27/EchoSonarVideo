#!/usr/bin/env bash
# Build the verl RL parquet for GRPO from the VQA jsonl + assembled frame tree.
#
#   bash scripts/build_grpo_parquet.sh
#
# Produces:
#   build/rl.jsonl          all train-VQA studies as rl_records  (echo_rl.cli)
#   build/rl_train.parquet  verl rows                            (echo_verl.generate_trainset)
#   build/eval.jsonl        all test-VQA studies as rl_records
#   build/rl_val.parquet    verl rows for validation
#
# Split note: we do NOT use echojepa_study_split_full.csv. train_vqa and test_vqa
# have zero study overlap (5061 + 1215 = 6276 distinct), so "--split all" over
# each file is already the clean train / test partition.
set -xeuo pipefail

REPO=${REPO:-/home/mashrafimonon/EchoSonarVideo}
cd "$REPO"

source /hdd2/ahmedaly/echogrpo/env.sh
PY=${PY:-/hdd2/ahmedaly/echogrpo/venv/bin/python}   # needs pyarrow (from vllm install)
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

mkdir -p build

# 1. rl_records from the two VQA files (joins to ECHO_PREPROCESSED_DIR frame tree)
"$PY" -m echo_rl.cli build-rl   --split all
"$PY" -m echo_rl.cli build-eval --split all

# 2. jsonl -> verl parquet
"$PY" echo_verl/generate_trainset.py --rl-jsonl build/rl.jsonl   --out build/rl_train.parquet
"$PY" echo_verl/generate_trainset.py --rl-jsonl build/eval.jsonl --out build/rl_val.parquet

wc -l build/rl.jsonl build/eval.jsonl
ls -la build/rl_train.parquet build/rl_val.parquet
