#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/mnt/nvme/projects/R-HAN/outputs/stage2_final_design_deductive_qwen_gsm8k_math_20260426_120327"
SOURCE_ROOT="${RUN_ROOT}/code/deductive_qwen_source"
OUTPUT_ROOT="${RUN_ROOT}/deductive_qwen_gsm8k_math_3x"
PYTHON_BIN="/home/maozhifang/miniconda3/bin/python"
DATA_ROOT="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_aflow_four_20260318"
STAGE1_ROOT="/mnt/nvme/projects/R-HAN/outputs/stage1_selected_ckpts_v2_20260330"
ROUTER_PORT="8127"

export PYTHONUNBUFFERED=1
export EMBED_API_BASE="http://127.0.0.1:8018"
export EMBED_MODEL="/mnt/nvme/Qwen3-Embedding-8B"
export EMBED_DIM="256"

cd "${SOURCE_ROOT}"

echo "[launch] $(date -Iseconds) run_root=${RUN_ROOT}"
echo "[launch] source_root=${SOURCE_ROOT}"
echo "[launch] output_root=${OUTPUT_ROOT}"
echo "[launch] data_root=${DATA_ROOT}"
echo "[launch] router_port=${ROUTER_PORT}"
echo "[launch] embed=${EMBED_API_BASE} model=${EMBED_MODEL} dim=${EMBED_DIM}"
echo "[launch] datasets=gsm8k,math mode=3x background-safe snapshot"

"${PYTHON_BIN}" "${SOURCE_ROOT}/Stage2-GCR+/run_stage2_gcr_plus_parallel.py" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset gsm8k \
  --dataset math \
  --workers 3 \
  --mode 3x \
  --x3-backend "b1=http://127.0.0.1:8041@1" \
  --x3-backend "b2=http://127.0.0.1:8042@1" \
  --x3-backend "b3=http://127.0.0.1:8043@1" \
  --router-port "${ROUTER_PORT}" \
  --python-bin "${PYTHON_BIN}" \
  --train-script train_mas_stage2_v4_4_target_suite.py \
  --cwd "${SOURCE_ROOT}" \
  "--train-arg=--seed 7" \
  "--train-arg=--search-iterations 8" \
  "--train-arg=--candidate-core-k 4" \
  "--train-arg=--candidate-explore-k 3" \
  "--train-arg=--candidate-max-k 8" \
  "--train-arg=--tier1-repeats 1" \
  "--train-arg=--tier2-repeats 1" \
  "--train-arg=--stage2-turn-count 5" \
  "--train-arg=--memory-top-k 4" \
  "--train-arg=--soft-prune-top-k 3" \
  "--train-arg=--soft-prune-threshold 0.38" \
  "--train-arg=--hard-prune-after-turn 4" \
  "--train-arg=--periodic-every 1000000" \
  "--train-arg=--periodic-size 0" \
  "--train-arg=--max-train -1" \
  "--train-arg=--max-validation 0" \
  "--train-arg=--max-test 0" \
  "--train-arg=--stage1-checkpoint-root ${STAGE1_ROOT}" \
  "--train-arg=--tier1-max-tokens 256" \
  "--train-arg=--tier2-max-tokens 896"
