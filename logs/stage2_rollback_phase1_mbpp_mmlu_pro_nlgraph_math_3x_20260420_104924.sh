#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/projects/R-HAN"
DATA_ROOT="/tmp/stage2_ucc_phase2_train180_plusmath_20260419_224402"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTAMP="20260420_104924"

cd "$ROOT"

run_dataset() {
  local dataset="$1"
  local output_name="$2"
  echo "[$(date '+%F %T')] start dataset=${dataset} output=${output_name}"
  "$PYTHON_BIN" "$ROOT/stage2-rollback/run_stage2_rollback_stage1_parallel.py" \
    --data-root "$DATA_ROOT" \
    --output-root "$ROOT/outputs/${output_name}" \
    --dataset "$dataset" \
    --workers 3 \
    --seed 7 \
    --max-train 180
  echo "[$(date '+%F %T')] done dataset=${dataset} output=${output_name}"
}

run_dataset "mbpp" "stage2-rollback-phase1_mbpp_3x_${TIMESTAMP}"
run_dataset "mmlu_pro" "stage2-rollback-phase1_mmlu-pro_3x_${TIMESTAMP}"
run_dataset "nlgraph" "stage2-rollback-phase1_nlgraph_3x_${TIMESTAMP}"
run_dataset "math" "stage2-rollback-phase1_math_3x_${TIMESTAMP}"

echo "[$(date '+%F %T')] all rollback phase1 datasets finished"
