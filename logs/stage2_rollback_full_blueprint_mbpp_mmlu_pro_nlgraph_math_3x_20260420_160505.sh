#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/nvme/projects/R-HAN"
DATA_ROOT="/tmp/stage2_ucc_phase2_train180_plusmath_20260419_224402"
MASTER_LOG="${ROOT}/logs/stage2_rollback_full_blueprint_20260420_160505.log"
PYTHON_BIN="${PYTHON_BIN:-python}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MASTER_LOG}"
}

run_dataset() {
  local dataset="$1"
  local output_dir="$2"
  log "starting dataset=${dataset} output=${output_dir}"
  "${PYTHON_BIN}" "${ROOT}/stage2-rollback/run_stage2_rollback_full_blueprint_parallel.py" \
    --data-root "${DATA_ROOT}" \
    --output-root "${output_dir}" \
    --dataset "${dataset}" \
    --workers 3 \
    --gpu-ids 0,1,2 \
    --seed 7 \
    --max-train 180 \
    --debug-limit 3 \
    --epochs-verifier 40 \
    --epochs-boundary 40 \
    --epochs-diffusion 40 \
    --epochs-selector 40
  log "completed dataset=${dataset}"
}

log "rollback full blueprint suite begin"
run_dataset "mbpp"      "${ROOT}/outputs/stage2-rollback-full_mbpp_3x_20260420_160505"
run_dataset "mmlu_pro"  "${ROOT}/outputs/stage2-rollback-full_mmlu-pro_3x_20260420_160505"
run_dataset "nlgraph"   "${ROOT}/outputs/stage2-rollback-full_nlgraph_3x_20260420_160505"
run_dataset "math"      "${ROOT}/outputs/stage2-rollback-full_math_3x_20260420_160505"
log "rollback full blueprint suite finished"
