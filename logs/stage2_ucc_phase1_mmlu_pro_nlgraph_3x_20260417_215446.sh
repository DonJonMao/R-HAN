#!/usr/bin/env bash
set -euo pipefail

export TZ=Asia/Shanghai

REPO_ROOT="/mnt/nvme/projects/R-HAN"
TMP_ROOT="/tmp/stage2_ucc_phase1_train180_20260417_113144"
LOG_DIR="${REPO_ROOT}/logs"
OUTPUT_DIR="${REPO_ROOT}/outputs"
PYTHON_BIN="/home/maozhifang/miniconda3/bin/python"
RUNNER="${REPO_ROOT}/Stage2-UCC/run_stage2_phase3a_unified_parallel.py"
TRAIN_SCRIPT="${REPO_ROOT}/Stage2-UCC/train_mas_stage2_phase1_semantic_safe_override_target_suite.py"
STAGE1_CHECKPOINT_ROOT="${REPO_ROOT}/outputs/mas_treesearch_dataset_suite"
RUN_TAG="20260417_215446"
MASTER_LOG="${LOG_DIR}/stage2_ucc_phase1_mmlu_pro_nlgraph_3x_${RUN_TAG}.log"
MASTER_PID="${LOG_DIR}/stage2_ucc_phase1_mmlu_pro_nlgraph_3x_${RUN_TAG}.pid"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
echo "$$" > "${MASTER_PID}"
exec >>"${MASTER_LOG}" 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

launch_dataset() {
  local dataset="$1"
  local output_slug="$2"
  local output_root="${OUTPUT_DIR}/stage2-ucc- phase1_${output_slug}_3x_${RUN_TAG}"
  local top_log="${LOG_DIR}/stage2_ucc_phase1_${output_slug}_3x_${RUN_TAG}.log"
  local status=0
  local cmd=()

  mkdir -p "${output_root}"
  {
    printf '[launch] %s\n' "${RUN_TAG}"
    printf '[dataset] %s\n' "${dataset}"
    printf '[output_root] %s\n' "${output_root}"
    printf '[data_root] %s\n' "${TMP_ROOT}"
    printf '[stage1_checkpoint_root] %s\n' "${STAGE1_CHECKPOINT_ROOT}"
  } > "${top_log}"

  cmd=(
    env
    PYTHONUNBUFFERED=1
    PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Stage2-UCC:${REPO_ROOT}/Stage2-UCC/vendor"
    LLM_JUDGE_API_BASE=http://127.0.0.1:8045
    EMBED_API_BASE=http://127.0.0.1:8018
    "${PYTHON_BIN}"
    "${RUNNER}"
    --data-root "${TMP_ROOT}"
    --output-root "${output_root}"
    --dataset "${dataset}"
    --workers 3
    --seed 7
    --mode 3x
    --x3-backend gpu0=http://127.0.0.1:8041@1
    --x3-backend gpu1=http://127.0.0.1:8042@1
    --x3-backend gpu2=http://127.0.0.1:8043@1
    --router-host 127.0.0.1
    --router-port 8039
    --python-bin "${PYTHON_BIN}"
    --train-script "${TRAIN_SCRIPT}"
    --cwd "${REPO_ROOT}"
    --train-arg "--search-iterations 8"
    --train-arg "--candidate-core-k 4"
    --train-arg "--candidate-explore-k 3"
    --train-arg "--candidate-max-k 8"
    --train-arg "--tier1-max-tokens 256"
    --train-arg "--tier2-max-tokens 896"
    --train-arg "--tier1-repeats 1"
    --train-arg "--tier2-repeats 1"
    --train-arg "--stage2-turn-count 5"
    --train-arg "--memory-top-k 4"
    --train-arg "--soft-prune-top-k 3"
    --train-arg "--soft-prune-threshold 0.38"
    --train-arg "--hard-prune-after-turn 4"
    --train-arg "--periodic-every 1000000"
    --train-arg "--periodic-size 0"
    --train-arg "--max-train 180"
    --train-arg "--max-validation 0"
    --train-arg "--max-test 0"
    --train-arg "--frontier-top-k 4"
    --train-arg "--max-prompt-chars 16000"
    --train-arg "--stage1-checkpoint-root ${STAGE1_CHECKPOINT_ROOT}"
  )

  log "starting dataset=${dataset} output_root=${output_root} log=${top_log}"
  if "${cmd[@]}" >> "${top_log}" 2>&1; then
    status=0
  else
    status=$?
  fi
  log "finished dataset=${dataset} status=${status} log=${top_log}"
  return "${status}"
}

main() {
  local overall=0
  log "scheduler started pid=$$ tmp_root=${TMP_ROOT}"

  if [[ ! -d "${TMP_ROOT}/mmlu_pro" || ! -d "${TMP_ROOT}/nlgraph" ]]; then
    log "required tmp dataset root is missing: ${TMP_ROOT}"
    exit 2
  fi

  launch_dataset "mmlu_pro" "mmlu-pro" || overall=$?
  launch_dataset "nlgraph" "nlgraph" || overall=$?

  log "scheduler finished overall_status=${overall}"
  exit "${overall}"
}

main "$@"
