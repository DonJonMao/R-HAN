#!/usr/bin/env bash
set -euo pipefail

export TZ=Asia/Shanghai

REPO_ROOT="/mnt/nvme/projects/R-HAN"
LOG_DIR="${REPO_ROOT}/logs"
OUTPUT_DIR="${REPO_ROOT}/outputs"
PYTHON_BIN="/home/maozhifang/miniconda3/bin/python"
RUNNER="${REPO_ROOT}/Stage2-UCC/run_stage2_phase3a_unified_parallel.py"
TRAIN_SCRIPT="${REPO_ROOT}/Stage2-UCC/train_mas_stage2_phase3a_unified_target_suite.py"
DATA_ROOT_GENERAL="${REPO_ROOT}/dataset/mas_treesearch_threeway_general_trainview_20260319"
STAGE1_CHECKPOINT_ROOT="${REPO_ROOT}/outputs/stage1_selected_ckpts_v2_20260330"

CURRENT_MBPP_OUTPUT_ROOT="${REPO_ROOT}/outputs/stage2_ucc_phase3a_unified_mbpp_3x_20260416_181035"
CURRENT_MBPP_PID_FILE="${REPO_ROOT}/logs/stage2_ucc_phase3a_unified_mbpp_3x_20260416_181035.pid"

TARGET_TIME="2026-04-17 02:00:00"
SCRIPT_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/schedule_stage2_ucc_mmlu_pro_then_nlgraph_20260417_0200_${SCRIPT_TS}.log"
MASTER_PID="${LOG_DIR}/schedule_stage2_ucc_mmlu_pro_then_nlgraph_20260417_0200_${SCRIPT_TS}.pid"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
echo "$$" > "${MASTER_PID}"
exec >>"${MASTER_LOG}" 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

wait_until_target() {
  local target_epoch
  local now_epoch
  local sleep_seconds
  target_epoch="$(date -d "${TARGET_TIME}" +%s)"
  now_epoch="$(date +%s)"
  if (( now_epoch < target_epoch )); then
    sleep_seconds=$(( target_epoch - now_epoch ))
    log "sleeping ${sleep_seconds}s until ${TARGET_TIME} ${TZ}"
    sleep "${sleep_seconds}"
  else
    log "target time ${TARGET_TIME} ${TZ} already passed; continuing immediately"
  fi
}

stop_current_mbpp() {
  local raw_pids=()
  local unique_pids=()
  local pid
  local still_running=()

  if [[ -f "${CURRENT_MBPP_PID_FILE}" ]]; then
    while IFS= read -r pid; do
      [[ -n "${pid}" ]] && raw_pids+=("${pid}")
    done < "${CURRENT_MBPP_PID_FILE}"
  fi

  while IFS= read -r pid; do
    [[ -n "${pid}" ]] && raw_pids+=("${pid}")
  done < <(pgrep -f "${CURRENT_MBPP_OUTPUT_ROOT}" || true)

  if (( ${#raw_pids[@]} == 0 )); then
    log "no matching MBPP processes found for ${CURRENT_MBPP_OUTPUT_ROOT}"
    return
  fi

  mapfile -t unique_pids < <(printf '%s\n' "${raw_pids[@]}" | awk '!seen[$0]++')
  log "stopping current MBPP run: ${unique_pids[*]}"
  kill "${unique_pids[@]}" || true
  sleep 10

  still_running=()
  for pid in "${unique_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      still_running+=("${pid}")
    fi
  done

  if (( ${#still_running[@]} > 0 )); then
    log "forcing termination for remaining MBPP processes: ${still_running[*]}"
    kill -9 "${still_running[@]}" || true
  fi
}

launch_dataset() {
  local dataset="$1"
  local run_ts
  local output_root
  local top_log
  local top_pid
  local run_pid
  local status
  local cmd=()

  run_ts="$(date +%Y%m%d_%H%M%S)"
  output_root="${OUTPUT_DIR}/stage2_ucc_phase3a_unified_${dataset}_3x_${run_ts}"
  top_log="${LOG_DIR}/stage2_ucc_phase3a_unified_${dataset}_3x_${run_ts}.log"
  top_pid="${LOG_DIR}/stage2_ucc_phase3a_unified_${dataset}_3x_${run_ts}.pid"

  mkdir -p "${output_root}"
  {
    printf '[launch] %s\n' "${run_ts}"
    printf '[dataset] %s\n' "${dataset}"
    printf '[output_root] %s\n' "${output_root}"
  } > "${top_log}"

  cmd=(
    env
    PYTHONUNBUFFERED=1
    PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Stage2-UCC:${REPO_ROOT}/Stage2-UCC/vendor"
    LLM_JUDGE_API_BASE=http://127.0.0.1:8045
    EMBED_API_BASE=http://127.0.0.1:8018
    "${PYTHON_BIN}"
    "${RUNNER}"
    --data-root "${DATA_ROOT_GENERAL}"
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
    --train-arg "--max-validation 0"
    --train-arg "--max-test 0"
    --train-arg "--max-prompt-chars 16000"
    --train-arg "--stage1-checkpoint-root ${STAGE1_CHECKPOINT_ROOT}"
  )

  log "starting ${dataset} training, log=${top_log}, output=${output_root}"
  "${cmd[@]}" >> "${top_log}" 2>&1 &
  run_pid=$!
  echo "${run_pid}" > "${top_pid}"
  log "${dataset} runner pid=${run_pid}"

  if wait "${run_pid}"; then
    status=0
  else
    status=$?
  fi

  log "${dataset} finished with status=${status}"
  return "${status}"
}

main() {
  log "scheduler started; pid file=${MASTER_PID}"
  wait_until_target
  stop_current_mbpp
  launch_dataset "mmlu_pro"
  launch_dataset "nlgraph"
  log "all scheduled training jobs completed"
}

main "$@"
