#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/nvme/projects/R-HAN
LOG_DIR="${PROJECT_ROOT}/logs"

V41_RUN_PID=3537011
V41_TRAIN_PID=3537021
V41_DATASET_LOG="${LOG_DIR}/stage2_v4_1_gsm8k_20260404.log"
V41_MASTER_LOG="${LOG_DIR}/stage2_v4_1_serial_20260404_master.log"
V41_SUITE_PROGRESS="${PROJECT_ROOT}/outputs/mas_stage2_v4_1_selected_runs_20260404/gsm8k/suite_progress.json"

V42_RUNNER="${PROJECT_ROOT}/run_stage2_v4_2_selected_serial_20260404.sh"
V42_MASTER_LOG="${LOG_DIR}/stage2_v4_2_serial_20260404_master.log"
V42_RUNNER_PID_FILE="${LOG_DIR}/stage2_v4_2_runner_20260404.pid"

POLL_SECONDS=60

timestamp() {
  date -Iseconds
}

log() {
  echo "[$(timestamp)] $*"
}

pid_alive() {
  local pid="$1"
  if [[ -z "${pid}" || "${pid}" == "0" ]]; then
    return 1
  fi
  kill -0 "${pid}" 2>/dev/null
}

v41_running() {
  pid_alive "${V41_RUN_PID}" || pid_alive "${V41_TRAIN_PID}"
}

v41_completed() {
  if [[ -f "${V41_SUITE_PROGRESS}" ]] && grep -q '"status": "completed"' "${V41_SUITE_PROGRESS}"; then
    return 0
  fi
  if [[ -f "${V41_DATASET_LOG}" ]] && grep -q '\[done\]\[gsm8k\]' "${V41_DATASET_LOG}"; then
    return 0
  fi
  if [[ -f "${V41_MASTER_LOG}" ]] && grep -q '\[runner-done\]' "${V41_MASTER_LOG}"; then
    return 0
  fi
  return 1
}

v42_already_running() {
  pgrep -f "train_mas_stage2_v4_2_target_suite.py" >/dev/null 2>&1 || \
    pgrep -f "run_stage2_v4_2_selected_serial_20260404.sh" >/dev/null 2>&1
}

start_v42() {
  if v42_already_running; then
    log "Detected existing v4.2 process, skip auto-start."
    return 0
  fi
  : > "${V42_MASTER_LOG}"
  log "Launching v4.2 runner: ${V42_RUNNER}" | tee -a "${V42_MASTER_LOG}"
  nohup /bin/bash "${V42_RUNNER}" >> "${V42_MASTER_LOG}" 2>&1 &
  local runner_pid=$!
  echo "${runner_pid}" > "${V42_RUNNER_PID_FILE}"
  log "v4.2 runner started with pid=${runner_pid}" | tee -a "${V42_MASTER_LOG}"
}

main() {
  log "Watcher started for v4.1 -> v4.2 handoff."
  log "Tracking v4.1 run_pid=${V41_RUN_PID} train_pid=${V41_TRAIN_PID}."
  while v41_running; do
    log "v4.1 still running; sleeping ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
  done

  log "v4.1 process exited; checking completion markers."
  sleep 10
  if v41_completed; then
    log "v4.1 completed cleanly; auto-starting v4.2."
    start_v42
    exit 0
  fi

  log "v4.1 exited without a clean completion marker; not starting v4.2 automatically."
  exit 1
}

main "$@"
