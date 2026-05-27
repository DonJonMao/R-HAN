#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/mnt/nvme/projects/R-HAN
LOG_DIR="${PROJECT_ROOT}/logs"

CURRENT_RUNNER_PATH="${PROJECT_ROOT}/run_stage2_v4_3_selected_serial_mbpp_20260407.sh"
CURRENT_DATASET_LOG="${LOG_DIR}/stage2_v4_3_mbpp_20260407.log"
CURRENT_SUITE_PROGRESS="${PROJECT_ROOT}/outputs/mas_stage2_v4_3_selected_runs_20260407/mbpp/suite_progress.json"
CURRENT_REPORT_JSON="${PROJECT_ROOT}/outputs/mas_stage2_v4_3_selected_runs_20260407/mbpp/mbpp/report.json"
CURRENT_RUNNER_PID="${CURRENT_RUNNER_PID:-3975597}"
CURRENT_TRAIN_PID="${CURRENT_TRAIN_PID:-3975609}"
CURRENT_TRAIN_PATTERN="train_mas_stage2_v4_3_target_suite.py --data-root /mnt/nvme/projects/R-HAN/dataset/mas_treesearch_aflow_four_20260318 --output-root /mnt/nvme/projects/R-HAN/outputs/mas_stage2_v4_3_selected_runs_20260407/mbpp --dataset mbpp"

V44_RUNNER="${PROJECT_ROOT}/run_stage2_v4_4_selected_serial_gsm8k_mbpp_nlgraph_20260409.sh"
V44_MASTER_LOG="${LOG_DIR}/stage2_v4_4_selected_serial_20260409_master.log"
V44_RUNNER_PID_FILE="${LOG_DIR}/stage2_v4_4_runner_20260409.pid"

POLL_SECONDS="${POLL_SECONDS:-60}"

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

resolve_pid() {
  local pattern="$1"
  pgrep -fo "${pattern}" 2>/dev/null || true
}

refresh_current_pids() {
  if ! pid_alive "${CURRENT_RUNNER_PID}"; then
    CURRENT_RUNNER_PID="$(resolve_pid "${CURRENT_RUNNER_PATH}")"
  fi
  if ! pid_alive "${CURRENT_TRAIN_PID}"; then
    CURRENT_TRAIN_PID="$(resolve_pid "${CURRENT_TRAIN_PATTERN}")"
  fi
}

current_mbpp_running() {
  refresh_current_pids
  pid_alive "${CURRENT_RUNNER_PID}" || pid_alive "${CURRENT_TRAIN_PID}"
}

json_status_value() {
  local file_path="$1"
  if [[ ! -f "${file_path}" ]]; then
    return 0
  fi
  grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' "${file_path}" | tail -n 1 | sed -E 's/.*"status"[[:space:]]*:[[:space:]]*"([^"]*)"/\1/' || true
}

current_mbpp_completed() {
  if [[ -f "${CURRENT_SUITE_PROGRESS}" ]] && grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "${CURRENT_SUITE_PROGRESS}"; then
    return 0
  fi
  if [[ -f "${CURRENT_REPORT_JSON}" ]]; then
    return 0
  fi
  if [[ -f "${CURRENT_DATASET_LOG}" ]] && grep -q '\[done\]\[mbpp\]' "${CURRENT_DATASET_LOG}"; then
    return 0
  fi
  return 1
}

current_status_snapshot() {
  refresh_current_pids
  local runner_state="dead"
  local train_state="dead"
  local suite_status="missing"
  local last_line="none"

  if pid_alive "${CURRENT_RUNNER_PID}"; then
    runner_state="alive"
  fi
  if pid_alive "${CURRENT_TRAIN_PID}"; then
    train_state="alive"
  fi
  if [[ -f "${CURRENT_SUITE_PROGRESS}" ]]; then
    suite_status="$(json_status_value "${CURRENT_SUITE_PROGRESS}")"
  fi
  if [[ -f "${CURRENT_DATASET_LOG}" ]]; then
    last_line="$(grep -E '\[(train|periodic-validation|validation|test)\]\[mbpp\]' "${CURRENT_DATASET_LOG}" | tail -n 1 || true)"
    if [[ -z "${last_line}" ]]; then
      last_line="$(tail -n 1 "${CURRENT_DATASET_LOG}" || true)"
    fi
  fi
  last_line="${last_line:0:280}"
  log "mbpp-status runner_pid=${CURRENT_RUNNER_PID:-0}(${runner_state}) train_pid=${CURRENT_TRAIN_PID:-0}(${train_state}) suite_status=${suite_status} last='${last_line}'"
}

v44_already_running() {
  pgrep -f "${V44_RUNNER}" >/dev/null 2>&1 || pgrep -f "train_mas_stage2_v4_4_target_suite.py" >/dev/null 2>&1
}

start_v44() {
  if v44_already_running; then
    log "Detected existing v4.4 process, skip auto-start."
    return 0
  fi
  : > "${V44_MASTER_LOG}"
  log "Launching v4.4 runner: ${V44_RUNNER}" | tee -a "${V44_MASTER_LOG}"
  nohup /bin/bash "${V44_RUNNER}" >> "${V44_MASTER_LOG}" 2>&1 &
  local runner_pid=$!
  echo "${runner_pid}" > "${V44_RUNNER_PID_FILE}"
  log "v4.4 runner started with pid=${runner_pid}" | tee -a "${V44_MASTER_LOG}"
}

main() {
  log "Watcher started for current mbpp(v4.3) -> v4.4 handoff."
  current_status_snapshot

  while current_mbpp_running; do
    current_status_snapshot
    sleep "${POLL_SECONDS}"
  done

  log "Detected mbpp processes exited; waiting 10s for completion markers."
  sleep 10

  if current_mbpp_completed; then
    log "Current mbpp run completed cleanly; auto-starting v4.4(gsm8k, mbpp, nlgraph)."
    start_v44
    exit 0
  fi

  current_status_snapshot
  log "Current mbpp run exited without clean completion markers; not starting v4.4 automatically."
  exit 1
}

main "$@"
