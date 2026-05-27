#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/nvme/projects/R-HAN}
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

PYTHON_BIN=${PYTHON_BIN:-/home/maozhifang/miniconda3/bin/python}
SERVE_SCRIPT=${SERVE_SCRIPT:-${PROJECT_ROOT}/scripts/serve_qwen3_8b_train.sh}
GCR_RUNNER=${GCR_RUNNER:-${PROJECT_ROOT}/Stage2-GCR+/run_stage2_gcr_plus_parallel.py}

CURRENT_RUNNER_PATH=${CURRENT_RUNNER_PATH:-${PROJECT_ROOT}/run_stage2_v4_4_selected_serial_gsm8k_mbpp_nlgraph_20260409.sh}
CURRENT_MBPP_TRAIN_PATTERN=${CURRENT_MBPP_TRAIN_PATTERN:-train_mas_stage2_v4_4_target_suite.py --data-root /mnt/nvme/projects/R-HAN/dataset/mas_treesearch_aflow_four_20260318 --output-root /mnt/nvme/projects/R-HAN/outputs/mas_stage2_v4_4_selected_runs_20260410_manual1/mbpp --dataset mbpp}
CURRENT_SUITE_PROGRESS=${CURRENT_SUITE_PROGRESS:-${PROJECT_ROOT}/outputs/mas_stage2_v4_4_selected_runs_20260410_manual1/mbpp/suite_progress.json}
CURRENT_REPORT_JSON=${CURRENT_REPORT_JSON:-${PROJECT_ROOT}/outputs/mas_stage2_v4_4_selected_runs_20260410_manual1/mbpp/mbpp/report.json}
CURRENT_DATASET_LOG=${CURRENT_DATASET_LOG:-${PROJECT_ROOT}/logs/stage2_v4_4_mbpp_20260410_manual1.log}

TP4_CHAT_PATTERN=${TP4_CHAT_PATTERN:-python -m vllm.entrypoints.openai.api_server --model /mnt/nvme/Qwen3-8B --served-model-name qwen3-8b-train --port 8043 --host 0.0.0.0 --tensor-parallel-size 4}
TP4_CHAT_PORT=${TP4_CHAT_PORT:-8043}

X3_GPUS=(${X3_GPUS:-0 1 2})
X3_PORTS=(${X3_PORTS:-8041 8042 8043})
X3_NAMES=(${X3_NAMES:-b1 b2 b3})

TRAIN_GPU=${TRAIN_GPU-}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
POLL_SECONDS=${POLL_SECONDS:-60}
ENABLE_DEADLINE=${ENABLE_DEADLINE:-0}
DEADLINE_LOCAL=${DEADLINE_LOCAL:-2026-04-14 05:00:00}
DEADLINE_TZ=${DEADLINE_TZ:-Asia/Shanghai}
DRY_RUN=${DRY_RUN:-0}

SERVICE_PID_DIR="${LOG_DIR}/stage2_gcr_3x_service_pids_${RUN_TAG}"
SERVICE_LOG_DIR="${LOG_DIR}/stage2_gcr_3x_service_logs_${RUN_TAG}"
TRAIN_OUTPUT_ROOT=${TRAIN_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/stage2_gcr_plus_mbpp_3x_${RUN_TAG}}
TRAIN_MASTER_LOG=${TRAIN_MASTER_LOG:-${LOG_DIR}/stage2_gcr_plus_mbpp_3x_${RUN_TAG}.log}
TRAIN_PID_FILE=${TRAIN_PID_FILE:-${LOG_DIR}/stage2_gcr_plus_mbpp_3x_${RUN_TAG}.pid}
WATCHER_LOG=${WATCHER_LOG:-${LOG_DIR}/watch_v4_4_mbpp_then_stage2_gcr_3x_${RUN_TAG}.log}

mkdir -p "${SERVICE_PID_DIR}" "${SERVICE_LOG_DIR}"
: > "${WATCHER_LOG}"

COMMON_TRAIN_ARGS=(
  --resume
  --seed 7
  --search-iterations 8
  --candidate-core-k 4
  --candidate-explore-k 3
  --candidate-max-k 8
  --tier1-repeats 1
  --tier2-repeats 1
  --stage2-turn-count 5
  --memory-top-k 4
  --soft-prune-top-k 3
  --soft-prune-threshold 0.38
  --hard-prune-after-turn 4
  --periodic-every 1000000
  --periodic-size 0
  --max-train -1
  --max-validation 0
  --max-test -1
  --stage1-checkpoint-root /mnt/nvme/projects/R-HAN/outputs/stage1_selected_ckpts_v2_20260330
)

if [[ "${ENABLE_DEADLINE}" == "1" ]]; then
  DEADLINE_EPOCH=$(TZ="${DEADLINE_TZ}" date -d "${DEADLINE_LOCAL}" +%s)
else
  DEADLINE_EPOCH=0
fi

_timestamp() {
  TZ="${DEADLINE_TZ}" date '+%Y-%m-%dT%H:%M:%S%z'
}

log() {
  local line="[$(_timestamp)] $*"
  echo "${line}" | tee -a "${WATCHER_LOG}"
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "[dry-run] $*"
    return 0
  fi
  "$@"
}

pid_alive() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

resolve_pid() {
  local pattern="$1"
  pgrep -fo "$pattern" 2>/dev/null || true
}

port_listener_pid() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | head -n 1 || true
    return 0
  fi
  ss -ltnp 2>/dev/null | awk -v p=":${port}" '$4 ~ p {print $NF}' | sed -E 's/.*pid=([0-9]+).*/\1/' | head -n 1 || true
}

pid_cmdline() {
  local pid="$1"
  ps -p "${pid}" -o cmd= 2>/dev/null || true
}

wait_for_pid_exit() {
  local pid="$1"
  local timeout_s="${2:-30}"
  local start
  start=$(date +%s)
  while pid_alive "${pid}"; do
    if (( $(date +%s) - start >= timeout_s )); then
      return 1
    fi
    sleep 1
  done
  return 0
}

stop_pid_gracefully() {
  local pid="$1"
  local label="$2"
  if ! pid_alive "${pid}"; then
    return 0
  fi
  log "Stopping ${label} pid=${pid} with SIGTERM"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "[dry-run] kill ${pid}"
    return 0
  fi
  run_cmd kill "${pid}"
  if wait_for_pid_exit "${pid}" 30; then
    return 0
  fi
  log "${label} pid=${pid} did not exit in time; escalating to SIGKILL"
  run_cmd kill -9 "${pid}"
  wait_for_pid_exit "${pid}" 10 || true
}

ensure_port_safe_for_qwen8b() {
  local port="$1"
  local pid
  pid=$(port_listener_pid "${port}")
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  local cmd
  cmd=$(pid_cmdline "${pid}")
  if [[ "${cmd}" == *"/mnt/nvme/Qwen3-8B"* && "${cmd}" == *"vllm.entrypoints.openai.api_server"* ]]; then
    log "Port ${port} is occupied by Qwen3-8B pid=${pid}; stopping it before relaunch."
    stop_pid_gracefully "${pid}" "qwen3-8b-port-${port}"
    return 0
  fi
  log "Port ${port} is occupied by non-target pid=${pid}. Refusing to continue. cmd=${cmd}"
  return 1
}

wait_for_http_ready() {
  local url="$1"
  local label="$2"
  local timeout_s="${3:-600}"
  local start
  start=$(date +%s)
  while true; do
    if curl -sf "${url}" >/dev/null 2>&1; then
      log "Ready: ${label} (${url})"
      return 0
    fi
    if (( $(date +%s) - start >= timeout_s )); then
      log "Timeout waiting for ${label} (${url})"
      return 1
    fi
    sleep 5
  done
}

current_mbpp_train_pid() {
  resolve_pid "${CURRENT_MBPP_TRAIN_PATTERN}"
}

current_v44_runner_pid() {
  resolve_pid "${CURRENT_RUNNER_PATH}"
}

current_mbpp_completed_marker() {
  if [[ -f "${CURRENT_REPORT_JSON}" ]]; then
    return 0
  fi
  if [[ -f "${CURRENT_SUITE_PROGRESS}" ]] && grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "${CURRENT_SUITE_PROGRESS}"; then
    return 0
  fi
  if [[ -f "${CURRENT_DATASET_LOG}" ]] && grep -q '\[done\]\[mbpp\]' "${CURRENT_DATASET_LOG}"; then
    return 0
  fi
  return 1
}

current_status_snapshot() {
  local train_pid runner_pid train_state runner_state suite_status deadline_status
  train_pid=$(current_mbpp_train_pid)
  runner_pid=$(current_v44_runner_pid)
  train_state="dead"
  runner_state="dead"
  suite_status="missing"
  deadline_status="disabled"
  if [[ "${ENABLE_DEADLINE}" == "1" ]]; then
    deadline_status=$(( DEADLINE_EPOCH - $(date +%s) ))
  fi
  if pid_alive "${train_pid}"; then
    train_state="alive"
  fi
  if pid_alive "${runner_pid}"; then
    runner_state="alive"
  fi
  if [[ -f "${CURRENT_SUITE_PROGRESS}" ]]; then
    suite_status=$(grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' "${CURRENT_SUITE_PROGRESS}" | tail -n 1 | sed -E 's/.*"status"[[:space:]]*:[[:space:]]*"([^"]*)"/\1/' || true)
    suite_status=${suite_status:-unknown}
  fi
  log "monitor train_pid=${train_pid:-0}(${train_state}) runner_pid=${runner_pid:-0}(${runner_state}) suite_status=${suite_status} deadline=${deadline_status}"
}


force_stop_current_v44_if_needed() {
  local train_pid runner_pid
  train_pid=$(current_mbpp_train_pid)
  runner_pid=$(current_v44_runner_pid)
  if pid_alive "${train_pid}"; then
    stop_pid_gracefully "${train_pid}" "current-v4.4-mbpp-train"
  fi
  if pid_alive "${runner_pid}"; then
    stop_pid_gracefully "${runner_pid}" "current-v4.4-runner"
  fi
}

stop_current_tp4_qwen() {
  local pid
  pid=$(resolve_pid "${TP4_CHAT_PATTERN}")
  if [[ -z "${pid}" ]]; then
    pid=$(port_listener_pid "${TP4_CHAT_PORT}")
    if [[ -n "${pid}" ]]; then
      local cmd
      cmd=$(pid_cmdline "${pid}")
      if [[ "${cmd}" != *"/mnt/nvme/Qwen3-8B"* || "${cmd}" != *"tensor-parallel-size 4"* ]]; then
        log "Port ${TP4_CHAT_PORT} is not the expected TP4 qwen3-8b service. cmd=${cmd}"
        return 1
      fi
    fi
  fi
  if [[ -z "${pid}" ]]; then
    log "No TP4 qwen3-8b service found on port ${TP4_CHAT_PORT}; continuing."
    return 0
  fi
  stop_pid_gracefully "${pid}" "qwen3-8b-tp4"
}

start_single_qwen_services() {
  local idx gpu port name log_path pid_file pid
  for idx in 0 1 2; do
    gpu="${X3_GPUS[$idx]}"
    port="${X3_PORTS[$idx]}"
    name="${X3_NAMES[$idx]}"
    ensure_port_safe_for_qwen8b "${port}"
    log_path="${SERVICE_LOG_DIR}/qwen3_8b_${name}_gpu${gpu}_port${port}.log"
    pid_file="${SERVICE_PID_DIR}/qwen3_8b_${name}_gpu${gpu}_port${port}.pid"
    log "Launching qwen3-8b single-card backend ${name} on gpu=${gpu} port=${port}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo 0 > "${pid_file}"
      continue
    fi
    nohup env CUDA_VISIBLE_DEVICES="${gpu}" PORT="${port}" TP_SIZE=1 HOST=0.0.0.0 \
      "${SERVE_SCRIPT}" > "${log_path}" 2>&1 < /dev/null &
    pid=$!
    echo "${pid}" > "${pid_file}"
    log "Started backend ${name} pid=${pid} log=${log_path}"
  done

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  for idx in 0 1 2; do
    port="${X3_PORTS[$idx]}"
    name="${X3_NAMES[$idx]}"
    wait_for_http_ready "http://127.0.0.1:${port}/v1/models" "qwen3-8b-${name}" 900
  done
}

start_stage2_gcr_training() {
  local cmd=()
  cmd=(
    "${PYTHON_BIN}" "${GCR_RUNNER}"
    --data-root /mnt/nvme/projects/R-HAN/dataset/mas_treesearch_aflow_four_20260318
    --output-root "${TRAIN_OUTPUT_ROOT}"
    --dataset mbpp
    --workers 3
    --mode 3x
    --x3-backend "${X3_NAMES[0]}=http://127.0.0.1:${X3_PORTS[0]}@1"
    --x3-backend "${X3_NAMES[1]}=http://127.0.0.1:${X3_PORTS[1]}@1"
    --x3-backend "${X3_NAMES[2]}=http://127.0.0.1:${X3_PORTS[2]}@1"
    --train-script train_mas_stage2_v4_4_target_suite.py
    --train-arg=--resume
    --train-arg=--seed 7
    --train-arg=--search-iterations 8
    --train-arg=--candidate-core-k 4
    --train-arg=--candidate-explore-k 3
    --train-arg=--candidate-max-k 8
    --train-arg=--tier1-repeats 1
    --train-arg=--tier2-repeats 1
    --train-arg=--stage2-turn-count 5
    --train-arg=--memory-top-k 4
    --train-arg=--soft-prune-top-k 3
    --train-arg=--soft-prune-threshold 0.38
    --train-arg=--hard-prune-after-turn 4
    --train-arg=--periodic-every 1000000
    --train-arg=--periodic-size 0
    --train-arg=--max-train -1
    --train-arg=--max-validation 0
    --train-arg=--max-test -1
    --train-arg=--stage1-checkpoint-root /mnt/nvme/projects/R-HAN/outputs/stage1_selected_ckpts_v2_20260330
    --train-arg=--tier1-max-tokens 256
    --train-arg=--tier2-max-tokens 896
    --cwd "${PROJECT_ROOT}"
  )

  local train_device_label
  local -a train_env=(
    PYTHONUNBUFFERED=1
    PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/Stage2-GCR+"
    LLM_JUDGE_API_BASE="http://127.0.0.1:8045"
    EMBED_API_BASE="http://127.0.0.1:8018"
  )
  if [[ -n "${TRAIN_GPU}" ]]; then
    train_device_label="CUDA_VISIBLE_DEVICES=${TRAIN_GPU}"
    train_env=(CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" "${train_env[@]}")
  else
    train_device_label="CPU-only (CUDA_VISIBLE_DEVICES='')"
    train_env=(CUDA_VISIBLE_DEVICES="" "${train_env[@]}")
  fi

  log "Launching Stage2-GCR+ MBPP 3x training with ${train_device_label} output_root=${TRAIN_OUTPUT_ROOT}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "[dry-run] ${cmd[*]}"
    return 0
  fi
  : > "${TRAIN_MASTER_LOG}"
  nohup env     "${train_env[@]}"     "${cmd[@]}" >> "${TRAIN_MASTER_LOG}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${TRAIN_PID_FILE}"
  log "Stage2-GCR+ launcher started pid=${pid} log=${TRAIN_MASTER_LOG}"
}


execute_operation_1() {
  log "Executing operation 1: stop TP4 qwen, launch 3x single-card qwen on GPUs 0-2, keep GPU3 free, start Stage2-GCR+ MBPP training in CPU-only mode unless TRAIN_GPU is explicitly set."
  force_stop_current_v44_if_needed
  stop_current_tp4_qwen
  start_single_qwen_services
  start_stage2_gcr_training
  log "Operation 1 completed."
}

main() {
  if [[ "${ENABLE_DEADLINE}" == "1" ]]; then
    log "Watcher started. deadline=${DEADLINE_LOCAL} ${DEADLINE_TZ}. DRY_RUN=${DRY_RUN}. TRAIN_GPU=${TRAIN_GPU:-<cpu-only>}"
  else
    log "Watcher started. deadline=disabled. DRY_RUN=${DRY_RUN}. TRAIN_GPU=${TRAIN_GPU:-<cpu-only>}"
  fi
  current_status_snapshot

  while true; do
    local now train_pid
    now=$(date +%s)
    train_pid=$(current_mbpp_train_pid)

    if [[ "${ENABLE_DEADLINE}" == "1" ]] && (( now >= DEADLINE_EPOCH )); then
      log "Deadline reached before observing MBPP train exit. Forcing operation 1 now."
      execute_operation_1
      return 0
    fi

    if [[ -z "${train_pid}" ]] || ! pid_alive "${train_pid}"; then
      if current_mbpp_completed_marker; then
        log "Observed current MBPP v4.4 training completed/exited. Starting operation 1."
      else
        log "Observed current MBPP v4.4 training process missing/exited without completion marker. Starting operation 1 anyway."
      fi
      execute_operation_1
      return 0
    fi

    current_status_snapshot
    sleep "${POLL_SECONDS}"
  done
}


main "$@"
