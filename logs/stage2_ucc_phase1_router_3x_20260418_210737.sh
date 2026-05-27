#!/usr/bin/env bash
set -euo pipefail

export TZ=Asia/Shanghai

REPO_ROOT="/mnt/nvme/projects/R-HAN"
LOG_DIR="${REPO_ROOT}/logs"
PYTHON_BIN="/home/maozhifang/miniconda3/bin/python"
ROUTER_SCRIPT="${REPO_ROOT}/Stage2-GCR+/run_stage2_gcr_plus_router.py"
RUN_TAG="20260418_210737"
ROUTER_LOG="${LOG_DIR}/stage2_ucc_phase1_router_3x_${RUN_TAG}.log"
ROUTER_PID="${LOG_DIR}/stage2_ucc_phase1_router_3x_${RUN_TAG}.pid"

mkdir -p "${LOG_DIR}"
echo "$$" > "${ROUTER_PID}"
exec >>"${ROUTER_LOG}" 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

if curl -sf "http://127.0.0.1:8039/v1/models" >/dev/null 2>&1; then
  log "router already healthy on 127.0.0.1:8039; no-op"
  exit 0
fi

log "starting router on 127.0.0.1:8039"
exec env \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Stage2-GCR+" \
  "${PYTHON_BIN}" "${ROUTER_SCRIPT}" \
  --backend gpu0=http://127.0.0.1:8041@1 \
  --backend gpu1=http://127.0.0.1:8042@1 \
  --backend gpu2=http://127.0.0.1:8043@1 \
  --host 127.0.0.1 \
  --port 8039
