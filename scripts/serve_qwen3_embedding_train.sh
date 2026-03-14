#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/mnt/nvme/code/PettingLLMs/pettingllms_venv/bin/python}
MODEL_PATH=${MODEL_PATH:-/mnt/nvme/Qwen3-Embedding-8B}
PORT=${PORT:-8041}
HOST=${HOST:-0.0.0.0}
TP_SIZE=${TP_SIZE:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name qwen3-embedding-8b-train \
  --dtype auto \
  --port "${PORT}" \
  --host "${HOST}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --disable-log-requests
