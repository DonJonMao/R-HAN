#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/mnt/nvme/code/PettingLLMs/pettingllms_venv/bin/python}
MODEL_PATH=${MODEL_PATH:-/mnt/nvme/Qwen3-8B}
PORT=${PORT:-8043}
HOST=${HOST:-0.0.0.0}
TP_SIZE=${TP_SIZE:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.88}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-32768}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,7}

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name qwen3-8b-train \
  --port "${PORT}" \
  --host "${HOST}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --enable-chunked-prefill \
  --disable-log-requests
