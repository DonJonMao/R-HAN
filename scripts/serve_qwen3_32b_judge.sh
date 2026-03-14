#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/mnt/nvme/code/PettingLLMs/pettingllms_venv/bin/python}
MODEL_PATH=${MODEL_PATH:-/mnt/nvme/Qwen3-32B}
PORT=${PORT:-8045}
HOST=${HOST:-0.0.0.0}
TP_SIZE=${TP_SIZE:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-16384}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2}

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name qwen3-32b-judge \
  --port "${PORT}" \
  --host "${HOST}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --enable-chunked-prefill \
  --disable-log-requests
