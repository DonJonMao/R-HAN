#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/dataset/mas_treesearch_aflow_four_20260318}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/mas_treesearch_aflow_four_structure_$(date +%Y%m%d_%H%M%S)}"

export LLM_API_BASE="${LLM_API_BASE:-http://127.0.0.1:8043}"
export LLM_MODEL="${LLM_MODEL:-qwen3-8b-train}"
export LLM_JUDGE_API_BASE="${LLM_JUDGE_API_BASE:-http://127.0.0.1:8045}"
export LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-qwen3-32b-judge}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://127.0.0.1:8018}"
export EMBED_MODEL="${EMBED_MODEL:-/mnt/nvme/Qwen3-Embedding-8B}"

echo "[healthcheck] chat=${LLM_API_BASE} judge=${LLM_JUDGE_API_BASE} embed=${EMBED_API_BASE}"
curl -sf "${LLM_API_BASE}/v1/models" >/dev/null
curl -sf "${LLM_JUDGE_API_BASE}/v1/models" >/dev/null
curl -sf "${EMBED_API_BASE}/v1/models" >/dev/null

echo "[prepare] ${DATA_ROOT}"
"${PYTHON_BIN}" "$ROOT/prepare_mas_treesearch_aflow_suite.py" --output-root "${DATA_ROOT}"

echo "[train] output=${OUTPUT_ROOT}"
"${PYTHON_BIN}" "$ROOT/train_mas_treesearch_target_suite.py" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset humaneval \
  --dataset mbpp \
  --dataset gsm8k \
  --dataset math \
  --plan-preset aflow_four \
  --pipeline-mode structure_only \
  --search-iterations 8 \
  --candidate-core-k 4 \
  --candidate-explore-k 3 \
  --candidate-max-k 8 \
  --tier1-max-tokens 192 \
  --tier2-max-tokens 512 \
  --tier1-repeats 1 \
  --tier2-repeats 1 \
  --checkpoint-every 25 \
  --resume \
  "$@"
