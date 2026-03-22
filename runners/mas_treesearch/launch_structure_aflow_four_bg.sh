#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/dataset/mas_treesearch_aflow_four_20260318}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/mas_treesearch_aflow_four_bg_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/mas_treesearch_aflow_four_bg_${RUN_STAMP}}"

export LLM_API_BASE="${LLM_API_BASE:-http://127.0.0.1:8043}"
export LLM_MODEL="${LLM_MODEL:-qwen3-8b-train}"
export LLM_JUDGE_API_BASE="${LLM_JUDGE_API_BASE:-http://127.0.0.1:8045}"
export LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-qwen3-32b-judge}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://127.0.0.1:8018}"
export EMBED_MODEL="${EMBED_MODEL:-/mnt/nvme/Qwen3-Embedding-8B}"
SKIP_HEALTHCHECK="${SKIP_HEALTHCHECK:-0}"

CODE_LLM_MAX_TOKENS="${CODE_LLM_MAX_TOKENS:-1024}"
CODE_JUDGE_MAX_TOKENS="${CODE_JUDGE_MAX_TOKENS:-224}"
CODE_TIER1_MAX_TOKENS="${CODE_TIER1_MAX_TOKENS:-256}"
CODE_TIER2_MAX_TOKENS="${CODE_TIER2_MAX_TOKENS:-896}"

NUMERIC_LLM_MAX_TOKENS="${NUMERIC_LLM_MAX_TOKENS:-768}"
NUMERIC_JUDGE_MAX_TOKENS="${NUMERIC_JUDGE_MAX_TOKENS:-192}"
NUMERIC_TIER1_MAX_TOKENS="${NUMERIC_TIER1_MAX_TOKENS:-224}"
NUMERIC_TIER2_MAX_TOKENS="${NUMERIC_TIER2_MAX_TOKENS:-640}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=(humaneval mbpp gsm8k math)
fi

if [ "${SKIP_HEALTHCHECK}" != "1" ]; then
  echo "[healthcheck] chat=${LLM_API_BASE} judge=${LLM_JUDGE_API_BASE} embed=${EMBED_API_BASE}"
  curl -sf "${LLM_API_BASE}/v1/models" >/dev/null
  curl -sf "${LLM_JUDGE_API_BASE}/v1/models" >/dev/null
  curl -sf "${EMBED_API_BASE}/v1/models" >/dev/null
else
  echo "[healthcheck] skipped"
fi

echo "[prepare] ${DATA_ROOT}"
"${PYTHON_BIN}" "$ROOT/prepare_mas_treesearch_aflow_suite.py" --output-root "${DATA_ROOT}"

MANIFEST_PATH="${RUN_ROOT}/launch_manifest.json"
{
  echo "{"
  echo "  \"run_stamp\": \"${RUN_STAMP}\","
  echo "  \"data_root\": \"${DATA_ROOT}\","
  echo "  \"run_root\": \"${RUN_ROOT}\","
  echo "  \"log_root\": \"${LOG_ROOT}\","
  echo "  \"datasets\": ["
  for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    suffix=","
    if [ "$i" -eq "$(( ${#DATASETS[@]} - 1 ))" ]; then
      suffix=""
    fi
    echo "    \"${dataset}\"${suffix}"
  done
  echo "  ]"
  echo "}"
} > "${MANIFEST_PATH}"

for dataset in "${DATASETS[@]}"; do
  DATASET_OUTPUT_ROOT="${RUN_ROOT}/${dataset}"
  DATASET_LOG_PATH="${LOG_ROOT}/${dataset}.log"
  mkdir -p "${DATASET_OUTPUT_ROOT}"
  case "${dataset}" in
    humaneval|mbpp)
      export LLM_MAX_TOKENS="${CODE_LLM_MAX_TOKENS}"
      export LLM_JUDGE_MAX_TOKENS="${CODE_JUDGE_MAX_TOKENS}"
      TIER1_MAX_TOKENS="${CODE_TIER1_MAX_TOKENS}"
      TIER2_MAX_TOKENS="${CODE_TIER2_MAX_TOKENS}"
      ;;
    gsm8k|math)
      export LLM_MAX_TOKENS="${NUMERIC_LLM_MAX_TOKENS}"
      export LLM_JUDGE_MAX_TOKENS="${NUMERIC_JUDGE_MAX_TOKENS}"
      TIER1_MAX_TOKENS="${NUMERIC_TIER1_MAX_TOKENS}"
      TIER2_MAX_TOKENS="${NUMERIC_TIER2_MAX_TOKENS}"
      ;;
    *)
      export LLM_MAX_TOKENS="${NUMERIC_LLM_MAX_TOKENS}"
      export LLM_JUDGE_MAX_TOKENS="${NUMERIC_JUDGE_MAX_TOKENS}"
      TIER1_MAX_TOKENS="${NUMERIC_TIER1_MAX_TOKENS}"
      TIER2_MAX_TOKENS="${NUMERIC_TIER2_MAX_TOKENS}"
      ;;
  esac
  {
    echo "[start][${dataset}] $(date --iso-8601=seconds)"
    echo "[paths][${dataset}] output=${DATASET_OUTPUT_ROOT} log=${DATASET_LOG_PATH}"
    echo "[models][${dataset}] chat=${LLM_API_BASE} judge=${LLM_JUDGE_API_BASE} embed=${EMBED_API_BASE}"
    echo "[tokens][${dataset}] llm_max=${LLM_MAX_TOKENS} judge_max=${LLM_JUDGE_MAX_TOKENS} tier1_max=${TIER1_MAX_TOKENS} tier2_max=${TIER2_MAX_TOKENS}"
  } | tee -a "${DATASET_LOG_PATH}"

  "${PYTHON_BIN}" "$ROOT/train_mas_treesearch_target_suite.py" \
    --data-root "${DATA_ROOT}" \
    --output-root "${DATASET_OUTPUT_ROOT}" \
    --dataset "${dataset}" \
    --plan-preset aflow_four \
    --pipeline-mode structure_only \
    --search-iterations 8 \
    --candidate-core-k 4 \
    --candidate-explore-k 3 \
    --candidate-max-k 8 \
    --tier1-max-tokens "${TIER1_MAX_TOKENS}" \
    --tier2-max-tokens "${TIER2_MAX_TOKENS}" \
    --tier1-repeats 1 \
    --tier2-repeats 1 \
    --checkpoint-every 25 \
    --resume \
    >> "${DATASET_LOG_PATH}" 2>&1

  echo "[done][${dataset}] $(date --iso-8601=seconds)" | tee -a "${DATASET_LOG_PATH}"
done
