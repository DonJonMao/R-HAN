#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-$ROOT/dataset/mas_treesearch_threeway_general_20260319}"
DATA_ROOT="${DATA_ROOT:-$ROOT/dataset/mas_treesearch_threeway_general_trainview_20260319}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/mas_treesearch_general_threeway_bg_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/mas_treesearch_general_threeway_bg_${RUN_STAMP}}"

export LLM_API_BASE="${LLM_API_BASE:-http://127.0.0.1:8043}"
export LLM_MODEL="${LLM_MODEL:-qwen3-8b-train}"
export LLM_JUDGE_API_BASE="${LLM_JUDGE_API_BASE:-http://127.0.0.1:8045}"
export LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-qwen3-32b-judge}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://127.0.0.1:8018}"
export EMBED_MODEL="${EMBED_MODEL:-/mnt/nvme/Qwen3-Embedding-8B}"
SKIP_HEALTHCHECK="${SKIP_HEALTHCHECK:-0}"

GENERAL_LLM_MAX_TOKENS="${GENERAL_LLM_MAX_TOKENS:-1024}"
GENERAL_JUDGE_MAX_TOKENS="${GENERAL_JUDGE_MAX_TOKENS:-224}"
GENERAL_TIER1_MAX_TOKENS="${GENERAL_TIER1_MAX_TOKENS:-256}"
GENERAL_TIER2_MAX_TOKENS="${GENERAL_TIER2_MAX_TOKENS:-896}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=(mmlu_pro nlgraph knowledge_crosswords)
fi

if [ "${SKIP_HEALTHCHECK}" != "1" ]; then
  echo "[healthcheck] chat=${LLM_API_BASE} judge=${LLM_JUDGE_API_BASE} embed=${EMBED_API_BASE}"
  python - <<'PY'
import urllib.request
for url in (
    "http://127.0.0.1:8043/v1/models",
    "http://127.0.0.1:8045/v1/models",
    "http://127.0.0.1:8018/v1/models",
):
    with urllib.request.urlopen(url, timeout=10) as resp:
        if resp.status != 200:
            raise RuntimeError(f"healthcheck failed for {url}: {resp.status}")
PY
else
  echo "[healthcheck] skipped"
fi

echo "[prepare-source] ${SOURCE_DATA_ROOT}"
echo "[prepare-trainview] ${DATA_ROOT}"
"${PYTHON_BIN}" "$ROOT/prepare_mas_treesearch_threeway_general_trainview.py" \
  --source-root "${SOURCE_DATA_ROOT}" \
  --output-root "${DATA_ROOT}"

MANIFEST_PATH="${RUN_ROOT}/launch_manifest.json"
{
  echo "{"
  echo "  \"run_stamp\": \"${RUN_STAMP}\","
  echo "  \"source_data_root\": \"${SOURCE_DATA_ROOT}\","
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

  export LLM_MAX_TOKENS="${GENERAL_LLM_MAX_TOKENS}"
  export LLM_JUDGE_MAX_TOKENS="${GENERAL_JUDGE_MAX_TOKENS}"
  TIER1_MAX_TOKENS="${GENERAL_TIER1_MAX_TOKENS}"
  TIER2_MAX_TOKENS="${GENERAL_TIER2_MAX_TOKENS}"

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
    --plan-preset general_threeway \
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
