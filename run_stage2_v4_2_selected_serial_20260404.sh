#!/bin/bash
set -euo pipefail

cd /mnt/nvme/projects/R-HAN

export LLM_API_BASE=http://127.0.0.1:8043
export LLM_JUDGE_API_BASE=http://127.0.0.1:8045
export EMBED_API_BASE=http://127.0.0.1:8018
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
PYTHON_BIN=${PYTHON_BIN:-/home/maozhifang/miniconda3/bin/python}

RUN_TAG=20260404
LOG_DIR=/mnt/nvme/projects/R-HAN/logs
OUTPUT_ROOT=/mnt/nvme/projects/R-HAN/outputs/mas_stage2_v4_2_selected_runs_${RUN_TAG}

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_ROOT}"

COMMON_ARGS=(
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

run_dataset() {
  local dataset_name="$1"
  local data_root="$2"
  local output_root="$3"
  local tier1_tokens="$4"
  local tier2_tokens="$5"
  local dataset_log="$LOG_DIR/stage2_v4_2_${dataset_name}_${RUN_TAG}.log"

  : > "${dataset_log}"
  echo "[start][${dataset_name}] $(date -Iseconds) output_root=${output_root}" | tee -a "${dataset_log}"
  echo "[config][${dataset_name}] tier1_tokens=${tier1_tokens} tier2_tokens=${tier2_tokens}" | tee -a "${dataset_log}"

  "${PYTHON_BIN}" train_mas_stage2_v4_2_target_suite.py \
    --data-root "${data_root}" \
    --output-root "${output_root}" \
    --dataset "${dataset_name}" \
    --tier1-max-tokens "${tier1_tokens}" \
    --tier2-max-tokens "${tier2_tokens}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "${dataset_log}"

  echo "[done][${dataset_name}] $(date -Iseconds)" | tee -a "${dataset_log}"
}

echo "[runner-start] $(date -Iseconds) output_root=${OUTPUT_ROOT}"
echo "[runner-config] ${COMMON_ARGS[*]}"
echo "[runner-scope] gsm8k only"

run_dataset \
  gsm8k \
  /mnt/nvme/projects/R-HAN/dataset/mas_treesearch_gsm8k_official_test_20_80_20260323 \
  "${OUTPUT_ROOT}/gsm8k" \
  192 \
  512

echo "[runner-done] $(date -Iseconds)"
