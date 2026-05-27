#!/bin/bash
set -euo pipefail

cd /mnt/nvme/projects/R-HAN

export LLM_API_BASE=http://127.0.0.1:8043
export LLM_JUDGE_API_BASE=http://127.0.0.1:8045
export EMBED_API_BASE=http://127.0.0.1:8018
export PYTHONPATH=.

COMMON_ARGS=(
  --resume
  --seed 7
  --search-iterations 8
  --candidate-core-k 4
  --candidate-explore-k 3
  --candidate-max-k 8
  --tier1-repeats 1
  --tier2-repeats 1
  --stage2-version v2
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

python train_mas_stage2_target_suite.py \
  --data-root /mnt/nvme/projects/R-HAN/dataset/mas_treesearch_gsm8k_official_test_20_80_20260323 \
  --output-root /mnt/nvme/projects/R-HAN/outputs/mas_stage2_v2_selected_runs_20260330/gsm8k \
  --dataset gsm8k \
  --tier1-max-tokens 192 \
  --tier2-max-tokens 512 \
  "${COMMON_ARGS[@]}" \
  >> /mnt/nvme/projects/R-HAN/logs/GSM8K二阶段V2.log 2>&1

python train_mas_stage2_target_suite.py \
  --data-root /mnt/nvme/projects/R-HAN/dataset/mas_treesearch_aflow_four_20260318 \
  --output-root /mnt/nvme/projects/R-HAN/outputs/mas_stage2_v2_selected_runs_20260330/humaneval \
  --dataset humaneval \
  --tier1-max-tokens 256 \
  --tier2-max-tokens 896 \
  "${COMMON_ARGS[@]}" \
  >> /mnt/nvme/projects/R-HAN/logs/Humaneval二阶段V2.log 2>&1

python train_mas_stage2_target_suite.py \
  --data-root /mnt/nvme/projects/R-HAN/dataset/mas_treesearch_aflow_four_20260318 \
  --output-root /mnt/nvme/projects/R-HAN/outputs/mas_stage2_v2_selected_runs_20260330/mbpp \
  --dataset mbpp \
  --tier1-max-tokens 256 \
  --tier2-max-tokens 896 \
  "${COMMON_ARGS[@]}" \
  >> /mnt/nvme/projects/R-HAN/logs/MBPP二阶段V2.log 2>&1
