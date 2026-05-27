#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/nvme/projects/R-HAN"
EXP_ROOT="${REPO_ROOT}/outputs/stage2_final_design_union_top2_operator_ablation_8ds_matched_3x_20260514_112258"
DATA_ROOT="${EXP_ROOT}/data"
STAGE1_CKPT_ROOT="${EXP_ROOT}/stage1_checkpoints"
LOG_PATH="${EXP_ROOT}/logs/launch.log"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/stage2_final_design:${REPO_ROOT}/stage2_final_design/Stage2-GCR+:${PYTHONPATH:-}"
export EMBED_API_BASE="http://127.0.0.1:8018"
export EMBED_MODEL="/mnt/nvme/Qwen3-Embedding-8B"
export EMBED_DIM="256"
export LLM_TIMEOUT_S="120"
export LLM_MAX_RETRIES="2"

mkdir -p "$(dirname "${LOG_PATH}")"

cd "${REPO_ROOT}"
exec /home/maozhifang/miniconda3/bin/python \
  stage2_final_design/Stage2-GCR+/run_stage2_gcr_plus_parallel.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${EXP_ROOT}" \
  --dataset mbpp \
  --dataset humaneval \
  --dataset math_level5 \
  --dataset math_full_math_ops \
  --dataset mmlu_pro \
  --dataset mmlu_as_mmlu_pro_ops \
  --dataset knowledge_crosswords \
  --dataset nlgraph \
  --workers 3 \
  --mode 3x \
  --x3-backend b1=http://127.0.0.1:8041@1 \
  --x3-backend b2=http://127.0.0.1:8042@1 \
  --x3-backend b3=http://127.0.0.1:8043@1 \
  --router-port 8060 \
  --router-timeout-s 180 \
  --python-bin /home/maozhifang/miniconda3/bin/python \
  --train-script train_mas_stage2_v4_4_target_suite.py \
  --cwd "${REPO_ROOT}/stage2_final_design" \
  "--train-arg=--seed 7" \
  "--train-arg=--search-iterations 8" \
  "--train-arg=--candidate-core-k 4" \
  "--train-arg=--candidate-explore-k 3" \
  "--train-arg=--candidate-max-k 8" \
  "--train-arg=--tier1-repeats 1" \
  "--train-arg=--tier2-repeats 1" \
  "--train-arg=--stage1-final-graph-mode union" \
  "--train-arg=--selected-topology-k 2" \
  "--train-arg=--stage2-turn-count 5" \
  "--train-arg=--memory-top-k 4" \
  "--train-arg=--soft-prune-top-k 3" \
  "--train-arg=--soft-prune-threshold 0.38" \
  "--train-arg=--hard-prune-after-turn 4" \
  "--train-arg=--periodic-every 1000000" \
  "--train-arg=--periodic-size 0" \
  "--train-arg=--max-train -1" \
  "--train-arg=--max-validation 0" \
  "--train-arg=--max-test -1" \
  "--train-arg=--stage1-checkpoint-root ${STAGE1_CKPT_ROOT}" \
  "--train-arg=--tier1-max-tokens 256" \
  "--train-arg=--tier2-max-tokens 896" \
  "--train-arg=--checkpoint-every 10" \
  "--train-arg=--resume" \
  2>&1 | tee "${LOG_PATH}"
