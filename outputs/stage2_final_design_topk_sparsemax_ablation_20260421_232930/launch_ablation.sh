#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/mnt/nvme/projects/R-HAN/outputs/stage2_final_design_topk_sparsemax_ablation_20260421_232930"
PYTHON_BIN="/home/maozhifang/miniconda3/bin/python"
DATA_ROOT="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_aflow_four_20260318"
STAGE1_ROOT="/mnt/nvme/projects/R-HAN/outputs/stage1_selected_ckpts_v2_20260330"
BACKENDS=(
  "b1=http://127.0.0.1:8041@1"
  "b2=http://127.0.0.1:8042@1"
  "b3=http://127.0.0.1:8043@1"
)
COMMON_TRAIN_ARGS=(
  "--resume"
  "--seed 7"
  "--search-iterations 8"
  "--candidate-core-k 4"
  "--candidate-explore-k 3"
  "--candidate-max-k 8"
  "--tier1-repeats 1"
  "--tier2-repeats 1"
  "--stage2-turn-count 5"
  "--memory-top-k 4"
  "--soft-prune-top-k 3"
  "--soft-prune-threshold 0.38"
  "--hard-prune-after-turn 4"
  "--periodic-every 1000000"
  "--periodic-size 0"
  "--max-train -1"
  "--max-validation 0"
  "--max-test 0"
  "--stage1-checkpoint-root ${STAGE1_ROOT}"
  "--tier1-max-tokens 256"
  "--tier2-max-tokens 896"
)

run_group() {
  local name="$1"
  local source_root="$2"
  local out_root="$3"
  local router_port="$4"
  echo "[group-start] ${name} $(date -Iseconds) source=${source_root} output=${out_root}"
  mkdir -p "${out_root}"
  local cmd=(
    "${PYTHON_BIN}"
    "${source_root}/Stage2-GCR+/run_stage2_gcr_plus_parallel.py"
    "--data-root" "${DATA_ROOT}"
    "--output-root" "${out_root}"
    "--dataset" "mbpp"
    "--dataset" "humaneval"
    "--workers" "3"
    "--mode" "3x"
    "--x3-backend" "${BACKENDS[0]}"
    "--x3-backend" "${BACKENDS[1]}"
    "--x3-backend" "${BACKENDS[2]}"
    "--router-port" "${router_port}"
    "--python-bin" "${PYTHON_BIN}"
    "--train-script" "train_mas_stage2_v4_4_target_suite.py"
    "--cwd" "${source_root}"
  )
  for arg in "${COMMON_TRAIN_ARGS[@]}"; do
    cmd+=("--train-arg" "${arg}")
  done
  printf '[group-cmd] ' && printf '%q ' "${cmd[@]}" && printf '\n'
  "${cmd[@]}"
  echo "[group-done] ${name} $(date -Iseconds)"
}

run_group "topk_baseline" "${RUN_ROOT}/code/topk_source" "${RUN_ROOT}/topk_mbpp_humaneval_3x" "8101"
run_group "sparsemax_only" "${RUN_ROOT}/code/sparsemax_source" "${RUN_ROOT}/sparsemax_mbpp_humaneval_3x" "8102"

echo "[all-done] $(date -Iseconds)"
