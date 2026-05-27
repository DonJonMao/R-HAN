#!/usr/bin/env bash
set -u

RUN_ROOT="/mnt/nvme/projects/R-HAN/outputs/stage2_final_design_bdc1ae0_rerun_nocache_20260425_084155"
SOURCE_ROOT="${RUN_ROOT}/code/bdc1ae0_source"
OUTPUT_ROOT="${RUN_ROOT}/bdc1ae0_rerun_nocache_mbpp_humaneval_3x"
BASELINE_ROOT="/mnt/nvme/projects/R-HAN/outputs/stage2_final_design_slot_mask_phase2_20260423_094157/slot_mask_phase2_mbpp_humaneval_3x"
PYTHON_BIN="/home/maozhifang/miniconda3/bin/python"
DATA_ROOT="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_aflow_four_20260318"
STAGE1_ROOT="/mnt/nvme/projects/R-HAN/outputs/stage1_selected_ckpts_v2_20260330"

export PYTHONUNBUFFERED=1

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "[launch-error] output_root already exists; refusing to reuse cached checkpoints: ${OUTPUT_ROOT}"
  exit 2
fi

cd "${SOURCE_ROOT}" || exit 1

echo "[launch] $(date -Iseconds) run_root=${RUN_ROOT}"
echo "[launch] source_root=${SOURCE_ROOT}"
echo "[launch] output_root=${OUTPUT_ROOT}"
echo "[launch] baseline_root=${BASELINE_ROOT}"
echo "[launch] commit=bdc1ae088e04269656bed748f617349955877df7"
echo "[launch] router_port=8118"
echo "[launch] cache_policy=no --resume, --disable-structure-cache, fresh output root"

set +e
"${PYTHON_BIN}" "${SOURCE_ROOT}/Stage2-GCR+/run_stage2_gcr_plus_parallel.py" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset mbpp \
  --dataset humaneval \
  --workers 3 \
  --mode 3x \
  --x3-backend "b1=http://127.0.0.1:8041@1" \
  --x3-backend "b2=http://127.0.0.1:8042@1" \
  --x3-backend "b3=http://127.0.0.1:8043@1" \
  --router-port 8118 \
  --python-bin "${PYTHON_BIN}" \
  --train-script train_mas_stage2_v4_4_target_suite.py \
  --cwd "${SOURCE_ROOT}" \
  "--train-arg=--seed 7" \
  "--train-arg=--search-iterations 8" \
  "--train-arg=--candidate-core-k 4" \
  "--train-arg=--candidate-explore-k 3" \
  "--train-arg=--candidate-max-k 8" \
  "--train-arg=--tier1-repeats 1" \
  "--train-arg=--tier2-repeats 1" \
  "--train-arg=--stage2-turn-count 5" \
  "--train-arg=--memory-top-k 4" \
  "--train-arg=--soft-prune-top-k 3" \
  "--train-arg=--soft-prune-threshold 0.38" \
  "--train-arg=--hard-prune-after-turn 4" \
  "--train-arg=--periodic-every 1000000" \
  "--train-arg=--periodic-size 0" \
  "--train-arg=--max-train -1" \
  "--train-arg=--max-validation 0" \
  "--train-arg=--max-test 0" \
  "--train-arg=--stage1-checkpoint-root ${STAGE1_ROOT}" \
  "--train-arg=--disable-structure-cache" \
  "--train-arg=--tier1-max-tokens 256" \
  "--train-arg=--tier2-max-tokens 896"
train_rc=$?
echo "[train-exit] $(date -Iseconds) returncode=${train_rc}"

"${PYTHON_BIN}" "${RUN_ROOT}/compare_stage2_runs.py" \
  --baseline "${BASELINE_ROOT}" \
  --candidate "${OUTPUT_ROOT}" \
  --output "${RUN_ROOT}/comparison.json" \
  --markdown "${RUN_ROOT}/comparison.md"
compare_rc=$?
echo "[compare-exit] $(date -Iseconds) returncode=${compare_rc}"

exit "${train_rc}"
