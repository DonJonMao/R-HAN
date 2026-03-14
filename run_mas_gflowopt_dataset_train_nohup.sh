#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/projects/R-HAN"
cd "$ROOT_DIR"

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$ROOT_DIR/logs/mas_gflowopt_dataset_train_full_run_${TS}.log"

CMD=(
  python demo_train_datasets.py
  --all-datasets
  --max-train-per-dataset 1000
  --test-ratio 0.1
  --agent-top-k 6
  --gflownet-train-epochs 1
  --gflownet-batch-size 1
  --num-sampled-dags 1
  --contribution-mode none
  --one-eval-per-trajectory
  --disable-refine
  --batch-eval
  --batch-eval-workers 12
  --batch-eval-size 24
  --eval-every 1000
  --periodic-test-size 100
  --embedding-api-base http://localhost:8018
  --embedding-model /mnt/nvme/Qwen3-Embedding-8B
)

{
  echo "[launcher] start_ts=${TS}"
  echo "[launcher] workdir=${ROOT_DIR}"
  printf '[launcher] command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
} >"$LOG_PATH"

nohup setsid env PYTHONUNBUFFERED=1 "${CMD[@]}" >>"$LOG_PATH" 2>&1 < /dev/null &
PID=$!

echo "pid=${PID}"
echo "log=${LOG_PATH}"
echo "tail_cmd=tail -f ${LOG_PATH}"
