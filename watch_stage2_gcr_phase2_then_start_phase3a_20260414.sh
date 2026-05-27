#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/nvme/projects/R-HAN"
LOG_DIR="$REPO_ROOT/logs"
OUTPUTS_DIR="$REPO_ROOT/outputs"
CURRENT_LAUNCHER_PID="125955"
CURRENT_WRAPPER_PID="125958"
CURRENT_RUN_ROOT="$OUTPUTS_DIR/stage2_gcr_plus_mbpp_3x_manual_20260414_174024"
DEADLINE_LOCAL="2026-04-15 01:00:00"
DEADLINE_EPOCH="$(date -d "$DEADLINE_LOCAL CST" +%s)"
POLL_INTERVAL_S=30

mkdir -p "$LOG_DIR" "$OUTPUTS_DIR"

SCRIPT_LOG="$LOG_DIR/watch_stage2_gcr_phase2_then_start_phase3a_20260414.log"

timestamp() {
  date '+%F %T %Z'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$SCRIPT_LOG"
}

process_alive() {
  local pid="$1"
  if [[ -z "$pid" || "$pid" == "0" ]]; then
    return 1
  fi
  kill -0 "$pid" >/dev/null 2>&1
}

current_training_alive() {
  process_alive "$CURRENT_WRAPPER_PID"
}

kill_current_training() {
  log "deadline reached; stopping current phase2 training rooted at $CURRENT_RUN_ROOT"
  local child_pids=""
  child_pids="$(pgrep -P "$CURRENT_WRAPPER_PID" || true)"
  if [[ -n "$child_pids" ]]; then
    log "sending TERM to current worker children: $child_pids"
    kill $child_pids >/dev/null 2>&1 || true
  fi
  if process_alive "$CURRENT_WRAPPER_PID"; then
    log "sending TERM to current wrapper pid $CURRENT_WRAPPER_PID"
    kill "$CURRENT_WRAPPER_PID" >/dev/null 2>&1 || true
  fi
  if process_alive "$CURRENT_LAUNCHER_PID"; then
    log "sending TERM to current launcher pid $CURRENT_LAUNCHER_PID"
    kill "$CURRENT_LAUNCHER_PID" >/dev/null 2>&1 || true
  fi
  sleep 8
  child_pids="$(pgrep -P "$CURRENT_WRAPPER_PID" || true)"
  if [[ -n "$child_pids" ]]; then
    log "forcing KILL on remaining worker children: $child_pids"
    kill -9 $child_pids >/dev/null 2>&1 || true
  fi
  if process_alive "$CURRENT_WRAPPER_PID"; then
    log "forcing KILL on current wrapper pid $CURRENT_WRAPPER_PID"
    kill -9 "$CURRENT_WRAPPER_PID" >/dev/null 2>&1 || true
  fi
  if process_alive "$CURRENT_LAUNCHER_PID"; then
    log "forcing KILL on current launcher pid $CURRENT_LAUNCHER_PID"
    kill -9 "$CURRENT_LAUNCHER_PID" >/dev/null 2>&1 || true
  fi
}

pick_router_port() {
  local candidate
  for candidate in 8091 8092 8093 8094; do
    if ! ss -tnlp 2>/dev/null | grep -q ":${candidate} "; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "8095"
}

start_phase3a() {
  local run_stamp run_name output_root master_log launch_script pid_file router_port cmd_string
  run_stamp="$(date +%Y%m%d_%H%M%S)"
  run_name="stage2_gcr_plus_phase3a_mbpp_3x_manual_${run_stamp}"
  output_root="$OUTPUTS_DIR/$run_name"
  master_log="$LOG_DIR/${run_name}.log"
  launch_script="$LOG_DIR/${run_name}.launch.sh"
  pid_file="$LOG_DIR/${run_name}.pid"
  router_port="$(pick_router_port)"

  mkdir -p "$output_root"

  cmd_string="cd $REPO_ROOT && mkdir -p $LOG_DIR $output_root && touch $master_log && nohup env CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 PYTHONPATH=$REPO_ROOT:$REPO_ROOT/Stage2-GCR+ LLM_JUDGE_API_BASE=http://127.0.0.1:8045 EMBED_API_BASE=http://127.0.0.1:8018 /home/maozhifang/miniconda3/bin/python $REPO_ROOT/Stage2-GCR+/run_stage2_gcr_plus_parallel.py --data-root $REPO_ROOT/dataset/mas_treesearch_aflow_four_20260318 --output-root $output_root --dataset mbpp --workers 3 --mode 3x --x3-backend b1=http://127.0.0.1:8041@1 --x3-backend b2=http://127.0.0.1:8042@1 --x3-backend b3=http://127.0.0.1:8043@1 --router-host 127.0.0.1 --router-port $router_port --train-script train_mas_stage2_v4_4_target_suite.py --train-arg=--resume --train-arg='--seed 7' --train-arg='--search-iterations 8' --train-arg='--candidate-core-k 4' --train-arg='--candidate-explore-k 3' --train-arg='--candidate-max-k 8' --train-arg='--tier1-repeats 1' --train-arg='--tier2-repeats 1' --train-arg='--stage2-turn-count 5' --train-arg='--memory-top-k 4' --train-arg='--soft-prune-top-k 3' --train-arg='--soft-prune-threshold 0.38' --train-arg='--hard-prune-after-turn 4' --train-arg='--periodic-every 1000000' --train-arg='--periodic-size 0' --train-arg='--max-train -1' --train-arg='--max-validation 0' --train-arg='--max-test -1' --train-arg='--stage1-checkpoint-root $REPO_ROOT/outputs/stage1_selected_ckpts_v2_20260330' --train-arg='--tier1-max-tokens 256' --train-arg='--tier2-max-tokens 896' --cwd $REPO_ROOT >> $master_log 2>&1 < /dev/null & PID=\$! && echo \"\$PID\" > $pid_file && echo \"\$PID\""

  cat > "$launch_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
$cmd_string
EOF
  chmod +x "$launch_script"

  log "starting phase3A training; output_root=$output_root router_port=$router_port"
  local new_pid
  new_pid="$(/bin/bash "$launch_script")"
  log "phase3A launcher pid=$new_pid"
  log "phase3A master log=$master_log"
}

main() {
  log "watcher started; monitoring current wrapper pid $CURRENT_WRAPPER_PID until $DEADLINE_LOCAL CST"
  while true; do
    if ! current_training_alive; then
      log "current phase2 training has ended before deadline"
      start_phase3a
      return 0
    fi
    if [[ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]]; then
      kill_current_training
      start_phase3a
      return 0
    fi
    sleep "$POLL_INTERVAL_S"
  done
}

main "$@"
