#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/nvme/projects/R-HAN"
EXP_ROOT="${REPO_ROOT}/outputs/stage2_final_design_union_top2_operator_ablation_8ds_matched_3x_20260514_112258"
LOG_PATH="${EXP_ROOT}/logs/worker_2_resume_20260515_182338.log"
PYTHON_BIN="/home/maozhifang/miniconda3/bin/python"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/stage2_final_design:${REPO_ROOT}/stage2_final_design/Stage2-GCR+:${PYTHONPATH:-}"
export EMBED_API_BASE="http://127.0.0.1:8018"
export EMBED_MODEL="/mnt/nvme/Qwen3-Embedding-8B"
export EMBED_DIM="256"
export LLM_TIMEOUT_S="120"
export LLM_MAX_RETRIES="2"
export LLM_400_FALLBACK_MESSAGE_CHARS="24000,16000,12000,8000"

mkdir -p "$(dirname "${LOG_PATH}")"

cd "${REPO_ROOT}/stage2_final_design"
exec "${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${LOG_PATH}"
import os
import subprocess
import sys
import time

from stage2_gcr_plus.orchestration.proxy import RouterProxyServer
from stage2_gcr_plus.orchestration.routing import parse_backend_specs

repo_root = "/mnt/nvme/projects/R-HAN"
exp_root = f"{repo_root}/outputs/stage2_final_design_union_top2_operator_ablation_8ds_matched_3x_20260514_112258"
python_bin = "/home/maozhifang/miniconda3/bin/python"
router_port = 8061

backend_specs = parse_backend_specs(
    [
        "b1=http://127.0.0.1:8041@1",
        "b2=http://127.0.0.1:8042@1",
        "b3=http://127.0.0.1:8043@1",
    ]
)
proxy = RouterProxyServer(
    backends=backend_specs,
    host="127.0.0.1",
    port=router_port,
    timeout_s=180,
    health_interval_s=15,
)

cmd = [
    python_bin,
    "train_mas_stage2_v4_4_target_suite.py",
    "--data-root",
    f"{exp_root}/sample_shards/shard_2",
    "--output-root",
    f"{exp_root}/workers/worker_2",
    "--dataset",
    "mbpp",
    "--dataset",
    "humaneval",
    "--dataset",
    "math_level5",
    "--dataset",
    "math_full_math_ops",
    "--dataset",
    "mmlu_pro",
    "--dataset",
    "mmlu_as_mmlu_pro_ops",
    "--dataset",
    "knowledge_crosswords",
    "--dataset",
    "nlgraph",
    "--seed",
    "7",
    "--search-iterations",
    "8",
    "--candidate-core-k",
    "4",
    "--candidate-explore-k",
    "3",
    "--candidate-max-k",
    "8",
    "--tier1-repeats",
    "1",
    "--tier2-repeats",
    "1",
    "--stage1-final-graph-mode",
    "union",
    "--selected-topology-k",
    "2",
    "--stage2-turn-count",
    "5",
    "--memory-top-k",
    "4",
    "--soft-prune-top-k",
    "3",
    "--soft-prune-threshold",
    "0.38",
    "--hard-prune-after-turn",
    "4",
    "--periodic-every",
    "1000000",
    "--periodic-size",
    "0",
    "--max-train",
    "-1",
    "--max-validation",
    "0",
    "--max-test",
    "-1",
    "--stage1-checkpoint-root",
    f"{exp_root}/stage1_checkpoints",
    "--tier1-max-tokens",
    "256",
    "--tier2-max-tokens",
    "896",
    "--checkpoint-every",
    "10",
    "--resume",
]

env = dict(os.environ)
env["LLM_API_BASE"] = f"http://127.0.0.1:{router_port}"
env["OPENAI_API_BASE"] = env["LLM_API_BASE"]
env["STAGE2_CHAT_API_BASE"] = env["LLM_API_BASE"]
env["STAGE2_JUDGE_API_BASE"] = env["LLM_API_BASE"]

print(f"[worker2-resume-start] {time.strftime('%Y-%m-%d %H:%M:%S')} router={env['LLM_API_BASE']}", flush=True)
print("[worker2-resume-cmd] " + " ".join(cmd), flush=True)
proxy.start()
try:
    result = subprocess.run(
        cmd,
        cwd=f"{repo_root}/stage2_final_design",
        env=env,
        text=True,
        check=False,
    )
finally:
    proxy.close()
print(f"[worker2-resume-exit] returncode={result.returncode}", flush=True)
sys.exit(result.returncode)
PY
