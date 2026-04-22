from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence


RUN_ROOT = Path("/mnt/nvme/projects/R-HAN/outputs/stage2_final_design_topk_sparsemax_ablation_20260421_232930")
PYTHON_BIN = "/home/maozhifang/miniconda3/bin/python"
DATA_ROOT = "/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_aflow_four_20260318"
STAGE1_ROOT = "/mnt/nvme/projects/R-HAN/outputs/stage1_selected_ckpts_v2_20260330"
BACKENDS = [
    "b1=http://127.0.0.1:8041@1",
    "b2=http://127.0.0.1:8042@1",
    "b3=http://127.0.0.1:8043@1",
]
COMMON_TRAIN_ARGS = [
    "--resume",
    "--seed 7",
    "--search-iterations 8",
    "--candidate-core-k 4",
    "--candidate-explore-k 3",
    "--candidate-max-k 8",
    "--tier1-repeats 1",
    "--tier2-repeats 1",
    "--stage2-turn-count 5",
    "--memory-top-k 4",
    "--soft-prune-top-k 3",
    "--soft-prune-threshold 0.38",
    "--hard-prune-after-turn 4",
    "--periodic-every 1000000",
    "--periodic-size 0",
    "--max-train -1",
    "--max-validation 0",
    "--max-test 0",
    f"--stage1-checkpoint-root {STAGE1_ROOT}",
    "--tier1-max-tokens 256",
    "--tier2-max-tokens 896",
]


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def runner_command(source_root: Path, output_root: Path, router_port: int) -> list[str]:
    command = [
        PYTHON_BIN,
        str(source_root / "Stage2-GCR+" / "run_stage2_gcr_plus_parallel.py"),
        "--data-root",
        DATA_ROOT,
        "--output-root",
        str(output_root),
        "--dataset",
        "mbpp",
        "--dataset",
        "humaneval",
        "--workers",
        "3",
        "--mode",
        "3x",
        "--router-port",
        str(router_port),
        "--python-bin",
        PYTHON_BIN,
        "--train-script",
        "train_mas_stage2_v4_4_target_suite.py",
        "--cwd",
        str(source_root),
    ]
    for backend in BACKENDS:
        command.extend(["--x3-backend", backend])
    for arg in COMMON_TRAIN_ARGS:
        command.append(f"--train-arg={arg}")
    return command


def complete(output_root: Path) -> bool:
    summary_path = output_root / "parallel_runner_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return False
    workers = summary.get("worker_results", [])
    return bool(workers) and all(int(worker.get("returncode", 1)) == 0 for worker in workers)


def process_lines() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.splitlines()


def active_for(output_root: Path) -> bool:
    needle = str(output_root)
    for line in process_lines():
        if needle in line and "supervise_ablation.py" not in line and "rg " not in line:
            return True
    return False


def launch_ablation_active() -> bool:
    for line in process_lines():
        if "launch_ablation.py" in line and "supervise_ablation.py" not in line and "rg " not in line:
            return True
    return False


def run_or_wait(name: str, source_root: Path, output_root: Path, router_port: int) -> None:
    while not complete(output_root):
        if active_for(output_root) or launch_ablation_active():
            log(f"{name}: active process detected; waiting")
            time.sleep(60)
            continue
        command = runner_command(source_root, output_root, router_port)
        log(f"{name}: launching {' '.join(command)}")
        completed = subprocess.run(command, cwd=str(source_root), check=False, text=True)
        log(f"{name}: runner exited returncode={completed.returncode}")
        if completed.returncode != 0:
            time.sleep(60)
        else:
            break
    if complete(output_root):
        log(f"{name}: completed")
    else:
        raise SystemExit(f"{name}: incomplete after runner exit")


def main() -> None:
    run_or_wait(
        "topk_baseline",
        RUN_ROOT / "code" / "topk_source",
        RUN_ROOT / "topk_mbpp_humaneval_3x",
        8101,
    )
    run_or_wait(
        "sparsemax_only",
        RUN_ROOT / "code" / "sparsemax_source",
        RUN_ROOT / "sparsemax_mbpp_humaneval_3x",
        8102,
    )
    log("all done")


if __name__ == "__main__":
    main()
