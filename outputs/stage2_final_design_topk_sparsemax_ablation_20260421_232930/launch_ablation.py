from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


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


def run_group(name: str, source_root: Path, output_root: Path, router_port: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
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

    print(f"[group-start] {name} {stamp()} source={source_root} output={output_root}", flush=True)
    print("[group-cmd] " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(source_root), check=False, text=True)
    print(f"[group-exit] {name} returncode={completed.returncode} {stamp()}", flush=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    run_group(
        "topk_baseline",
        RUN_ROOT / "code" / "topk_source",
        RUN_ROOT / "topk_mbpp_humaneval_3x",
        8101,
    )
    run_group(
        "sparsemax_only",
        RUN_ROOT / "code" / "sparsemax_source",
        RUN_ROOT / "sparsemax_mbpp_humaneval_3x",
        8102,
    )
    print(f"[all-done] {stamp()}", flush=True)


if __name__ == "__main__":
    main()
