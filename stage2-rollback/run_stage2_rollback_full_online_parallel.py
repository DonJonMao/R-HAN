from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
for candidate in (PACKAGE_ROOT, REPO_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rollback full online evaluation in parallel shards.")
    parser.add_argument("--trained-suite-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--debug-limit", type=int, default=3)
    parser.add_argument("--chat-api-bases", default="http://127.0.0.1:8041,http://127.0.0.1:8042,http://127.0.0.1:8043")
    args = parser.parse_args()

    bases = [item.strip() for item in str(args.chat_api_bases).split(",") if item.strip()]
    if len(bases) < args.workers:
        raise ValueError(f"Need at least {args.workers} chat api bases, got {bases}")
    trained_suite_root = Path(args.trained_suite_root)
    output_root = Path(args.output_root)
    worker_root = output_root / "workers"
    log_root = output_root / "logs"
    worker_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    procs: List[subprocess.Popen[str]] = []
    handles = []
    for worker_idx in range(args.workers):
        shard_root = trained_suite_root / "sample_shards" / f"shard_{worker_idx}"
        trained_root = trained_suite_root / "workers" / f"worker_{worker_idx}"
        worker_output = worker_root / f"worker_{worker_idx}"
        worker_output.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"worker_{worker_idx}.log"
        cmd = [
            args.python_bin,
            str(PACKAGE_ROOT / "evaluate_stage2_rollback_full_online_target_suite.py"),
            "--data-root",
            str(shard_root),
            "--trained-root",
            str(trained_root),
            "--output-root",
            str(worker_output),
            "--dataset",
            args.dataset,
            "--debug-limit",
            str(args.debug_limit),
            "--chat-api-base",
            bases[worker_idx],
        ]
        handle = log_path.open("w", encoding="utf-8")
        handles.append(handle)
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True)
        procs.append(proc)
    exit_codes = [proc.wait() for proc in procs]
    for handle in handles:
        handle.close()

    metrics: List[Dict[str, Any]] = []
    for worker_idx in range(args.workers):
        metrics_path = worker_root / f"worker_{worker_idx}" / args.dataset / "metrics.json"
        if metrics_path.exists():
            metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    _write_json(
        output_root / "parallel_runner_summary.json",
        {
            "dataset": args.dataset,
            "exit_codes": exit_codes,
            "metrics": metrics,
            "chat_api_bases": bases[: args.workers],
        },
    )
    if any(code != 0 for code in exit_codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
