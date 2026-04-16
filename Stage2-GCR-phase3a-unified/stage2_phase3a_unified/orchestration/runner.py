from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from stage2_gcr_plus.orchestration.gate import MbppGateStatus, check_mbpp_completion, normalize_statuses, wait_for_mbpp_completion
from stage2_gcr_plus.orchestration.proxy import RouterProxyServer
from stage2_gcr_plus.orchestration.routing import parse_backend_specs
from stage2_gcr_plus.orchestration.sharding import ShardSpec, build_sample_shards


def resolve_execution_mode(requested_mode: str, *, mbpp_ready: bool) -> str:
    mode = str(requested_mode or "auto").strip().lower()
    if mode in {"tp4", "3x"}:
        return mode
    if mode != "auto":
        raise ValueError(f"unsupported mode: {requested_mode}")
    return "3x" if mbpp_ready else "tp4"


def _list_datasets(data_root: Path) -> list[str]:
    if not data_root.exists():
        return []
    names = [item.name for item in data_root.iterdir() if item.is_dir() and not item.name.startswith(".")]
    return sorted(names)


def _resolve_datasets(data_root: Path, requested: Sequence[str]) -> list[str]:
    if requested:
        seen: set[str] = set()
        ordered: list[str] = []
        for name in requested:
            text = str(name or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered
    return _list_datasets(data_root)


def _flatten_train_args(values: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(shlex.split(value))
    return tokens


def _build_worker_command(
    *,
    python_bin: str,
    train_script: str,
    shard: ShardSpec,
    worker_output_root: Path,
    datasets: Sequence[str],
    train_args: Sequence[str],
) -> list[str]:
    command = [
        python_bin,
        train_script,
        "--data-root",
        str(shard.data_root),
        "--output-root",
        str(worker_output_root),
    ]
    for dataset_name in datasets:
        command.extend(["--dataset", dataset_name])
    command.extend(train_args)
    return command


def _worker_env(base_url: str) -> dict[str, str]:
    env = dict(os.environ)
    env["OPENAI_API_BASE"] = base_url
    env["LLM_API_BASE"] = base_url
    env["STAGE2_CHAT_API_BASE"] = base_url
    env["STAGE2_JUDGE_API_BASE"] = base_url
    return env


def _run_worker(
    *,
    worker_index: int,
    command: Sequence[str],
    log_path: Path,
    base_url: str,
    cwd: Path,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = _worker_env(base_url)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[worker-start] idx={worker_index} base_url={base_url}\n")
        handle.write(f"[worker-cmd] {' '.join(shlex.quote(part) for part in command)}\n")
        handle.flush()
        result = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        handle.write(f"\n[worker-exit] idx={worker_index} returncode={result.returncode}\n")
    return {
        "worker_index": worker_index,
        "command": list(command),
        "log_path": str(log_path),
        "returncode": int(result.returncode),
        "base_url": base_url,
    }


def _resolve_mbpp_gate(args: argparse.Namespace) -> MbppGateStatus:
    statuses = normalize_statuses(args.mbpp_complete_status)
    if not args.mbpp_progress_path:
        return MbppGateStatus(
            ready=False,
            status="missing",
            reason="mbpp progress path not provided",
            progress_path="",
        )
    if args.wait_for_mbpp:
        timeout_s = None if args.mbpp_timeout_s <= 0 else float(args.mbpp_timeout_s)
        return wait_for_mbpp_completion(
            args.mbpp_progress_path,
            dataset_name=args.mbpp_dataset,
            accepted_statuses=statuses,
            poll_interval_s=float(args.mbpp_poll_interval_s),
            timeout_s=timeout_s,
        )
    return check_mbpp_completion(
        args.mbpp_progress_path,
        dataset_name=args.mbpp_dataset,
        accepted_statuses=statuses,
    )


def run_sample_parallel(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    gate_status = _resolve_mbpp_gate(args)
    mode = resolve_execution_mode(args.mode, mbpp_ready=gate_status.ready)
    datasets = _resolve_datasets(data_root, args.dataset)
    if not datasets:
        raise ValueError(f"No datasets found under {data_root}")

    train_args = _flatten_train_args(args.train_arg)
    shard_count = max(1, int(args.workers))
    shard_specs = build_sample_shards(
        data_root=data_root,
        datasets=datasets,
        shard_root=output_root / "sample_shards",
        shard_count=shard_count,
        seed=int(args.seed),
    )

    proxy: RouterProxyServer | None = None
    base_url = str(args.tp4_backend).rstrip("/")
    if mode == "3x":
        backend_specs = parse_backend_specs(args.x3_backend)
        if len(backend_specs) != 3:
            raise ValueError("3x mode requires exactly 3 backends via --x3-backend")
        proxy = RouterProxyServer(
            backends=backend_specs,
            host=args.router_host,
            port=int(args.router_port),
            timeout_s=float(args.router_timeout_s),
            health_interval_s=float(args.router_health_interval_s),
        )
        proxy.start()
        base_url = proxy.listen_url

    worker_output_root = output_root / "workers"
    worker_output_root.mkdir(parents=True, exist_ok=True)

    worker_results: list[dict[str, Any]] = []
    error: Exception | None = None
    try:
        with ThreadPoolExecutor(max_workers=shard_count) as pool:
            futures = []
            for spec in shard_specs:
                command = _build_worker_command(
                    python_bin=args.python_bin,
                    train_script=args.train_script,
                    shard=spec,
                    worker_output_root=worker_output_root / f"worker_{spec.shard_index}",
                    datasets=datasets,
                    train_args=train_args,
                )
                future = pool.submit(
                    _run_worker,
                    worker_index=spec.shard_index,
                    command=command,
                    log_path=output_root / "logs" / f"worker_{spec.shard_index}.log",
                    base_url=base_url,
                    cwd=Path(args.cwd).resolve(),
                )
                futures.append(future)
            for future in as_completed(futures):
                worker_results.append(future.result())
    except Exception as exc:
        error = exc
    finally:
        if proxy is not None:
            proxy.close()

    worker_results.sort(key=lambda item: int(item.get("worker_index", 0)))

    summary = {
        "mode_requested": args.mode,
        "mode_resolved": mode,
        "mbpp_gate": {
            "ready": gate_status.ready,
            "status": gate_status.status,
            "reason": gate_status.reason,
            "progress_path": gate_status.progress_path,
        },
        "datasets": datasets,
        "workers": shard_count,
        "base_url": base_url,
        "train_script": args.train_script,
        "worker_results": worker_results,
    }
    summary_path = output_root / "parallel_runner_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if error is not None:
        raise RuntimeError(f"parallel run failed: {error}") from error
    failed = [item for item in worker_results if int(item.get("returncode", 1)) != 0]
    if failed:
        failed_ids = ",".join(str(item.get("worker_index")) for item in failed)
        raise RuntimeError(f"workers failed: {failed_ids}. See logs under {output_root / 'logs'}")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase3a unified sample-parallel runner with MBPP-gated TP4->3x switch")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", action="append", default=[], help="Repeatable dataset selection. Default: all datasets under data-root")
    parser.add_argument("--workers", type=int, default=3, help="Sample shard count / worker count")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--mode", choices=("auto", "tp4", "3x"), default="auto")
    parser.add_argument("--mbpp-progress-path", default="")
    parser.add_argument("--mbpp-dataset", default="mbpp")
    parser.add_argument("--mbpp-complete-status", action="append", default=["completed"])
    parser.add_argument("--wait-for-mbpp", action="store_true")
    parser.add_argument("--mbpp-poll-interval-s", type=float, default=120.0)
    parser.add_argument("--mbpp-timeout-s", type=float, default=0.0)

    parser.add_argument("--tp4-backend", default="http://127.0.0.1:8039")
    parser.add_argument(
        "--x3-backend",
        action="append",
        default=[],
        help="Repeatable backend spec for 3x mode. Format: name=http://host:port@weight",
    )
    parser.add_argument("--router-host", default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=8039)
    parser.add_argument("--router-timeout-s", type=float, default=120.0)
    parser.add_argument("--router-health-interval-s", type=float, default=15.0)

    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--train-script",
        default=str(Path(__file__).resolve().parents[1] / "train_mas_stage2_phase3a_unified_target_suite.py"),
    )
    parser.add_argument(
        "--train-arg",
        action="append",
        default=[],
        help="Pass-through args for the train script. Repeatable; each value supports shell-style splitting.",
    )
    parser.add_argument("--cwd", default=str(Path.cwd()))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    summary = run_sample_parallel(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
