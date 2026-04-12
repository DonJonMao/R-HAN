from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_COMPLETE_STATUSES: tuple[str, ...] = ("completed",)


@dataclass(frozen=True)
class MbppGateStatus:
    ready: bool
    status: str
    reason: str
    progress_path: str


def _normalize_status(value: object) -> str:
    text = str(value or "").strip().lower()
    return text


def _extract_dataset_status(payload: dict, dataset_name: str) -> str:
    datasets = payload.get("datasets")
    if isinstance(datasets, dict):
        dataset_state = datasets.get(dataset_name)
        if isinstance(dataset_state, dict):
            status = _normalize_status(dataset_state.get("status"))
            if status:
                return status
    return _normalize_status(payload.get("status"))


def check_mbpp_completion(
    progress_path: str | Path,
    *,
    dataset_name: str = "mbpp",
    accepted_statuses: Sequence[str] = DEFAULT_COMPLETE_STATUSES,
) -> MbppGateStatus:
    path = Path(progress_path)
    accepted = {_normalize_status(item) for item in accepted_statuses if _normalize_status(item)}
    if not accepted:
        accepted = {"completed"}
    if not path.exists():
        return MbppGateStatus(
            ready=False,
            status="missing",
            reason=f"progress file not found: {path}",
            progress_path=str(path),
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return MbppGateStatus(
            ready=False,
            status="invalid",
            reason=f"invalid progress json: {exc}",
            progress_path=str(path),
        )
    if not isinstance(payload, dict):
        return MbppGateStatus(
            ready=False,
            status="invalid",
            reason="progress json root is not an object",
            progress_path=str(path),
        )
    status = _extract_dataset_status(payload, dataset_name)
    if not status:
        return MbppGateStatus(
            ready=False,
            status="unknown",
            reason=f"dataset status missing for {dataset_name}",
            progress_path=str(path),
        )
    if status in accepted:
        return MbppGateStatus(
            ready=True,
            status=status,
            reason=f"dataset={dataset_name} reached accepted status={status}",
            progress_path=str(path),
        )
    return MbppGateStatus(
        ready=False,
        status=status,
        reason=f"dataset={dataset_name} not ready, status={status}",
        progress_path=str(path),
    )


def wait_for_mbpp_completion(
    progress_path: str | Path,
    *,
    dataset_name: str = "mbpp",
    accepted_statuses: Sequence[str] = DEFAULT_COMPLETE_STATUSES,
    poll_interval_s: float = 60.0,
    timeout_s: float | None = None,
) -> MbppGateStatus:
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be > 0")
    started = time.monotonic()
    while True:
        status = check_mbpp_completion(
            progress_path,
            dataset_name=dataset_name,
            accepted_statuses=accepted_statuses,
        )
        if status.ready:
            return status
        if timeout_s is not None and timeout_s > 0 and time.monotonic() - started >= timeout_s:
            return MbppGateStatus(
                ready=False,
                status=status.status,
                reason=f"timeout after {timeout_s:.1f}s waiting for MBPP completion ({status.reason})",
                progress_path=status.progress_path,
            )
        time.sleep(poll_interval_s)


def normalize_statuses(values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(item for item in (_normalize_status(v) for v in values) if item)
    return normalized or DEFAULT_COMPLETE_STATUSES
