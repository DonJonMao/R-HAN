from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class BackendTarget:
    name: str
    base_url: str
    weight: float = 1.0
    healthy: bool = True
    active_requests: int = 0
    failures: int = 0
    successes: int = 0
    last_error: str = ""

    def sanitized_base_url(self) -> str:
        return self.base_url.rstrip("/")


class BackendRouter:
    def __init__(self, backends: Sequence[BackendTarget]) -> None:
        if not backends:
            raise ValueError("BackendRouter requires at least one backend")
        self._backends = [BackendTarget(**vars(item)) for item in backends]
        self._by_name = {item.name: item for item in self._backends}
        if len(self._by_name) != len(self._backends):
            raise ValueError("backend names must be unique")
        self._lock = threading.Lock()

    def _candidate_pool(self, *, exclude: set[str], allow_unhealthy: bool) -> list[BackendTarget]:
        candidates = [item for item in self._backends if item.name not in exclude]
        if not allow_unhealthy:
            healthy = [item for item in candidates if item.healthy]
            if healthy:
                return healthy
        return candidates

    @staticmethod
    def _priority_key(item: BackendTarget) -> tuple[float, int, int, str]:
        weight = item.weight if item.weight > 0 else 1.0
        load = float(item.active_requests) / weight
        return (load, item.failures, -item.successes, item.name)

    def choose_backend(
        self,
        *,
        exclude: Iterable[str] | None = None,
        allow_unhealthy: bool = False,
    ) -> BackendTarget:
        exclude_set = set(exclude or ())
        with self._lock:
            candidates = self._candidate_pool(exclude=exclude_set, allow_unhealthy=allow_unhealthy)
            if not candidates:
                raise RuntimeError("no backend available")
            selected = min(candidates, key=self._priority_key)
            return BackendTarget(**vars(selected))

    def start_request(self, backend_name: str) -> None:
        with self._lock:
            backend = self._by_name[backend_name]
            backend.active_requests += 1

    def finish_request(self, backend_name: str, *, success: bool, error: str = "") -> None:
        with self._lock:
            backend = self._by_name[backend_name]
            backend.active_requests = max(0, backend.active_requests - 1)
            if success:
                backend.successes += 1
                backend.last_error = ""
                backend.healthy = True
            else:
                backend.failures += 1
                backend.last_error = error.strip()
                backend.healthy = False

    def mark_backend_health(self, backend_name: str, healthy: bool, *, reason: str = "") -> None:
        with self._lock:
            backend = self._by_name[backend_name]
            backend.healthy = bool(healthy)
            if healthy:
                backend.last_error = ""
            elif reason:
                backend.last_error = str(reason)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name": item.name,
                    "base_url": item.base_url,
                    "weight": item.weight,
                    "healthy": item.healthy,
                    "active_requests": item.active_requests,
                    "failures": item.failures,
                    "successes": item.successes,
                    "last_error": item.last_error,
                }
                for item in self._backends
            ]


def parse_backend_specs(specs: Sequence[str]) -> list[BackendTarget]:
    backends: list[BackendTarget] = []
    for idx, raw_spec in enumerate(specs):
        text = str(raw_spec or "").strip()
        if not text:
            continue
        weight = 1.0
        left = text
        if "@" in text:
            left, right = text.rsplit("@", 1)
            right = right.strip()
            if right:
                try:
                    weight = float(right)
                except ValueError as exc:
                    raise ValueError(f"invalid backend weight in spec: {text}") from exc
        name: str
        url: str
        if "=" in left:
            name, url = left.split("=", 1)
            name = name.strip()
            url = url.strip()
        else:
            name = f"backend_{idx}"
            url = left.strip()
        if not name:
            raise ValueError(f"backend name missing in spec: {text}")
        if not url.startswith("http://") and not url.startswith("https://"):
            raise ValueError(f"backend url must start with http:// or https:// : {text}")
        if weight <= 0:
            raise ValueError(f"backend weight must be > 0 in spec: {text}")
        backends.append(BackendTarget(name=name, base_url=url.rstrip("/"), weight=weight))
    if not backends:
        raise ValueError("no valid backend specs provided")
    names = [item.name for item in backends]
    if len(names) != len(set(names)):
        raise ValueError("backend names must be unique")
    return backends
