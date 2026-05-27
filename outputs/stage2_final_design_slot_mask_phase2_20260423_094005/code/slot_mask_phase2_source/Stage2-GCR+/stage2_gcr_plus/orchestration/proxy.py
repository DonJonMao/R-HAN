from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable, Optional

from .routing import BackendRouter, BackendTarget


_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


class RouterProxyServer:
    def __init__(
        self,
        *,
        backends: Iterable[BackendTarget],
        host: str = "127.0.0.1",
        port: int = 8039,
        timeout_s: float = 120.0,
        health_interval_s: float = 15.0,
    ) -> None:
        self.router = BackendRouter(list(backends))
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.health_interval_s = max(1.0, float(health_interval_s))
        self._server = ThreadingHTTPServer((self.host, self.port), self._build_handler())
        self._server.daemon_threads = True
        self._serve_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def listen_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _build_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "Stage2GCRProxy/1.0"

            def do_GET(self) -> None:  # noqa: N802
                outer._handle_http(self, head_only=False)

            def do_HEAD(self) -> None:  # noqa: N802
                outer._handle_http(self, head_only=True)

            def do_POST(self) -> None:  # noqa: N802
                outer._handle_http(self, head_only=False)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        return Handler

    def _read_request_body(self, request: BaseHTTPRequestHandler) -> bytes:
        raw = request.headers.get("Content-Length")
        if not raw:
            return b""
        try:
            size = int(raw)
        except ValueError:
            return b""
        if size <= 0:
            return b""
        return request.rfile.read(size)

    def _copy_response(
        self,
        request: BaseHTTPRequestHandler,
        *,
        status: int,
        headers: list[tuple[str, str]],
        body: bytes,
        head_only: bool,
    ) -> None:
        request.send_response(status)
        for key, value in headers:
            if key.lower() in _HOP_BY_HOP_HEADERS:
                continue
            request.send_header(key, value)
        request.send_header("Content-Length", str(len(body)))
        request.end_headers()
        if not head_only:
            request.wfile.write(body)

    def _health_response(self, request: BaseHTTPRequestHandler, *, head_only: bool) -> None:
        payload = {
            "ok": True,
            "listen_url": self.listen_url,
            "backends": self.router.snapshot(),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._copy_response(
            request,
            status=200,
            headers=[("Content-Type", "application/json; charset=utf-8")],
            body=body,
            head_only=head_only,
        )

    def _forward_request(
        self,
        *,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        tried: set[str] = set()
        errors: list[str] = []
        backend_count = len(self.router.snapshot())
        for _ in range(max(1, backend_count)):
            try:
                backend = self.router.choose_backend(exclude=tried)
            except RuntimeError:
                backend = self.router.choose_backend(exclude=tried, allow_unhealthy=True)
            tried.add(backend.name)
            target_url = f"{backend.sanitized_base_url()}{path}"
            req = urllib.request.Request(
                target_url,
                data=body if method.upper() in {"POST", "PUT", "PATCH", "DELETE"} else None,
                method=method.upper(),
            )
            for key, value in headers.items():
                if key.lower() in _HOP_BY_HOP_HEADERS:
                    continue
                req.add_header(key, value)
            self.router.start_request(backend.name)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    status = int(resp.status)
                    response_headers = list(resp.getheaders())
                    response_body = resp.read()
                self.router.finish_request(backend.name, success=True)
                return status, response_headers, response_body
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                response_headers = list(exc.headers.items())
                response_body = exc.read()
                if status >= 500:
                    self.router.finish_request(backend.name, success=False, error=f"http_{status}")
                    errors.append(f"{backend.name}:http_{status}")
                    continue
                self.router.finish_request(backend.name, success=True)
                return status, response_headers, response_body
            except Exception as exc:
                self.router.finish_request(backend.name, success=False, error=str(exc))
                errors.append(f"{backend.name}:{exc}")
                continue
        payload = {
            "error": "all_backends_failed",
            "details": errors[-8:],
        }
        return 502, [("Content-Type", "application/json; charset=utf-8")], json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _handle_http(self, request: BaseHTTPRequestHandler, *, head_only: bool) -> None:
        if request.path in {"/healthz", "/_healthz"}:
            self._health_response(request, head_only=head_only)
            return
        body = self._read_request_body(request)
        incoming_headers = {k: v for k, v in request.headers.items()}
        status, response_headers, response_body = self._forward_request(
            method=request.command,
            path=request.path,
            headers=incoming_headers,
            body=body,
        )
        self._copy_response(
            request,
            status=status,
            headers=response_headers,
            body=response_body,
            head_only=head_only,
        )

    def _health_probe_once(self) -> None:
        snapshot = self.router.snapshot()
        for item in snapshot:
            url = f"{str(item['base_url']).rstrip('/')}/v1/models"
            req = urllib.request.Request(url, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=min(5.0, self.timeout_s)) as resp:
                    healthy = int(resp.status) < 500
                self.router.mark_backend_health(str(item["name"]), healthy)
            except Exception as exc:
                self.router.mark_backend_health(str(item["name"]), False, reason=str(exc))

    def _health_probe_loop(self) -> None:
        while not self._stop_event.wait(self.health_interval_s):
            self._health_probe_once()

    def start(self) -> None:
        if self._serve_thread is not None:
            return
        self._stop_event.clear()
        self._serve_thread = threading.Thread(target=self._server.serve_forever, name="stage2-gcr-proxy", daemon=True)
        self._serve_thread.start()
        self._health_thread = threading.Thread(target=self._health_probe_loop, name="stage2-gcr-proxy-health", daemon=True)
        self._health_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._serve_thread is not None:
            self._serve_thread.join(timeout=3.0)
            self._serve_thread = None
        if self._health_thread is not None:
            self._health_thread.join(timeout=3.0)
            self._health_thread = None

    def __enter__(self) -> "RouterProxyServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
