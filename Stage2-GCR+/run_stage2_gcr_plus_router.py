from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
for candidate in (PACKAGE_ROOT, REPO_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from stage2_gcr_plus.orchestration.proxy import RouterProxyServer
from stage2_gcr_plus.orchestration.routing import parse_backend_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage2-GCR+ 4x router/load-balancer")
    parser.add_argument("--backend", action="append", required=True, help="name=http://host:port@weight")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8039)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--health-interval-s", type=float, default=15.0)
    args = parser.parse_args()

    backends = parse_backend_specs(args.backend)
    server = RouterProxyServer(
        backends=backends,
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
        health_interval_s=args.health_interval_s,
    )
    server.start()
    print(json.dumps({"listen_url": server.listen_url, "backends": args.backend}, ensure_ascii=False), flush=True)

    stop = {"value": False}

    def _stop_handler(signum, frame):
        del signum, frame
        stop["value"] = True

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    try:
        while not stop["value"]:
            time.sleep(1.0)
    finally:
        server.close()


if __name__ == "__main__":
    main()
