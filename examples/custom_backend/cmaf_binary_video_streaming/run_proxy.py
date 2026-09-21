# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve ``client.html`` and a running Dynamo frontend under one browser origin.

The frontend sends no CORS headers, so a page served from another port cannot
POST JSON to it: the browser refuses the preflight and ``fetch`` fails with
``TypeError: Failed to fetch`` before the request goes out. Answering both from
one port avoids that, and lets the browser sit on a different machine.

    python examples/custom_backend/cmaf_binary_video_streaming/run_proxy.py \
        --bind 0.0.0.0 --proxy-port 8080 --frontend-port 8000
    # then open http://<host>:8080/

Development only: no auth, no TLS, one upstream.
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import mimetypes
import sys
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
CLIENT_HTML = EXAMPLE_DIR / "client.html"
# The first byte lands only after the whole clip has been denoised, so the socket
# timeout has to cover a generation, not just the gaps between fragments.
DEFAULT_UPSTREAM_TIMEOUT = 600.0
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class DemoProxyServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        frontend_host: str,
        frontend_port: int,
        upstream_timeout: float = DEFAULT_UPSTREAM_TIMEOUT,
    ) -> None:
        super().__init__(server_address, DemoProxyHandler)
        self.frontend_host = frontend_host
        self.frontend_port = frontend_port
        self.static_root = EXAMPLE_DIR
        self.upstream_timeout = upstream_timeout


class DemoProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "cmaf-binary-demo/0.1"

    def do_GET(self) -> None:
        self._handle()

    def do_HEAD(self) -> None:
        self._handle(head_only=True)

    def do_POST(self) -> None:
        self._handle()

    def do_OPTIONS(self) -> None:
        self._handle()

    def _handle(self, head_only: bool = False) -> None:
        # Route on the path alone; `self.path` keeps the query string because
        # upstream needs it verbatim. Matching the raw target would 404 on
        # `/?debug=1`.
        route = self._route_path()

        if route in {"/", "/client.html"}:
            self._serve_static(CLIENT_HTML, head_only=head_only)
            return

        if route.startswith("/v1/") or route in {
            "/live",
            "/ready",
            "/health",
            "/metrics",
        }:
            self._forward(head_only=head_only)
            return

        candidate = self._safe_static_path(route)
        if candidate and candidate.is_file():
            self._serve_static(candidate, head_only=head_only)
            return

        self.send_error(404, "Not found")

    def _route_path(self) -> str:
        """The request target with any query string and fragment removed."""
        return self.path.split("?", 1)[0].split("#", 1)[0]

    def _safe_static_path(self, route: str) -> Path | None:
        cleaned = route.lstrip("/")
        if not cleaned:
            return None
        candidate = (self.server.static_root / cleaned).resolve()
        try:
            candidate.relative_to(self.server.static_root)
        except ValueError:
            return None
        return candidate

    def _serve_static(self, path: Path, head_only: bool = False) -> None:
        body = path.read_bytes()
        mime_type, _ = mimetypes.guess_type(path.name)
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        # A cached copy silently hides edits to the page while bringing it up.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _forward(self, head_only: bool = False) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else None

        upstream_headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        upstream_headers["Host"] = (
            f"{self.server.frontend_host}:{self.server.frontend_port}"
        )

        conn = http.client.HTTPConnection(
            self.server.frontend_host,
            self.server.frontend_port,
            timeout=self.server.upstream_timeout,
        )
        try:
            conn.request(self.command, self.path, body=body, headers=upstream_headers)
            response = conn.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() in HOP_BY_HOP_HEADERS:
                    continue
                self.send_header(key, value)
            self.end_headers()

            if head_only:
                response.read()
                return

            while True:
                # read1, not read(n): read(n) keeps pulling until it has n bytes,
                # so it would hold a finished fragment back waiting for the next.
                chunk = response.read1(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away or reloaded mid-stream
        except (TimeoutError, OSError) as exc:
            sys.stderr.write(f"[demo-proxy] upstream failed: {exc!r}\n")
        finally:
            conn.close()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[demo-proxy] {self.address_string()} - {fmt % args}\n")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the CMAF demo page and proxy an already-running "
        "Dynamo frontend under one browser origin."
    )
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Browser-facing bind address for the demo proxy (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=8080,
        help="Browser-facing port for the demo proxy (default: 8080)",
    )
    parser.add_argument(
        "--frontend-host",
        default="127.0.0.1",
        help="Host where the running Dynamo frontend listens (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=8000,
        help="Port of the running Dynamo frontend to proxy (default: 8000)",
    )
    parser.add_argument(
        "--frontend-timeout-seconds",
        type=float,
        default=30.0,
        help="How long to wait for the frontend to answer before serving (default: 30)",
    )
    parser.add_argument(
        "--upstream-timeout-seconds",
        type=float,
        default=DEFAULT_UPSTREAM_TIMEOUT,
        help="Socket timeout when talking to the frontend; must cover a whole "
        f"generation (default: {DEFAULT_UPSTREAM_TIMEOUT:.0f})",
    )
    return parser


def wait_for_frontend(host: str, port: int, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=1)
            conn.request("GET", "/live")
            response = conn.getresponse()
            response.read()
            conn.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(
        f"Timed out waiting for frontend on http://{host}:{port} ({last_error})"
    )


def main() -> int:
    args = create_parser().parse_args()

    wait_for_frontend(
        args.frontend_host,
        args.frontend_port,
        args.frontend_timeout_seconds,
    )

    server = DemoProxyServer(
        (args.bind, args.proxy_port),
        frontend_host=args.frontend_host,
        frontend_port=args.frontend_port,
        upstream_timeout=args.upstream_timeout_seconds,
    )

    print()
    print("CMAF demo proxy is ready")
    print(f"  Browser URL: http://{args.bind}:{args.proxy_port}/")
    print(f"  Proxied frontend: http://{args.frontend_host}:{args.frontend_port}/")
    print("  The browser sees one origin, so no Dynamo CORS change is needed.")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
