# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import http.client
import http.server
import mimetypes
import os
import subprocess
import sys
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[2]
CLIENT_HTML = EXAMPLE_DIR / "client.html"
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
    ) -> None:
        super().__init__(server_address, DemoProxyHandler)
        self.frontend_host = frontend_host
        self.frontend_port = frontend_port
        self.static_root = EXAMPLE_DIR


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
        if self.path == "/" or self.path == "/client.html":
            self._serve_static(CLIENT_HTML, head_only=head_only)
            return

        if self.path.startswith("/v1/") or self.path in {
            "/live",
            "/ready",
            "/health",
            "/metrics",
        }:
            self._proxy_request(head_only=head_only)
            return

        candidate = self._safe_static_path(self.path)
        if candidate and candidate.is_file():
            self._serve_static(candidate, head_only=head_only)
            return

        self.send_error(404, "Not found")

    def _safe_static_path(self, raw_path: str) -> Path | None:
        cleaned = raw_path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
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
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _proxy_request(self, head_only: bool = False) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else None
        upstream_headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        upstream_headers[
            "Host"
        ] = f"{self.server.frontend_host}:{self.server.frontend_port}"

        conn = http.client.HTTPConnection(
            self.server.frontend_host,
            self.server.frontend_port,
            timeout=300,
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
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except BrokenPipeError:
            pass
        finally:
            conn.close()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[demo-proxy] {self.address_string()} - {fmt % args}\n")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the CMAF binary demo page and proxy Dynamo under one browser origin."
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
        help="Internal host where the Dynamo frontend listens (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=18001,
        help="Internal port for the Dynamo frontend subprocess or existing frontend (default: 18001)",
    )
    parser.add_argument(
        "--frontend-timeout-seconds",
        type=float,
        default=30.0,
        help="How long to wait for the frontend to start responding (default: 30)",
    )
    parser.add_argument(
        "--discovery-backend",
        default=os.environ.get("DYN_DISCOVERY_BACKEND", "file"),
        help="Discovery backend used when auto-starting the frontend (default: file)",
    )
    parser.add_argument(
        "--request-plane",
        default=os.environ.get("DYN_REQUEST_PLANE", "tcp"),
        help="Request plane used when auto-starting the frontend (default: tcp)",
    )
    parser.add_argument(
        "--no-start-frontend",
        action="store_true",
        help="Reuse an already-running frontend instead of starting a subprocess",
    )
    parser.add_argument(
        "frontend_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to `python -m dynamo.frontend` after `--`",
    )
    return parser


def build_frontend_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "dynamo.frontend",
        "--http-port",
        str(args.frontend_port),
        "--discovery-backend",
        args.discovery_backend,
    ]
    if args.frontend_args:
        extras = list(args.frontend_args)
        if extras and extras[0] == "--":
            extras = extras[1:]
        cmd.extend(extras)
    return cmd


def wait_for_frontend(
    host: str, port: int, timeout_seconds: float, proc: subprocess.Popen[str] | None
) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Frontend exited early with code {proc.returncode}")

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


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    args = create_parser().parse_args()

    frontend_proc: subprocess.Popen[str] | None = None
    if not args.no_start_frontend:
        frontend_env = os.environ.copy()
        frontend_env.setdefault("DYN_DISCOVERY_BACKEND", args.discovery_backend)
        frontend_env.setdefault("DYN_REQUEST_PLANE", args.request_plane)
        cmd = build_frontend_command(args)
        print(f"Starting Dynamo frontend: {' '.join(cmd)}")
        frontend_proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=frontend_env)
        wait_for_frontend(
            args.frontend_host,
            args.frontend_port,
            args.frontend_timeout_seconds,
            frontend_proc,
        )
    else:
        wait_for_frontend(
            args.frontend_host,
            args.frontend_port,
            args.frontend_timeout_seconds,
            None,
        )

    server = DemoProxyServer(
        (args.bind, args.proxy_port),
        frontend_host=args.frontend_host,
        frontend_port=args.frontend_port,
    )

    print()
    print("CMAF binary demo proxy is ready")
    print(f"  Browser URL: http://{args.bind}:{args.proxy_port}/")
    print(f"  Proxied frontend: http://{args.frontend_host}:{args.frontend_port}/")
    print("  The browser now sees one origin, so no Dynamo CORS change is needed.")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if frontend_proc is not None:
            terminate_process(frontend_proc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
