# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KVBM Management HTTP Server.

Provides a lightweight REST API for managing KVBM pools when KVBM_DEV_MODE=TRUE.

Endpoints:
    POST /v1/cache/clear          Clear a specific pool (body: {"pool": "cpu"})
    POST /v1/cache/clear_all      Clear all pools (device, host, disk)
    POST /v1/cache/cpu_lookup     Enable/disable CPU cache lookup (body: {"disabled": true})
    GET  /v1/cache/status          Get pool status summary
    GET  /v1/cache/cpu_lookup     Get CPU cache lookup status
    GET  /v1/health                Health check

Environment variables:
    KVBM_DEV_MODE               Must be TRUE/true/1 to enable destructive operations.
    KVBM_MANAGEMENT_PORT        HTTP port for the management server (default: 6881).
    KVBM_MANAGEMENT_ENABLED     Set to TRUE/true/1 to auto-start the server (default: same as KVBM_DEV_MODE).

Usage (auto-start with connector leader):
    The management server starts automatically when the KvConnectorLeader is created
    and KVBM_DEV_MODE=TRUE, listening on KVBM_MANAGEMENT_PORT.

Usage (standalone):
    python -m kvbm.management --port 6881

Usage (curl):
    # Clear CPU pool
    curl -X POST http://localhost:6881/v1/cache/clear -d '{"pool": "cpu"}'

    # Clear GPU pool
    curl -X POST http://localhost:6881/v1/cache/clear -d '{"pool": "gpu"}'

    # Clear all pools
    curl -X POST http://localhost:6881/v1/cache/clear_all

    # Disable CPU cache lookup
    curl -X POST http://localhost:6881/v1/cache/cpu_lookup -d '{"disabled": true}'

    # Enable CPU cache lookup
    curl -X POST http://localhost:6881/v1/cache/cpu_lookup -d '{"disabled": false}'

    # CPU cache lookup status
    curl http://localhost:6881/v1/cache/cpu_lookup

    # Health check
    curl http://localhost:6881/v1/health
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_MANAGEMENT_PORT = 6881

# Valid pool names (case-insensitive)
VALID_POOLS = {"gpu", "device", "cpu", "host", "disk"}


def _is_dev_mode() -> bool:
    """Check if KVBM_DEV_MODE is enabled."""
    return os.environ.get("KVBM_DEV_MODE", "").lower() in ("true", "1")


def _is_management_enabled() -> bool:
    """Check if the management server should be started."""
    val = os.environ.get("KVBM_MANAGEMENT_ENABLED", "")
    if val:
        return val.lower() in ("true", "1")
    # Default to dev-mode setting
    return _is_dev_mode()


def _get_management_port() -> int:
    """Get the management server port."""
    try:
        return int(os.environ.get("KVBM_MANAGEMENT_PORT", str(DEFAULT_MANAGEMENT_PORT)))
    except ValueError:
        return DEFAULT_MANAGEMENT_PORT


class _PoolClearFunc:
    """Holds references to the clear_pool callable(s) registered by KVBM components."""

    def __init__(self):
        self._clear_pool: Optional[Callable[[str], None]] = None
        self._get_status: Optional[Callable[[], dict[str, Any]]] = None
        self._set_cpu_lookup_disabled: Optional[Callable[[bool], None]] = None
        self._get_cpu_lookup_status: Optional[Callable[[], dict[str, Any]]] = None

    def register_clear_pool(self, fn: Callable[[str], None]) -> None:
        self._clear_pool = fn

    def register_get_status(self, fn: Callable[[], dict[str, Any]]) -> None:
        self._get_status = fn

    def register_set_cpu_lookup_disabled(self, fn: Callable[[bool], None]) -> None:
        self._set_cpu_lookup_disabled = fn

    def register_get_cpu_lookup_status(self, fn: Callable[[], dict[str, Any]]) -> None:
        self._get_cpu_lookup_status = fn

    def clear_pool(self, pool: str) -> None:
        if self._clear_pool is None:
            raise RuntimeError("No clear_pool handler registered. Is the KVBM leader initialized?")
        self._clear_pool(pool)

    def get_status(self) -> dict[str, Any]:
        if self._get_status is None:
            return {"status": "no status handler registered"}
        return self._get_status()

    def set_cpu_lookup_disabled(self, disabled: bool) -> None:
        if self._set_cpu_lookup_disabled is None:
            raise RuntimeError("No cpu_lookup setter registered. Is the KVBM leader initialized?")
        self._set_cpu_lookup_disabled(disabled)

    def get_cpu_lookup_status(self) -> dict[str, Any]:
        if self._get_cpu_lookup_status is None:
            return {"status": "no cpu_lookup status handler registered"}
        return self._get_cpu_lookup_status()


# Module-level singleton that handlers register against.
_pool_ops = _PoolClearFunc()


def register_clear_pool(fn: Callable[[str], None]) -> None:
    """Register the clear_pool callable from a KVBM component.

    This is called during leader initialization so the HTTP handler
    can delegate pool-clearing to the actual block manager.
    """
    _pool_ops.register_clear_pool(fn)


def register_get_status(fn: Callable[[], dict[str, Any]]) -> None:
    """Register a status callback that returns pool status information."""
    _pool_ops.register_get_status(fn)


def register_set_cpu_lookup_disabled(fn: Callable[[bool], None]) -> None:
    """Register a callback to enable/disable CPU cache lookup."""
    _pool_ops.register_set_cpu_lookup_disabled(fn)


def register_get_cpu_lookup_status(fn: Callable[[], dict[str, Any]]) -> None:
    """Register a status callback for CPU cache lookup."""
    _pool_ops.register_get_cpu_lookup_status(fn)


class _ManagementHandler(BaseHTTPRequestHandler):
    """HTTP request handler for KVBM management endpoints."""

    def _send_json(self, status: int, body: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body, indent=2).encode())

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    # ------------------------------------------------------------------
    # GET routes
    # ------------------------------------------------------------------

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/health":
            self._send_json(200, {
                "status": "ok",
                "dev_mode": _is_dev_mode(),
            })

        elif self.path == "/v1/cache/status":
            try:
                status = _pool_ops.get_status()
                self._send_json(200, {"status": "ok", "pools": status})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif self.path == "/v1/cache/cpu_lookup":
            try:
                status = _pool_ops.get_cpu_lookup_status()
                self._send_json(200, {"status": "ok", **status})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        else:
            self._send_json(404, {"error": f"not found: {self.path}"})

    # ------------------------------------------------------------------
    # POST routes
    # ------------------------------------------------------------------

    def do_POST(self):  # noqa: N802
        if self.path == "/v1/cache/clear":
            self._handle_clear_pool()

        elif self.path == "/v1/cache/clear_all":
            self._handle_clear_all()

        elif self.path == "/v1/cache/cpu_lookup":
            self._handle_cpu_lookup_toggle()

        else:
            self._send_json(404, {"error": f"not found: {self.path}"})

    def _handle_clear_pool(self) -> None:
        if not _is_dev_mode():
            self._send_json(403, {
                "error": "KVBM_DEV_MODE is not enabled. Set KVBM_DEV_MODE=TRUE to allow destructive operations."
            })
            return

        raw = self._read_body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        pool = body.get("pool", "").strip().lower()
        if pool not in VALID_POOLS:
            self._send_json(400, {
                "error": f"invalid pool '{pool}'. Must be one of: {sorted(VALID_POOLS)}"
            })
            return

        try:
            _pool_ops.clear_pool(pool)
            self._send_json(200, {"status": "ok", "pool": pool, "action": "cleared"})
        except Exception as e:
            logger.error("clear_pool(%s) failed: %s", pool, e)
            self._send_json(500, {"error": str(e)})

    def _handle_clear_all(self) -> None:
        if not _is_dev_mode():
            self._send_json(403, {
                "error": "KVBM_DEV_MODE is not enabled. Set KVBM_DEV_MODE=TRUE to allow destructive operations."
            })
            return

        errors: dict[str, str] = {}
        cleared: list[str] = []

        for pool in ["gpu", "cpu", "disk"]:
            try:
                _pool_ops.clear_pool(pool)
                cleared.append(pool)
            except Exception as e:
                errors[pool] = str(e)

        status = 200 if not errors else 207  # 207 Multi-Status
        self._send_json(status, {
            "status": "partial" if errors else "ok",
            "cleared": cleared,
            "errors": errors,
        })

    def _handle_cpu_lookup_toggle(self) -> None:
        if not _is_dev_mode():
            self._send_json(403, {
                "error": "KVBM_DEV_MODE is not enabled. Set KVBM_DEV_MODE=TRUE to allow dev-only operations."
            })
            return

        raw = self._read_body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        if "disabled" not in body:
            self._send_json(400, {"error": "missing required field: disabled"})
            return

        disabled = bool(body.get("disabled"))
        try:
            _pool_ops.set_cpu_lookup_disabled(disabled)
            status = _pool_ops.get_cpu_lookup_status()
            self._send_json(200, {"status": "ok", **status})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def log_message(self, format: str, *args) -> None:
        """Override to use Python logging instead of stderr."""
        logger.info("[kvbm-mgmt] %s", format % args)


# ------------------------------------------------------------------
# Server lifecycle
# ------------------------------------------------------------------

_server_instance: Optional[HTTPServer] = None
_server_thread: Optional[threading.Thread] = None


def start_management_server(port: Optional[int] = None) -> Optional[int]:
    """Start the management HTTP server in a background daemon thread.

    Args:
        port: Port to listen on. Defaults to KVBM_MANAGEMENT_PORT env or 6881.

    Returns:
        The port the server is listening on, or None if it was not started.
    """
    global _server_instance, _server_thread

    if _server_instance is not None:
        logger.warning("KVBM management server is already running")
        return _server_instance.server_address[1]

    if not _is_management_enabled():
        logger.debug("KVBM management server not enabled (set KVBM_DEV_MODE=TRUE or KVBM_MANAGEMENT_ENABLED=TRUE)")
        return None

    if port is None:
        port = _get_management_port()

    try:
        server = HTTPServer(("0.0.0.0", port), _ManagementHandler)
    except OSError as e:
        logger.error("Failed to start KVBM management server on port %d: %s", port, e)
        return None

    _server_instance = server

    _server_thread = threading.Thread(
        target=server.serve_forever,
        name="kvbm-management",
        daemon=True,
    )
    _server_thread.start()

    logger.info(
        "KVBM management server started on port %d (dev_mode=%s)",
        port,
        _is_dev_mode(),
    )
    return port


def stop_management_server() -> None:
    """Stop the management server if running."""
    global _server_instance, _server_thread

    if _server_instance is not None:
        _server_instance.shutdown()
        _server_instance = None

    if _server_thread is not None:
        _server_thread.join(timeout=5)
        _server_thread = None

    logger.info("KVBM management server stopped")


# ------------------------------------------------------------------
# CLI entry point: python -m kvbm.management
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import signal
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="KVBM Management HTTP Server")
    parser.add_argument("--port", type=int, default=_get_management_port(), help="Port to listen on")
    args = parser.parse_args()

    if not _is_dev_mode():
        logger.warning("KVBM_DEV_MODE is not set. Destructive operations will be rejected.")

    # Force enable for CLI mode
    os.environ["KVBM_MANAGEMENT_ENABLED"] = "TRUE"

    port = start_management_server(args.port)
    if port is None:
        logger.error("Failed to start management server")
        sys.exit(1)

    print(f"KVBM Management Server listening on http://0.0.0.0:{port}")
    print(f"  POST /v1/cache/clear      - Clear a pool (body: {{\"pool\": \"cpu\"}})")
    print(f"  POST /v1/cache/clear_all   - Clear all pools")
    print(f"  POST /v1/cache/cpu_lookup  - Toggle CPU lookup (body: {{\"disabled\": true}})")
    print(f"  GET  /v1/cache/status      - Pool status")
    print(f"  GET  /v1/cache/cpu_lookup  - CPU lookup status")
    print(f"  GET  /v1/health            - Health check")

    # Block until SIGINT/SIGTERM
    evt = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: evt.set())
    signal.signal(signal.SIGTERM, lambda *_: evt.set())
    evt.wait()

    stop_management_server()
