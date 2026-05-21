# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Thin pass-through supervisor for the SGLang gRPC bridge.

    python -m dynamo.sglang_grpc [--spawn-sglang] <bridge args...> [-- <sglang args...>]

Tokens before ``--`` (minus the supervisor's own ``--spawn-sglang`` flag) are
forwarded verbatim to ``dynamo._core.run_sglang_bridge_worker``. Tokens after
``--`` are forwarded verbatim to ``python -m sglang.launch_server`` when
``--spawn-sglang`` is set. This module knows nothing about either tool's flag
surface; both upstreams own their own CLIs.

The bridge installs its own tokio signal handlers via ``dynamo_backend_common``
and drives graceful shutdown. The supervisor's Python signal handler is a
belt-and-braces forwarder that runs after the bridge returns, ensuring the
spawned ``sglang.launch_server`` child is gone before the supervisor exits.
"""

import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from typing import Optional

from dynamo._core import run_sglang_bridge_worker
from dynamo.runtime.logging import configure_dynamo_logging

SGLANG_DEFAULT_GRPC = "http://127.0.0.1:30000"
PORT_WAIT_TIMEOUT_SECS = 600.0
CHILD_TERMINATE_GRACE_SECS = 10.0

configure_dynamo_logging(service_name="dynamo.sglang_grpc")
logger = logging.getLogger(__name__)


def _split_argv(argv: list[str]) -> tuple[bool, list[str], Optional[list[str]]]:
    """Split into ``(spawn_sglang, bridge_args, sglang_args)``.

    ``sglang_args is None`` iff no ``--`` separator was given. ``--spawn-sglang``
    may appear anywhere on the supervisor side of ``--``.
    """
    spawn = False
    if "--" in argv:
        idx = argv.index("--")
        left, sglang_args = argv[:idx], argv[idx + 1 :]
    else:
        left, sglang_args = argv, None

    bridge_args = []
    for arg in left:
        if arg == "--spawn-sglang":
            spawn = True
        else:
            bridge_args.append(arg)
    return spawn, bridge_args, sglang_args


def _resolve_grpc_endpoint(bridge_args: list[str]) -> str:
    """Find ``--sglang-grpc-endpoint`` in bridge args, else env, else default."""
    flag = "--sglang-grpc-endpoint"
    it = iter(bridge_args)
    for arg in it:
        if arg == flag:
            return next(it, SGLANG_DEFAULT_GRPC)
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return os.environ.get("SGLANG_GRPC_ENDPOINT", SGLANG_DEFAULT_GRPC)


def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except OSError:
                pass
        time.sleep(1.0)
    return False


def _terminate_child(child: subprocess.Popen) -> None:
    """SIGTERM with grace, then SIGKILL. Idempotent."""
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=CHILD_TERMINATE_GRACE_SECS)
    except subprocess.TimeoutExpired:
        child.kill()
        try:
            child.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def _spawn_sglang(sglang_args: list[str], grpc_endpoint: str) -> subprocess.Popen:
    """Spawn ``sglang.launch_server`` and block until its gRPC port opens."""
    child = subprocess.Popen([sys.executable, "-m", "sglang.launch_server", *sglang_args])

    url = urllib.parse.urlparse(grpc_endpoint)
    host, port = url.hostname or "127.0.0.1", url.port or 30000
    logger.info("Waiting for sglang gRPC %s:%d", host, port)
    try:
        if not _wait_for_port(host, port, PORT_WAIT_TIMEOUT_SECS):
            raise RuntimeError(
                f"sglang gRPC {host}:{port} did not open in {PORT_WAIT_TIMEOUT_SECS}s"
            )
        if child.poll() is not None:
            raise RuntimeError(f"sglang exited during startup (rc={child.returncode})")
    except BaseException:
        _terminate_child(child)
        raise
    logger.info("sglang gRPC up on %s:%d", host, port)
    return child


def _watch_child(child: subprocess.Popen) -> None:
    """If the child exits, signal the supervisor so the bridge tears down."""
    rc = child.wait()
    logger.warning("sglang child exited rc=%d; signalling supervisor shutdown", rc)
    os.kill(os.getpid(), signal.SIGTERM)


def main() -> None:
    spawn, bridge_args, sglang_args = _split_argv(sys.argv[1:])
    if spawn and sglang_args is None:
        logger.error("--spawn-sglang requires args after `--` for sglang.launch_server")
        sys.exit(2)

    sglang = _spawn_sglang(sglang_args, _resolve_grpc_endpoint(bridge_args)) if spawn else None

    def _shutdown(signum, _frame):
        logger.info("Received signal %d, shutting down", signum)
        if sglang is not None:
            _terminate_child(sglang)
        sys.exit(128 + signum)

    if sglang is not None:
        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)
        threading.Thread(target=_watch_child, args=(sglang,), daemon=True).start()

    try:
        run_sglang_bridge_worker(bridge_args)
    finally:
        if sglang is not None:
            _terminate_child(sglang)


if __name__ == "__main__":
    main()
