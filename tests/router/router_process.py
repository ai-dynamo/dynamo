# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import re
import socket
import ssl
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from tests.utils.http_checks import check_http_ok
from tests.utils.managed_process import ManagedProcess
from tests.utils.router_logs import completed_utf8_log_prefix

logger = logging.getLogger(__name__)

_EPP_BINARY_ENV = "DYN_EPP_BINARY"
_DEFAULT_EPP_BINARY = "/epp"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_INIT_OBSERVE_POLL_S = 0.05


class SlotTrackerProcess(ManagedProcess):
    """Manages a standalone dynamo.slot_tracker process."""

    def __init__(self, request, port: int):
        super().__init__(
            command=[
                sys.executable,
                "-m",
                "dynamo.slot_tracker",
                "--port",
                str(port),
            ],
            timeout=10,
            display_output=False,
            health_check_ports=[port],
            health_check_urls=[f"http://localhost:{port}/health"],
            log_dir=request.node.name,
            terminate_all_matching_process_names=False,
            display_name="dynamo-slot-tracker",
        )
        self.port = port


class FrontendRouterProcess(ManagedProcess):
    """Manages a dynamo.frontend process with configurable --router-mode.

    Supports all router modes (round-robin, random, kv, direct) and all
    KV-specific options (block size, thresholds, disagg).
    block_size is only sent to the CLI when router_mode is "kv".
    """

    def __init__(
        self,
        request,
        block_size: int,
        frontend_port: int,
        namespace: str,
        store_backend: str = "etcd",
        blocks_threshold: float | str | None = None,
        tokens_threshold: int | str | None = None,
        tokens_threshold_frac: float | str | None = None,
        router_queue_threshold: float | str | None = None,
        request_plane: str = "nats",
        router_mode: str = "kv",
        min_initial_workers: int | None = None,
        router_ais_config: dict[str, str | int] | None = None,
        serve_indexer: bool = False,
        use_remote_indexer: bool = False,
        event_plane: str | None = None,
        session_affinity_ttl_secs: int | None = None,
        extra_env: Mapping[str, str | None] | None = None,
        require_readiness: bool = True,
    ):
        command = [
            sys.executable,
            "-m",
            "dynamo.frontend",
            "--router-mode",
            router_mode,
            "--http-port",
            str(frontend_port),
            "--discovery-backend",
            store_backend,
            "--namespace",
            namespace,
        ]

        if router_mode == "kv":
            command.extend(["--kv-cache-block-size", str(block_size)])

        if blocks_threshold is not None:
            command.extend(["--active-decode-blocks-threshold", str(blocks_threshold)])

        if tokens_threshold is not None:
            command.extend(["--active-prefill-tokens-threshold", str(tokens_threshold)])

        if tokens_threshold_frac is not None:
            command.extend(
                ["--active-prefill-tokens-threshold-frac", str(tokens_threshold_frac)]
            )

        if router_queue_threshold is not None:
            command.extend(["--router-queue-threshold", str(router_queue_threshold)])

        if serve_indexer:
            command.append("--serve-indexer")

        if use_remote_indexer:
            command.append("--use-remote-indexer")

        if session_affinity_ttl_secs is not None:
            command.extend(
                ["--router-session-affinity-ttl-secs", str(session_affinity_ttl_secs)]
            )

        if router_ais_config is not None:
            command.extend(
                [
                    "--router-track-prefill-tokens",
                    "--router-prefill-load-model",
                    "ais",
                    "--ais-perf-config",
                    json.dumps(router_ais_config),
                ]
            )

        env = os.environ.copy()
        env["DYN_REQUEST_PLANE"] = request_plane
        if event_plane is not None:
            env["DYN_EVENT_PLANE"] = event_plane
        if event_plane == "zmq" and request_plane != "nats":
            env.pop("NATS_SERVER", None)
        if min_initial_workers is not None:
            env["DYN_ROUTER_MIN_INITIAL_WORKERS"] = str(min_initial_workers)
        for name, value in (extra_env or {}).items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value

        super().__init__(
            command=command,
            env=env,
            timeout=60,
            display_output=True,
            health_check_ports=[frontend_port] if require_readiness else [],
            health_check_urls=(
                [(f"http://localhost:{frontend_port}/v1/models", check_http_ok)]
                if require_readiness
                else []
            ),
            log_dir=request.node.name,
            terminate_all_matching_process_names=False,
            display_name=f"dynamo-frontend-{router_mode}-{frontend_port}",
        )
        self.port = frontend_port
        self.router_mode = router_mode
        self.require_readiness = require_readiness


# Backward-compatible alias so existing callers that import KVRouterProcess
# continue to work without changes.
KVRouterProcess = FrontendRouterProcess


def validate_loopback_kubeconfig(kubeconfig: str | os.PathLike[str]) -> Path:
    """Require KUBECONFIG to be this test's loopback fixture, not a cluster file."""
    raw = str(kubeconfig).strip()
    if not raw:
        raise RuntimeError(
            "KUBECONFIG must point at the allocated loopback Kubernetes API fixture"
        )
    path = Path(raw)
    if not path.is_file():
        raise RuntimeError(
            "KUBECONFIG must be an existing loopback fixture file, not a cluster default"
        )
    text = path.read_text(encoding="utf-8")
    if "https://" in text:
        raise RuntimeError("KUBECONFIG must not target a TLS or cluster Kubernetes API")
    if (
        "server: http://127.0.0.1:" not in text
        and "server: http://localhost:" not in text
    ):
        raise RuntimeError(
            "KUBECONFIG must target the allocated loopback Kubernetes API on 127.0.0.1"
        )
    return path


def fetch_loopback_tls_cert(host: str, port: int, timeout_s: float) -> str:
    """Read a loopback self-signed peer cert with a bounded handshake.

    Uses a local SSLContext for this connection only; it does not change
    process-global TLS settings.
    """
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("TLS cert fetch is limited to loopback")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with socket.create_connection((host, port), timeout=timeout_s) as raw:
        raw.settimeout(timeout_s)
        with context.wrap_socket(raw, server_hostname="localhost") as tls:
            der = tls.getpeercert(binary_form=True)
    if not der:
        raise OSError("loopback TLS peer certificate missing")
    return ssl.DER_cert_to_PEM_cert(der)


def epp_loopback_tls_ready(grpc_port: int, timeout: float | None = None) -> bool:
    """ManagedProcess readiness callback for the standard /epp TLS listener."""
    budget = 5.0 if timeout is None else max(float(timeout), 0.1)
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        remaining = max(deadline - time.monotonic(), 0.05)
        try:
            fetch_loopback_tls_cert("127.0.0.1", grpc_port, min(remaining, 2.0))
            return True
        except OSError as exc:
            logger.debug("EPP TLS listener not ready on %s: %s", grpc_port, exc)
            time.sleep(0.2)
    return False


class RuntimeEppProcess(ManagedProcess):
    """Thin ManagedProcess wrapper around the standard Runtime EPP binary."""

    def __init__(
        self,
        request,
        *,
        binary: str | None = None,
        grpc_port: int,
        health_port: int,
        extra_env: Mapping[str, str | None] | None = None,
        timeout: float = 240,
        kubeconfig: str | os.PathLike[str] | None = None,
        require_readiness: bool = True,
    ):
        resolved_binary = (binary or os.environ.get(_EPP_BINARY_ENV, "")).strip() or (
            _DEFAULT_EPP_BINARY
        )
        if not Path(resolved_binary).exists():
            raise RuntimeError(
                f"Runtime EPP binary {resolved_binary!r} is not present. "
                f"Set {_EPP_BINARY_ENV} or install the standard /epp binary; "
                "this test does not skip on absence."
            )
        if kubeconfig is not None:
            validate_loopback_kubeconfig(kubeconfig)

        env = os.environ.copy()
        for name, value in (extra_env or {}).items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        validate_loopback_kubeconfig(env.get("KUBECONFIG", ""))
        env["DYN_EPP_MODE"] = "dynamo"
        env["DYN_EPP_METRICS_PORT"] = "0"

        super().__init__(
            command=[resolved_binary],
            env=env,
            timeout=timeout,
            display_output=True,
            health_check_ports=[health_port] if require_readiness else [],
            health_check_urls=[],
            health_check_funcs=(
                [lambda remaining=None: epp_loopback_tls_ready(grpc_port, remaining)]
                if require_readiness
                else []
            ),
            log_dir=request.node.name,
            terminate_all_matching_process_names=False,
            display_name="dynamo-runtime-epp",
        )
        self.require_readiness = require_readiness


def observe_owned_initialization_diagnostic(
    process: ManagedProcess,
    *,
    diagnostic: str,
    deadline_s: float,
    node_name: str,
    label: str,
) -> tuple[int, Path]:
    """Watch one already-started ManagedProcess log for an init diagnostic.

    The process must already be entered through the shared ManagedProcess
    lifecycle. This helper does not start, retry, or replace that lifecycle.
    A process may stay alive after a failed router/model construction. The
    only success is the exact diagnostic in a completed newline-terminated
    line before the single deadline. An exited parent may still flush the
    completed log inside that same deadline; a partial last line is not a
    match. Readiness timeout, a missing model, an arbitrary HTTP error, a
    killed process, or any other startup error is a hard failure.
    """
    if not diagnostic:
        raise AssertionError(f"{label} diagnostic must be nonempty")
    if deadline_s <= 0:
        raise AssertionError(f"{label} deadline_s={deadline_s!r} must be positive")
    if process is None:
        raise AssertionError(f"{label} was not saved")
    proc = process.proc
    if proc is None:
        raise AssertionError(f"{label} process was not started")
    log_path = process.log_path
    if not log_path:
        raise AssertionError(f"{label} process has no log_path")
    log = Path(log_path)
    if node_name not in log.as_posix():
        raise AssertionError(f"{label} log_path {log} is not this node {node_name}")
    boundary = re.compile(re.escape(diagnostic) + r"(?![0-9A-Za-z.])")
    deadline = time.monotonic() + deadline_s
    while True:
        pid = proc.pid
        proc.poll()
        raw = log.read_bytes() if log.is_file() else b""
        text = completed_utf8_log_prefix(raw, label=f"{label} pid={pid} log={log}")
        if boundary.search(_ANSI_RE.sub("", text)):
            logger.info(
                "%s pid=%s log=%s initialization diagnostic observed",
                label,
                pid,
                log,
            )
            return pid, log
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"{label} pid={pid} log={log} reached the initialization "
                "deadline without the consumed-key diagnostic "
                f"{diagnostic!r}; this is not an expected readiness timeout, "
                "missing model, HTTP error, or other startup failure"
            )
        remaining = deadline - time.monotonic()
        time.sleep(min(_INIT_OBSERVE_POLL_S, max(remaining, 0.0)))
