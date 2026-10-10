# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability-gated proof that a predecessor can no longer access shared HBM."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUQuiescenceProof:
    quiesced: bool
    provider: str
    detail: str = ""

    elapsed_ms: float = 0.0


def _backend_env(backend_name: str, suffix: str) -> str:
    backend = backend_name.upper().replace("-", "_")
    return f"DYN_{backend}_GMS_{suffix}"


def _configured_command(backend_name: str) -> str | None:
    return os.environ.get(
        _backend_env(backend_name, "GPU_QUIESCENCE_COMMAND")
    ) or os.environ.get("DYN_GMS_GPU_QUIESCENCE_COMMAND")


def gpu_quiescence_provider_configured(backend_name: str) -> bool:
    provider = os.environ.get(
        _backend_env(backend_name, "GPU_QUIESCENCE_PROVIDER"),
        os.environ.get("DYN_GMS_GPU_QUIESCENCE_PROVIDER", ""),
    )
    return provider.strip().lower() == "gms-mps" or bool(
        (_configured_command(backend_name) or "").strip()
    )


def gms_mps_provider_enabled(backend_name: str) -> bool:
    provider = os.environ.get(
        _backend_env(backend_name, "GPU_QUIESCENCE_PROVIDER"),
        os.environ.get("DYN_GMS_GPU_QUIESCENCE_PROVIDER", ""),
    )
    return provider.strip().lower() == "gms-mps"


def gpu_crash_interlock_enabled(backend_name: str) -> bool:
    raw = os.environ.get(
        _backend_env(backend_name, "GPU_CRASH_INTERLOCK"),
        os.environ.get("DYN_GMS_GPU_CRASH_INTERLOCK", "0"),
    )
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _socket_path(backend_name: str, device: int) -> str:
    backend = backend_name.upper().replace("-", "_")
    explicit = os.environ.get(f"GMS_{backend}_VMM_IPC_SOCKET") or os.environ.get(
        "DYN_GMS_PERSISTENT_KV_SOCKET"
    )
    if explicit:
        return explicit
    from gpu_memory_service.common.utils import get_socket_path

    return get_socket_path(device, "kv_cache")


def _process_start_time(pid: int) -> str:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1]
        return fields.split()[19]
    except (FileNotFoundError, PermissionError, IndexError, OSError) as exc:
        raise RuntimeError("cannot determine CUDA worker process identity") from exc


def register_gpu_client(
    *, backend_name: str, device: int, cohort: str, rank: int = 0
) -> int | None:
    """Register this CUDA writer with its persistent GMS daemon.

    Registration is required only for the GMS-MPS provider and happens before
    the engine initializes CUDA. The daemon intentionally retains it after this
    short RPC session disconnects so a successor can identify a crashed cohort.
    """
    if not gms_mps_provider_enabled(backend_name):
        if gpu_crash_interlock_enabled(backend_name):
            raise RuntimeError("GPU crash interlock requires the GMS MPS provider")
        return None
    from gpu_memory_service.client.session import _GMSClientSession
    from gpu_memory_service.common.locks import RequestedLockType

    pid = os.getpid()
    with _GMSClientSession(
        _socket_path(backend_name, device), RequestedLockType.RW_PERSISTENT, 1_000
    ) as session:
        arguments = {
            "backend": backend_name,
            "cohort": cohort,
            "client_pid": pid,
            "process_start_time": _process_start_time(pid),
            "rank": rank,
        }
        if gpu_crash_interlock_enabled(backend_name):
            return session.register_gpu_client_with_crash_interlock(**arguments)
        if not session.register_gpu_client(**arguments):
            raise RuntimeError("GMS rejected CUDA worker registration")
    return None


def arm_gpu_crash_interlock(notification_fd: int | None, *, backend_name: str) -> None:
    """Install the native one-shot handler after CUDA/MPS initialization.

    Ownership of the notification FD transfers to the native extension. The
    handler performs only async-signal-safe syscalls: one fixed-size pipe write,
    SIGSTOP, and pause. GMS proves MPS client termination before it SIGKILLs
    the stopped host process.
    """
    if notification_fd is None:
        return
    try:
        import signal

        import gms_rust_ring  # type: ignore[import-not-found]

        signals = [
            getattr(signal, name)
            for name in (
                "SIGSEGV",
                "SIGABRT",
                "SIGBUS",
                "SIGILL",
                "SIGFPE",
            )
            if hasattr(signal, name)
        ]
        gms_rust_ring.install_gpu_crash_interlock(notification_fd, signals)
        logger.info("Armed native GMS GPU crash interlock for %s", backend_name)
    except Exception:
        os.close(notification_fd)
        raise


def _timeout_s(backend_name: str) -> float:
    raw = os.environ.get(
        _backend_env(backend_name, "GPU_QUIESCENCE_TIMEOUT_SECS"),
        os.environ.get("DYN_GMS_GPU_QUIESCENCE_TIMEOUT_SECS", "2.0"),
    )
    try:
        value = float(raw)
        return max(0.05, value) if math.isfinite(value) else 1.0
    except ValueError:
        logger.warning("Ignoring invalid GPU quiescence timeout %r", raw)
        return 1.0


def prove_predecessor_gpu_quiescence_sync(
    *,
    backend_name: str,
    predecessor_cohort: str | None,
    device: int = 0,
    terminate_host: bool = False,
) -> GPUQuiescenceProof:
    """Synchronously request the GMS-owned proof for one local GPU pool.

    A None predecessor means every registered cohort other than the current
    successor. That form is deliberately scoped by the per-device KV GMS
    socket and is used by each TP worker immediately before remapping its pool.
    """
    if not gms_mps_provider_enabled(backend_name):
        return GPUQuiescenceProof(False, "quarantine-only", "GMS MPS is disabled")
    cohort_env = f"GMS_{backend_name.upper().replace('-', '_')}_WRITER_COHORT_PATH"
    successor_cohort = os.environ.get(cohort_env, "").strip()
    if not successor_cohort:
        raise RuntimeError("current writer cohort is unavailable")

    from gpu_memory_service.client.session import _GMSClientSession
    from gpu_memory_service.common.locks import RequestedLockType

    with _GMSClientSession(
        _socket_path(backend_name, device), RequestedLockType.RW_PERSISTENT, 1_000
    ) as session:
        response = session.quiesce_gpu_cohort(
            backend=backend_name,
            predecessor_cohort=predecessor_cohort,
            successor_cohort=successor_cohort,
            terminate_host=terminate_host,
        )
    return GPUQuiescenceProof(
        response.quiesced,
        response.provider,
        response.detail,
        response.elapsed_ms,
    )


def wait_for_predecessor_gpu_quiescence_sync(
    *,
    backend_name: str,
    predecessor_cohort: str | None,
    device: int = 0,
    timeout_s: float | None = None,
    retry_interval_s: float = 0.01,
    terminate_host: bool = False,
) -> GPUQuiescenceProof:
    """Wait for an authoritative local proof before shared-HBM remapping.

    TP ranks do not finish CUDA-context teardown simultaneously. A successor
    may therefore reach its local remap while MPS is still retiring the old
    client. Retrying the *proof* is safe; proceeding after elapsed time is not.
    The final rejected proof is returned at the deadline so callers can fail
    closed with the provider's diagnostic.
    """

    timeout = _timeout_s(backend_name) if timeout_s is None else timeout_s
    timeout = max(0.0, float(timeout))
    interval = max(0.001, float(retry_interval_s))
    deadline = time.monotonic() + timeout
    last = prove_predecessor_gpu_quiescence_sync(
        backend_name=backend_name,
        predecessor_cohort=predecessor_cohort,
        device=device,
        terminate_host=terminate_host,
    )
    while not last.quiesced and time.monotonic() < deadline:
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
        last = prove_predecessor_gpu_quiescence_sync(
            backend_name=backend_name,
            predecessor_cohort=predecessor_cohort,
            device=device,
            terminate_host=terminate_host,
        )
    return last


def terminate_current_gpu_cohort_sync(
    *, backend_name: str, device: int = 0
) -> GPUQuiescenceProof:
    """Prove and retire this process tree's local CUDA cohort.

    A surviving TP leader calls this when another rank is lost, while its own
    CUDA worker is still alive and visible to MPS.  Waiting for ordinary vLLM
    teardown can make MPS forget the client before GMS obtains an authoritative
    ``terminate_client`` result, which correctly forces the successor cold.
    """
    if not gms_mps_provider_enabled(backend_name):
        return GPUQuiescenceProof(False, "quarantine-only", "GMS MPS is disabled")
    cohort_env = f"GMS_{backend_name.upper().replace('-', '_')}_WRITER_COHORT_PATH"
    current_cohort = os.environ.get(cohort_env, "").strip()
    if not current_cohort:
        raise RuntimeError("current writer cohort is unavailable")

    from gpu_memory_service.client.session import _GMSClientSession
    from gpu_memory_service.common.locks import RequestedLockType

    with _GMSClientSession(
        _socket_path(backend_name, device), RequestedLockType.RW_PERSISTENT, 1_000
    ) as session:
        response = session.quiesce_gpu_cohort(
            backend=backend_name,
            predecessor_cohort=current_cohort,
            # This identity is never registered. It only makes the requested
            # predecessor unambiguous to the existing protocol.
            successor_cohort=f"{current_cohort}.rank-loss-successor",
            terminate_host=True,
        )
    return GPUQuiescenceProof(
        response.quiesced,
        response.provider,
        response.detail,
        response.elapsed_ms,
    )


async def prove_predecessor_gpu_quiescence(
    *, backend_name: str, predecessor_cohort: str | None, device: int = 0
) -> GPUQuiescenceProof:
    """Request the configured platform proof without delaying safe serving.

    No command means quarantine-only recovery. A provider must exit zero only
    after the predecessor CUDA context is unable to issue or complete accesses
    to the shared allocation. Process death, heartbeats, traffic cessation and
    elapsed time are explicitly insufficient evidence.

    ``gms-mps`` delegates exact-cohort termination to the persistent GMS daemon.
    The legacy external-command provider remains for deployment-specific context
    authorities; it splits arguments with :func:`shlex.split` and substitutes
    ``{backend}`` and ``{cohort}`` per argument.
    """
    if gms_mps_provider_enabled(backend_name):
        return await asyncio.to_thread(
            prove_predecessor_gpu_quiescence_sync,
            backend_name=backend_name,
            predecessor_cohort=predecessor_cohort,
            device=device,
        )

    command = _configured_command(backend_name)
    if not command:
        return GPUQuiescenceProof(False, "quarantine-only", "no provider configured")
    try:
        argv = [
            arg.format(
                backend=backend_name,
                cohort=predecessor_cohort or "",
            )
            for arg in shlex.split(command)
        ]
    except (ValueError, KeyError) as exc:
        raise RuntimeError("invalid GPU quiescence command") from exc
    if not argv:
        raise RuntimeError("GPU quiescence command is empty")

    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=_timeout_s(backend_name)
        )
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise RuntimeError("GPU quiescence provider timed out") from exc
    detail = (stdout or stderr).decode("utf-8", errors="replace").strip()[:512]
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if process.returncode != 0:
        logger.warning(
            "GPU quiescence provider rejected reclaim backend=%s rc=%d detail=%s",
            backend_name,
            process.returncode,
            detail,
        )
        return GPUQuiescenceProof(False, "external-command", detail, elapsed_ms)
    return GPUQuiescenceProof(True, "external-command", detail, elapsed_ms)
