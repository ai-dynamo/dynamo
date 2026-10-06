# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability-gated proof that a predecessor can no longer access shared HBM."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import shlex
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_memory_service.common.utils import is_truthy_env

logger = logging.getLogger(__name__)

# Reuse the registration RPC connection for takeover proof. At TP16, opening
# fresh GMS sessions after a crash adds avoidable handshake latency.
_proof_session_lock = threading.Lock()
_proof_sessions: dict[tuple[int, str], Any] = {}


def _replace_proof_session(socket_path: str, session: Any | None) -> None:
    key = (os.getpid(), socket_path)
    previous = _proof_sessions.pop(key, None)
    if previous is not None:
        previous.close()
    if session is not None:
        _proof_sessions[key] = session


def _discard_inherited_proof_sessions() -> None:
    global _proof_session_lock
    _proof_session_lock = threading.Lock()
    for session in _proof_sessions.values():
        # Avoid session.close() here: its logger may hold a parent-thread lock
        # at fork. Only the inherited socket needs closing in this child.
        transport = getattr(session, "_transport", None)
        if transport is not None:
            with contextlib.suppress(OSError):
                transport.close()
    _proof_sessions.clear()


os.register_at_fork(after_in_child=_discard_inherited_proof_sessions)


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


def _provider_name(backend_name: str) -> str:
    configured = (
        os.environ.get(
            _backend_env(backend_name, "GPU_QUIESCENCE_PROVIDER"),
            os.environ.get("DYN_GMS_GPU_QUIESCENCE_PROVIDER", ""),
        )
        .strip()
        .lower()
    )
    if configured:
        return configured
    from gpu_memory_service.common.gpu_isolation import default_quiescence_provider

    mode_default = default_quiescence_provider()
    if mode_default is not None:
        return mode_default
    return (
        "external-command" if _configured_command(backend_name) else "quarantine-only"
    )


def _process_lifetime_enabled() -> bool:
    """The process-lifetime provider is opt-in: the isolation mode or the flag."""
    from gpu_memory_service.common.gpu_isolation import gpu_isolation_mode

    return (
        gpu_isolation_mode() == "process"
        or os.environ.get("DYN_GMS_EXPERIMENTAL_PROCESS_LIFETIME_RECLAIM") == "1"
    )


def mps_client_possible() -> bool:
    return _mps_client_possible()


def _mps_client_possible() -> bool:
    # CUDA clients also use this default directory when no explicit pipe
    # directory is exported. A false positive is safer than reclaiming a live
    # MPS server's memory after its client process has disappeared.
    return (
        bool(
            os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
            or os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
            or os.environ.get("DYN_GMS_MPS_PIPE_DIRECTORY")
        )
        or Path("/tmp/nvidia-mps/control").exists()
    )


def gpu_quiescence_provider_configured(backend_name: str) -> bool:
    provider = _provider_name(backend_name)
    return (
        (
            provider == "process-lifetime"
            and _process_lifetime_enabled()
            and not _mps_client_possible()
        )
        or provider == "gms-mps"
        or bool((_configured_command(backend_name) or "").strip())
    )


def gms_mps_provider_enabled(backend_name: str) -> bool:
    return _provider_name(backend_name) == "gms-mps"


# Held (shared) for the engine's lifetime once it has claimed its MPS domain.
_mps_domain_claim_fd: int | None = None


def _claim_mps_domain_after_dead_engine(pipe: str, backend_name: str) -> bool:
    """Claim this MPS domain for the engine; True only for a dead predecessor.

    A domain's first engine finds no claim file. A live engine holds the
    claim shared, so another starter cannot take it exclusively. Only an
    engine restarted after its predecessor exited finds a stale claim. A new
    pod gets a fresh domain directory and therefore a first start.
    """
    import fcntl

    global _mps_domain_claim_fd
    if _mps_domain_claim_fd is not None:
        return False
    path = os.path.join(pipe, f".dynamo-{backend_name}-engine.claim")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return False
    restarted = os.pread(fd, 1, 0) == b"C"
    if not restarted and os.pwrite(fd, b"C", 0) != 1:
        os.close(fd)
        raise OSError(f"could not write MPS domain claim {path}")
    fcntl.flock(fd, fcntl.LOCK_SH)
    _mps_domain_claim_fd = fd
    return restarted


def recycle_idle_mps_servers(backend_name: str) -> int:
    """Shut down a dead engine's leftover MPS server before CUDA init.

    After an MPS client is killed, its server can stay up in a faulted state
    with no clients, and the control daemon hands every new client to it, so a
    restarted engine fails its first CUDA call (cudaErrorDevicesUnavailable).

    The client list is not a liveness proof: a client that is still inside
    cuInit is attached to the server but not yet listed, and shutting the
    server down then fails that cuInit with CUDA_ERROR_NO_DEVICE. So this only
    acts when an engine restarts after its predecessor in the same domain died,
    never on a first start (when GMS servers attach concurrently), and only on
    a server that stays client-less across a settle interval.
    """
    import subprocess

    pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if not pipe or not gms_mps_provider_enabled(backend_name):
        return 0
    try:
        if not _claim_mps_domain_after_dead_engine(pipe, backend_name):
            return 0
    except OSError:
        logger.warning(
            "[GMS MPS] could not claim MPS domain %s; not recycling servers",
            pipe,
            exc_info=True,
        )
        return 0
    binary = os.environ.get("DYN_GMS_MPS_CONTROL_BINARY", "nvidia-cuda-mps-control")
    env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe}

    def control(command: str) -> list[str]:
        result = subprocess.run(
            [binary],
            input=command + "\n",
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return [token for token in result.stdout.split() if token.isdigit()]

    # A restarted engine can race its predecessor's last clients detaching.
    wait = float(os.environ.get("DYN_GMS_MPS_RECYCLE_WAIT_SECS", "10"))
    settle = float(os.environ.get("DYN_GMS_MPS_RECYCLE_SETTLE_SECS", "1"))
    deadline = time.monotonic() + max(0.0, wait)
    idle_since: dict[str, float] = {}
    recycled: set[str] = set()
    try:
        while True:
            busy = False
            now = time.monotonic()
            for server in control("get_server_list"):
                if server in recycled:
                    busy = True  # still shutting down
                elif control(f"get_client_list {server}"):
                    idle_since.pop(server, None)
                    busy = True
                elif now - idle_since.setdefault(server, now) < settle:
                    busy = True
                else:
                    logger.warning(
                        "[GMS MPS] recycling client-less MPS server %s in %s "
                        "before CUDA init",
                        server,
                        pipe,
                    )
                    control(f"shutdown_server {server} -f")
                    recycled.add(server)
                    busy = True
            if not busy or time.monotonic() >= deadline:
                if busy:
                    logger.warning(
                        "[GMS MPS] MPS servers in %s still have clients after %.0fs; "
                        "starting without recycling them",
                        pipe,
                        wait,
                    )
                return len(recycled)
            time.sleep(0.2)
    except (OSError, subprocess.SubprocessError):
        logger.warning(
            "[GMS MPS] could not inspect MPS servers in %s", pipe, exc_info=True
        )
        return 0


def _prove_process_lifetime(predecessor_cohort: str | None) -> GPUQuiescenceProof:
    """Experimental process-lifetime assumption for non-MPS trials only.

    The writer guard includes every CUDA worker and cannot be retired while a
    registered process remains alive. This is a CPU/process fence, *not* a
    documented CUDA-stream quiescence proof. It must never be selected by
    default or silently enabled in a production deployment. MPS is excluded
    because its server owns a context independently of the client process.
    """
    if not _process_lifetime_enabled():
        return GPUQuiescenceProof(
            False,
            "process-lifetime",
            "process-lifetime reclaim was not enabled "
            "(set DYN_GMS_GPU_ISOLATION=process)",
        )
    if _mps_client_possible():
        return GPUQuiescenceProof(
            False,
            "process-lifetime",
            "MPS may be active; select the gms-mps provider instead",
        )
    if predecessor_cohort is None:
        return GPUQuiescenceProof(
            False, "process-lifetime", "predecessor cohort identity is missing"
        )
    from gpu_memory_service.integrations.common.process_lifecycle import (
        retired_writer_cohort_has_no_processes,
    )

    try:
        retired = retired_writer_cohort_has_no_processes(Path(predecessor_cohort))
    except (OSError, ValueError) as exc:
        return GPUQuiescenceProof(False, "process-lifetime", str(exc))
    return GPUQuiescenceProof(
        retired,
        "process-lifetime",
        (
            "all predecessor writer guards exited and cohort is tombstoned"
            if retired
            else "predecessor writer cohort is still live or not retired"
        ),
    )


def gpu_crash_interlock_enabled(backend_name: str) -> bool:
    from gpu_memory_service.common.gpu_isolation import default_crash_interlock

    raw = os.environ.get(
        _backend_env(backend_name, "GPU_CRASH_INTERLOCK"),
        os.environ.get("DYN_GMS_GPU_CRASH_INTERLOCK", default_crash_interlock()),
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


def preconnect_gpu_quiescence_session(*, backend_name: str, device: int = 0) -> None:
    """Open the shadow controller's proof channel before waiting on its lock.

    The parent controller has no CUDA client to register, unlike each worker.
    Opening this independent RW_PERSISTENT session while the primary serves
    avoids a contended GMS handshake after the crash. It grants no KV access;
    the explicit MPS proof is still required after the ownership lock.
    """
    if not gms_mps_provider_enabled(backend_name):
        return
    from gpu_memory_service.client.session import _GMSClientSession
    from gpu_memory_service.common.locks import RequestedLockType

    socket_path = _socket_path(backend_name, device)
    with _proof_session_lock:
        if (os.getpid(), socket_path) not in _proof_sessions:
            session = _GMSClientSession(
                socket_path, RequestedLockType.RW_PERSISTENT, 1_000
            )
            _replace_proof_session(socket_path, session)


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
    socket_path = _socket_path(backend_name, device)
    session = _GMSClientSession(socket_path, RequestedLockType.RW_PERSISTENT, 1_000)
    try:
        arguments = {
            "backend": backend_name,
            "cohort": cohort,
            "client_pid": pid,
            "process_start_time": _process_start_time(pid),
            "rank": rank,
            "failure_notify_addr": os.environ.get(
                "DYN_GMS_RANK_LIVENESS_CONNECT_ADDR", ""
            ),
            "mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY", ""),
        }
        if gpu_crash_interlock_enabled(backend_name):
            notification_fd = session.register_gpu_client_with_crash_interlock(
                **arguments
            )
        else:
            if not session.register_gpu_client(**arguments):
                raise RuntimeError("GMS rejected CUDA worker registration")
            notification_fd = None
        with _proof_session_lock:
            _replace_proof_session(socket_path, session)
        return notification_fd
    except BaseException:
        session.close()
        raise


def arm_gpu_crash_interlock(notification_fd: int | None, *, backend_name: str) -> None:
    """Install the native one-shot handler after CUDA/MPS initialization.

    Ownership of the notification FD transfers to the native extension. The
    handler performs only async-signal-safe syscalls: one fixed-size socket
    write and parking the reporting thread. GMS proves MPS client termination
    before it terminates the host process.
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
    start_gpu_fault_watchdog(backend_name)


def _gpu_fault_watchdog_interval_s() -> float:
    raw = os.environ.get("DYN_GMS_GPU_FAULT_WATCHDOG_MS", "100")
    try:
        value = float(raw)
    except ValueError:
        value = 100.0
    return max(0.0, value) / 1000.0 if math.isfinite(value) else 0.1


def _relax_thread_stream_capture_mode() -> None:
    """Keep this thread's CUDA calls out of other threads' graph captures.

    Engines capture CUDA graphs in global mode, where an unsafe call from any
    thread is rejected and can invalidate the capture. Relaxed mode applies
    only to the calling thread, so the watchdog's stream queries neither fail
    nor disturb an engine capture.
    """
    import ctypes

    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        mode = ctypes.c_int(2)  # CU_STREAM_CAPTURE_MODE_RELAXED
        result = libcuda.cuThreadExchangeStreamCaptureMode(ctypes.byref(mode))
        if result != 0:
            logger.warning(
                "[GMS] could not relax stream-capture mode for the fault "
                "watchdog (CUresult %d)",
                result,
            )
    except OSError:
        logger.warning("[GMS] libcuda unavailable for the fault watchdog")


def start_gpu_fault_watchdog(backend_name: str, on_fault=None) -> bool:
    """Turn a sticky CUDA context error into a fail-stop crash.

    A GPU fault (for example an illegal address) poisons the CUDA context.
    Under MPS, the server also marks the device unavailable to every client
    it hosts. It raises no signal and kills no process. The worker can then
    sit idle while its TP peers wait on it, and neither the crash interlock
    nor rank heartbeats notice. Poll an idle private stream, which surfaces
    the sticky error in about 2 microseconds per query. On a fault, kill the
    worker. Its interlock socket closes, which reports the crash through the
    normal takeover path. DYN_GMS_GPU_FAULT_WATCHDOG_MS=0 disables the
    watchdog.
    """
    interval = _gpu_fault_watchdog_interval_s()
    if interval <= 0:
        return False
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return False
    device = torch.cuda.current_device()
    stream = torch.cuda.Stream(device=device)

    def fail_stop(exc: BaseException) -> None:
        import signal

        logger.critical(
            "[GMS] %s CUDA context faulted (%s); killing this worker",
            backend_name,
            str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
        )
        # Not SIGABRT: the interlock would park this process so GMS can certify
        # MPS termination, which a faulted context can never provide (MPS
        # answers 700, 806, then 201). The parked process only delayed the
        # writer fence by about 0.6 s at TP16. Process death closes the
        # interlock socket, which reports the crash just as fast.
        os.kill(os.getpid(), signal.SIGKILL)

    handler = on_fault or fail_stop

    def watch() -> None:
        torch.cuda.set_device(device)
        _relax_thread_stream_capture_mode()
        failures = 0
        while True:
            time.sleep(interval)
            try:
                stream.query()
                failures = 0
            except Exception as exc:  # noqa: BLE001
                # A context fault is sticky. Require it twice so a transient
                # error never stops a healthy worker.
                failures += 1
                if failures >= 2:
                    handler(exc)
                    return

    threading.Thread(target=watch, name="gms-gpu-fault-watchdog", daemon=True).start()
    logger.info(
        "Started GMS GPU fault watchdog for %s (every %.0f ms)",
        backend_name,
        interval * 1000,
    )
    return True


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
    if _provider_name(backend_name) == "process-lifetime":
        return _prove_process_lifetime(predecessor_cohort)
    if not gms_mps_provider_enabled(backend_name):
        return GPUQuiescenceProof(False, "quarantine-only", "GMS MPS is disabled")
    cohort_env = f"GMS_{backend_name.upper().replace('-', '_')}_WRITER_COHORT_PATH"
    successor_cohort = os.environ.get(cohort_env, "").strip()
    if not successor_cohort:
        raise RuntimeError("current writer cohort is unavailable")

    from gpu_memory_service.client.session import _GMSClientSession
    from gpu_memory_service.common.locks import RequestedLockType

    started = time.monotonic()
    socket_path = _socket_path(backend_name, device)
    with _proof_session_lock:
        session = _proof_sessions.get((os.getpid(), socket_path))
        if session is None:
            session = _GMSClientSession(
                socket_path, RequestedLockType.RW_PERSISTENT, 1_000
            )
            _replace_proof_session(socket_path, session)
        connected = time.monotonic()
        try:
            response = session.quiesce_gpu_cohort(
                backend=backend_name,
                predecessor_cohort=predecessor_cohort,
                successor_cohort=successor_cohort,
                terminate_host=terminate_host,
            )
        except BaseException:
            _replace_proof_session(socket_path, None)
            raise
    if os.environ.get("DYN_GMS_GPU_QUIESCENCE_TIMING") == "1":
        logger.info(
            "GPU proof RPC timing backend=%s session_ms=%.2f request_ms=%.2f",
            backend_name,
            (connected - started) * 1000,
            (time.monotonic() - connected) * 1000,
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
        timeout_kwarg = (
            {
                # The peer-rank handler kills its process tree after this RPC.
                # A one-second response deadline could kill the CUDA client
                # while GMS's configured MPS command was still running (1.5s
                # for vLLM), losing the only chance for CUDA_SUCCESS proof.
                # Leave a small allowance for the inventory check and reply.
                "response_timeout_ms": math.ceil(
                    (_timeout_s(backend_name) + 0.5) * 1_000
                )
            }
            if is_truthy_env("DYN_GMS_FAILOVER_FROZEN_PREDECESSOR")
            else {}
        )
        response = session.quiesce_gpu_cohort(
            backend=backend_name,
            predecessor_cohort=current_cohort,
            # This identity is never registered. It only makes the requested
            # predecessor unambiguous to the existing protocol.
            successor_cohort=f"{current_cohort}.rank-loss-successor",
            terminate_host=True,
            # A stalled GMS/MPS response must not hold the writer lock forever.
            # Frozen recovery continues with old GPU pages quarantined.
            **timeout_kwarg,
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
    if _provider_name(backend_name) == "process-lifetime":
        return await asyncio.to_thread(_prove_process_lifetime, predecessor_cohort)
    if gms_mps_provider_enabled(backend_name):
        queued = time.monotonic()

        def prove() -> GPUQuiescenceProof:
            started = time.monotonic()
            result = prove_predecessor_gpu_quiescence_sync(
                backend_name=backend_name,
                predecessor_cohort=predecessor_cohort,
                device=device,
            )
            if os.environ.get("DYN_GMS_GPU_QUIESCENCE_TIMING") == "1":
                logger.info(
                    "GPU proof dispatch timing backend=%s executor_wait_ms=%.2f total_ms=%.2f",
                    backend_name,
                    (started - queued) * 1000,
                    (time.monotonic() - queued) * 1000,
                )
            return result

        return await asyncio.to_thread(prove)

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
