# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Daemon-owned CUDA MPS client termination for shared-KV recovery."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import signal
import struct
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from gpu_memory_service.common.gpu_failure_marker import publish_gpu_failure_marker

logger = logging.getLogger(__name__)

_CRASH_MAGIC = 0x47534D43  # "GSMC": GMS crash notification.
_CRASH_VERSION = 1
_CRASH_RECORD = struct.Struct("=IIii")


def process_start_time(pid: int) -> str | None:
    """Return Linux's process birth identity, or ``None`` outside its namespace."""
    try:
        # comm may contain spaces and parentheses. Fields following the final ')'
        # start at proc field 3; starttime is field 22, hence index 19 here.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1]
        return fields.split()[19]
    except (FileNotFoundError, PermissionError, IndexError, OSError):
        return None


def process_state(pid: int) -> str | None:
    """Return the single-letter Linux process state for ``pid``."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, PermissionError, IndexError, OSError):
        return None


@dataclass(frozen=True)
class GPUClient:
    backend: str
    cohort: str
    pid: int
    process_start_time: str
    rank: int
    failure_notify_addr: str = ""


def signal_client(client: GPUClient, sig: int) -> None:
    """Signal a birth-checked process without a check-to-kill PID reuse race."""
    fd = os.pidfd_open(client.pid)
    try:
        if process_start_time(client.pid) != client.process_start_time:
            raise ProcessLookupError("GPU client exited or its PID was reused")
        signal.pidfd_send_signal(fd, sig)
    finally:
        os.close(fd)


def _inventory_pids(output: str) -> set[str]:
    """Reject diagnostics rather than interpreting them as an empty inventory."""
    lines = output.split()
    if any(not re.fullmatch(r"[1-9][0-9]*", item) for item in lines):
        raise RuntimeError(f"Invalid MPS PID inventory: {output[:512]}")
    return set(lines)


@dataclass(frozen=True)
class QuiescenceResult:
    quiesced: bool
    provider: str
    client_count: int
    detail: str
    elapsed_ms: float


class GPUQuiescenceManager:
    """Tracks CUDA writers and terminates one exact cohort through MPS.

    Registrations intentionally outlive RPC sessions: the interesting client is
    normally already disconnected or dead when its successor requests recovery.
    A cohort UUID is never reused, and a successful result is therefore safe to
    cache for idempotent takeover retries.
    """

    def __init__(self) -> None:
        self._clients: dict[tuple[str, str, int], GPUClient] = {}
        self._proofs: dict[tuple[str, str], QuiescenceResult] = {}
        self._retired: set[tuple[str, str]] = set()
        # Partial multi-client termination is retryable, but only against the
        # same MPS server. A restarted server must never inherit old proof.
        self._terminated_clients: set[tuple[str, str, int, str, str]] = set()
        self._crash_tasks: dict[tuple[str, str, int], asyncio.Task[None]] = {}
        self._crash_read_fds: dict[tuple[str, str, int], int] = {}
        self._lock = asyncio.Lock()
        self._crashed: set[tuple[str, str, int]] = set()
        self._interlock_eof: set[tuple[str, str, int]] = set()
        self._failure_notifiers: dict[str, object] = {}

    def register(
        self,
        *,
        backend: str,
        cohort: str,
        pid: int,
        process_start_time_value: str,
        rank: int,
        failure_notify_addr: str = "",
        crash_interlock: bool = False,
    ) -> int:
        backend = backend.strip().lower()
        cohort = cohort.strip()
        if not backend or not cohort:
            raise ValueError("GPU client backend and cohort must be non-empty")
        if (backend, cohort) in self._retired:
            raise ValueError("GPU client cohort is already retired")
        if pid <= 0 or rank < 0:
            raise ValueError("GPU client pid and rank must be non-negative")
        if not process_start_time_value:
            raise ValueError("GPU client process start time must be non-empty")
        observed = process_start_time(pid)
        if observed is None:
            raise ValueError(
                "GPU client is outside the GMS PID namespace; deploy GMS and "
                "engine workers in a shared PID namespace"
            )
        if observed != process_start_time_value:
            raise ValueError("GPU client PID birth identity does not match")
        for registered in self._clients.values():
            if (
                registered.backend == backend
                and registered.pid == pid
                and registered.process_start_time == process_start_time_value
                and registered.cohort != cohort
            ):
                raise ValueError("live GPU client cannot join multiple cohorts")
        key = (backend, cohort, pid)
        client = GPUClient(
            backend=backend,
            cohort=cohort,
            pid=pid,
            process_start_time=process_start_time_value,
            rank=rank,
            failure_notify_addr=failure_notify_addr.strip(),
        )
        previous = self._clients.get(key)
        if previous is not None and previous != client:
            raise ValueError(
                "GPU client PID was already registered with other metadata"
            )
        if crash_interlock and not self.configured(backend):
            raise ValueError(
                "GPU crash interlock requires DYN_GMS_GPU_QUIESCENCE_PROVIDER=gms-mps"
            )
        if crash_interlock:
            self._ensure_failure_notifier(client.failure_notify_addr)
        if crash_interlock and not (
            hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
        ):
            raise ValueError("GPU crash interlock requires Linux pidfd signaling")
        if crash_interlock and key in self._crash_tasks:
            raise ValueError("GPU client crash interlock is already armed")
        if not crash_interlock:
            self._clients[key] = client
            # A late registration invalidates a cached proof. The writer-cohort
            # flock prevents this once takeover fencing has completed, but keep
            # this component independently fail-closed.
            self._proofs.pop((backend, cohort), None)
            return -1

        read_fd, write_fd = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
        watcher = self._watch_crash_interlock(client, read_fd)
        try:
            task = asyncio.create_task(
                watcher,
                name=f"gms-crash-interlock-{backend}-{rank}-{pid}",
            )
        except Exception:
            watcher.close()
            os.close(read_fd)
            os.close(write_fd)
            raise
        self._clients[key] = client
        self._proofs.pop((backend, cohort), None)
        self._crash_tasks[key] = task
        self._crash_read_fds[key] = read_fd

        def finish(completed: asyncio.Task[None]) -> None:
            self._crash_tasks.pop(key, None)
            self._crash_read_fds.pop(key, None)
            # Also runs if cancellation happened before the coroutine started.
            with suppress(OSError):
                os.close(read_fd)
            try:
                completed.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "GPU crash interlock task failed backend=%s cohort=%s pid=%d",
                    backend,
                    cohort,
                    pid,
                )

        task.add_done_callback(finish)
        return write_fd

    def _ensure_failure_notifier(self, address: str) -> None:
        """Connect a persistent crash-hint socket while the client is healthy."""

        if not address or address in self._failure_notifiers:
            return
        socket = None
        try:
            import zmq

            socket = zmq.Context.instance().socket(zmq.DEALER)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDHWM, 16)
            identity = (
                f"gms-crash-{os.uname().nodename}-{os.getpid()}-"
                f"{len(self._failure_notifiers)}"
            )
            socket.setsockopt(zmq.IDENTITY, identity.encode())
            socket.connect(address)
        except Exception:
            logger.exception(
                "Failed to connect GPU crash notifier address=%s; "
                "shared marker remains available",
                address,
            )
            with suppress(Exception):
                if socket is not None:
                    socket.close(0)
            return
        self._failure_notifiers[address] = socket
        logger.info("Connected GPU crash notifier address=%s", address)

    def _notify_gpu_failure(self, client: GPUClient, source: str) -> None:
        """Send a best-effort hint; this never constitutes fencing proof."""

        socket = self._failure_notifiers.get(client.failure_notify_addr)
        if socket is None:
            return
        try:
            import zmq

            socket.send_multipart(
                [
                    b"gpu-failed-v1",
                    client.cohort.encode(),
                    str(client.rank).encode(),
                    str(client.pid).encode(),
                    source.encode(),
                ],
                flags=zmq.NOBLOCK,
            )
        except Exception:
            logger.exception(
                "Failed to send GPU crash notification backend=%s cohort=%s "
                "rank=%d pid=%d; shared marker remains available",
                client.backend,
                client.cohort,
                client.rank,
                client.pid,
            )

    async def _watch_crash_interlock(self, client: GPUClient, read_fd: int) -> None:
        """Quiesce a cohort when its native handler reports a catchable crash.

        EOF is also actionable: it means the registered process closed its last
        write descriptor. MPS may still know the exiting client, so the daemon
        makes one strict termination attempt. Failure never becomes proof.
        """
        loop = asyncio.get_running_loop()
        readable: asyncio.Future[bytes] = loop.create_future()

        def read_record() -> None:
            try:
                data = os.read(read_fd, _CRASH_RECORD.size)
            except BlockingIOError:
                return
            except OSError as exc:
                if not readable.done():
                    readable.set_exception(exc)
                return
            if not readable.done():
                readable.set_result(data)

        loop.add_reader(read_fd, read_record)
        try:
            data = await readable
        finally:
            loop.remove_reader(read_fd)

        signal_number: int | None = None
        native_record = bool(data)
        source = "pipe-eof"
        if data:
            if len(data) != _CRASH_RECORD.size:
                logger.error(
                    "Ignoring malformed GPU crash interlock record pid=%d bytes=%d",
                    client.pid,
                    len(data),
                )
                return
            magic, version, signal_number, reporting_pid = _CRASH_RECORD.unpack(data)
            if (
                magic != _CRASH_MAGIC
                or version != _CRASH_VERSION
                or reporting_pid != client.pid
            ):
                logger.error(
                    "Ignoring invalid GPU crash interlock record expected_pid=%d "
                    "reported_pid=%d magic=%#x version=%d",
                    client.pid,
                    reporting_pid,
                    magic,
                    version,
                )
                return
            source = f"signal-{signal_number}" if signal_number else "process-exit"

        logger.critical(
            "GPU crash interlock fired backend=%s cohort=%s rank=%d pid=%d source=%s",
            client.backend,
            client.cohort,
            client.rank,
            client.pid,
            source,
        )
        self._notify_gpu_failure(client, source)
        # Wake the leader immediately. This marker is deliberately only a
        # latency hint: takeover still waits for writer-cohort retirement and
        # authoritative ring/CUDA recovery before granting writable ownership.
        try:
            publish_gpu_failure_marker(
                client.cohort,
                rank=client.rank,
                pid=client.pid,
                source=source,
            )
        except (OSError, ValueError):
            logger.exception(
                "Failed to publish GPU crash marker backend=%s cohort=%s pid=%d",
                client.backend,
                client.cohort,
                client.pid,
            )
        if not native_record:
            self._interlock_eof.add((client.backend, client.cohort, client.pid))
        async with self._lock:
            key = (client.backend, client.cohort)
            # A successor may have completed recovery before this notification.
            if key in self._proofs:
                return
            self._retired.add(key)
            if native_record:
                self._crashed.add((client.backend, client.cohort, client.pid))
                # Only the reporting process ran the native handler. Other
                # local TP ranks can still be live; terminate them through MPS.
                if not await self._wait_native_stop(client):
                    logger.critical("GPU crash client did not stop pid=%d", client.pid)
                    return
            result = await self._quiesce_locked(
                backend=client.backend,
                predecessor_cohort=client.cohort,
                terminate_host=True,
            )
        if not result.quiesced:
            logger.critical(
                "GPU crash interlock failed closed backend=%s cohort=%s pid=%d "
                "detail=%s; process remains stopped when the native handler fired",
                client.backend,
                client.cohort,
                client.pid,
                result.detail,
            )
            return

        logger.info(
            "GPU crash interlock completed backend=%s cohort=%s clients=%d "
            "elapsed_ms=%.2f",
            client.backend,
            client.cohort,
            result.client_count,
            result.elapsed_ms,
        )

    async def _wait_native_stop(self, client: GPUClient) -> bool:
        deadline = time.monotonic() + self._timeout(client.backend)
        while True:
            if process_start_time(client.pid) != client.process_start_time:
                return False
            if process_state(client.pid) in {"T", "t"}:
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(min(0.001, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def configured(backend: str) -> bool:
        backend_env = backend.upper().replace("-", "_")
        provider = os.environ.get(
            f"DYN_{backend_env}_GMS_GPU_QUIESCENCE_PROVIDER",
            os.environ.get("DYN_GMS_GPU_QUIESCENCE_PROVIDER", ""),
        )
        return provider.strip().lower() == "gms-mps"

    @staticmethod
    def _timeout(backend: str) -> float:
        backend_env = backend.upper().replace("-", "_")
        raw = os.environ.get(
            f"DYN_{backend_env}_GMS_GPU_QUIESCENCE_TIMEOUT_SECS",
            os.environ.get("DYN_GMS_GPU_QUIESCENCE_TIMEOUT_SECS", "2.0"),
        )
        try:
            value = float(raw)
            return max(0.05, value) if math.isfinite(value) else 1.0
        except ValueError:
            return 1.0

    @staticmethod
    def _allow_survivor_inventory_retirement(backend: str) -> bool:
        """Allow cooperative TP survivors to retire without terminate_client.

        ``terminate_client`` remains mandatory for the rank that actually
        crashed. Healthy peer ranks are different: forcing their termination
        while a warm successor shares the MPS server can poison the successor
        context on some drivers. This explicit best-effort mode kills the
        birth-checked host process and waits for authoritative MPS inventory
        retirement instead.

        The mode is opt-in because inventory retirement is weaker evidence
        than a successful ``terminate_client`` response.
        """
        backend_env = backend.upper().replace("-", "_")
        raw = os.environ.get(
            f"DYN_{backend_env}_GMS_ALLOW_SURVIVOR_INVENTORY_RETIREMENT",
            os.environ.get("DYN_GMS_ALLOW_SURVIVOR_INVENTORY_RETIREMENT", "0"),
        )
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _binary() -> str:
        return os.environ.get("DYN_GMS_MPS_CONTROL_BINARY", "nvidia-cuda-mps-control")

    async def _control(self, backend: str, *command: str) -> tuple[int, str]:
        control_env = os.environ.copy()
        # The protocol is numeric ASCII. Do not let a missing deployment
        # locale make the wrapper emit a diagnostic on stderr and invalidate
        # an otherwise authoritative response.
        control_env.update(LC_ALL="C", LANG="C")
        process = await asyncio.create_subprocess_exec(
            self._binary(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=control_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate((" ".join(command) + "\n").encode()),
                timeout=self._timeout(backend),
            )
        except asyncio.CancelledError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise
        except asyncio.TimeoutError as exc:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise RuntimeError("MPS control command timed out") from exc
        # Parse the complete stdout. Truncation can hide the target client or
        # turn a diagnostic into a plausible numeric success prefix.
        stdout_detail = stdout.decode("utf-8", errors="replace").strip()
        stderr_detail = stderr.decode("utf-8", errors="replace").strip()
        if stderr_detail:
            raise RuntimeError(f"MPS control diagnostic: {stderr_detail[:512]}")
        return int(process.returncode), stdout_detail

    async def _server_pid(self, backend: str) -> str:
        backend_env = backend.upper().replace("-", "_")
        configured = os.environ.get(
            f"DYN_{backend_env}_GMS_MPS_SERVER_PID",
            os.environ.get("DYN_GMS_MPS_SERVER_PID", ""),
        ).strip()
        if configured:
            if not configured.isdigit() or int(configured) <= 0:
                raise RuntimeError("configured MPS server PID is invalid")
            return configured

        rc, detail = await self._control(backend, "get_server_list")
        if rc != 0:
            raise RuntimeError(f"MPS server discovery failed: {detail}")
        servers = sorted(_inventory_pids(detail))
        if len(servers) != 1:
            raise RuntimeError(
                "MPS server discovery requires exactly one server; configure "
                "DYN_GMS_MPS_SERVER_PID when multiple servers are active"
            )
        return servers[0]

    async def _wait_client_absent(
        self, backend: str, server_pid: str, client_pid: int
    ) -> tuple[bool, str]:
        deadline = time.monotonic() + self._timeout(backend)
        last_inventory = ""
        while True:
            rc, last_inventory = await self._control(
                backend, "get_client_list", server_pid
            )
            client_pids = _inventory_pids(last_inventory)
            if rc == 0 and str(client_pid) not in client_pids:
                return True, last_inventory
            if time.monotonic() >= deadline:
                return False, last_inventory
            await asyncio.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    async def quiesce(
        self,
        *,
        backend: str,
        predecessor_cohort: str | None,
        successor_cohort: str,
        terminate_host: bool = False,
    ) -> QuiescenceResult:
        backend = backend.strip().lower()
        predecessor_cohort = (
            None if predecessor_cohort is None else predecessor_cohort.strip()
        )
        successor_cohort = successor_cohort.strip()
        if predecessor_cohort == "" or not successor_cohort:
            raise ValueError("specified cohorts must be non-empty")
        if predecessor_cohort == successor_cohort:
            raise ValueError("refusing to terminate the successor GPU cohort")
        if not self.configured(backend):
            return QuiescenceResult(
                False, "quarantine-only", 0, "GMS MPS provider is not enabled", 0.0
            )

        async with self._lock:
            if predecessor_cohort is None:
                predecessors = sorted(
                    {
                        cohort
                        for registered_backend, cohort, _pid in self._clients
                        if registered_backend == backend and cohort != successor_cohort
                    }
                    | {
                        cohort
                        for registered_backend, cohort in self._proofs
                        if registered_backend == backend and cohort != successor_cohort
                    }
                )
                if not predecessors:
                    return QuiescenceResult(
                        False,
                        "gms-mps",
                        0,
                        "no predecessor CUDA cohort registered for this pool",
                        0.0,
                    )
                # Freeze the whole snapshot before the first subprocess await.
                self._retired.update((backend, cohort) for cohort in predecessors)
                allow_inventory_retirement = (
                    terminate_host
                    and self._allow_survivor_inventory_retirement(backend)
                )
                results = [
                    await self._quiesce_locked(
                        backend=backend,
                        predecessor_cohort=cohort,
                        terminate_host=terminate_host,
                        trigger_interlock=(
                            terminate_host and not allow_inventory_retirement
                        ),
                        allow_inventory_retirement=allow_inventory_retirement,
                    )
                    for cohort in predecessors
                ]
                failed = next(
                    (result for result in results if not result.quiesced), None
                )
                if failed is not None:
                    return failed
                if any(
                    registered_backend == backend
                    and cohort != successor_cohort
                    and cohort not in predecessors
                    for registered_backend, cohort, _pid in self._clients
                ):
                    return QuiescenceResult(
                        False,
                        "gms-mps",
                        0,
                        "predecessor inventory changed during termination; retry",
                        0.0,
                    )
                provider = (
                    "gms-mps-inventory"
                    if any(result.provider == "gms-mps-inventory" for result in results)
                    else "gms-mps"
                )
                return QuiescenceResult(
                    True,
                    provider,
                    sum(result.client_count for result in results),
                    "; ".join(result.detail for result in results)[:512],
                    sum(result.elapsed_ms for result in results),
                )
            allow_inventory_retirement = (
                terminate_host and self._allow_survivor_inventory_retirement(backend)
            )
            return await self._quiesce_locked(
                backend=backend,
                predecessor_cohort=predecessor_cohort,
                terminate_host=terminate_host,
                trigger_interlock=terminate_host and not allow_inventory_retirement,
                allow_inventory_retirement=allow_inventory_retirement,
            )

    async def _quiesce_locked(
        self,
        *,
        backend: str,
        predecessor_cohort: str,
        terminate_host: bool = False,
        trigger_interlock: bool = False,
        allow_inventory_retirement: bool = False,
    ) -> QuiescenceResult:
        """Keep resume, MPS proof, and failure restop under the same lock."""
        resumed: list[GPUClient] = []
        result = None
        try:
            result = await self._terminate_locked(
                backend=backend,
                predecessor_cohort=predecessor_cohort,
                terminate_host=terminate_host,
                trigger_interlock=trigger_interlock,
                allow_inventory_retirement=allow_inventory_retirement,
                resumed=resumed,
            )
            return result
        finally:
            if result is None or not result.quiesced:
                for client in resumed:
                    try:
                        signal_client(client, signal.SIGSTOP)
                    except ProcessLookupError:
                        pass

    async def _terminate_locked(
        self,
        *,
        backend: str,
        predecessor_cohort: str,
        terminate_host: bool = False,
        trigger_interlock: bool = False,
        allow_inventory_retirement: bool = False,
        resumed: list[GPUClient],
    ) -> QuiescenceResult:
        """Quiesce one exact cohort while ``self._lock`` is held."""
        key = (backend, predecessor_cohort)
        cached = self._proofs.get(key)
        if cached is not None:
            return cached
        # This is deliberately permanent. CPU fencing says no legitimate
        # member can join after takeover starts; rejecting late registration
        # makes that invariant explicit even while MPS control awaits.
        self._retired.add(key)
        clients = sorted(
            (
                client
                for client in self._clients.values()
                if client.backend == backend and client.cohort == predecessor_cohort
            ),
            key=lambda client: (client.rank, client.pid),
        )
        if not clients:
            return QuiescenceResult(
                False,
                "gms-mps",
                0,
                "no CUDA clients registered for predecessor cohort",
                0.0,
            )

        started = time.monotonic()
        server_pid = await self._server_pid(backend)
        details: list[str] = []
        for client in clients:
            client_identity = (client.backend, client.cohort, client.pid)
            stopped_before_termination = process_state(client.pid) in {"T", "t"}
            if trigger_interlock and not stopped_before_termination:
                # Cooperative rank-loss fencing reaches otherwise healthy TP
                # workers while they may be blocked in a broken NCCL kernel.
                # Trip the same native fail-closed interlock used for a real
                # crash before asking MPS to retire the CUDA client. Merely
                # killing the launcher can make MPS forget the client before
                # it returns an authoritative termination result.
                if client_identity not in self._crash_tasks:
                    return QuiescenceResult(
                        False,
                        "gms-mps",
                        len(clients),
                        f"client {client.pid} has no armed crash interlock",
                        (time.monotonic() - started) * 1000.0,
                    )
                try:
                    signal_client(client, signal.SIGABRT)
                except ProcessLookupError:
                    return QuiescenceResult(
                        False,
                        "gms-mps",
                        len(clients),
                        f"client {client.pid} exited before interlock fencing",
                        (time.monotonic() - started) * 1000.0,
                    )
                if not await self._wait_native_stop(client):
                    return QuiescenceResult(
                        False,
                        "gms-mps",
                        len(clients),
                        f"client {client.pid} did not stop in its crash interlock",
                        (time.monotonic() - started) * 1000.0,
                    )
                self._crashed.add(client_identity)
                stopped_before_termination = True
            observed = process_start_time(client.pid)
            if observed is not None and observed != client.process_start_time:
                return QuiescenceResult(
                    False,
                    "gms-mps",
                    len(clients),
                    f"PID {client.pid} was reused before MPS termination",
                    (time.monotonic() - started) * 1000.0,
                )
            client_key = (
                client.backend,
                client.cohort,
                client.pid,
                client.process_start_time,
                server_pid,
            )
            if allow_inventory_retirement:
                # Healthy ranks are cooperatively fenced after a peer failure.
                # Avoid terminate_client here: TP=8 validation showed it can
                # reset/poison an initialized warm-shadow context. Host death
                # prevents new submissions; MPS inventory retirement bounds
                # when the old client is no longer tracked by the server.
                observed = process_start_time(client.pid)
                if observed == client.process_start_time:
                    try:
                        signal_client(client, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif observed is not None:
                    return QuiescenceResult(
                        False,
                        "gms-mps-inventory",
                        len(clients),
                        f"PID {client.pid} was reused before survivor retirement",
                        (time.monotonic() - started) * 1000.0,
                    )
                absent, inventory = await self._wait_client_absent(
                    backend, server_pid, client.pid
                )
                if not absent:
                    return QuiescenceResult(
                        False,
                        "gms-mps-inventory",
                        len(clients),
                        f"client {client.pid} remained in MPS inventory: "
                        f"{inventory[:256]}",
                        (time.monotonic() - started) * 1000.0,
                    )
                details.append(f"pid={client.pid}:absent from inventory")
                continue
            if stopped_before_termination:
                signal_client(client, signal.SIGCONT)
                resumed.append(client)
            if client_key not in self._terminated_clients:
                rc, detail = await self._control(
                    backend, "terminate_client", server_pid, str(client.pid)
                )
                # The executable normally exits zero even when the MPS command
                # fails. Its stdout is the CUDA result. CUDA_SUCCESS is the
                # only proof that the predecessor GPU work is terminated.
                # Interlock EOF, host-PID absence, and MPS-inventory absence
                # prove process/bookkeeping retirement, but not GPU quiescence:
                # TP=8 validation observed CUDA_ERROR_INVALID_CONTEXT (201),
                # followed by inventory absence and then an illegal access in
                # the successor immediately after remapping the shared pool.
                if rc != 0 or detail.strip() != "0":
                    return QuiescenceResult(
                        False,
                        "gms-mps",
                        len(clients),
                        f"MPS did not certify termination of client {client.pid}: "
                        f"control_rc={rc} cuda_result={detail or '<empty>'}",
                        (time.monotonic() - started) * 1000.0,
                    )
                # Remember individual success before the inventory check. This
                # makes cancellation and partial multi-rank failure retry-safe:
                # a second terminate_client would return INVALID_CONTEXT even
                # though the first command already supplied authoritative proof.
                self._terminated_clients.add(client_key)
            # CUDA_SUCCESS authorizes host teardown. A crash-interlocked client
            # cannot disappear from MPS inventory while its signal handler is
            # stopped/paused outside CUDA, so kill the exact birth-checked PID
            # before waiting for inventory retirement. This is not a fallback:
            # an uncorroborated unsuccessful terminate_client never reaches this branch.
            if (
                terminate_host
                or stopped_before_termination
                or (client.backend, client.cohort, client.pid) in self._crashed
            ):
                observed = process_start_time(client.pid)
                if observed == client.process_start_time:
                    try:
                        signal_client(client, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif observed is not None:
                    return QuiescenceResult(
                        False,
                        "gms-mps",
                        len(clients),
                        f"PID {client.pid} was reused after MPS termination",
                        (time.monotonic() - started) * 1000.0,
                    )
            absent, inventory = await self._wait_client_absent(
                backend, server_pid, client.pid
            )
            if not absent:
                return QuiescenceResult(
                    False,
                    "gms-mps",
                    len(clients),
                    f"client {client.pid} remains in MPS inventory after "
                    f"certified termination: {inventory}",
                    (time.monotonic() - started) * 1000.0,
                )
            details.append(f"pid={client.pid}:cuda_result=0:absent from inventory")

        result = QuiescenceResult(
            True,
            "gms-mps-inventory" if allow_inventory_retirement else "gms-mps",
            len(clients),
            "; ".join(details)[:512],
            (time.monotonic() - started) * 1000.0,
        )
        for client in clients:
            self._clients.pop((client.backend, client.cohort, client.pid), None)
            self._crashed.discard((client.backend, client.cohort, client.pid))
            self._interlock_eof.discard((client.backend, client.cohort, client.pid))
            self._terminated_clients.discard(
                (
                    client.backend,
                    client.cohort,
                    client.pid,
                    client.process_start_time,
                    server_pid,
                )
            )
        self._proofs[key] = result
        return result
