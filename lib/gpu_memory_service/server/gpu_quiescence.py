# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Daemon-owned CUDA MPS client termination for shared-KV recovery."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def process_start_time(pid: int) -> str | None:
    """Return Linux's process birth identity, or ``None`` outside its namespace."""
    try:
        # comm may contain spaces and parentheses. Fields following the final ')'
        # start at proc field 3; starttime is field 22, hence index 19 here.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1]
        return fields.split()[19]
    except (FileNotFoundError, PermissionError, IndexError, OSError):
        return None


@dataclass(frozen=True)
class GPUClient:
    backend: str
    cohort: str
    pid: int
    process_start_time: str
    rank: int


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
        self._terminated_clients: set[tuple[str, str, int, str]] = set()
        self._lock = asyncio.Lock()

    def register(
        self,
        *,
        backend: str,
        cohort: str,
        pid: int,
        process_start_time_value: str,
        rank: int,
    ) -> None:
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
        )
        previous = self._clients.get(key)
        if previous is not None and previous != client:
            raise ValueError(
                "GPU client PID was already registered with other metadata"
            )
        self._clients[key] = client
        # A late registration invalidates a cached proof. The writer-cohort flock
        # prevents this once takeover fencing has completed, but invalidation keeps
        # this component independently fail-closed.
        self._proofs.pop((backend, cohort), None)

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
            os.environ.get("DYN_GMS_GPU_QUIESCENCE_TIMEOUT_SECS", "1.0"),
        )
        try:
            return max(0.05, float(raw))
        except ValueError:
            return 1.0

    @staticmethod
    def _binary() -> str:
        return os.environ.get("DYN_GMS_MPS_CONTROL_BINARY", "nvidia-cuda-mps-control")

    async def _control(self, backend: str, *command: str) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            self._binary(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate((" ".join(command) + "\n").encode()),
                timeout=self._timeout(backend),
            )
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise RuntimeError("MPS control command timed out") from exc
        stdout_detail = stdout.decode("utf-8", errors="replace").strip()[:512]
        stderr_detail = stderr.decode("utf-8", errors="replace").strip()[:512]
        return int(process.returncode), stdout_detail or stderr_detail

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
        servers = re.findall(r"(?m)^\s*(\d+)\s*$", detail)
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
            client_pids = set(re.findall(r"(?m)^\s*(\d+)\s*$", last_inventory))
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
                results = [
                    await self._quiesce_locked(
                        backend=backend,
                        predecessor_cohort=cohort,
                    )
                    for cohort in predecessors
                ]
                failed = next(
                    (result for result in results if not result.quiesced), None
                )
                if failed is not None:
                    return failed
                return QuiescenceResult(
                    True,
                    "gms-mps",
                    sum(result.client_count for result in results),
                    "; ".join(result.detail for result in results)[:512],
                    sum(result.elapsed_ms for result in results),
                )
            return await self._quiesce_locked(
                backend=backend,
                predecessor_cohort=predecessor_cohort,
            )

    async def _quiesce_locked(
        self, *, backend: str, predecessor_cohort: str
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
            )
            if client_key not in self._terminated_clients:
                rc, detail = await self._control(
                    backend, "terminate_client", server_pid, str(client.pid)
                )
                # The executable normally exits zero even when the MPS command
                # fails. Its stdout is the CUDA result. Only CUDA_SUCCESS proves
                # that every context became INACTIVE and teardown completed.
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
            "gms-mps",
            len(clients),
            "; ".join(details)[:512],
            (time.monotonic() - started) * 1000.0,
        )
        for client in clients:
            self._clients.pop((client.backend, client.cohort, client.pid), None)
            self._terminated_clients.discard(
                (client.backend, client.cohort, client.pid, client.process_start_time)
            )
        self._proofs[key] = result
        return result
