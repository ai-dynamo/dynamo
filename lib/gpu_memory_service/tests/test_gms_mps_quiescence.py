# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import signal
import struct

import pytest
from gpu_memory_service.server import gpu_quiescence as quiescence
from gpu_memory_service.server.gpu_quiescence import (
    GPUClient,
    GPUQuiescenceManager,
    QuiescenceResult,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.none]


class _Process:
    def __init__(self, returncode=0, output=b"0", *, hangs=False):
        self.returncode = returncode
        self.output = output
        self.hangs = hangs
        self.killed = False
        self.input = None

    async def communicate(self, input=None):
        self.input = input
        if self.hangs:
            await asyncio.Event().wait()
        if input and input.startswith(b"get_client_list") and self.output == b"0":
            return b"", b""
        return self.output, b""

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_control_exit_race_preserves_timeout_or_cancellation(
    monkeypatch, cancelled
):
    class ExitedProcess(_Process):
        async def communicate(self, input=None):
            if cancelled:
                raise asyncio.CancelledError
            raise asyncio.TimeoutError

        def kill(self):
            raise ProcessLookupError

    async def create(*args, **kwargs):
        return ExitedProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    expected = asyncio.CancelledError if cancelled else RuntimeError
    with pytest.raises(expected):
        await GPUQuiescenceManager()._control("vllm", "get_server_list")


def _register_current(manager, cohort="old", rank=0):
    pid = os.getpid()
    start = quiescence.process_start_time(pid)
    assert start is not None
    manager.register(
        backend="vllm",
        cohort=cohort,
        pid=pid,
        process_start_time_value=start,
        rank=rank,
    )
    return pid


@pytest.mark.asyncio
async def test_mps_terminates_exact_registered_cohort_and_caches_proof(monkeypatch):
    manager = GPUQuiescenceManager()
    pid = _register_current(manager)
    calls = []
    processes = []

    async def create(*args, **_kwargs):
        calls.append(args)
        process = _Process()
        processes.append(process)
        return process

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    first = await manager.quiesce(
        backend="vllm", predecessor_cohort="old", successor_cohort="new"
    )
    second = await manager.quiesce(
        backend="vllm", predecessor_cohort="old", successor_cohort="new"
    )

    assert first.quiesced and second == first
    assert first.client_count == 1
    assert calls == [
        ("nvidia-cuda-mps-control",),
        ("nvidia-cuda-mps-control",),
    ]
    assert [process.input for process in processes] == [
        f"terminate_client 444 {pid}\n".encode(),
        b"get_client_list 444\n",
    ]
    with pytest.raises(ValueError, match="retired"):
        _register_current(manager)


@pytest.mark.asyncio
async def test_proactive_teardown_kills_host_only_after_cuda_success(monkeypatch):
    manager = GPUQuiescenceManager()
    pid = 12345
    client = GPUClient("vllm", "old", pid, "birth", 0)
    manager._clients[("vllm", "old", pid)] = client
    manager._crash_tasks[("vllm", "old", pid)] = object()
    stopped = False
    events = []

    async def control(_backend, *command):
        events.append(command)
        if command[0] == "terminate_client":
            return 0, "0"
        assert events[-2][0] == "terminate_client"
        assert signals[-1] == (pid, signal.SIGKILL)
        return 0, ""

    signals = []
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setattr(manager, "_control", control)
    monkeypatch.setattr(
        quiescence,
        "process_start_time",
        lambda candidate: "birth" if candidate == pid else None,
    )
    monkeypatch.setattr(
        quiescence, "process_state", lambda _pid: "T" if stopped else "R"
    )

    def send(client, sig):
        nonlocal stopped
        signals.append((client.pid, sig))
        if sig == signal.SIGABRT:
            stopped = True

    monkeypatch.setattr(quiescence, "signal_client", send)

    result = await manager.quiesce(
        backend="vllm",
        predecessor_cohort="old",
        successor_cohort="new",
        terminate_host=True,
    )

    assert result.quiesced
    assert signals == [
        (pid, signal.SIGABRT),
        (pid, signal.SIGCONT),
        (pid, signal.SIGKILL),
    ]


@pytest.mark.asyncio
async def test_opt_in_survivor_inventory_retirement_avoids_mps_terminate(monkeypatch):
    manager = GPUQuiescenceManager()
    pid = 12345
    client = GPUClient("vllm", "old", pid, "birth", 0)
    manager._clients[("vllm", "old", pid)] = client
    controls = []
    signals = []

    async def control(_backend, *command):
        controls.append(command)
        if command[0] == "get_client_list":
            assert signals == [(pid, signal.SIGKILL)]
            return 0, ""
        raise AssertionError(f"unexpected MPS command: {command}")

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setenv("DYN_GMS_ALLOW_SURVIVOR_INVENTORY_RETIREMENT", "1")
    monkeypatch.setattr(manager, "_control", control)
    monkeypatch.setattr(
        quiescence,
        "process_start_time",
        lambda candidate: "birth" if candidate == pid else None,
    )
    monkeypatch.setattr(
        quiescence,
        "signal_client",
        lambda client, sig: signals.append((client.pid, sig)),
    )

    result = await manager.quiesce(
        backend="vllm",
        predecessor_cohort="old",
        successor_cohort="new",
        terminate_host=True,
    )

    assert result.quiesced
    assert result.provider == "gms-mps-inventory"
    assert signals == [(pid, signal.SIGKILL)]
    assert controls == [("get_client_list", "444")]


@pytest.mark.asyncio
async def test_aggregate_preserves_inventory_retirement_provider(monkeypatch):
    manager = GPUQuiescenceManager()
    manager._proofs[("vllm", "old-a")] = QuiescenceResult(
        True, "gms-mps", 1, "pid=1:cuda_result=0:absent from inventory", 1.0
    )
    manager._proofs[("vllm", "old-b")] = QuiescenceResult(
        True, "gms-mps-inventory", 1, "pid=2:absent from inventory", 2.0
    )
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")

    result = await manager.quiesce(
        backend="vllm",
        predecessor_cohort=None,
        successor_cohort="new",
    )

    assert result.quiesced
    assert result.provider == "gms-mps-inventory"
    assert result.client_count == 2


@pytest.mark.asyncio
async def test_mps_fails_closed_without_registered_predecessor(monkeypatch):
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")

    result = await manager.quiesce(
        backend="vllm", predecessor_cohort="missing", successor_cohort="new"
    )

    assert not result.quiesced
    assert result.client_count == 0


def test_registration_requires_shared_pid_namespace(monkeypatch):
    manager = GPUQuiescenceManager()
    monkeypatch.setattr(quiescence, "process_start_time", lambda _pid: None)

    with pytest.raises(ValueError, match="shared PID namespace"):
        manager.register(
            backend="vllm",
            cohort="old",
            pid=123,
            process_start_time_value="456",
            rank=0,
        )


@pytest.mark.asyncio
async def test_mps_rejects_successor_as_predecessor(monkeypatch):
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")

    with pytest.raises(ValueError, match="successor"):
        await manager.quiesce(
            backend="vllm", predecessor_cohort="same", successor_cohort="same"
        )


@pytest.mark.asyncio
async def test_mps_failure_and_pid_reuse_fail_closed(monkeypatch):
    manager = GPUQuiescenceManager()
    pid = _register_current(manager)
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")

    calls = 0

    async def reject(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Process(returncode=1, output=b"termination rejected")
        return _Process(returncode=0, output=f"{pid}\n".encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", reject)
    rejected = await manager.quiesce(
        backend="vllm", predecessor_cohort="old", successor_cohort="new"
    )
    assert not rejected.quiesced
    assert "did not certify termination" in rejected.detail

    monkeypatch.setattr(
        quiescence,
        "process_start_time",
        lambda candidate: "reused" if candidate == pid else None,
    )
    reused = await manager.quiesce(
        backend="vllm", predecessor_cohort="old", successor_cohort="new"
    )
    assert not reused.quiesced
    assert "reused" in reused.detail


@pytest.mark.asyncio
async def test_mps_zero_exit_is_not_proof_while_client_remains(monkeypatch):
    manager = GPUQuiescenceManager()
    pid = _register_current(manager)
    calls = 0

    async def create(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        output = b"Server 0 not found" if calls == 1 else f"{pid}\n".encode()
        return _Process(returncode=0, output=output)

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_TIMEOUT_SECS", "0.05")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    result = await manager.quiesce(
        backend="vllm", predecessor_cohort="old", successor_cohort="new"
    )

    assert not result.quiesced
    assert "did not certify termination" in result.detail


@pytest.mark.asyncio
async def test_mps_rejects_absent_client_without_cuda_success(monkeypatch):
    manager = GPUQuiescenceManager()
    _register_current(manager)
    calls = 0

    async def create(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        # The control program itself exits zero, but 201 is the CUDA command
        # result. This is the exact false-proof shape observed during TP=2
        # process teardown.
        return _Process(returncode=0, output=b"201")

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    result = await manager.quiesce(
        backend="vllm", predecessor_cohort="old", successor_cohort="new"
    )

    assert not result.quiesced
    assert "cuda_result=201" in result.detail
    assert calls == 1


@pytest.mark.asyncio
async def test_interlock_eof_and_inventory_absence_are_not_gpu_proof(monkeypatch):
    manager = GPUQuiescenceManager()
    pid = _register_current(manager)
    manager._interlock_eof.add(("vllm", "old", pid))
    calls = 0

    async def create(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        # CUDA rejected terminate_client after SIGKILL raced MPS cleanup.
        # Host/interlock retirement must not upgrade this to GPU quiescence.
        return _Process(returncode=0, output=b"201")

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(quiescence, "process_start_time", lambda _pid: None)

    result = await manager.quiesce(
        backend="vllm", predecessor_cohort="old", successor_cohort="new"
    )

    assert not result.quiesced
    assert "cuda_result=201" in result.detail
    assert calls == 1


@pytest.mark.asyncio
async def test_mps_timeout_kills_control_process(monkeypatch):
    manager = GPUQuiescenceManager()
    _register_current(manager)
    process = _Process(hangs=True)

    async def create(*_args, **_kwargs):
        return process

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_TIMEOUT_SECS", "0.05")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    with pytest.raises(RuntimeError, match="timed out"):
        await manager.quiesce(
            backend="vllm", predecessor_cohort="old", successor_cohort="new"
        )
    assert process.killed


@pytest.mark.asyncio
async def test_mps_partial_multi_rank_proof_is_retry_safe(monkeypatch):
    manager = GPUQuiescenceManager()
    starts = {1001: "a", 1002: "b"}
    monkeypatch.setattr(quiescence, "process_start_time", lambda pid: starts.get(pid))
    for rank, pid in enumerate(starts):
        manager.register(
            backend="vllm",
            cohort="old",
            pid=pid,
            process_start_time_value=starts[pid],
            rank=rank,
        )

    attempts = {1001: 0, 1002: 0}

    async def control(_backend, *command):
        if command[0] == "get_client_list":
            return 0, ""
        pid = int(command[2])
        attempts[pid] += 1
        if pid == 1002 and attempts[pid] == 1:
            return 0, "999"
        return 0, "0"

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setattr(manager, "_control", control)

    first = await manager.quiesce(
        backend="vllm", predecessor_cohort="old", successor_cohort="new"
    )
    second = await manager.quiesce(
        backend="vllm", predecessor_cohort="old", successor_cohort="new"
    )

    assert not first.quiesced
    assert second.quiesced
    assert attempts == {1001: 1, 1002: 2}


@pytest.mark.asyncio
async def test_mps_quiesces_every_predecessor_but_not_successor(monkeypatch):
    manager = GPUQuiescenceManager()
    starts = {1001: "a", 1002: "b", 2001: "c"}
    monkeypatch.setattr(quiescence, "process_start_time", lambda pid: starts.get(pid))
    for cohort, pid in (("old-a", 1001), ("old-b", 1002), ("new", 2001)):
        manager.register(
            backend="vllm",
            cohort=cohort,
            pid=pid,
            process_start_time_value=starts[pid],
            rank=0,
        )

    processes = []

    async def create(*_args, **_kwargs):
        process = _Process()
        processes.append(process)
        return process

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    result = await manager.quiesce(
        backend="vllm", predecessor_cohort=None, successor_cohort="new"
    )

    assert result.quiesced
    assert result.client_count == 2
    commands = [process.input for process in processes]
    assert b"terminate_client 444 1001\n" in commands
    assert b"terminate_client 444 1002\n" in commands
    assert b"terminate_client 444 2001\n" not in commands

    repeated = await manager.quiesce(
        backend="vllm", predecessor_cohort=None, successor_cohort="new"
    )
    assert repeated.quiesced
    assert len(processes) == 4


@pytest.mark.asyncio
async def test_mps_all_predecessors_fails_closed_without_local_registration(
    monkeypatch,
):
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")

    result = await manager.quiesce(
        backend="vllm", predecessor_cohort=None, successor_cohort="new"
    )

    assert not result.quiesced
    assert result.detail == "no predecessor CUDA cohort registered for this pool"


@pytest.mark.asyncio
async def test_crash_interlock_quiesces_then_kills_registered_cohort(monkeypatch):
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    pid = os.getpid()
    start = quiescence.process_start_time(pid)
    assert start is not None
    killed = []

    async def control(_backend, *command):
        if command[0] == "get_client_list":
            assert killed[-1] == (pid, signal.SIGKILL)
            return 0, ""
        return 0, "0"

    monkeypatch.setattr(manager, "_control", control)
    monkeypatch.setattr(quiescence, "process_state", lambda _pid: "T")
    monkeypatch.setattr(
        quiescence,
        "signal_client",
        lambda target, sig: killed.append((target.pid, sig)),
    )
    write_fd = manager.register(
        backend="vllm",
        cohort="old",
        pid=pid,
        process_start_time_value=start,
        rank=0,
        crash_interlock=True,
    )
    task = manager._crash_tasks[("vllm", "old", pid)]
    os.write(write_fd, struct.pack("=IIii", 0x47534D43, 1, signal.SIGABRT, pid))
    await asyncio.wait_for(task, timeout=1)
    os.close(write_fd)

    assert killed == [(pid, signal.SIGCONT), (pid, signal.SIGKILL)]
    assert ("vllm", "old") in manager._proofs


@pytest.mark.asyncio
async def test_terminate_all_predecessors_trips_native_interlock(monkeypatch):
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    pid = os.getpid()
    start = quiescence.process_start_time(pid)
    assert start is not None
    stopped = False
    signals = []

    async def control(_backend, *command):
        return (0, "") if command[0] == "get_client_list" else (0, "0")

    def send(client, sig):
        nonlocal stopped
        signals.append((client.pid, sig))
        if sig == signal.SIGABRT:
            stopped = True

    monkeypatch.setattr(manager, "_control", control)
    monkeypatch.setattr(
        quiescence, "process_state", lambda _pid: "T" if stopped else "S"
    )
    monkeypatch.setattr(quiescence, "signal_client", send)
    write_fd = manager.register(
        backend="vllm",
        cohort="old",
        pid=pid,
        process_start_time_value=start,
        rank=0,
        crash_interlock=True,
    )
    result = await manager.quiesce(
        backend="vllm",
        predecessor_cohort=None,
        successor_cohort="new",
        terminate_host=True,
    )
    os.close(write_fd)

    assert result.quiesced
    assert signals == [
        (pid, signal.SIGABRT),
        (pid, signal.SIGCONT),
        (pid, signal.SIGKILL),
    ]


@pytest.mark.asyncio
async def test_crash_interlock_leaves_process_stopped_without_mps_proof(monkeypatch):
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    pid = os.getpid()
    start = quiescence.process_start_time(pid)
    assert start is not None

    async def control(_backend, *_command):
        return 0, "201"

    killed = []
    monkeypatch.setattr(manager, "_control", control)
    monkeypatch.setattr(quiescence, "process_state", lambda _pid: "T")
    monkeypatch.setattr(
        quiescence,
        "signal_client",
        lambda target, sig: killed.append((target.pid, sig)),
    )
    write_fd = manager.register(
        backend="sglang",
        cohort="old",
        pid=pid,
        process_start_time_value=start,
        rank=0,
        crash_interlock=True,
    )
    task = manager._crash_tasks[("sglang", "old", pid)]
    os.write(write_fd, struct.pack("=IIii", 0x47534D43, 1, signal.SIGSEGV, pid))
    await asyncio.wait_for(task, timeout=1)
    os.close(write_fd)

    assert killed == [(pid, signal.SIGCONT), (pid, signal.SIGSTOP)]
    assert ("sglang", "old") not in manager._proofs


@pytest.mark.asyncio
async def test_pipe_eof_without_mps_proof_fails_closed(monkeypatch):
    """SIGKILL/exit EOF cannot become a warm-reuse proof by itself."""
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    pid = os.getpid()
    start = quiescence.process_start_time(pid)
    assert start is not None

    async def control(_backend, *_command):
        return 0, "201"

    monkeypatch.setattr(manager, "_control", control)
    write_fd = manager.register(
        backend="vllm",
        cohort="old",
        pid=pid,
        process_start_time_value=start,
        rank=0,
        crash_interlock=True,
    )
    task = manager._crash_tasks[("vllm", "old", pid)]
    os.close(write_fd)
    await asyncio.wait_for(task, timeout=1)

    assert ("vllm", "old") in manager._retired
    assert ("vllm", "old") not in manager._proofs


@pytest.mark.asyncio
async def test_crash_interlock_restops_resumed_members_if_later_resume_fails(
    monkeypatch,
):
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    pid = os.getpid()
    start = quiescence.process_start_time(pid)
    assert start is not None
    missing_pid = pid + 1_000_000
    starts = {pid: start, missing_pid: "missing-start"}
    signals = []

    async def control(_backend, *command):
        return (0, "") if command[0] == "get_client_list" else (0, "0")

    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setattr(manager, "_control", control)

    def kill(target, sig):
        signals.append((target, sig))
        if target == missing_pid and sig == signal.SIGCONT:
            raise ProcessLookupError

    monkeypatch.setattr(
        quiescence, "process_start_time", lambda target: starts.get(target)
    )
    monkeypatch.setattr(quiescence, "process_state", lambda _pid: "T")
    monkeypatch.setattr(
        quiescence, "signal_client", lambda client, sig: kill(client.pid, sig)
    )
    write_fd = manager.register(
        backend="vllm",
        cohort="old",
        pid=pid,
        process_start_time_value=start,
        rank=0,
        crash_interlock=True,
    )
    missing = GPUClient("vllm", "old", missing_pid, "missing-start", 1)
    manager._clients[("vllm", "old", missing_pid)] = missing
    task = manager._crash_tasks[("vllm", "old", pid)]
    os.write(write_fd, struct.pack("=IIii", 0x47534D43, 1, signal.SIGABRT, pid))
    with pytest.raises(ProcessLookupError):
        await asyncio.wait_for(task, timeout=1)
    os.close(write_fd)

    assert signals == [
        (pid, signal.SIGCONT),
        (pid, signal.SIGKILL),
        (missing_pid, signal.SIGCONT),
        (pid, signal.SIGSTOP),
    ]
    assert ("vllm", "old") not in manager._proofs


@pytest.mark.asyncio
async def test_control_preserves_long_inventory(monkeypatch):
    output = "\n".join(str(pid) for pid in range(1000, 1300))

    async def create(*_args, **_kwargs):
        return _Process(output=output.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    rc, inventory = await GPUQuiescenceManager()._control(
        "vllm", "get_client_list", "1"
    )
    assert rc == 0
    assert inventory == output
    assert "1299" in quiescence._inventory_pids(inventory)


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["Server 444 not found", "1001\nERROR", "0", "-1"])
async def test_inventory_diagnostics_cannot_prove_absence(monkeypatch, output):
    manager = GPUQuiescenceManager()

    async def control(*_args):
        return 0, output

    monkeypatch.setattr(manager, "_control", control)
    with pytest.raises(RuntimeError, match="Invalid MPS PID inventory"):
        await manager._wait_client_absent("vllm", "444", 9999)


def test_pidfd_signal_closes_handle_and_rejects_reused_pid(monkeypatch):
    client = GPUClient("vllm", "old", 1234, "birth", 0)
    events = []
    monkeypatch.setattr(
        os, "pidfd_open", lambda pid: events.append(("open", pid)) or 99
    )
    monkeypatch.setattr(os, "close", lambda fd: events.append(("close", fd)))
    monkeypatch.setattr(
        signal, "pidfd_send_signal", lambda fd, sig: events.append((fd, sig))
    )
    monkeypatch.setattr(quiescence, "process_start_time", lambda pid: "birth")
    quiescence.signal_client(client, signal.SIGCONT)
    assert events == [("open", 1234), (99, signal.SIGCONT), ("close", 99)]
    events.clear()
    monkeypatch.setattr(quiescence, "process_start_time", lambda pid: "reused")
    with pytest.raises(ProcessLookupError):
        quiescence.signal_client(client, signal.SIGKILL)
    assert events == [("open", 1234), ("close", 99)]


@pytest.mark.asyncio
async def test_crashed_rank_does_not_wait_for_healthy_rank_to_stop(monkeypatch):
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setattr(quiescence, "process_start_time", lambda pid: str(pid))
    monkeypatch.setattr(
        quiescence, "process_state", lambda pid: "T" if pid == 1001 else "S"
    )
    events = []
    monkeypatch.setattr(
        quiescence,
        "signal_client",
        lambda client, sig: events.append((client.pid, sig)),
    )

    async def control(_backend, *command):
        events.append(command)
        return (0, "") if command[0] == "get_client_list" else (0, "0")

    monkeypatch.setattr(manager, "_control", control)
    manager.register(
        backend="vllm", cohort="old", pid=1002, process_start_time_value="1002", rank=1
    )
    fd = manager.register(
        backend="vllm",
        cohort="old",
        pid=1001,
        process_start_time_value="1001",
        rank=0,
        crash_interlock=True,
    )
    task = manager._crash_tasks[("vllm", "old", 1001)]
    try:
        os.write(fd, struct.pack("=IIii", 0x47534D43, 1, signal.SIGABRT, 1001))
        await asyncio.wait_for(task, 1)
    finally:
        os.close(fd)
    assert (1002, signal.SIGCONT) not in events
    assert ("terminate_client", "444", "1002") in events
    assert manager._proofs[("vllm", "old")].client_count == 2


@pytest.mark.asyncio
async def test_cancelled_termination_restops_before_unlock(monkeypatch):
    manager = GPUQuiescenceManager()
    _register_current(manager)
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setattr(quiescence, "process_state", lambda pid: "T")
    events = []
    monkeypatch.setattr(
        quiescence,
        "signal_client",
        lambda client, sig: events.append((sig, manager._lock.locked())),
    )
    entered = asyncio.Event()

    async def control(*_args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_control", control)
    task = asyncio.create_task(
        manager.quiesce(
            backend="vllm", predecessor_cohort="old", successor_cohort="new"
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == [(signal.SIGCONT, True), (signal.SIGSTOP, True)]
    assert not manager._proofs


@pytest.mark.parametrize("raw", ["inf", "nan", "-inf"])
def test_timeout_is_finite(monkeypatch, raw):
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_TIMEOUT_SECS", raw)
    assert GPUQuiescenceManager._timeout("vllm") == 1.0


@pytest.mark.asyncio
async def test_all_predecessors_rejects_a_new_cohort_during_proof(monkeypatch):
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    monkeypatch.setattr(quiescence, "process_start_time", lambda pid: str(pid))
    manager.register(
        backend="vllm", cohort="old", pid=1001, process_start_time_value="1001", rank=0
    )

    async def control(_backend, *command):
        if command[0] == "terminate_client":
            manager.register(
                backend="vllm",
                cohort="late",
                pid=1002,
                process_start_time_value="1002",
                rank=0,
            )
            return 0, "0"
        return 0, ""

    monkeypatch.setattr(manager, "_control", control)
    result = await manager.quiesce(
        backend="vllm", predecessor_cohort=None, successor_cohort="new"
    )
    assert not result.quiesced
    assert "inventory changed" in result.detail


@pytest.mark.asyncio
async def test_cancel_before_watcher_starts_closes_notification_reader(monkeypatch):
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    pid = os.getpid()
    fd = manager.register(
        backend="vllm",
        cohort="old",
        pid=pid,
        process_start_time_value=quiescence.process_start_time(pid),
        rank=0,
        crash_interlock=True,
    )
    key = ("vllm", "old", pid)
    reader = manager._crash_read_fds[key]
    task = manager._crash_tasks[key]
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(OSError):
            os.fstat(reader)
    finally:
        os.close(fd)


@pytest.mark.asyncio
async def test_registration_failure_rolls_back_pipe_and_client(monkeypatch):
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    opened: list[int] = []
    real_pipe2 = os.pipe2

    def pipe2(flags):
        pair = real_pipe2(flags)
        opened.extend(pair)
        return pair

    def fail_create(*_args, **_kwargs):
        raise RuntimeError("no loop")

    monkeypatch.setattr(os, "pipe2", pipe2)
    monkeypatch.setattr(asyncio, "create_task", fail_create)
    pid = os.getpid()
    with pytest.raises(RuntimeError, match="no loop"):
        manager.register(
            backend="vllm",
            cohort="old",
            pid=pid,
            process_start_time_value=quiescence.process_start_time(pid),
            rank=0,
            crash_interlock=True,
        )
    assert not manager._clients
    assert not manager._crash_tasks
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.asyncio
async def test_native_watcher_and_successor_share_one_termination(monkeypatch):
    manager = GPUQuiescenceManager()
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    pid = os.getpid()
    monkeypatch.setattr(quiescence, "process_state", lambda _pid: "T")
    monkeypatch.setattr(quiescence, "signal_client", lambda *_args: None)
    entered = asyncio.Event()
    release = asyncio.Event()
    terminate_calls = 0

    async def control(_backend, *command):
        nonlocal terminate_calls
        if command[0] == "terminate_client":
            terminate_calls += 1
            entered.set()
            await release.wait()
            return 0, "0"
        return 0, ""

    monkeypatch.setattr(manager, "_control", control)
    fd = manager.register(
        backend="vllm",
        cohort="old",
        pid=pid,
        process_start_time_value=quiescence.process_start_time(pid),
        rank=0,
        crash_interlock=True,
    )
    watcher = manager._crash_tasks[("vllm", "old", pid)]
    try:
        os.write(fd, struct.pack("=IIii", 0x47534D43, 1, signal.SIGABRT, pid))
        await asyncio.wait_for(entered.wait(), 1)
        successor = asyncio.create_task(
            manager.quiesce(
                backend="vllm", predecessor_cohort="old", successor_cohort="new"
            )
        )
        await asyncio.sleep(0)
        assert not successor.done()
        release.set()
        await asyncio.wait_for(watcher, 1)
        result = await asyncio.wait_for(successor, 1)
    finally:
        os.close(fd)
    assert result.quiesced
    assert terminate_calls == 1


@pytest.mark.asyncio
async def test_partial_proof_is_scoped_to_mps_server(monkeypatch):
    manager = GPUQuiescenceManager()
    _register_current(manager)
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "444")
    terminate_servers: list[str] = []
    first_inventory = True

    async def control(_backend, *command):
        nonlocal first_inventory
        if command[0] == "terminate_client":
            terminate_servers.append(command[1])
            return 0, "0"
        if first_inventory:
            first_inventory = False
            raise RuntimeError("MPS server restarted")
        return 0, ""

    monkeypatch.setattr(manager, "_control", control)
    with pytest.raises(RuntimeError, match="restarted"):
        await manager.quiesce(
            backend="vllm", predecessor_cohort="old", successor_cohort="new"
        )
    monkeypatch.setenv("DYN_GMS_MPS_SERVER_PID", "555")
    result = await manager.quiesce(
        backend="vllm", predecessor_cohort="old", successor_cohort="new"
    )
    assert result.quiesced
    assert terminate_servers == ["444", "555"]
    assert result.client_count == 1


@pytest.mark.asyncio
async def test_control_stderr_is_not_cuda_success(monkeypatch):
    class DiagnosticProcess(_Process):
        async def communicate(self, input=None):
            return b"0", b"MPS control failed"

    async def create(*_args, **_kwargs):
        return DiagnosticProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    with pytest.raises(RuntimeError, match="MPS control diagnostic"):
        await GPUQuiescenceManager()._control("vllm", "terminate_client", "1", "2")


def test_failure_notifier_is_connected_before_crash_and_sends_hint(monkeypatch):
    import zmq

    connected: list[str] = []
    sent: list[list[bytes]] = []

    class Socket:
        def setsockopt(self, *_args):
            pass

        def connect(self, address):
            connected.append(address)

        def send_multipart(self, frames, **_kwargs):
            sent.append(frames)

        def close(self, _linger):
            pass

    socket = Socket()
    monkeypatch.setattr(
        zmq,
        "Context",
        type(
            "Context",
            (),
            {
                "instance": staticmethod(
                    lambda: type("C", (), {"socket": lambda self, _kind: socket})()
                )
            },
        ),
    )
    manager = GPUQuiescenceManager()
    pid = os.getpid()
    address = "tcp://leader.example:29555"
    manager._ensure_failure_notifier(address)
    client = GPUClient(
        "vllm",
        "/shared/cohort",
        pid,
        quiescence.process_start_time(pid),
        7,
        address,
    )

    manager._notify_gpu_failure(client, "signal-11")

    assert connected == [address]
    assert sent == [
        [
            b"gpu-failed-v1",
            b"/shared/cohort",
            b"7",
            str(pid).encode(),
            b"signal-11",
        ]
    ]
