# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import pytest
from gpu_memory_service.server import gpu_quiescence as quiescence
from gpu_memory_service.server.gpu_quiescence import GPUQuiescenceManager

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
        return self.output, b""

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


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
