# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for production GMS process and listener topology."""

from __future__ import annotations

import asyncio
import os

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.cli import dual_server, runner, server
from gpu_memory_service.cli.args import Config
from gpu_memory_service.server import allocations as server_allocations
from gpu_memory_service.server.rpc import GMSRPCServer

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


class _Process:
    def __init__(self, pid: int, exit_code: int | None = None) -> None:
        self.pid = pid
        self.exit_code = exit_code
        self.terminated = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True


def test_supervisor_starts_one_dual_tag_child_per_device(monkeypatch):
    commands: list[list[str]] = []
    processes: list[_Process] = []

    def popen(command: list[str]) -> _Process:
        commands.append(command)
        process = _Process(1000 + len(processes))
        processes.append(process)
        return process

    monkeypatch.setattr(server.subprocess, "Popen", popen)
    monkeypatch.setattr(server.sys, "executable", "/venv/bin/python")

    assert server._start_processes([0, 1, 2]) == processes
    assert commands == [
        [
            "/venv/bin/python",
            "-m",
            "gpu_memory_service.cli.dual_server",
            "--device",
            str(device),
        ]
        for device in range(3)
    ]


def test_supervisor_terminates_siblings_when_child_exits(monkeypatch):
    processes = [_Process(1000, exit_code=17), _Process(1001)]

    monkeypatch.setattr(server, "list_devices", lambda: [0, 1])
    monkeypatch.setattr(server, "_start_processes", lambda _devices: processes)
    monkeypatch.setattr(server.signal, "signal", lambda *_args: None)

    with pytest.raises(SystemExit) as exc_info:
        server.main()

    assert exc_info.value.code == 17
    assert processes[1].terminated


def test_dual_tag_configs_have_independent_socket_paths(monkeypatch):
    monkeypatch.setattr(
        dual_server,
        "get_socket_path",
        lambda device, tag: f"/sockets/{device}-{tag}.sock",
    )

    configs = dual_server.make_server_configs(
        3,
        allocation_retry_interval=0.25,
        allocation_retry_timeout=10.0,
        verbose=True,
    )

    assert [config.tag for config in configs] == ["weights", "kv_cache"]
    assert [config.socket_path for config in configs] == [
        "/sockets/3-weights.sock",
        "/sockets/3-kv_cache.sock",
    ]
    assert all(config.device == 3 for config in configs)


@pytest.mark.asyncio
async def test_dual_tag_servers_bind_independent_listeners(monkeypatch, tmp_path):
    monkeypatch.setattr(server_allocations, "cuda_ensure_initialized", lambda: None)
    monkeypatch.setattr(
        server_allocations,
        "cumem_get_allocation_granularity",
        lambda _device: 4096,
    )

    instances: list[GMSRPCServer] = []

    def make_server(*args, **kwargs) -> GMSRPCServer:
        instance = GMSRPCServer(*args, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(runner, "GMSRPCServer", make_server)
    socket_paths = [
        str(tmp_path / "weights.sock"),
        str(tmp_path / "kv_cache.sock"),
    ]
    configs = [
        Config(
            device=0,
            tag=tag,
            socket_path=socket_path,
            alloc_retry_interval=0.5,
            alloc_retry_timeout=60.0,
            verbose=False,
        )
        for tag, socket_path in zip(dual_server.TAGS, socket_paths)
    ]

    task = asyncio.create_task(runner.serve_configs(configs))
    try:

        async def wait_for_listeners() -> None:
            while not all(os.path.exists(path) for path in socket_paths):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_listeners(), timeout=2)
        assert len(instances) == 2
        assert instances[0]._gms is not instances[1]._gms
        assert instances[0]._gms._allocations is not instances[1]._gms._allocations
        assert instances[0]._gms._sessions is not instances[1]._gms._sessions
        assert instances[0]._gms._metadata is not instances[1]._gms._metadata
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_server_failure_cancels_other_listener(monkeypatch):
    both_started = asyncio.Event()
    started = 0
    sibling_cancelled = asyncio.Event()

    class FailingServer:
        def __init__(self, socket_path: str, **_kwargs) -> None:
            self.socket_path = socket_path

        async def serve(self) -> None:
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await both_started.wait()
            if self.socket_path == "weights":
                raise RuntimeError("listener failed")
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

    monkeypatch.setattr(runner, "GMSRPCServer", FailingServer)
    configs = [
        Config(
            device=0,
            tag=tag,
            socket_path=tag,
            alloc_retry_interval=0.5,
            alloc_retry_timeout=60.0,
            verbose=False,
        )
        for tag in dual_server.TAGS
    ]

    with pytest.raises(RuntimeError, match="listener failed"):
        await runner.serve_configs(configs)

    assert sibling_cancelled.is_set()
