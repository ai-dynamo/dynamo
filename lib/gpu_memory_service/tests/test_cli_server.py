# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for production GMS process and listener topology."""

from __future__ import annotations

import asyncio

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.cli import args as cli_args
from gpu_memory_service.cli import runner, server
from gpu_memory_service.cli.args import Config, parse_args

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


def test_supervisor_starts_one_multi_tag_child_per_device(monkeypatch):
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
            "gpu_memory_service",
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


def test_parse_args_defaults_to_one_config_per_production_tag(monkeypatch):
    monkeypatch.setattr(
        cli_args,
        "get_socket_path",
        lambda device, tag: f"/sockets/{device}-{tag}.sock",
    )

    configs = parse_args(["--device", "3"])

    assert [config.tag for config in configs] == ["weights", "kv_cache"]
    assert [config.socket_path for config in configs] == [
        "/sockets/3-weights.sock",
        "/sockets/3-kv_cache.sock",
    ]
    assert all(config.device == 3 for config in configs)

    (config,) = parse_args(["--device", "3", "--tag", "kv_cache"])
    assert config.tag == "kv_cache"

    with pytest.raises(SystemExit):
        # --socket-path cannot name one socket for multiple tags.
        parse_args(["--device", "3", "--socket-path", "/tmp/gms.sock"])


def _config(tag: str, socket_path: str) -> Config:
    return Config(
        device=0,
        tag=tag,
        socket_path=socket_path,
        alloc_retry_interval=0.5,
        alloc_retry_timeout=60.0,
        verbose=False,
    )


@pytest.mark.timeout(10)
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
    configs = [_config(tag, tag) for tag in ("weights", "kv_cache")]

    with pytest.raises(RuntimeError, match="listener failed"):
        await runner.serve_configs(configs)

    assert sibling_cancelled.is_set()
