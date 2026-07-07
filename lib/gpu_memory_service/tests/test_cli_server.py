# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for production GMS process and listener topology."""

from __future__ import annotations

import asyncio
import sys

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.cli import args as cli_args
from gpu_memory_service.cli import runner, server
from gpu_memory_service.cli.args import parse_args

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_child_command_launches_default_multi_tag_runner():
    assert server._child_command(3) == [
        sys.executable,
        "-m",
        "gpu_memory_service",
        "--device",
        "3",
    ]


class _Process:
    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        self.terminated = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True


def test_supervisor_terminates_siblings_when_child_exits():
    processes = [_Process(exit_code=17), _Process()]

    assert server._supervise(processes) == 17
    assert processes[1].terminated


def test_parse_args_defaults_to_one_config_per_production_tag(monkeypatch):
    # get_socket_path queries the GPU UUID through NVML; stub the hardware.
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

    with pytest.raises(SystemExit):
        # Tags outside GMS_TAGS are rejected.
        parse_args(["--device", "3", "--tag", "weight"])


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_server_failure_cancels_other_listener():
    both_started = asyncio.Event()
    started = 0
    sibling_cancelled = asyncio.Event()

    class FailingServer:
        def __init__(self, fails: bool) -> None:
            self.fails = fails

        async def serve(self) -> None:
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await both_started.wait()
            if self.fails:
                raise RuntimeError("listener failed")
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

    with pytest.raises(RuntimeError, match="listener failed"):
        await runner.run_servers(
            [FailingServer(fails=True), FailingServer(fails=False)]
        )

    assert sibling_cancelled.is_set()
