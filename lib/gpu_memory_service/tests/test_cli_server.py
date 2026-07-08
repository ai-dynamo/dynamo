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

    # A clean exit (poll() returning 0) is an exit, not "still running".
    clean = [_Process(exit_code=0), _Process()]
    assert server._supervise(clean) == 0
    assert clean[1].terminated


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


def test_parse_args_single_tag_honors_explicit_socket_path():
    (config,) = parse_args(
        ["--device", "3", "--tag", "weights", "--socket-path", "/run/gms.sock"]
    )

    assert config.socket_path == "/run/gms.sock"


@pytest.mark.parametrize(
    "argv",
    [
        ["--device", "3", "--tag", "weight"],
        ["--device", "3", "--tag", "weights", "--tag", "bogus"],
    ],
)
def test_parse_args_rejects_unknown_tags(argv, capsys):
    with pytest.raises(SystemExit):
        parse_args(argv)

    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        # Default tags: one socket path cannot serve both listeners.
        ["--device", "3", "--socket-path", "/run/gms.sock"],
        # Explicit multiple tags with one socket path.
        [
            "--device",
            "3",
            "--tag",
            "weights",
            "--tag",
            "kv_cache",
            "--socket-path",
            "/run/gms.sock",
        ],
    ],
)
def test_parse_args_rejects_socket_path_for_multiple_tags(argv, capsys):
    with pytest.raises(SystemExit):
        parse_args(argv)

    assert "requires exactly one --tag" in capsys.readouterr().err


def test_parse_args_rejects_duplicate_tags(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--device", "3", "--tag", "weights", "--tag", "weights"])

    assert "must be unique" in capsys.readouterr().err


@pytest.mark.timeout(10)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("crash", "match"),
    [
        pytest.param(True, "listener failed", id="listener-crash"),
        pytest.param(False, "stopped unexpectedly", id="clean-stop"),
    ],
)
async def test_server_stop_cancels_other_listener(crash, match):
    both_started = asyncio.Event()
    started = 0
    sibling_cancelled = asyncio.Event()

    class Server:
        def __init__(self, stops: bool) -> None:
            self.stops = stops

        async def serve(self) -> None:
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await both_started.wait()
            if self.stops:
                if crash:
                    raise RuntimeError("listener failed")
                return  # A clean return must still be fail-closed.
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

    with pytest.raises(RuntimeError, match=match):
        await runner.run_servers([Server(stops=True), Server(stops=False)])

    assert sibling_cancelled.is_set()
