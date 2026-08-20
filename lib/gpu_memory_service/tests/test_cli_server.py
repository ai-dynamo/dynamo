# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for production GMS process and listener topology."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

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
from gpu_memory_service.v1 import device as v1_device

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_child_command_launches_default_multi_tag_runner():
    assert server._child_command(3, "cuda") == [
        sys.executable,
        "-m",
        "gpu_memory_service",
        "--device",
        "3",
        "--device-type",
        "cuda",
    ]


def test_v1_cli_dispatches_cuda_only_child(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: (
            calls.append(("import", name))
            or SimpleNamespace(main=lambda argv: calls.append(("main", argv)))
        ),
    )

    runner.main(["--use-v1", "--device", "3"])

    assert calls == [
        ("import", "gpu_memory_service.v1.cli"),
        ("main", ["--device", "3"]),
    ]
    assert server._child_command(3, "cuda", use_v1=True) == [
        sys.executable,
        "-m",
        "gpu_memory_service",
        "--use-v1",
        "--device",
        "3",
    ]
    with pytest.raises(SystemExit):
        server.main(["--use-v1", "--device-type", "xpu"])
    assert "--use-v1 only supports --device-type=cuda" in capsys.readouterr().err


def test_enable_loader_requires_use_v1(capsys):
    with pytest.raises(SystemExit):
        server.main(["--enable-loader", "--checkpoint-dir", "/ckpt"])
    assert "--enable-loader requires --use-v1" in capsys.readouterr().err


def test_enable_loader_cannot_combine_with_probe(capsys):
    with pytest.raises(SystemExit):
        server.main(["--use-v1", "--probe-restore-ready", "--enable-loader"])
    assert "--probe-restore-ready cannot be combined with --enable-loader" in (
        capsys.readouterr().err
    )


def test_v1_socket_path_rejects_af_unix_overflow(monkeypatch):
    monkeypatch.setenv("GMS_SOCKET_DIR", "/" + "s" * 200)
    monkeypatch.setattr(v1_device, "get_device_uuid", lambda _device: "GPU-0")

    with pytest.raises(ValueError, match="too long for AF_UNIX"):
        v1_device.get_socket_path(0, "weights")


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
    assert server._supervise(clean) == 1
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
