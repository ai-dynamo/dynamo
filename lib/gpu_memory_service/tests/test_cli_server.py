# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GMS process topology."""

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

from gpu_memory_service.cli import runner, server
from gpu_memory_service.v1 import device as v1_device

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_v1_socket_path_rejects_af_unix_overflow(monkeypatch):
    monkeypatch.setenv("GMS_SOCKET_DIR", "/" + "s" * 200)
    monkeypatch.setattr(v1_device, "get_device_uuid", lambda _device: "GPU-0")

    with pytest.raises(ValueError, match="too long for AF_UNIX"):
        v1_device.get_socket_path(0, "weights")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ("weights", "kv_cache")),
        ("kv_cache", ("kv_cache",)),
        (" weights, kv_cache ", ("weights", "kv_cache")),
    ],
)
def test_server_tags_from_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("GMS_SERVER_TAGS", raising=False)
    else:
        monkeypatch.setenv("GMS_SERVER_TAGS", raw)

    assert server._tags_from_env() == expected


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (" , ", "at least one tag"),
        ("weights,unknown", "unsupported"),
        ("kv_cache,kv_cache", "duplicate"),
    ],
)
def test_server_tags_from_env_rejects_invalid_values(monkeypatch, raw, match):
    monkeypatch.setenv("GMS_SERVER_TAGS", raw)

    with pytest.raises(RuntimeError, match=match):
        server._tags_from_env()


def test_directory_command_uses_distinct_protocol_server():
    assert server._directory_command("/run/gms/directory.sock") == [
        sys.executable,
        "-m",
        "gms_kv_ring.daemon.directory_server",
        "/run/gms/directory.sock",
    ]


def test_main_supervises_optional_directory_with_cuda_servers(monkeypatch):
    commands = []
    supervised = []

    class Process:
        pid = 123

        def poll(self):
            return None

        def terminate(self):
            return None

    class Vmm:
        @staticmethod
        def list_devices():
            return [3]

    monkeypatch.delenv("DYN_GMS_USE_V1", raising=False)
    monkeypatch.setenv("GMS_DIRECTORY_SOCKET", "/run/gms/directory.sock")
    monkeypatch.setenv("GMS_SERVER_TAGS", "kv_cache")
    monkeypatch.setattr(server, "init_vmm", lambda *_args: None)
    monkeypatch.setattr(server, "get_vmm", lambda: Vmm())
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda command: (commands.append(command), Process())[1],
    )
    monkeypatch.setattr(server.signal, "signal", lambda *_args: None)

    def supervise(servers, loaders):
        supervised.extend(servers)
        assert loaders == []
        return 0

    monkeypatch.setattr(server, "_supervise", supervise)

    with pytest.raises(SystemExit) as exc:
        server.main([])

    assert exc.value.code == 0
    assert commands == [
        server._directory_command("/run/gms/directory.sock"),
        [
            sys.executable,
            "-m",
            "gpu_memory_service",
            "--device",
            "3",
            "--device-type",
            "cuda",
            "--tag",
            "kv_cache",
        ],
    ]
    assert len(supervised) == 2


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
