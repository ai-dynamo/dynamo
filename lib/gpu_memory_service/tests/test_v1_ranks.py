# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GMS V1 per-GPU server processes and their Landlock confinement."""

from __future__ import annotations

import ctypes
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.cli import server
from gpu_memory_service.v1 import device, landlock, ranks

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def _write_card(proc_gpus: Path, bus_id: str, uuid: str, minor: int) -> None:
    card = proc_gpus / bus_id
    card.mkdir(parents=True)
    (card / "information").write_text(
        "Model: \t\t NVIDIA B200\n"
        f"GPU UUID: \t {uuid}\n"
        f"Bus Location: \t {bus_id}\n"
        f"Device Minor: \t {minor}\n"
    )


@pytest.fixture
def node(tmp_path):
    """Four cards whose minors are not in bus order. Two can be opened."""
    proc_gpus, dev = tmp_path / "gpus", tmp_path / "dev"
    dev.mkdir()
    _write_card(proc_gpus, "0000:29:00.0", "GPU-b", 0)
    _write_card(proc_gpus, "0000:18:00.0", "GPU-a", 6)
    _write_card(proc_gpus, "0000:3a:00.0", "GPU-hidden", 3)
    for minor in (0, 6):
        (dev / f"nvidia{minor}").touch()
    _write_card(proc_gpus, "0000:4b:00.0", "GPU-denied", 2)
    # A node that exists but does not open, like one the device cgroup denies.
    (dev / "nvidia2").mkdir()
    return proc_gpus, dev


def test_node_gpus_lists_openable_cards_in_bus_order(node):
    assert device.node_gpus(*node) == [
        device.Gpu("0000:18:00.0", 6, "GPU-a"),
        device.Gpu("0000:29:00.0", 0, "GPU-b"),
    ]


def test_gms_device_n_is_the_nth_gpu_in_bus_order(node, monkeypatch):
    gpus = device.node_gpus(*node)
    monkeypatch.setattr(device, "node_gpus", lambda: gpus)

    assert [device.gpu_uuid(n) for n in range(2)] == ["GPU-a", "GPU-b"]
    with pytest.raises(ValueError, match="GMS device 2 is not one of 2 GPUs"):
        device.gpu_uuid(2)


@pytest.mark.parametrize("confine", [True, False])
def test_each_device_process_is_pinned_to_its_gpu(node, monkeypatch, confine):
    started = []

    class Popen:
        def __init__(self, command, env, preexec_fn):
            self.pid = len(started)
            started.append((command, env, preexec_fn))

    monkeypatch.setattr(ranks.subprocess, "Popen", Popen)

    ranks.start_ranks(
        device.node_gpus(*node),
        lambda n, gpu: [f"device{n}", gpu.uuid],
        confine=confine,
    )

    assert [command for command, _, _ in started] == [
        ["device0", "GPU-a"],
        ["device1", "GPU-b"],
    ]
    assert [env["CUDA_VISIBLE_DEVICES"] for _, env, _ in started] == [
        "GPU-a",
        "GPU-b",
    ]
    if confine:
        assert [fn.args for _, _, fn in started] == [(6,), (0,)]
    else:
        assert [fn for _, _, fn in started] == [None, None]


def test_supervisor_without_landlock_runs_unconfined(node, monkeypatch, caplog):
    def unavailable():
        raise OSError(38, "Landlock: Function not implemented")

    started = []

    def start_ranks(gpus, command, *, confine):
        started.append(([command(n, gpu) for n, gpu in enumerate(gpus)], confine))
        return []

    monkeypatch.setattr(landlock, "abi_version", unavailable)
    monkeypatch.setattr(server, "node_gpus", lambda: device.node_gpus(*node))
    monkeypatch.setattr(server.ranks, "start_ranks", start_ranks)
    monkeypatch.setattr(server.signal, "signal", lambda *_args: None)

    assert server._run_v1_ranks(["--checkpoint-dir", "/ckpt"], confine=True) == 0

    [(commands, confine)] = started
    assert not confine
    assert "Running them unconfined" in caplog.text
    # Each process is told its GPU, ahead of the loader's arguments.
    assert commands[1][-6:] == [
        "1",
        "--gpu-uuid",
        "GPU-b",
        "--enable-loader",
        "--checkpoint-dir",
        "/ckpt",
    ]


def test_cuda_ordinal_finds_the_gpu_whatever_order_cuda_uses(monkeypatch):
    class _Cuda:
        class CUresult:
            CUDA_SUCCESS = 0

        @staticmethod
        def cuDeviceGetCount():
            return 0, 3

    cuda_order = ["GPU-c", "GPU-a", "GPU-b"]
    monkeypatch.setattr(device, "cuda", _Cuda)
    monkeypatch.setattr(device, "cuda_ensure_initialized", lambda: None)
    monkeypatch.setattr(device, "get_device_uuid", cuda_order.__getitem__)

    assert device.cuda_ordinal("GPU-b") == 2
    with pytest.raises(RuntimeError, match="GPU-d is not visible to CUDA"):
        device.cuda_ordinal("GPU-d")


def _landlock_available() -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    return libc.syscall(444, None, ctypes.c_size_t(0), ctypes.c_uint32(1)) >= 1


@pytest.mark.skipif(not _landlock_available(), reason="kernel without Landlock")
def test_confined_process_opens_only_its_own_gpu_node(tmp_path):
    dev = tmp_path / "dev"
    dev.mkdir()
    for name in ("nvidia0", "nvidia1", "nvidiactl", "null"):
        (dev / name).write_text("")
    (tmp_path / "other").write_text("")
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from gpu_memory_service.v1.landlock import confine_to_gpu
        dev = Path({str(dev)!r})
        confine_to_gpu(0, dev)
        for name in ("nvidia0", "nvidiactl", "null", "../other"):
            (dev / name).open("r+").close()
        # File permissions still allow it, which is why GPUs are listed by
        # opening their nodes, and before confining.
        assert os.access(dev / "nvidia1", os.R_OK | os.W_OK)
        try:
            (dev / "nvidia1").open("r+")
        except PermissionError:
            print("blocked")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "blocked"
