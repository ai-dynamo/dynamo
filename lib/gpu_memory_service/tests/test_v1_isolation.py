# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GMS V1 rank isolation and in-process loading arguments."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from concurrent.futures import Future
from pathlib import Path

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.snapshot.backends import mx
from gpu_memory_service.v1 import isolation
from gpu_memory_service.v1.snapshot import loader

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
    """A node with three cards whose minors are not in bus order; two are visible."""
    proc_gpus, dev = tmp_path / "gpus", tmp_path / "dev"
    dev.mkdir()
    _write_card(proc_gpus, "0000:29:00.0", "GPU-b", 0)
    _write_card(proc_gpus, "0000:18:00.0", "GPU-a", 6)
    _write_card(proc_gpus, "0000:3a:00.0", "GPU-hidden", 3)
    for minor in (0, 6):
        (dev / f"nvidia{minor}").touch()
    return proc_gpus, dev


def test_node_gpus_lists_openable_cards_in_bus_order(node):
    proc_gpus, dev = node

    gpus = isolation.node_gpus(proc_gpus, dev)

    assert gpus == [
        isolation.Gpu("0000:18:00.0", 6, "GPU-a"),
        isolation.Gpu("0000:29:00.0", 0, "GPU-b"),
    ]


class _Popen:
    started: list[_Popen] = []

    def __init__(self, command, **kwargs) -> None:
        self.command, self.kwargs, self.pid = command, kwargs, 1000 + len(self.started)
        self.started.append(self)


@pytest.fixture
def popen(monkeypatch):
    _Popen.started = []
    monkeypatch.setattr(isolation.subprocess, "Popen", _Popen)
    return _Popen


def test_unprivileged_ranks_confine_themselves(node, tmp_path, popen, monkeypatch):
    proc_gpus, dev = node
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 1000)
    socket_dir = tmp_path / "sockets"
    socket_dir.mkdir()

    isolation.start_ranks(
        isolation.node_gpus(proc_gpus, dev),
        str(socket_dir),
        lambda r: [f"rank{r}"],
        dev,
    )

    assert [p.command for p in popen.started] == [["rank0"], ["rank1"]]
    for rank, (process, minor) in enumerate(zip(popen.started, (6, 0))):
        env = process.kwargs["env"]
        assert env[isolation.CONFINE_ENV] == str(minor)
        assert env["GMS_SOCKET_DIR"] == str(socket_dir / f"rank-{rank}")
        for domain in ("weights", "kv_cache"):
            link = socket_dir / f"gms_{rank}_{domain}.sock"
            assert os.readlink(link) == f"rank-{rank}/gms_{rank}_{domain}.sock"


def test_root_runs_each_rank_as_its_own_user(node, tmp_path, popen, monkeypatch):
    proc_gpus, dev = node
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 0)
    events = []
    monkeypatch.setattr(
        isolation.os,
        "chown",
        lambda path, uid, gid: events.append(("chown", str(path), gid)),
    )
    monkeypatch.setattr(
        isolation.os,
        "chmod",
        lambda path, mode: events.append(("chmod", str(path), mode)),
    )
    socket_dir = tmp_path / "sockets"
    socket_dir.mkdir()
    gpus = isolation.node_gpus(proc_gpus, dev)
    # Each rank's GPU node belongs to its group alone before any rank starts.
    node_events = [
        ("chown", str(dev / "nvidia6"), 20000),
        ("chmod", str(dev / "nvidia6"), 0o660),
        ("chown", str(dev / "nvidia0"), 20001),
        ("chmod", str(dev / "nvidia0"), 0o660),
    ]

    isolation.start_ranks(gpus, str(socket_dir), lambda r: [f"rank{r}"], dev)

    assert events[:4] == node_events
    assert ("chmod", str(socket_dir / "rank-1"), 0o700) in events
    assert events[-1] == ("chmod", str(socket_dir), 0o755)
    for rank, process in enumerate(popen.started):
        assert process.kwargs["user"] == process.kwargs["group"] == 20000 + rank
        assert process.kwargs["extra_groups"] == []
        assert process.kwargs["preexec_fn"] is isolation._set_no_new_privs
        assert isolation.CONFINE_ENV not in process.kwargs["env"]


def _landlock_available() -> bool:
    import ctypes

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
        from pathlib import Path
        from gpu_memory_service.v1.isolation import confine_to_gpu
        dev = Path({str(dev)!r})
        confine_to_gpu(0, dev)
        for name in ("nvidia0", "nvidiactl", "null", "../other"):
            (dev / name).open("r+").close()
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


def test_loader_rank_selects_peer_rank_and_port():
    args = loader.parse_args(
        [
            "--checkpoint-dir",
            "/ckpt",
            "--transfer-backend",
            "mx",
            "--mx-model",
            "m",
            "--mx-tensor-parallel-size",
            "8",
            "--mx-listen-port",
            "25555",
            "--device",
            "0",
            "--rank",
            "3",
        ]
    )

    assert args.device == 0
    assert args.backend_params["mx_rank"] == 3
    assert args.backend_params["mx_listen_port"] == 25558


def test_loader_rank_defaults_to_device():
    args = loader.parse_args(["--checkpoint-dir", "/ckpt", "--device", "2"])

    assert args.rank == 2
    assert args.backend_params == {}


def test_mx_backend_accepts_a_future_nixl_manager():
    manager = object()
    future: Future = Future()
    future.set_result(manager)

    assert mx._resolve(future) is manager
    assert mx._resolve(manager) is manager
