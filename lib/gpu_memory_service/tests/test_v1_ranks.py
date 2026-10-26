# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GMS V1 per-GPU server processes."""

from __future__ import annotations

from pathlib import Path

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.v1 import device, ranks

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
    """Three cards whose minors are not in bus order; two are visible."""
    proc_gpus, dev = tmp_path / "gpus", tmp_path / "dev"
    dev.mkdir()
    _write_card(proc_gpus, "0000:29:00.0", "GPU-b", 0)
    _write_card(proc_gpus, "0000:18:00.0", "GPU-a", 6)
    _write_card(proc_gpus, "0000:3a:00.0", "GPU-hidden", 3)
    for minor in (0, 6):
        (dev / f"nvidia{minor}").touch()
    return proc_gpus, dev


def test_node_gpus_lists_openable_cards_in_bus_order(node):
    assert ranks.node_gpus(*node) == [
        ranks.Gpu("0000:18:00.0", 6, "GPU-a"),
        ranks.Gpu("0000:29:00.0", 0, "GPU-b"),
    ]


def test_each_device_process_is_pinned_to_its_gpu(node, monkeypatch):
    started = []

    class Popen:
        def __init__(self, command, env):
            self.pid = len(started)
            started.append((command, env))

    monkeypatch.setattr(ranks.subprocess, "Popen", Popen)

    ranks.start_ranks(ranks.node_gpus(*node), lambda n: [f"device{n}"])

    assert [command for command, _ in started] == [["device0"], ["device1"]]
    assert [env["CUDA_VISIBLE_DEVICES"] for _, env in started] == ["GPU-a", "GPU-b"]


@pytest.mark.parametrize(
    ("visible", "expected"),
    [(None, 3), ("GPU-a", 0), ("3", 0), ("2,3", 3)],
)
def test_cuda_ordinal_of_a_pinned_process_is_zero(monkeypatch, visible, expected):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    assert device.cuda_ordinal(3) == expected
