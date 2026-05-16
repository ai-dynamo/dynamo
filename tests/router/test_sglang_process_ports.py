# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tests.router import test_router_e2e_with_sglang as sglang_e2e
from tests.utils.constants import DefaultPort

pytestmark = [
    pytest.mark.unit,
    pytest.mark.router,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _FakeRequest:
    node = SimpleNamespace(name="test_sglang_process_ports")

    def __init__(self):
        self.finalizers = []

    def addfinalizer(self, finalizer):
        self.finalizers.append(finalizer)


def _kv_events_config(command: list[str]) -> str:
    return command[command.index("--kv-events-config") + 1]


def test_sglang_dp_workers_use_contiguous_kv_event_port_blocks(monkeypatch):
    def fake_allocate_ports(count: int, start_port: int) -> list[int]:
        assert (count, start_port) == (2, DefaultPort.SYSTEM1.value)
        return [10000, 10010]

    contiguous_calls = []

    def fake_allocate_contiguous_ports(
        count: int, block_size: int, start_port: int
    ) -> list[int]:
        contiguous_calls.append((count, block_size, start_port))
        return [11000, 11001, 11010, 11011]

    monkeypatch.setattr(sglang_e2e, "allocate_ports", fake_allocate_ports)
    monkeypatch.setattr(
        sglang_e2e, "allocate_contiguous_ports", fake_allocate_contiguous_ports
    )

    process = sglang_e2e.SGLangProcess(
        _FakeRequest(),
        sglang_args=sglang_e2e.SGLANG_ARGS,
        num_workers=2,
        data_parallel_size=2,
    )

    assert contiguous_calls == [(2, 2, DefaultPort.SYSTEM1.value)]
    assert '"endpoint":"tcp://*:11000"' in _kv_events_config(
        process.worker_processes[0].command
    )
    assert '"endpoint":"tcp://*:11010"' in _kv_events_config(
        process.worker_processes[1].command
    )
