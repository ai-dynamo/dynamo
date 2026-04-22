# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tests.fault_tolerance.migration import test_sglang
from tests.utils.managed_process import ManagedProcess

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


class _DummyRequest:
    def __init__(self, node_name: str = "test_dynamic_sglang_ports"):
        self.node = SimpleNamespace(name=node_name)

    def getfixturevalue(self, name: str):
        if name != "request_plane":
            raise KeyError(name)
        return "tcp"


def _command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_prefill_worker_allocates_dynamic_ports_and_releases_them(monkeypatch):
    allocated = iter([9101, 12355, 24077])
    requested_start_ports: list[int] = []
    released_ports: list[list[int]] = []
    managed_process_calls: list[dict] = []

    def fake_allocate_port(start_port: int) -> int:
        requested_start_ports.append(start_port)
        return next(allocated)

    def fake_deallocate_ports(ports: list[int]) -> None:
        released_ports.append(list(ports))

    def fake_managed_process_init(self, **kwargs):
        managed_process_calls.append(kwargs)

    monkeypatch.setattr(test_sglang, "allocate_port", fake_allocate_port)
    monkeypatch.setattr(test_sglang, "deallocate_ports", fake_deallocate_ports)
    monkeypatch.setattr(ManagedProcess, "__init__", fake_managed_process_init)
    monkeypatch.setattr(ManagedProcess, "__exit__", lambda self, *args: False)

    worker = test_sglang.DynamoWorkerProcess(
        _DummyRequest(),
        "worker7",
        frontend_port=8000,
        disagg_mode="prefill",
    )

    assert requested_start_ports == [9100, 12340, 24000]
    assert worker.system_port == 9101
    assert worker.disaggregation_bootstrap_port == 12355
    assert worker.prefill_port == 24077
    assert worker._allocated_ports == [9101, 12355, 24077]

    command = managed_process_calls[0]["command"]
    assert _command_value(command, "--disaggregation-bootstrap-port") == "12355"
    assert _command_value(command, "--port") == "24077"
    assert "40000" not in command

    worker.__exit__(None, None, None)

    assert released_ports == [[9101, 12355, 24077]]
    assert worker._allocated_ports == []


def test_prefill_worker_releases_ports_when_init_fails(monkeypatch):
    allocated = iter([9102, 12356, 24078])
    released_ports: list[list[int]] = []

    monkeypatch.setattr(
        test_sglang, "allocate_port", lambda start_port: next(allocated)
    )
    monkeypatch.setattr(
        test_sglang,
        "deallocate_ports",
        lambda ports: released_ports.append(list(ports)),
    )

    def fail_managed_process_init(self, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ManagedProcess, "__init__", fail_managed_process_init)

    with pytest.raises(RuntimeError, match="boom"):
        test_sglang.DynamoWorkerProcess(
            _DummyRequest("test_dynamic_sglang_ports_failure"),
            "worker8",
            frontend_port=8000,
            disagg_mode="prefill",
        )

    assert released_ports == [[9102, 12356, 24078]]
