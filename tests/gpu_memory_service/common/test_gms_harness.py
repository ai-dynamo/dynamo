# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.gpu_memory_service.harness.gms import GMSServerProcess
from tests.utils.managed_process import ManagedProcess

pytestmark = [pytest.mark.unit]


@pytest.fixture
def request_stub():
    return SimpleNamespace(node=SimpleNamespace(name="gms-harness"))


def test_server_process_refuses_foreign_live_socket(monkeypatch, request_stub):
    server = GMSServerProcess(request_stub, device=0, tag="weights")

    monkeypatch.setattr(
        "tests.gpu_memory_service.harness.gms.os.path.exists", lambda path: True
    )
    monkeypatch.setattr(
        "tests.gpu_memory_service.harness.gms._socket_has_live_gms", lambda path: True
    )

    with pytest.raises(RuntimeError, match="already active"):
        server.__enter__()


def test_server_process_unlinks_only_stale_socket_on_exit(monkeypatch, request_stub):
    server = GMSServerProcess(request_stub, device=0, tag="weights")
    unlinked: list[str] = []

    monkeypatch.setattr(
        ManagedProcess,
        "__exit__",
        lambda self, exc_type, exc_val, exc_tb: False,
    )
    monkeypatch.setattr(
        "tests.gpu_memory_service.harness.gms.os.path.exists", lambda path: True
    )
    monkeypatch.setattr(
        "tests.gpu_memory_service.harness.gms.os.unlink", unlinked.append
    )

    monkeypatch.setattr(
        "tests.gpu_memory_service.harness.gms._socket_has_live_gms", lambda path: True
    )
    server.__exit__(None, None, None)
    assert unlinked == []

    monkeypatch.setattr(
        "tests.gpu_memory_service.harness.gms._socket_has_live_gms", lambda path: False
    )
    server.__exit__(None, None, None)
    assert unlinked == [server.socket_path]
