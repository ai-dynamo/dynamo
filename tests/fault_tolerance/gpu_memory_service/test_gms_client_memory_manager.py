# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from gpu_memory_service.client import memory_manager as memory_manager_module
from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.common.types import GrantedLockType

pytestmark = [
    pytest.mark.unit,
    pytest.mark.fault_tolerance,
]


class _FakeSession:
    def __init__(self):
        self.lock_type = GrantedLockType.RW
        self.committed = False
        self.closed = False

    @property
    def is_connected(self) -> bool:
        return not self.closed

    def get_memory_layout_hash(self) -> str:
        return ""

    def commit(self) -> bool:
        self.closed = True
        return True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(
        memory_manager_module, "cuda_set_current_device", lambda _device: None
    )
    monkeypatch.setattr(
        memory_manager_module, "cumem_get_allocation_granularity", lambda _device: 65536
    )
    monkeypatch.setattr(memory_manager_module, "cuda_synchronize", lambda: None)
    return GMSClientMemoryManager("/tmp/gms-test.sock", device=0)


def test_commit_clears_client_lock_state(manager):
    manager._client = _FakeSession()
    manager._granted_lock_type = GrantedLockType.RW

    assert manager.commit()

    assert manager.granted_lock_type is None
    assert not manager.is_connected
    assert manager.is_unmapped


def test_disconnect_clears_client_lock_state(manager):
    manager._client = _FakeSession()
    manager._granted_lock_type = GrantedLockType.RW

    manager.disconnect()

    assert manager.granted_lock_type is None
    assert not manager.is_connected
