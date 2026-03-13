# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from gpu_memory_service.client.torch import allocator as allocator_module
from gpu_memory_service.common.types import GrantedLockType, RequestedLockType

pytestmark = [
    pytest.mark.unit,
]


class _FakeManager:
    def __init__(self, socket_path: str, *, device: int):
        self.socket_path = socket_path
        self.device = device
        self._connected = False
        self._granted_lock_type: GrantedLockType | None = None
        self._mappings: dict[int, object] = {}
        self._unmapped = False

    @property
    def granted_lock_type(self) -> GrantedLockType | None:
        return self._granted_lock_type

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def mappings(self) -> dict[int, object]:
        return self._mappings

    @property
    def is_unmapped(self) -> bool:
        return self._unmapped

    def connect(self, mode: RequestedLockType, timeout_ms: int | None = None) -> None:
        del timeout_ms
        self._connected = True
        if mode == RequestedLockType.RW:
            self._granted_lock_type = GrantedLockType.RW
            return
        self._granted_lock_type = GrantedLockType.RO

    @property
    def _client_rpc(self):
        return self

    def get_lock_state(self) -> object:
        return object()


@pytest.fixture(autouse=True)
def clear_tag_states():
    allocator_module._tag_states.clear()
    yield
    allocator_module._tag_states.clear()


@pytest.fixture
def fake_allocator(monkeypatch):
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.GMSClientMemoryManager",
        _FakeManager,
    )
    monkeypatch.setattr(
        allocator_module,
        "_ensure_callbacks_initialized",
        lambda: None,
    )
    monkeypatch.setattr(allocator_module, "_create_mem_pool", lambda: object())


def test_tag_registry_rejects_socket_or_device_mismatch(fake_allocator):
    manager = allocator_module.get_or_create_gms_client_memory_manager(
        "/tmp/weights.sock",
        0,
        RequestedLockType.RO,
        tag="weights",
    )

    with pytest.raises(RuntimeError, match="initialized for"):
        allocator_module.get_or_create_gms_client_memory_manager(
            "/tmp/other.sock",
            0,
            RequestedLockType.RO,
            tag="weights",
        )

    with pytest.raises(RuntimeError, match="initialized for"):
        allocator_module.get_or_create_gms_client_memory_manager(
            "/tmp/weights.sock",
            1,
            RequestedLockType.RO,
            tag="weights",
        )

    assert manager.is_connected


def test_tag_registry_recreates_disconnected_empty_manager(fake_allocator):
    first = allocator_module.get_or_create_gms_client_memory_manager(
        "/tmp/weights.sock",
        0,
        RequestedLockType.RO,
        tag="weights",
    )
    first._connected = False
    first._granted_lock_type = None

    second = allocator_module.get_or_create_gms_client_memory_manager(
        "/tmp/weights.sock",
        0,
        RequestedLockType.RO,
        tag="weights",
    )

    assert second is not first
    assert second.is_connected
    assert second.granted_lock_type == GrantedLockType.RO


def test_tag_registry_rejects_disconnected_manager_with_preserved_state(
    fake_allocator,
):
    manager = allocator_module.get_or_create_gms_client_memory_manager(
        "/tmp/weights.sock",
        0,
        RequestedLockType.RO,
        tag="weights",
    )
    manager._connected = False
    manager._mappings[0x1000] = object()

    with pytest.raises(RuntimeError, match="preserved state"):
        allocator_module.get_or_create_gms_client_memory_manager(
            "/tmp/weights.sock",
            0,
            RequestedLockType.RO,
            tag="weights",
        )


def test_close_evicts_manager_from_tag_registry(fake_allocator):
    manager = allocator_module.get_or_create_gms_client_memory_manager(
        "/tmp/weights.sock",
        0,
        RequestedLockType.RO,
        tag="weights",
    )
    allocator_module.evict_gms_client_memory_manager(manager)
    assert allocator_module.get_gms_client_memory_manager("weights") is None
