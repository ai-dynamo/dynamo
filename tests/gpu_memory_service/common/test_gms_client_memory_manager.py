# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from gpu_memory_service.client import memory_manager as memory_manager_module
from gpu_memory_service.client.memory_manager import (
    GMSClientMemoryManager,
    LocalMapping,
)
from gpu_memory_service.common.types import GrantedLockType, RequestedLockType

pytestmark = [
    pytest.mark.unit,
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


class _TrackingSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FailingCommitSession:
    is_connected = True

    def commit(self) -> bool:
        raise ConnectionError("commit failed after local unmap")


class _SuccessfulCommitSession:
    is_connected = True

    def commit(self) -> bool:
        return True


class _CloseFailingSession:
    def close(self) -> None:
        raise ConnectionError("close failed")


def _make_mapping(
    allocation_id: str,
    va: int,
    *,
    handle: int,
    tag: str = "weights",
) -> LocalMapping:
    return LocalMapping(
        allocation_id=allocation_id,
        va=va,
        size=4096,
        aligned_size=4096,
        handle=handle,
        tag=tag,
    )


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


def _make_manager(
    monkeypatch,
    *,
    client: object | None,
    lock_type: GrantedLockType | None,
    mappings: list[LocalMapping] | None = None,
    unmapped: bool = False,
    va_preserved: bool = False,
    layout_hash: str = "",
) -> GMSClientMemoryManager:
    monkeypatch.setattr(
        memory_manager_module, "cuda_set_current_device", lambda _device: None
    )
    manager = object.__new__(GMSClientMemoryManager)
    manager.socket_path = "/tmp/gms-test.sock"
    manager.device = 0
    manager._client = client
    manager._granted_lock_type = lock_type
    manager._mappings = {}
    manager._inverse_mapping = {}
    manager._unmapped = unmapped
    manager._va_preserved = va_preserved
    manager._last_memory_layout_hash = layout_hash
    manager.granularity = 4096

    for mapping in mappings or []:
        manager._track_mapping(mapping)

    return manager


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


def test_connect_rejects_double_connect(monkeypatch):
    manager = _make_manager(
        monkeypatch,
        client=object(),
        lock_type=GrantedLockType.RO,
    )

    with pytest.raises(RuntimeError, match="already connected"):
        manager.connect(RequestedLockType.RO)


def test_disconnect_rejects_live_mappings(monkeypatch):
    session = _TrackingSession()
    manager = _make_manager(
        monkeypatch,
        client=session,
        lock_type=GrantedLockType.RO,
        mappings=[_make_mapping("alloc-1", 0x1000, handle=1234)],
    )

    with pytest.raises(RuntimeError, match="unmapped first"):
        manager.disconnect()

    assert not session.closed


def test_commit_failure_after_local_unmap_keeps_preserved_unmapped_state(monkeypatch):
    manager = _make_manager(
        monkeypatch,
        client=_FailingCommitSession(),
        lock_type=GrantedLockType.RW,
        mappings=[
            _make_mapping("alloc_1", 0x1000, handle=11),
            _make_mapping("alloc_2", 0x2000, handle=0),
        ],
    )

    unmapped_vas: list[int] = []

    monkeypatch.setattr(memory_manager_module, "cuda_synchronize", lambda: None)

    def fake_unmap_va(self, va: int) -> None:
        unmapped_vas.append(va)
        self._mappings[va] = self._mappings[va].with_handle(0)

    monkeypatch.setattr(GMSClientMemoryManager, "unmap_va", fake_unmap_va)

    with pytest.raises(ConnectionError, match="commit failed after local unmap"):
        manager.commit()

    assert unmapped_vas == [0x1000]
    assert manager._mappings[0x1000].handle == 0
    assert manager._mappings[0x2000].handle == 0
    assert manager._va_preserved
    assert manager._unmapped
    assert manager._client is not None


def test_successful_commit_clears_rw_mode_before_local_cleanup(monkeypatch):
    manager = _make_manager(
        monkeypatch,
        client=_SuccessfulCommitSession(),
        lock_type=GrantedLockType.RW,
        mappings=[_make_mapping("alloc_1", 0x1000, handle=11)],
    )

    free_handle_calls: list[str] = []

    monkeypatch.setattr(memory_manager_module, "cuda_synchronize", lambda: None)

    def fake_unmap_va(self, va: int) -> None:
        self._mappings[va] = self._mappings[va].with_handle(0)

    def fake_free_va(self, va: int) -> None:
        mapping = self._mappings.pop(va)
        self._inverse_mapping.pop(mapping.allocation_id, None)

    def fake_free_handle(self, allocation_id: str) -> bool:
        free_handle_calls.append(allocation_id)
        return True

    monkeypatch.setattr(GMSClientMemoryManager, "unmap_va", fake_unmap_va)
    monkeypatch.setattr(GMSClientMemoryManager, "free_va", fake_free_va)
    monkeypatch.setattr(GMSClientMemoryManager, "free_handle", fake_free_handle)

    assert manager.commit()
    assert manager._client is None
    assert manager._granted_lock_type is None
    assert manager._unmapped
    assert manager._va_preserved

    manager.destroy_mapping(0x1000)
    assert free_handle_calls == []


def test_destroy_mapping_keeps_local_state_when_server_free_fails(monkeypatch):
    mapping = _make_mapping("alloc_1", 0x1000, handle=77)
    manager = _make_manager(
        monkeypatch,
        client=None,
        lock_type=GrantedLockType.RW,
        mappings=[mapping],
    )

    monkeypatch.setattr(
        manager,
        "free_handle",
        lambda allocation_id: (_ for _ in ()).throw(RuntimeError("server free failed")),
    )

    with pytest.raises(RuntimeError, match="server free failed"):
        manager.destroy_mapping(mapping.va)

    assert manager.mappings[mapping.va] == mapping


def test_disconnect_clears_local_state_even_if_close_fails(monkeypatch):
    manager = _make_manager(
        monkeypatch,
        client=_CloseFailingSession(),
        lock_type=GrantedLockType.RO,
    )

    with pytest.raises(ConnectionError, match="close failed"):
        manager.disconnect()

    assert manager._client is None
    assert manager._granted_lock_type is None
