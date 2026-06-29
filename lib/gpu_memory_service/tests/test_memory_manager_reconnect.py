# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_aborted_persistent_kv_manager_refreshes_to_kv_cache_socket(monkeypatch):
    import gpu_memory_service.common.utils as common_utils
    from gpu_memory_service.client import memory_manager
    from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType

    session_calls = []

    class FakeSession:
        def __init__(self, socket_path, lock_type, timeout_ms):
            session_calls.append((socket_path, lock_type, timeout_ms))
            self.lock_type = GrantedLockType.RW_PERSISTENT
            self.committed = False
            self.is_connected = True

        def get_memory_layout_hash(self):
            return ""

    monkeypatch.setattr(memory_manager, "cuda_ensure_initialized", lambda: None)
    monkeypatch.setattr(
        memory_manager,
        "cumem_get_allocation_granularity",
        lambda device: 2 * 1024 * 1024,
    )
    monkeypatch.setattr(memory_manager, "_GMSClientSession", FakeSession)
    monkeypatch.setattr(common_utils, "invalidate_uuid_cache", lambda: None)
    monkeypatch.setattr(
        common_utils, "get_socket_path", lambda device, tag: f"socket:{device}:{tag}"
    )

    manager = memory_manager.GMSClientMemoryManager(
        "/tmp/gms_GPU-old_kv_cache.sock", device=2, tag="kv_pool:cuda2"
    )
    manager._aborted = True

    manager.connect(RequestedLockType.RW_PERSISTENT)

    assert session_calls == [
        ("socket:2:kv_cache", RequestedLockType.RW_PERSISTENT, None)
    ]
    assert manager.socket_path == "socket:2:kv_cache"


def test_aborted_weight_manager_refreshes_to_its_own_socket_tag(monkeypatch):
    import gpu_memory_service.common.utils as common_utils
    from gpu_memory_service.client import memory_manager
    from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType

    session_calls = []

    class FakeSession:
        def __init__(self, socket_path, lock_type, timeout_ms):
            session_calls.append((socket_path, lock_type, timeout_ms))
            self.lock_type = GrantedLockType.RO
            self.committed = False
            self.is_connected = True

        def get_memory_layout_hash(self):
            return ""

    monkeypatch.setattr(memory_manager, "cuda_ensure_initialized", lambda: None)
    monkeypatch.setattr(
        memory_manager,
        "cumem_get_allocation_granularity",
        lambda device: 2 * 1024 * 1024,
    )
    monkeypatch.setattr(memory_manager, "_GMSClientSession", FakeSession)
    monkeypatch.setattr(common_utils, "invalidate_uuid_cache", lambda: None)
    monkeypatch.setattr(
        common_utils, "get_socket_path", lambda device, tag: f"socket:{device}:{tag}"
    )

    manager = memory_manager.GMSClientMemoryManager(
        "/tmp/gms_GPU-old_weights.sock", device=1, tag="weights"
    )
    manager._aborted = True

    manager.connect(RequestedLockType.RO)

    assert session_calls == [("socket:1:weights", RequestedLockType.RO, None)]
    assert manager.socket_path == "socket:1:weights"


def test_aborted_manager_preserves_explicit_socket_path(monkeypatch):
    from gpu_memory_service.client import memory_manager
    from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType

    session_calls = []

    class FakeSession:
        def __init__(self, socket_path, lock_type, timeout_ms):
            session_calls.append(socket_path)
            self.lock_type = GrantedLockType.RW
            self.committed = False
            self.is_connected = True

    monkeypatch.setattr(memory_manager, "cuda_ensure_initialized", lambda: None)
    monkeypatch.setattr(
        memory_manager,
        "cumem_get_allocation_granularity",
        lambda device: 2 * 1024 * 1024,
    )
    monkeypatch.setattr(memory_manager, "_GMSClientSession", FakeSession)

    manager = memory_manager.GMSClientMemoryManager(
        "/run/test/custom.sock", device=0, tag="kv_pool"
    )
    manager._aborted = True
    manager.connect(RequestedLockType.RW)

    assert session_calls == ["/run/test/custom.sock"]
    assert manager.socket_path == "/run/test/custom.sock"
