# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import sys
import types

import pytest
from gpu_memory_service.client import memory_manager
from gpu_memory_service.client.torch import allocator
from gpu_memory_service.common.locks import GrantedLockType


@pytest.fixture(autouse=True)
def _clear_scratch_backed_env(monkeypatch):
    monkeypatch.delenv("GMS_PERSISTENT_DEFER_PHYSICAL_SCRATCH_BACKED", raising=False)
    yield
    monkeypatch.delenv("GMS_PERSISTENT_DEFER_PHYSICAL_SCRATCH_BACKED", raising=False)


class _FakeCuda:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def current_device(self) -> int:
        return 0

    @contextlib.contextmanager
    def use_mem_pool(self, mem_pool, *, device):
        self.calls.append((mem_pool, device))
        yield


def _install_fake_torch(monkeypatch):
    fake_cuda = _FakeCuda()
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_cuda


class _FakePersistentManager:
    def __init__(self) -> None:
        self.scratch_calls: list[tuple[int, str, bool]] = []
        self.persistent_calls: list[tuple[str, str, int, bool]] = []

    def create_scratch_mapping(
        self,
        *,
        size: int,
        tag: str,
        map_scratch: bool = True,
    ) -> int:
        self.scratch_calls.append((size, tag, map_scratch))
        return 0x1000 + len(self.scratch_calls)

    def create_persistent_mapping(
        self, *, engine_id: str, tag: str, size: int, shared: bool
    ) -> int:
        self.persistent_calls.append((engine_id, tag, size, shared))
        return 0x2000 + len(self.persistent_calls)


def _register_tag(tag: str, *, persistent: bool = True) -> object:
    mem_pool = object()
    allocator._tag_states[tag] = allocator._TagState(
        manager=object(),
        mem_pool=mem_pool,
        socket_path="/tmp/gms-test.sock",
        device=0,
        is_persistent=persistent,
        persistent_engine_id="engine",
    )
    return mem_pool


@pytest.fixture(autouse=True)
def _reset_allocator_state():
    saved = dict(allocator._tag_states)
    allocator._tag_states.clear()
    yield
    allocator._tag_states.clear()
    allocator._tag_states.update(saved)


def test_persistent_pool_context_is_reentrant_for_same_tag_and_device(monkeypatch):
    fake_cuda = _install_fake_torch(monkeypatch)
    mem_pool = _register_tag("kv_pool")

    with allocator.gms_use_persistent_pool("kv_pool", 0):
        with allocator.gms_use_persistent_pool("kv_pool", 0):
            assert allocator._active_tag.get() == "kv_pool"

    assert fake_cuda.calls == [(mem_pool, 0)]
    assert allocator._active_tag.get() is None
    assert allocator._active_pool.get() is None


def test_nested_pool_context_rejects_mismatched_tag(monkeypatch):
    fake_cuda = _install_fake_torch(monkeypatch)
    mem_pool = _register_tag("kv_pool")
    _register_tag("weights", persistent=False)

    with allocator.gms_use_persistent_pool("kv_pool", 0):
        with pytest.raises(RuntimeError, match="Nested GMS mempool contexts"):
            with allocator.gms_use_mem_pool("weights", 0):
                pass

    assert fake_cuda.calls == [(mem_pool, 0)]


def test_deferred_persistent_pool_uses_semantic_tag_plan():
    manager = _FakePersistentManager()
    allocator._tag_states["kv_pool"] = allocator._TagState(
        manager=manager,
        mem_pool=object(),
        socket_path="/tmp/gms-test.sock",
        device=0,
        is_persistent=True,
        persistent_engine_id="engine|bootstrap=shadow",
        persistent_defer_physical=True,
    )
    allocator.set_persistent_allocator_tag_plan(
        "kv_pool",
        ["kv_pool:v2:aaa", "kv_pool:v2:bbb"],
    )

    token = allocator._active_tag.set("kv_pool")
    try:
        assert allocator._gms_malloc(1024, 0, 0) == 0x1001
        assert allocator._gms_malloc(2048, 0, 0) == 0x1002
        assert allocator._gms_malloc(4096, 0, 0) == 0x1003
    finally:
        allocator._active_tag.reset(token)
        allocator.clear_persistent_allocator_tag_plan("kv_pool")

    assert manager.scratch_calls == [
        (1024, "kv_pool:v2:aaa", False),
        (2048, "kv_pool:v2:bbb", False),
        (4096, "kv_pool#2", False),
    ]


def test_deferred_persistent_pool_uses_shared_ordinal_tags():
    manager = _FakePersistentManager()
    allocator._tag_states["kv_pool"] = allocator._TagState(
        manager=manager,
        mem_pool=object(),
        socket_path="/tmp/gms-test.sock",
        device=0,
        is_persistent=True,
        persistent_engine_id="engine|bootstrap=shadow",
        persistent_defer_physical=True,
    )

    token = allocator._active_tag.set("kv_pool")
    try:
        assert allocator._gms_malloc(1024, 0, 0) == 0x1001
        assert allocator._gms_malloc(2048, 0, 0) == 0x1002
    finally:
        allocator._active_tag.reset(token)

    assert manager.scratch_calls == [
        (1024, "kv_pool#0", False),
        (2048, "kv_pool#1", False),
    ]


def test_persistent_scratch_promotion_accepts_rw_persistent_grant():
    manager = object.__new__(memory_manager.GMSClientMemoryManager)
    manager.tag = "kv_pool"
    manager._granted_lock_type = GrantedLockType.RW_PERSISTENT
    manager._mappings = {}
    manager._scratch_mappings = {
        0x1000: memory_manager._ScratchMapping(
            size=4096,
            aligned_size=4096,
            va_reserved_size=1 << 20,
            tag="kv_pool#0",
        )
    }
    state = allocator._TagState(
        manager=manager,
        mem_pool=object(),
        socket_path="/tmp/gms-test.sock",
        device=0,
        is_persistent=True,
        persistent_engine_id="sglang:gpu0:block-pool|bootstrap=1",
        persistent_defer_physical=True,
    )
    allocator._tag_states["kv_pool"] = state

    assert manager.prepare_scratch_for_reallocation() == 1
    assert not manager._scratch_mappings
    assert manager._mappings[0x1000].tag == "kv_pool#0"
