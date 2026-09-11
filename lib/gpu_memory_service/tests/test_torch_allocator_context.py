# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import sys
import types

import pytest
from gpu_memory_service.client.torch import allocator


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


def test_persistent_pool_rejects_mismatched_registered_device(monkeypatch):
    fake_cuda = _install_fake_torch(monkeypatch)
    _register_tag("kv_pool")

    with pytest.raises(RuntimeError, match="registered for CUDA device 0"):
        with allocator.gms_use_persistent_pool("kv_pool", 1):
            pass

    assert fake_cuda.calls == []


def test_nested_pool_context_rejects_mismatched_tag(monkeypatch):
    fake_cuda = _install_fake_torch(monkeypatch)
    mem_pool = _register_tag("kv_pool")
    _register_tag("weights", persistent=False)

    with allocator.gms_use_persistent_pool("kv_pool", 0):
        with pytest.raises(RuntimeError, match="Nested GMS mempool contexts"):
            with allocator.gms_use_mem_pool("weights", 0):
                pass

    assert fake_cuda.calls == [(mem_pool, 0)]


def test_semantic_persistent_tag_plan_must_be_fully_consumed():
    _register_tag("kv_pool")
    allocator.set_persistent_allocator_tag_plan("kv_pool", ["layer-0", "layer-1"])
    allocator._tag_states["kv_pool"].persistent_alloc_seq = 1

    with pytest.raises(RuntimeError, match="under-consumed.*consumed 1/2"):
        allocator.validate_persistent_allocator_tag_plan_consumed("kv_pool")


def test_semantic_persistent_tag_plan_accepts_exact_consumption():
    _register_tag("kv_pool")
    allocator.set_persistent_allocator_tag_plan("kv_pool", ["layer-0", "layer-1"])
    allocator._tag_states["kv_pool"].persistent_alloc_seq = 2

    allocator.validate_persistent_allocator_tag_plan_consumed("kv_pool")


def test_semantic_persistent_tag_plan_rejects_empty_plan():
    _register_tag("kv_pool")
    allocator.set_persistent_allocator_tag_plan("kv_pool", [])

    with pytest.raises(RuntimeError, match="empty semantic tag plan"):
        allocator.validate_persistent_allocator_tag_plan_consumed("kv_pool")
