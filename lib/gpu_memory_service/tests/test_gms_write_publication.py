# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deferred GMS write publication in the vLLM integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

from gpu_memory_service.integrations.common import utils as common_utils
from gpu_memory_service.integrations.vllm import model_loader

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


class _FakeAllocator:
    def __init__(self, events: list[str], *, fail_commit: bool = False):
        self.events = events
        self.fail_commit = fail_commit
        self.total_bytes = 100
        self.mappings = {
            1: SimpleNamespace(allocation_id="weight", aligned_size=60),
            2: SimpleNamespace(allocation_id="scratch", aligned_size=40),
        }

    def commit(self):
        self.events.append("commit")
        if self.fail_commit:
            raise RuntimeError("commit failed")

    def connect(self, lock_type):
        self.events.append("connect")

    def remap_all_vas(self):
        self.events.append("remap")

    def close(self, *, best_effort=False):
        self.events.append("close_best_effort" if best_effort else "close")


@pytest.fixture
def gms_write(monkeypatch):
    """A prepared (registered + pruned, uncommitted) fake GMS write."""
    events = []
    allocator = _FakeAllocator(events)

    def register(_allocator, _model):
        events.append("register")
        return {"weight"}

    def prune(_allocator, *, referenced_allocation_ids):
        assert referenced_allocation_ids == {"weight"}
        events.append("prune")
        _allocator.mappings.pop(2)
        _allocator.total_bytes = 60

    monkeypatch.setattr(common_utils, "register_module_tensors", register)
    monkeypatch.setattr(common_utils, "prune_allocations", prune)
    monkeypatch.setattr(
        common_utils,
        "rebind_nonparameter_tensors",
        lambda _allocator, _model, **_kwargs: events.append("rebind") or 12,
    )

    stats = common_utils.prepare_gms_write(allocator, object())
    return allocator, stats, events


@pytest.fixture(autouse=True)
def clear_pending_write(monkeypatch):
    monkeypatch.setattr(model_loader, "_pending_gms_client", None)
    monkeypatch.setattr(model_loader, "_pending_retained_gms_tensors", [])
    monkeypatch.setattr(model_loader, "_last_imported_weights_bytes", 0)
    monkeypatch.setattr(model_loader, "_last_model_memory_usage_offset_bytes", 0)


def test_prepare_registers_prunes_and_defers_commit(gms_write):
    """Prepare must not commit; accounting covers pruned and rebound bytes."""
    allocator, stats, events = gms_write

    assert events == ["register", "prune"]
    assert stats.committed_bytes == 60
    assert stats.pruned_bytes == 40
    assert stats.pruned_count == 1

    model_loader._store_pending_gms_write(allocator, stats, 12, [])

    assert model_loader.get_imported_weights_bytes() == 60
    assert model_loader.get_model_memory_usage_offset_bytes() == 52


def test_pending_write_publish_clears_pending(gms_write):
    allocator, stats, events = gms_write
    model_loader._store_pending_gms_write(allocator, stats, 12, [])
    assert model_loader.has_pending_gms_write()

    assert model_loader.publish_pending_gms_write()

    assert events == ["register", "prune", "commit", "connect", "remap"]
    assert not model_loader.has_pending_gms_write()
    assert not model_loader.publish_pending_gms_write()


def test_pending_write_abort_releases_writer_without_cuda_cleanup(gms_write):
    allocator, stats, events = gms_write
    retained = [object()]
    model_loader._store_pending_gms_write(allocator, stats, 12, retained)

    assert model_loader.abort_pending_gms_write()

    assert events == ["register", "prune", "close_best_effort"]
    assert "close" not in events
    assert model_loader._pending_retained_gms_tensors == []
    assert not model_loader.has_pending_gms_write()
    assert not model_loader.abort_pending_gms_write()


def test_publication_failure_releases_writer_and_preserves_error(gms_write):
    allocator, stats, events = gms_write
    allocator.fail_commit = True
    model_loader._store_pending_gms_write(allocator, stats, 12, [])

    with pytest.raises(RuntimeError, match="commit failed"):
        model_loader.publish_pending_gms_write()

    assert events == ["register", "prune", "commit", "close_best_effort"]
    assert not model_loader.has_pending_gms_write()


def test_eager_finalize_preserves_publish_then_rebind_order(gms_write):
    """SGLang/TRT-LLM keep the eager register->commit->reconnect->rebind flow."""
    allocator, _, events = gms_write
    events.clear()
    allocator.total_bytes = 100
    allocator.mappings[2] = SimpleNamespace(allocation_id="scratch", aligned_size=40)

    stats = common_utils.finalize_gms_write(allocator, object())

    assert stats.committed_bytes == 60
    assert stats.pruned_bytes == 40
    assert events == [
        "register",
        "prune",
        "commit",
        "connect",
        "remap",
        "rebind",
    ]
