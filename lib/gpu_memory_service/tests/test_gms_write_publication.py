# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deferred GMS write publication in the vLLM integration."""

from __future__ import annotations

import gc
import weakref
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

import torch
from gpu_memory_service.client.torch import module as torch_module
from gpu_memory_service.client.torch.module import (
    _refresh_cached_tensor_aliases,
    rebind_nonparameter_tensors,
)
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


def test_refresh_cached_tensor_aliases_obeys_plain_object_boundary():
    source = torch.empty(1)
    replacement = torch.ones(1)
    aliases = {id(source): (source, replacement)}

    class _RejectingSetattr:
        def __init__(self):
            self.__dict__["tensor"] = source

        def __setattr__(self, name, value):
            raise AssertionError(f"unexpected setter for {name}")

    class _TensorList(list):
        pass

    model = torch.nn.Module()
    cycle = SimpleNamespace(tensor=source)
    cycle.self = cycle
    model.cycle = cycle
    beyond_depth = SimpleNamespace()
    cursor = beyond_depth
    for _ in range(9):
        child = SimpleNamespace()
        cursor.child = child
        cursor = child
    cursor.tensor = source
    model.beyond_depth = beyond_depth
    model.rejecting_setattr = _RejectingSetattr()
    model.containers = SimpleNamespace(
        values=[source],
        custom=_TensorList([source]),
        mapping={"tensor": source},
        immutable=(source,),
    )

    _refresh_cached_tensor_aliases(model, aliases, max_depth=8)

    assert model.cycle.tensor is replacement
    assert model.rejecting_setattr.tensor is replacement
    assert cursor.tensor is source
    assert model.containers.values[0] is source
    assert model.containers.custom[0] is source
    assert model.containers.mapping["tensor"] is source
    assert model.containers.immutable[0] is source


def test_refresh_cached_tensor_aliases_revisits_with_more_remaining_depth():
    source = torch.empty(1)
    replacement = torch.ones(1)
    aliases = {id(source): (source, replacement)}
    helper = SimpleNamespace(child=SimpleNamespace(tensor=source))

    model = torch.nn.Module()
    model.deep = SimpleNamespace(helper=helper)
    model.shallow = helper

    _refresh_cached_tensor_aliases(model, aliases, max_depth=2)

    assert helper.child.tensor is replacement


@pytest.mark.parametrize("collision", [None, pytest.param(torch.empty(1), id="tensor")])
def test_rebound_tensor_owner_rejects_plain_attribute_collision(collision):
    model = torch.nn.Module()
    model.__dict__["_gms_rebound_tensor_owners"] = collision

    with pytest.raises(RuntimeError, match="Reserved GMS attribute.*model attribute"):
        rebind_nonparameter_tensors(SimpleNamespace(mappings={}), model)


def test_rebound_tensor_owner_rejects_registered_namespace_collision():
    model = torch.nn.Module()
    model._buffers["_gms_rebound_tensor_owners"] = torch.empty(1)

    with pytest.raises(RuntimeError, match=r"Reserved GMS attribute.*model\._buffers"):
        rebind_nonparameter_tensors(SimpleNamespace(mappings={}), model)


def test_rebound_tensor_owner_rejects_class_attribute_collision():
    class _ConflictingModule(torch.nn.Module):
        _gms_rebound_tensor_owners = None

    with pytest.raises(RuntimeError, match="Reserved GMS attribute.*class attribute"):
        rebind_nonparameter_tensors(
            SimpleNamespace(mappings={}),
            _ConflictingModule(),
        )


def test_rebound_tensor_owner_retains_sources_after_partial_failure(monkeypatch):
    first = torch.arange(1)
    second = torch.arange(1)
    model = torch.nn.Module()
    model.first = first
    model.second = second
    monkeypatch.setattr(
        torch_module,
        "_iter_module_tensors",
        lambda _model: iter(
            (
                ("first", first, "tensor_attr"),
                ("second", second, "tensor_attr"),
            )
        ),
    )
    resolve = torch_module._resolve_module_attr

    def fail_second(root, name):
        if name == "second":
            raise RuntimeError("resolve failed")
        return resolve(root, name)

    monkeypatch.setattr(
        torch_module,
        "_resolve_module_attr",
        fail_second,
    )
    manager = SimpleNamespace(
        mappings={
            tensor.data_ptr(): SimpleNamespace(
                aligned_size=tensor.numel() * tensor.element_size()
            )
            for tensor in (first, second)
        }
    )

    with pytest.raises(RuntimeError, match="resolve failed"):
        rebind_nonparameter_tensors(manager, model)

    owners = model.__dict__["_gms_rebound_tensor_owners"].tensors
    assert len(owners) == 1
    assert owners[0] is first


def test_rebound_tensor_owner_lifetime_is_model_scoped(monkeypatch):
    source = torch.arange(4)
    source_ref = weakref.ref(source)
    model = torch.nn.Module()
    model.runtime = source
    model.alias = source
    retained: list[torch.Tensor] = []

    monkeypatch.setattr(
        torch_module,
        "_iter_module_tensors",
        lambda target: iter(
            (
                ("runtime", target.runtime, "tensor_attr"),
                ("alias", target.alias, "tensor_attr"),
            )
        ),
    )
    manager = SimpleNamespace(
        mappings={
            source.data_ptr(): SimpleNamespace(
                aligned_size=source.numel() * source.element_size()
            )
        }
    )

    rebound_bytes = rebind_nonparameter_tensors(
        manager,
        model,
        retain_gms_tensors=retained,
    )
    holder = model.__dict__["_gms_rebound_tensor_owners"]
    holder_ref = weakref.ref(holder)

    assert rebound_bytes == source.numel() * source.element_size()
    assert model.runtime is model.alias
    assert len(retained) == 1
    assert retained[0] is source
    assert len(holder.tensors) == 1
    assert holder.tensors[0] is source
    assert model.state_dict() == {}

    repeated_retained: list[torch.Tensor] = []
    assert (
        rebind_nonparameter_tensors(
            manager,
            model,
            retain_gms_tensors=repeated_retained,
        )
        == 0
    )
    assert model.__dict__["_gms_rebound_tensor_owners"] is holder
    assert repeated_retained == []

    model_loader._pending_retained_gms_tensors = retained
    retained.clear()
    model_loader._pending_retained_gms_tensors = []
    del source
    gc.collect()
    assert source_ref() is not None

    del holder
    del model
    gc.collect()
    assert holder_ref() is None
    assert source_ref() is None
