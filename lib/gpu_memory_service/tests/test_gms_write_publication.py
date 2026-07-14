# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deferred GMS write publication in the vLLM integration."""

from __future__ import annotations

import json
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
    _iter_module_tensors,
    materialize_module_from_gms,
    rebind_nonparameter_tensors,
    register_module_tensors,
)
from gpu_memory_service.client.torch.tensor import GMSTensorSpec, TensorMetadata
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


def _manager_for(tensor):
    storage = tensor.untyped_storage()
    return SimpleNamespace(
        mappings={
            storage.data_ptr(): SimpleNamespace(
                allocation_id="allocation",
                aligned_size=storage.nbytes(),
            )
        }
    )


def _set_module_tensors(monkeypatch, *entries):
    monkeypatch.setattr(
        torch_module, "_iter_module_tensors", lambda _model: iter(entries)
    )


def _spec(
    tensor,
    tensor_type,
    *,
    allocation_id="allocation",
    offset_bytes=0,
):
    return SimpleNamespace(
        allocation_id=allocation_id,
        offset_bytes=offset_bytes,
        meta=TensorMetadata.from_tensor(tensor, tensor_type),
        materialize=lambda _manager, _device: tensor.detach().clone(),
    )


def _set_specs(monkeypatch, specs):
    monkeypatch.setattr(
        torch_module.GMSTensorSpec,
        "load_all",
        classmethod(lambda _cls, _manager: specs),
    )


def _component_spec(
    name,
    tensor,
    tensor_type,
    *,
    storage_group_id,
    object_group_id,
    storage_base_offset,
    storage_nbytes,
    buffer_persistent=False,
    allocation_id="allocation",
):
    meta = TensorMetadata.from_tensor(
        tensor,
        tensor_type,
        storage={
            "schema_version": 1,
            "storage_group_id": storage_group_id,
            "object_group_id": object_group_id,
            "storage_base_offset": storage_base_offset,
            "storage_nbytes": storage_nbytes,
            "storage_offset": tensor.storage_offset(),
            "buffer_persistent": buffer_persistent,
        },
    )
    element_size = tensor.element_size()
    return GMSTensorSpec(
        key=name,
        name=name,
        allocation_id=allocation_id,
        offset_bytes=storage_base_offset + tensor.storage_offset() * element_size,
        meta=meta,
    )


def test_tensor_iteration_ignores_read_only_buffer_alias_property():
    class RoutedExpertsLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.property_reads = 0
            self.register_buffer(
                "_expert_map",
                torch.empty(4, device="meta", dtype=torch.int32),
                persistent=False,
            )

        @property
        def expert_map(self):
            self.property_reads += 1
            return self._expert_map

    model = RoutedExpertsLike()

    assert list(_iter_module_tensors(model)) == []
    assert model.property_reads == 0
    assert "_expert_map" in model._non_persistent_buffers_set


def test_rebind_does_not_swap_parameter_alias(monkeypatch):
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    model.__dict__["parameter_alias"] = parameter
    _set_module_tensors(
        monkeypatch,
        ("weight", parameter, "parameter"),
        ("parameter_alias", parameter, "tensor_attr"),
    )
    parameter_ptr = parameter.data_ptr()
    retained = []

    assert (
        rebind_nonparameter_tensors(
            _manager_for(parameter), model, retain_gms_tensors=retained
        )
        == 0
    )
    assert model.weight is model.parameter_alias is parameter
    assert parameter.data_ptr() == parameter_ptr
    assert retained == []


def test_rebound_tensor_owner_is_collision_safe(monkeypatch):
    model = torch.nn.Module()
    model.__dict__["_gms_rebound_tensor_owners"] = None
    tensor = torch.zeros(4)
    model.runtime = tensor
    _set_module_tensors(monkeypatch, ("runtime", tensor, "tensor_attr"))

    with pytest.raises(RuntimeError, match="Reserved GMS attribute"):
        rebind_nonparameter_tensors(_manager_for(tensor), model)


def test_rebind_fails_closed_when_swap_rejects_weakref(monkeypatch):
    source = torch.arange(4)
    source_ref = weakref.ref(source)
    model = torch.nn.Module()
    model.runtime = source
    _set_module_tensors(monkeypatch, ("runtime", source, "tensor_attr"))
    source_ptr = source.data_ptr()

    with pytest.raises(RuntimeError, match="does not support weak references"):
        rebind_nonparameter_tensors(_manager_for(source), model)

    assert source_ref() is source
    assert source.data_ptr() == source_ptr


def test_rebind_supports_discovered_view_with_undiscovered_base(monkeypatch):
    base = torch.arange(8)
    view = base[::2]
    model = torch.nn.Module()
    model.view = view
    _set_module_tensors(monkeypatch, ("view", view, "tensor_attr"))

    assert rebind_nonparameter_tensors(_manager_for(base), model) == base.nbytes
    torch.testing.assert_close(model.view, torch.tensor([0, 2, 4, 6]))


def test_rebind_skips_ephemeral_property_view(monkeypatch):
    class Model(torch.nn.Module):
        @property
        def derived(self):
            return self.runtime[:]

    model = Model()
    model.runtime = torch.arange(4)
    runtime_ptr = model.runtime.data_ptr()

    def iter_tensors(_model):
        yield ("derived", model.derived, "tensor_attr")
        yield ("runtime", model.runtime, "tensor_attr")

    monkeypatch.setattr(torch_module, "_iter_module_tensors", iter_tensors)

    assert (
        rebind_nonparameter_tensors(_manager_for(model.runtime), model)
        == model.runtime.numel() * model.runtime.element_size()
    )
    assert model.runtime.data_ptr() != runtime_ptr


@pytest.mark.parametrize("alias_first", [False, True])
def test_reader_parameter_alias_keeps_parameter_identity(monkeypatch, alias_first):
    parameter = torch.nn.Parameter(torch.zeros(4))
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    model.__dict__["weight_alias"] = parameter
    expected = torch.arange(4, dtype=torch.float32)
    entries = [
        ("weight", _spec(expected, "parameter")),
        ("weight_alias", _spec(expected, "tensor_attr")),
    ]
    if alias_first:
        entries.reverse()
    _set_specs(monkeypatch, dict(entries))

    materialize_module_from_gms(object(), model, device_index=0)

    assert model.weight is model.weight_alias is parameter
    assert type(parameter) is torch.nn.Parameter
    assert model._parameters["weight"] is parameter
    torch.testing.assert_close(parameter, expected)


@pytest.mark.parametrize("alias_first", [False, True])
def test_reader_parameter_alias_conflict_fails_before_mutation(
    monkeypatch, alias_first
):
    parameter = torch.nn.Parameter(torch.zeros(4))
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    model.__dict__["weight_alias"] = parameter
    entries = [
        ("weight", _spec(torch.ones(4), "parameter")),
        (
            "weight_alias",
            _spec(torch.ones(4), "tensor_attr", offset_bytes=16),
        ),
    ]
    if alias_first:
        entries.reverse()
    _set_specs(monkeypatch, dict(entries))

    with pytest.raises(RuntimeError, match="aliases incompatible GMS entries"):
        materialize_module_from_gms(object(), model, device_index=0)

    assert model.weight is model.weight_alias is parameter
    assert type(parameter) is torch.nn.Parameter
    assert model._parameters["weight"] is parameter
    torch.testing.assert_close(parameter, torch.zeros(4))


def test_reader_preserves_nonpersistent_buffer(monkeypatch):
    model = torch.nn.Module()
    buffer = torch.zeros(4)
    model.register_buffer("expert_mask", buffer, persistent=False)
    _set_specs(monkeypatch, {"expert_mask": _spec(torch.ones(4), "buffer")})

    materialize_module_from_gms(object(), model, device_index=0)

    assert model.expert_mask is buffer
    assert model._buffers["expert_mask"] is buffer
    assert "expert_mask" in model._non_persistent_buffers_set
    assert "expert_mask" not in model.state_dict()
    torch.testing.assert_close(buffer, torch.ones(4))


def test_legacy_reader_rejects_discovered_view_placeholders(monkeypatch):
    model = torch.nn.Module()
    model.first = torch.zeros(4)
    model.view = model.first[:]
    _set_specs(
        monkeypatch,
        {
            "first": _spec(torch.ones(4), "tensor_attr"),
            "view": _spec(torch.ones(4), "tensor_attr"),
        },
    )

    with pytest.raises(RuntimeError, match="cannot materialize tensor view"):
        materialize_module_from_gms(object(), model, device_index=0)

    torch.testing.assert_close(model.first, torch.zeros(4))


def test_legacy_reader_rejects_distinct_shared_storage_placeholders(monkeypatch):
    model = torch.nn.Module()
    model.first = torch.zeros(4)
    model.second = torch.empty(0)
    model.second.set_(model.first.untyped_storage(), 0, (4,), (1,))
    assert model.second._base is None
    _set_specs(
        monkeypatch,
        {
            "first": _spec(torch.ones(4), "tensor_attr"),
            "second": _spec(torch.ones(4), "tensor_attr"),
        },
    )

    with pytest.raises(RuntimeError, match="distinct tensors that share storage"):
        materialize_module_from_gms(object(), model, device_index=0)

    torch.testing.assert_close(model.first, torch.zeros(4))


class _TensorSubclass(torch.Tensor):
    pass


class _LoaderParameter(torch.nn.Parameter):
    __slots__ = ("slot_state",)

    def __new__(
        cls,
        data,
        *,
        weight_loader,
        tp_rank,
        tp_size,
        requires_grad,
    ):
        return super().__new__(cls, data, requires_grad=requires_grad)

    def __init__(
        self,
        data,
        *,
        weight_loader,
        tp_rank,
        tp_size,
        requires_grad,
    ):
        self._weight_loader = weight_loader
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.slot_state = {"requires_constructor_arguments": True}


@pytest.mark.parametrize("alias_first", [False, True])
def test_reader_materializes_parameter_subclass_without_losing_state(
    monkeypatch, alias_first
):
    def weight_loader(tensor):
        return tensor

    parameter = _LoaderParameter(
        torch.zeros(4, device="meta"),
        weight_loader=weight_loader,
        tp_rank=2,
        tp_size=8,
        requires_grad=True,
    )
    parameter_state = parameter.__dict__
    slot_state = parameter.slot_state
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    model.__dict__["weight_alias"] = parameter
    expected = torch.arange(4, dtype=torch.float32)
    entries = [
        ("weight", _spec(expected, "parameter")),
        ("weight_alias", _spec(expected, "tensor_attr")),
    ]
    if alias_first:
        entries.reverse()
    _set_specs(monkeypatch, dict(entries))

    materialize_module_from_gms(object(), model, device_index=0)

    assert model.weight is model.weight_alias is parameter
    assert model._parameters["weight"] is parameter
    assert type(parameter) is _LoaderParameter
    assert parameter.__dict__ is parameter_state
    assert parameter._weight_loader is weight_loader
    assert parameter.tp_rank == 2
    assert parameter.tp_size == 8
    assert parameter.slot_state is slot_state
    assert parameter.requires_grad
    assert not parameter.is_meta
    torch.testing.assert_close(parameter, expected)


def test_reader_rejects_weak_parameter_before_mutation(monkeypatch):
    parameter = _LoaderParameter(
        torch.zeros(4, device="meta"),
        weight_loader=lambda tensor: tensor,
        tp_rank=0,
        tp_size=1,
        requires_grad=False,
    )
    parameter_ref = weakref.ref(parameter)
    model = torch.nn.Module()
    model.runtime = torch.zeros(4)
    model.register_parameter("weight", parameter)
    _set_specs(
        monkeypatch,
        {
            "runtime": _spec(torch.ones(4), "tensor_attr"),
            "weight": _spec(torch.ones(4), "parameter"),
        },
    )

    with pytest.raises(RuntimeError, match="does not support weak references"):
        materialize_module_from_gms(object(), model, device_index=0)

    assert parameter_ref() is parameter
    assert model._parameters["weight"] is parameter
    assert type(parameter) is _LoaderParameter
    assert parameter.is_meta
    torch.testing.assert_close(model.runtime, torch.zeros(4))


def test_nonparameter_component_copies_storage_once(monkeypatch):
    base = torch.arange(8, dtype=torch.float32)
    view = base[2::2]
    model = torch.nn.Module()
    model.base = base
    model.view = view
    manager = _manager_for(base)
    old_storage = base.untyped_storage()._cdata
    _set_module_tensors(
        monkeypatch,
        ("base", base, "tensor_attr"),
        ("view", view, "tensor_attr"),
    )

    rebound = rebind_nonparameter_tensors(manager, model)

    assert rebound == base.untyped_storage().nbytes()
    assert model.base.untyped_storage()._cdata != old_storage
    assert model.base.untyped_storage()._cdata == model.view.untyped_storage()._cdata
    model.view.add_(10)
    torch.testing.assert_close(model.base[2::2], model.view)


def test_mixed_dtype_component_reconstructs_shared_storage(monkeypatch):
    source_octets = torch.tensor([1, 0, 0, 0, 2, 0, 0, 0], dtype=torch.uint8)
    source_storage = source_octets.untyped_storage()
    source_words = torch.empty(0, dtype=torch.int32).set_(source_storage, 0, (2,), (1,))
    writer = torch.nn.Module()
    writer.octets = source_octets
    writer.words = source_words
    _set_module_tensors(
        monkeypatch,
        ("octets", source_octets, "tensor_attr"),
        ("words", source_words, "tensor_attr"),
    )

    specs = {}

    def metadata_put(*, key, allocation_id, offset_bytes, value):
        specs[key] = GMSTensorSpec(
            key=key,
            name=key,
            allocation_id=allocation_id,
            offset_bytes=offset_bytes,
            meta=TensorMetadata.from_bytes(value),
        )
        return True

    manager = _manager_for(source_octets)
    manager.metadata_put = metadata_put
    assert register_module_tensors(manager, writer) == {"allocation"}

    reader_octets = torch.zeros(8, dtype=torch.uint8)
    reader_storage = reader_octets.untyped_storage()
    reader_words = torch.empty(0, dtype=torch.int32).set_(reader_storage, 0, (2,), (1,))
    reader = torch.nn.Module()
    reader.octets = reader_octets
    reader.words = reader_words
    _set_specs(monkeypatch, specs)
    monkeypatch.setattr(
        torch_module,
        "_storage_from_pointer",
        lambda _ptr, _size, _device: source_storage,
    )

    materialize_module_from_gms(manager, reader, device_index=0)

    assert reader.octets is reader_octets and reader.words is reader_words
    assert (
        reader.octets.untyped_storage()._cdata == reader.words.untyped_storage()._cdata
    )
    torch.testing.assert_close(reader.octets, source_octets)
    torch.testing.assert_close(reader.words, torch.tensor([1, 2], dtype=torch.int32))
    reader.words[0] = 257
    torch.testing.assert_close(
        reader.octets, torch.tensor([1, 1, 0, 0, 2, 0, 0, 0], dtype=torch.uint8)
    )


def test_writer_rejects_distinct_overlapping_storage_ranges(monkeypatch):
    allocation = torch.arange(16, dtype=torch.uint8)
    ptr = allocation.data_ptr()
    first_storage = torch._C._construct_storage_from_data_pointer(
        ptr, torch.device("cpu"), 12
    )
    second_storage = torch._C._construct_storage_from_data_pointer(
        ptr + 4, torch.device("cpu"), 12
    )
    first = torch.empty(0, dtype=torch.uint8).set_(first_storage, 0, (12,), (1,))
    second = torch.empty(0, dtype=torch.uint8).set_(second_storage, 0, (12,), (1,))
    model = torch.nn.Module()
    model.first = first
    model.second = second
    manager = SimpleNamespace(
        mappings={
            ptr: SimpleNamespace(
                allocation_id="allocation",
                aligned_size=allocation.untyped_storage().nbytes(),
            )
        },
        metadata_put=lambda **_kwargs: pytest.fail("metadata was mutated"),
    )
    _set_module_tensors(
        monkeypatch,
        ("first", first, "tensor_attr"),
        ("second", second, "tensor_attr"),
    )

    with pytest.raises(RuntimeError, match="byte ranges overlap"):
        register_module_tensors(manager, model)


def test_writer_accepts_ordinary_trainable_leaf_parameter(monkeypatch):
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    _set_module_tensors(monkeypatch, ("weight", parameter, "parameter"))
    published = []
    manager = _manager_for(parameter)
    manager.metadata_put = lambda **kwargs: published.append(kwargs) or True

    assert register_module_tensors(manager, model) == {"allocation"}
    assert [entry["key"] for entry in published] == ["weight"]


def test_hidden_no_grad_view_fails_public_preflights(monkeypatch):
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    with torch.no_grad():
        hidden_view = parameter[:]
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    _set_module_tensors(monkeypatch, ("weight", parameter, "parameter"))
    manager = _manager_for(parameter)
    manager.metadata_put = lambda **_kwargs: pytest.fail("metadata was mutated")

    with pytest.raises(RuntimeError, match="TensorImpl use count"):
        register_module_tensors(manager, model)

    runtime = torch.zeros(4)
    model.runtime = runtime
    _set_specs(
        monkeypatch,
        {
            "runtime": _component_spec(
                "runtime",
                runtime,
                "tensor_attr",
                storage_group_id=0,
                object_group_id=0,
                storage_base_offset=0,
                storage_nbytes=runtime.untyped_storage().nbytes(),
            ),
            "weight": _component_spec(
                "weight",
                parameter,
                "parameter",
                storage_group_id=1,
                object_group_id=1,
                storage_base_offset=runtime.untyped_storage().nbytes(),
                storage_nbytes=parameter.untyped_storage().nbytes(),
            ),
        },
    )
    reader_manager = SimpleNamespace(
        mappings={},
        create_mapping=lambda **_kwargs: pytest.fail("reader mapping was created"),
    )

    with pytest.raises(RuntimeError, match="TensorImpl use count"):
        materialize_module_from_gms(reader_manager, model, device_index=0)

    assert hidden_view.shape == parameter.shape
    torch.testing.assert_close(runtime, torch.zeros(4))
    torch.testing.assert_close(parameter, torch.arange(4, dtype=torch.float32))


@pytest.mark.parametrize("trainable_view", [False, True])
def test_writer_rejects_unsupported_view_ownership(monkeypatch, trainable_view):
    base = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    model = torch.nn.Module()
    if trainable_view:
        model.register_parameter("weight", base)
        model.view = base[::2]
        entries = [("weight", base, "parameter"), ("view", model.view, "tensor_attr")]
    else:
        model.runtime = base.data
        model.helper = SimpleNamespace(view=model.runtime[::2])
        entries = [("runtime", model.runtime, "tensor_attr")]
    _set_module_tensors(monkeypatch, *entries)
    manager = _manager_for(base)
    manager.metadata_put = lambda **_kwargs: pytest.fail("metadata was mutated")

    with pytest.raises(RuntimeError, match="autograd tensor|TensorImpl use count"):
        register_module_tensors(manager, model)


def test_reader_rejects_overlapping_wire_components_before_mapping(monkeypatch):
    model = torch.nn.Module()
    model.first = torch.zeros(2)
    model.second = torch.zeros(2)
    specs = {
        "first": _component_spec(
            "first",
            model.first,
            "tensor_attr",
            storage_group_id=0,
            object_group_id=0,
            storage_base_offset=0,
            storage_nbytes=8,
        ),
        "second": _component_spec(
            "second",
            model.second,
            "tensor_attr",
            storage_group_id=1,
            object_group_id=1,
            storage_base_offset=4,
            storage_nbytes=8,
        ),
    }
    _set_specs(monkeypatch, specs)
    manager = SimpleNamespace(create_mapping=lambda **_kwargs: pytest.fail("mapped"))

    with pytest.raises(RuntimeError, match="byte ranges overlap"):
        materialize_module_from_gms(manager, model, device_index=0)

    torch.testing.assert_close(model.first, torch.zeros(2))


@pytest.mark.parametrize(
    "wire_shared",
    [True, False],
    ids=("wire-shared-reader-split", "wire-split-reader-shared"),
)
def test_reader_rejects_storage_impl_topology_before_mapping(monkeypatch, wire_shared):
    first = torch.zeros(2)
    if wire_shared:
        second = torch.ones(2)
        wire_topology = ((0, 0), (0, 0))
    else:
        second = torch.empty(0).set_(first.untyped_storage(), 0, (2,), (1,))
        wire_topology = ((0, 0), (1, first.untyped_storage().nbytes()))
    model = torch.nn.Module()
    model.first = first
    model.second = second
    identities = (first, second)
    specs = {
        name: _component_spec(
            name,
            tensor,
            "tensor_attr",
            storage_group_id=storage_group_id,
            object_group_id=object_group_id,
            storage_base_offset=storage_base_offset,
            storage_nbytes=tensor.numel() * tensor.element_size(),
        )
        for object_group_id, (
            name,
            tensor,
            (storage_group_id, storage_base_offset),
        ) in enumerate(
            zip(
                ("first", "second"),
                identities,
                wire_topology,
                strict=True,
            )
        )
    }
    _set_specs(monkeypatch, specs)
    manager = SimpleNamespace(
        mappings={},
        create_mapping=lambda **_kwargs: pytest.fail("reader mapping was created"),
    )

    with pytest.raises(RuntimeError, match="StorageImpl topology differs"):
        materialize_module_from_gms(manager, model, device_index=0)

    assert model.first is first
    assert model.second is second
    torch.testing.assert_close(first, torch.zeros(2))
    torch.testing.assert_close(second, torch.ones(2) if wire_shared else torch.zeros(2))


def test_component_schema_is_strict_and_keeps_legacy_offset():
    tensor = torch.arange(4, dtype=torch.float32)
    meta = TensorMetadata.from_tensor(
        tensor,
        "tensor_attr",
        storage={
            "schema_version": 1,
            "storage_group_id": 3,
            "object_group_id": 4,
            "storage_base_offset": 16,
            "storage_nbytes": 32,
            "storage_offset": 0,
            "buffer_persistent": False,
        },
    )
    encoded = json.loads(meta.to_bytes())
    assert encoded["storage"]["schema_version"] == 1
    assert encoded["storage"]["storage_base_offset"] == 16
    assert encoded["storage"]["storage_offset"] == 0

    null_stride = encoded.copy()
    null_stride["stride"] = None
    with pytest.raises(ValueError, match="stride must be a list"):
        TensorMetadata.from_bytes(json.dumps(null_stride).encode())

    del encoded["storage"]["storage_nbytes"]
    with pytest.raises(ValueError, match="fields must be exact"):
        TensorMetadata.from_bytes(json.dumps(encoded).encode())

    for legacy_stride in ("", ',"stride":null'):
        legacy = TensorMetadata.from_bytes(
            (
                '{"shape":[4],"dtype":"torch.float32",'
                f'"tensor_type":"parameter"{legacy_stride}}}'
            ).encode()
        )
        assert not legacy.is_storage_component
        assert legacy.stride == (1,)

    with pytest.raises(ValueError, match="Unknown tensor type"):
        TensorMetadata.from_bytes(
            b'{"shape":[4],"dtype":"torch.float32","tensor_type":"unknown"}'
        )


def test_reader_rejects_nontensor_slot_before_mapping(monkeypatch):
    tensor = torch.zeros(4)
    bad = torch.nn.Module()
    bad.sentinel = "not a tensor"
    _set_specs(
        monkeypatch,
        {
            "sentinel": _component_spec(
                "sentinel",
                tensor,
                "tensor_attr",
                storage_group_id=0,
                object_group_id=0,
                storage_base_offset=0,
                storage_nbytes=tensor.untyped_storage().nbytes(),
            )
        },
    )
    with pytest.raises(RuntimeError, match="preexisting direct Tensor"):
        materialize_module_from_gms(object(), bad, device_index=0)

    _set_specs(monkeypatch, {"sentinel": _spec(tensor, "tensor_attr")})
    with pytest.raises(RuntimeError, match="preexisting direct Tensor"):
        materialize_module_from_gms(object(), bad, device_index=0)


def test_component_allocation_bounds_failure_cleans_only_new_mapping(monkeypatch):
    model = torch.nn.Module()
    model.existing = torch.zeros(2)
    model.imported = torch.ones(2)
    identities = (model.existing, model.imported)
    _set_specs(
        monkeypatch,
        {
            "existing": _component_spec(
                "existing",
                model.existing,
                "tensor_attr",
                storage_group_id=0,
                object_group_id=0,
                storage_base_offset=0,
                storage_nbytes=model.existing.untyped_storage().nbytes(),
                allocation_id="existing",
            ),
            "imported": _component_spec(
                "imported",
                model.imported,
                "tensor_attr",
                storage_group_id=1,
                object_group_id=1,
                storage_base_offset=4,
                storage_nbytes=model.imported.untyped_storage().nbytes(),
                allocation_id="imported",
            ),
        },
    )
    existing_mapping = SimpleNamespace(allocation_id="existing", aligned_size=8)
    manager = SimpleNamespace(mappings={100: existing_mapping})
    freed = []

    def create_mapping(*, allocation_id):
        assert allocation_id == "imported"
        manager.mappings[200] = SimpleNamespace(
            allocation_id=allocation_id, aligned_size=8
        )
        return 200

    def free_va(va):
        freed.append(va)
        del manager.mappings[va]

    manager.create_mapping = create_mapping
    manager.free_va = free_va

    with pytest.raises(RuntimeError, match="exceeds allocation"):
        materialize_module_from_gms(manager, model, device_index=0)

    assert model.existing is identities[0]
    assert model.imported is identities[1]
    torch.testing.assert_close(model.existing, torch.zeros(2))
    torch.testing.assert_close(model.imported, torch.ones(2))
    assert manager.mappings == {100: existing_mapping}
    assert freed == [200]


@pytest.mark.parametrize("reader", [False, True])
def test_nonparameter_tensor_subclass_is_rejected(monkeypatch, reader):
    tensor = torch.Tensor._make_subclass(_TensorSubclass, torch.zeros(4))
    model = torch.nn.Module()
    model.runtime = tensor

    if reader:
        _set_specs(monkeypatch, {"runtime": _spec(torch.ones(4), "tensor_attr")})

        def call():
            materialize_module_from_gms(object(), model, device_index=0)

    else:
        _set_module_tensors(monkeypatch, ("runtime", tensor, "tensor_attr"))

        def call():
            rebind_nonparameter_tensors(_manager_for(tensor), model)

    with pytest.raises(RuntimeError, match="non-parameter Tensor subclass"):
        call()

    assert model.runtime is tensor
    torch.testing.assert_close(tensor, torch.zeros(4))


def test_rebind_failure_is_terminal(monkeypatch):
    model = torch.nn.Module()
    model.first = torch.arange(4, dtype=torch.float32)
    model.second = torch.arange(4, dtype=torch.float32) + 10
    retained = []
    manager = SimpleNamespace(
        mappings={
            model.first.data_ptr(): SimpleNamespace(
                allocation_id="first",
                aligned_size=model.first.untyped_storage().nbytes(),
            ),
            model.second.data_ptr(): SimpleNamespace(
                allocation_id="second",
                aligned_size=model.second.untyped_storage().nbytes(),
            ),
        }
    )
    _set_module_tensors(
        monkeypatch,
        ("first", model.first, "tensor_attr"),
        ("second", model.second, "tensor_attr"),
    )
    real_swap = torch_module._swap_tensor_contents
    calls = 0

    def fail_second(existing, replacement, *, name):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second swap failure")
        real_swap(existing, replacement, name=name)

    monkeypatch.setattr(torch_module, "_swap_tensor_contents", fail_second)

    with pytest.raises(RuntimeError, match="injected second swap failure"):
        rebind_nonparameter_tensors(manager, model, retain_gms_tensors=retained)

    assert retained == []
    assert "_gms_rebound_tensor_owners" not in model.__dict__


def test_failed_vllm_write_load_releases_lease(monkeypatch):
    client = SimpleNamespace(
        close=lambda *, best_effort: events.append(("close", best_effort))
    )
    events = []
    monkeypatch.setattr(
        model_loader,
        "_load_write_mode_impl",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("rebind failed")),
    )

    with pytest.raises(RuntimeError, match="rebind failed"):
        model_loader._load_write_mode(client, None, None, None, torch.device("cpu"))

    assert events == [("close", True)]
    assert not model_loader.has_pending_gms_write()


def test_failed_vllm_write_load_clears_pending_state(monkeypatch):
    events = []
    client = SimpleNamespace(
        close=lambda *, best_effort: events.append(("close", best_effort))
    )

    def fail_after_pending(*_args):
        model_loader._pending_gms_client = client
        model_loader._pending_retained_gms_tensors = [object()]
        raise RuntimeError("model eval failed")

    monkeypatch.setattr(model_loader, "_load_write_mode_impl", fail_after_pending)

    with pytest.raises(RuntimeError, match="model eval failed"):
        model_loader._load_write_mode(client, None, None, None, torch.device("cpu"))

    assert events == [("close", True)]
    assert model_loader._pending_retained_gms_tensors == []
    assert not model_loader.has_pending_gms_write()
