# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deferred GMS write publication in the vLLM integration."""

from __future__ import annotations

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
)
from gpu_memory_service.client.torch.tensor import TensorMetadata
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
        mappings={storage.data_ptr(): SimpleNamespace(aligned_size=storage.nbytes())}
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

    with pytest.raises(RuntimeError, match="swap_tensors failed:.*weakref"):
        rebind_nonparameter_tensors(_manager_for(source), model)

    assert source_ref() is source
    assert source.data_ptr() == source_ptr


def test_rebind_rejects_distinct_views(monkeypatch):
    base = torch.arange(8)
    view = base[::2]
    model = torch.nn.Module()
    model.view = view
    _set_module_tensors(monkeypatch, ("view", view, "tensor_attr"))

    with pytest.raises(RuntimeError, match="cannot rebind tensor view"):
        rebind_nonparameter_tensors(_manager_for(base), model)


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


def test_reader_rejects_observed_view_before_mutation(monkeypatch):
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


def test_reader_rejects_observed_shared_storage_before_mutation(monkeypatch):
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


def test_rebind_rolls_back_multi_candidate_failure_and_retries(monkeypatch):
    model = torch.nn.Module()
    model.first = torch.arange(4, dtype=torch.float32)
    model.second = torch.arange(4, dtype=torch.float32) + 10
    first_ptr = model.first.data_ptr()
    second_ptr = model.second.data_ptr()
    retained = []
    manager = SimpleNamespace(
        mappings={
            first_ptr: SimpleNamespace(
                aligned_size=model.first.untyped_storage().nbytes()
            ),
            second_ptr: SimpleNamespace(
                aligned_size=model.second.untyped_storage().nbytes()
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

    assert model.first.data_ptr() == first_ptr
    assert model.second.data_ptr() == second_ptr
    assert retained == []
    assert "_gms_rebound_tensor_owners" not in model.__dict__

    monkeypatch.setattr(torch_module, "_swap_tensor_contents", real_swap)
    assert (
        rebind_nonparameter_tensors(manager, model, retain_gms_tensors=retained)
        == 2 * model.first.numel() * model.first.element_size()
    )
    assert model.first.data_ptr() != first_ptr
    assert model.second.data_ptr() != second_ptr
    assert len(retained) == 2


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
