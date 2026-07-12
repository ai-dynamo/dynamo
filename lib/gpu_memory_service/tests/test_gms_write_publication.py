# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deferred GMS write publication in the vLLM integration."""

from __future__ import annotations

import contextlib
import gc
import importlib
import sys
import types
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
from gpu_memory_service.client.torch.tensor import TensorMetadata
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.integrations import common
from gpu_memory_service.integrations.common import utils as common_utils
from gpu_memory_service.integrations.vllm import model_loader, patches

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
    """Prepare must not commit; the caller supplies the final usage offset."""
    allocator, stats, events = gms_write

    assert events == ["register", "prune"]
    assert stats.committed_bytes == 60
    assert stats.pruned_bytes == 40
    assert stats.pruned_count == 1

    model_loader._store_pending_gms_write(allocator, stats, 52, [])

    assert model_loader.get_imported_weights_bytes() == 60
    assert model_loader.get_model_memory_usage_offset_bytes() == 52


def test_pending_write_publish_clears_pending(gms_write):
    allocator, stats, events = gms_write
    model_loader._store_pending_gms_write(allocator, stats, 52, [])
    assert model_loader.has_pending_gms_write()

    assert model_loader.publish_pending_gms_write()

    assert events == ["register", "prune", "commit", "connect", "remap"]
    assert not model_loader.has_pending_gms_write()
    assert not model_loader.publish_pending_gms_write()


def test_pending_write_abort_releases_writer_without_cuda_cleanup(gms_write):
    allocator, stats, events = gms_write
    retained = [object()]
    model_loader._store_pending_gms_write(allocator, stats, 52, retained)

    assert model_loader.abort_pending_gms_write()

    assert events == ["register", "prune", "close_best_effort"]
    assert "close" not in events
    assert model_loader._pending_retained_gms_tensors == []
    assert not model_loader.has_pending_gms_write()
    assert not model_loader.abort_pending_gms_write()


def test_publication_failure_releases_writer_and_preserves_error(gms_write):
    allocator, stats, events = gms_write
    allocator.fail_commit = True
    model_loader._store_pending_gms_write(allocator, stats, 52, [])

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


@pytest.fixture
def gms_worker_module(monkeypatch):
    """Import GMSWorker against the smallest observable vLLM worker boundary."""

    class Worker:
        def load_model(self, *_args, **_kwargs):
            self.parent_load_called = True

        def determine_available_memory(self):
            self.profiled_weights_memory = self.model_runner.model_memory_usage
            return self.requested_memory - self.profiled_weights_memory

    def register_model_loader(_load_format):
        return lambda loader: loader

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.model_loader": types.ModuleType(
            "vllm.model_executor.model_loader"
        ),
        "vllm.model_executor.model_loader.base_loader": types.ModuleType(
            "vllm.model_executor.model_loader.base_loader"
        ),
        "vllm.model_executor.model_loader.default_loader": types.ModuleType(
            "vllm.model_executor.model_loader.default_loader"
        ),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.gpu_worker": types.ModuleType("vllm.v1.worker.gpu_worker"),
    }
    modules[
        "vllm.model_executor.model_loader"
    ].register_model_loader = register_model_loader
    modules["vllm.model_executor.model_loader.base_loader"].BaseModelLoader = object
    modules[
        "vllm.model_executor.model_loader.default_loader"
    ].DefaultModelLoader = object
    modules["vllm.v1.worker.gpu_worker"].Worker = Worker
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(common, "patch_empty_cache", lambda: None)
    monkeypatch.setattr(patches, "patch_memory_snapshot", lambda: None)
    monkeypatch.setattr(patches, "apply_scratch_kv_patches", lambda: None)

    module_name = "gpu_memory_service.integrations.vllm.worker"
    package = sys.modules["gpu_memory_service.integrations.vllm"]
    previous = sys.modules.pop(module_name, None)
    previous_attribute = getattr(package, "worker", None)
    try:
        yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous
            package.worker = previous
        elif previous_attribute is not None:
            package.worker = previous_attribute
        else:
            package.__dict__.pop("worker", None)


def _assert_worker_profiles_loader_memory(
    gms_worker_module,
    *,
    imported_weights_bytes: int,
    memory_usage_offset_bytes: int,
) -> None:
    worker = object.__new__(gms_worker_module.GMSWorker)
    worker.model_runner = SimpleNamespace(model_memory_usage=-1)
    worker.requested_memory = 10_000

    worker.load_model()
    available = worker._determine_available_memory_before_gms_publish()

    expected_model_memory_usage = imported_weights_bytes + memory_usage_offset_bytes
    assert worker.parent_load_called
    assert worker.model_runner.model_memory_usage == expected_model_memory_usage
    assert worker.profiled_weights_memory == expected_model_memory_usage
    assert available == worker.requested_memory - expected_model_memory_usage


class _GLMLikeMLA(torch.nn.Module):
    def __init__(
        self,
        *,
        temporary_backing=False,
        device="cpu",
        build_derived=True,
        kv_lora_rank=2,
        num_heads=2,
        qk_nope_head_dim=1,
        v_head_dim=2,
        dtype=torch.float32,
    ):
        super().__init__()
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.temporary_backing = temporary_backing
        output_size = num_heads * (qk_nope_head_dim + v_head_dim)
        weight = torch.arange(
            output_size * kv_lora_rank,
            dtype=dtype,
            device=device,
        ).view(output_size, kv_lora_rank)
        self.kv_b_proj = torch.nn.Linear(
            kv_lora_rank,
            output_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.kv_b_proj.weight = torch.nn.Parameter(weight)
        if build_derived:
            self.process_weights_after_loading(dtype)

    def process_weights_after_loading(self, dtype):
        weight = self.kv_b_proj.weight
        if self.temporary_backing:
            weight = weight.detach().clone()
        projected = weight.to(dtype).T.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        w_uk, w_uv = projected.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        self.W_UV = w_uv.transpose(0, 1)
        self.W_UV_alias = self.W_UV
        self.W_UK_T = w_uk.permute(1, 2, 0)

    def forward(self, value):
        return torch.bmm(value, self.W_UV)


class _RegistrationManager:
    def __init__(self, tensor, *, allocation_id="weight"):
        storage = tensor.untyped_storage()
        self.mappings = {
            storage.data_ptr(): SimpleNamespace(
                allocation_id=allocation_id,
                aligned_size=storage.nbytes(),
            )
        }
        self.registered = []

    def metadata_put(self, **metadata):
        self.registered.append(metadata)


class _PruningManager(_RegistrationManager):
    def __init__(self, weight, temporary_source):
        super().__init__(weight)
        weight_mapping = next(iter(self.mappings.values()))
        weight_mapping.aligned_size = 100
        weight_mapping.handle = 1
        source_storage = temporary_source.untyped_storage()
        self.mappings[source_storage.data_ptr()] = SimpleNamespace(
            allocation_id="dequant",
            aligned_size=64,
            handle=1,
        )
        self.granted_lock_type = GrantedLockType.RW
        self.is_unmapped = False
        self.device = 0

    @property
    def total_bytes(self):
        return sum(mapping.aligned_size for mapping in self.mappings.values())

    def destroy_mapping(self, va):
        self.mappings.pop(va)


def _mla_entries(module):
    return (
        ("mla.kv_b_proj.weight", module.kv_b_proj.weight, "parameter"),
        ("mla.W_UV", module.W_UV, "tensor_attr"),
        ("mla.W_UV_alias", module.W_UV_alias, "tensor_attr"),
        ("mla.W_UK_T", module.W_UK_T, "tensor_attr"),
    )


def test_writer_normalizes_glm_mla_views_before_generic_registration(monkeypatch):
    mla = _GLMLikeMLA()
    model = torch.nn.Module()
    model.mla = mla
    manager = _RegistrationManager(mla.kv_b_proj.weight)
    original_w_uv = mla.W_UV
    original_w_uk_t = mla.W_UK_T
    expected_w_uv = mla.W_UV.clone()
    expected_w_uk_t = mla.W_UK_T.clone()
    value = torch.arange(12, dtype=torch.float32).view(2, 3, 2)
    expected_output = mla(value).clone()

    stats = model_loader._normalize_mla_derived_tensors(manager, model)

    assert mla.W_UV is mla.W_UV_alias is original_w_uv
    assert mla.W_UK_T is original_w_uk_t
    assert mla.W_UV._base is None
    assert mla.W_UK_T._base is None
    assert not model_loader._is_gms_mapped_tensor(manager, mla.W_UV)
    assert not model_loader._is_gms_mapped_tensor(manager, mla.W_UK_T)
    torch.testing.assert_close(mla.W_UV, expected_w_uv)
    torch.testing.assert_close(mla.W_UK_T, expected_w_uk_t)
    torch.testing.assert_close(mla(value), expected_output)
    assert stats.private_bytes == sum(
        tensor.numel() * tensor.element_size() for tensor in (mla.W_UV, mla.W_UK_T)
    )
    assert stats.source_allocation_ids == frozenset({"weight"})

    monkeypatch.setattr(
        torch_module,
        "_iter_module_tensors",
        lambda _model: iter(_mla_entries(mla)),
    )
    assert register_module_tensors(manager, model) == {"weight"}
    assert [entry["key"] for entry in manager.registered] == ["mla.kv_b_proj.weight"]
    assert rebind_nonparameter_tensors(manager, model) == 0


@pytest.mark.parametrize("attr_name", model_loader._MLA_DERIVED_TENSOR_ATTRS)
def test_writer_normalizes_every_known_mla_attr(attr_name):
    mla = _GLMLikeMLA(build_derived=False)
    source = mla.kv_b_proj.weight.detach().clone()
    derived = source[:, :1]
    mla.__dict__[attr_name] = derived
    model = torch.nn.Module()
    model.mla = mla
    manager = _RegistrationManager(source)

    stats = model_loader._normalize_mla_derived_tensors(manager, model)

    assert mla.__dict__[attr_name] is derived
    assert derived._base is None
    assert stats.private_bytes == derived.untyped_storage().nbytes()
    assert stats.source_allocation_ids == frozenset({"weight"})


def test_mla_private_accounting_counts_shared_views_once_and_excludes_weights():
    mla = _GLMLikeMLA(build_derived=False)
    private = torch.arange(16, dtype=torch.float32)
    mla.W_K = private[:8]
    mla.W_V = private[8:]
    mla.W_K_scale = mla.kv_b_proj.weight.view(-1)
    model = torch.nn.Module()
    model.mla = mla
    manager = _RegistrationManager(mla.kv_b_proj.weight)

    private_bytes = model_loader._mla_private_storage_bytes(
        manager,
        model_loader._mla_modules(model),
    )

    assert private_bytes == private.untyped_storage().nbytes()


def test_writer_normalizes_glm_tp8_mla_shapes_and_strides():
    mla = _GLMLikeMLA(
        kv_lora_rank=512,
        num_heads=4,
        qk_nope_head_dim=128,
        v_head_dim=128,
    )
    model = torch.nn.Module()
    model.mla = mla
    manager = _RegistrationManager(mla.kv_b_proj.weight)
    expected_w_uv = mla.W_UV.clone()
    expected_w_uk_t = mla.W_UK_T.clone()

    assert mla.W_UV.shape == (4, 512, 128)
    assert mla.W_UV.stride() == (131072, 1, 512)
    assert mla.W_UK_T.shape == (4, 128, 512)
    assert mla.W_UK_T.stride() == (131072, 512, 1)

    stats = model_loader._normalize_mla_derived_tensors(manager, model)

    torch.testing.assert_close(mla.W_UV, expected_w_uv)
    torch.testing.assert_close(mla.W_UK_T, expected_w_uk_t)
    assert stats.private_bytes == sum(
        tensor.untyped_storage().nbytes() for tensor in (mla.W_UV, mla.W_UK_T)
    )


@pytest.mark.parametrize("extra_reference", ["weakref", "tensor_impl"])
def test_writer_mla_normalization_fails_closed_on_swap_reference(extra_reference):
    mla = _GLMLikeMLA()
    model = torch.nn.Module()
    model.mla = mla
    manager = _RegistrationManager(mla.kv_b_proj.weight)
    tensor = mla.W_UV
    original_ptr = tensor.data_ptr()
    reference = (
        weakref.ref(tensor)
        if extra_reference == "weakref"
        else torch._to_functional_tensor(tensor)
    )

    with pytest.raises(RuntimeError, match="swap_tensors failed"):
        model_loader._normalize_mla_derived_tensors(manager, model)

    if extra_reference == "weakref":
        assert reference() is tensor
    else:
        assert reference is not None
    assert mla.W_UV is mla.W_UV_alias is tensor
    assert tensor.data_ptr() == original_ptr
    assert tensor._base is not None


def test_writer_mla_normalization_rolls_back_all_swaps(monkeypatch):
    mla = _GLMLikeMLA()
    model = torch.nn.Module()
    model.mla = mla
    manager = _RegistrationManager(mla.kv_b_proj.weight)
    originals = {
        name: (
            tensor,
            tensor.data_ptr(),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.clone(),
        )
        for name, tensor in (("W_UK_T", mla.W_UK_T), ("W_UV", mla.W_UV))
    }
    real_swap = torch_module._swap_tensor_contents
    calls = 0

    def fail_second(existing, replacement, *, name):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected MLA swap failure")
        real_swap(existing, replacement, name=name)

    monkeypatch.setattr(model_loader, "_swap_tensor_contents", fail_second)

    with pytest.raises(RuntimeError, match="injected MLA swap failure"):
        model_loader._normalize_mla_derived_tensors(manager, model)

    assert mla.W_UV is mla.W_UV_alias
    for name, (tensor, ptr, stride, offset, values) in originals.items():
        current = getattr(mla, name)
        assert current is tensor
        assert current.data_ptr() == ptr
        assert current.stride() == stride
        assert current.storage_offset() == offset
        assert current._base is not None
        torch.testing.assert_close(current, values)


def test_writer_mla_normalization_keeps_temporary_derived_values_alive():
    mla = _GLMLikeMLA(temporary_backing=True)
    model = torch.nn.Module()
    model.mla = mla
    temporary_owner = mla.W_UV._base
    assert temporary_owner is not None
    temporary_ref = weakref.ref(temporary_owner)
    expected = mla.W_UV.clone()
    manager = _RegistrationManager(temporary_owner)
    del temporary_owner

    model_loader._normalize_mla_derived_tensors(manager, model)
    gc.collect()

    assert temporary_ref() is None
    assert mla.W_UV._base is None
    assert mla.W_UK_T._base is None
    torch.testing.assert_close(mla.W_UV, expected)


@pytest.mark.parametrize("attr_name", model_loader._MLA_DERIVED_TENSOR_ATTRS)
def test_reader_clears_every_known_mla_attr_and_direct_alias(attr_name):
    mla = _GLMLikeMLA(device="meta", build_derived=False)
    derived = mla.kv_b_proj.weight.view(-1)
    mla.__dict__[attr_name] = derived
    mla.direct_alias = derived
    model = torch.nn.Module()
    model.mla = mla

    modules, cleared_attrs = model_loader._clear_mla_derived_tensors(model)

    assert modules == [("mla", mla)]
    assert cleared_attrs == 2
    assert attr_name not in mla.__dict__
    assert "direct_alias" not in mla.__dict__


def test_reader_clears_mla_views_before_materialization_and_rebuilds_after(
    monkeypatch,
    gms_worker_module,
):
    events = []
    model = torch.nn.Module()
    model.mla = _GLMLikeMLA(device="meta", temporary_backing=True)
    model.alias_owner = torch.nn.Module()
    model.alias_owner.cached_w_uv = model.mla.W_UV
    model_config = SimpleNamespace(dtype=torch.float32)
    gms_client = SimpleNamespace(
        total_bytes=48,
        mappings={},
        close=lambda: events.append("close"),
    )

    def create(*_args):
        events.append("create")
        return model

    def materialize(_client, loaded, *, device_index):
        assert loaded is model
        assert device_index == 0
        assert "W_UV" not in model.mla.__dict__
        assert "W_UV_alias" not in model.mla.__dict__
        assert "W_UK_T" not in model.mla.__dict__
        assert "cached_w_uv" not in model.alias_owner.__dict__
        parameter = model.mla.kv_b_proj.weight
        assert parameter._use_count() == 1
        replacement = torch.nn.Parameter(
            torch.arange(12, dtype=torch.float32).view(6, 2)
        )
        torch.utils.swap_tensors(parameter, replacement)
        events.append("materialize")

    original_hook = model.mla.process_weights_after_loading

    def rebuild(dtype):
        assert not model.mla.kv_b_proj.weight.is_meta
        events.append("mla")
        original_hook(dtype)

    model.mla.process_weights_after_loading = rebuild
    torch_utils = types.ModuleType("vllm.utils.torch_utils")
    torch_utils.set_default_torch_dtype = lambda _dtype: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)
    monkeypatch.setattr(model_loader, "_create_meta_model", create)
    monkeypatch.setattr(model_loader, "materialize_module_from_gms", materialize)
    monkeypatch.setattr(
        model_loader,
        "get_mx_load_context",
        lambda *_args: events.append("mx") or None,
    )

    loaded = model_loader._load_read_mode(
        gms_client,
        object(),
        model_config,
        device_index=0,
    )

    assert loaded is model
    assert events == ["create", "materialize", "mla", "mx"]
    assert model.mla.W_UV._base is not None
    assert model.mla.W_UK_T._base is not None
    assert model_loader.get_imported_weights_bytes() == 48
    assert model_loader.get_model_memory_usage_offset_bytes() == 48
    _assert_worker_profiles_loader_memory(
        gms_worker_module,
        imported_weights_bytes=48,
        memory_usage_offset_bytes=48,
    )


def test_reader_rebuilds_all_strict_mla_modules_in_model_order(monkeypatch):
    """An MLA hook that failed before creating attrs must still be rebuilt."""
    events = []
    model = torch.nn.Module()
    model.first = _GLMLikeMLA(
        device="meta",
        temporary_backing=True,
        build_derived=False,
    )
    model.second = _GLMLikeMLA(device="meta", temporary_backing=True)
    model_config = SimpleNamespace(dtype=torch.float32)
    gms_client = SimpleNamespace(total_bytes=96, mappings={}, close=lambda: None)

    def materialize(_client, loaded, *, device_index):
        assert loaded is model
        assert device_index == 0
        assert not any(
            attr in model.first.__dict__
            for attr in model_loader._MLA_DERIVED_TENSOR_ATTRS
        )
        for module in (model.first, model.second):
            parameter = module.kv_b_proj.weight
            replacement = torch.nn.Parameter(
                torch.arange(12, dtype=torch.float32).view(6, 2)
            )
            torch.utils.swap_tensors(parameter, replacement)
        events.append("materialize")

    for name, module in (("first", model.first), ("second", model.second)):
        original_hook = module.process_weights_after_loading

        def rebuild(dtype, *, module_name=name, hook=original_hook):
            events.append(module_name)
            hook(dtype)

        module.process_weights_after_loading = rebuild

    torch_utils = types.ModuleType("vllm.utils.torch_utils")
    torch_utils.set_default_torch_dtype = lambda _dtype: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)
    monkeypatch.setattr(model_loader, "_create_meta_model", lambda *_args: model)
    monkeypatch.setattr(model_loader, "materialize_module_from_gms", materialize)
    monkeypatch.setattr(
        model_loader,
        "get_mx_load_context",
        lambda *_args: events.append("mx") or None,
    )

    model_loader._load_read_mode(
        gms_client,
        object(),
        model_config,
        device_index=0,
    )

    assert events == ["materialize", "first", "second", "mx"]
    assert model_loader.get_model_memory_usage_offset_bytes() == 96


def test_write_mode_normalizes_mla_before_registration_and_counts_private_bytes(
    monkeypatch,
):
    events = []
    model = torch.nn.Module()
    model_config = SimpleNamespace(dtype=torch.float32)
    target_device = torch.device("cpu")
    gms_client = SimpleNamespace(
        mappings={
            1: SimpleNamespace(allocation_id="weight", aligned_size=100),
            2: SimpleNamespace(allocation_id="dequant", aligned_size=20),
        }
    )
    stats = common_utils.GMSCommittedMemoryStats(
        committed_bytes=100,
        pruned_bytes=20,
    )
    loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    loader_utils.initialize_model = lambda **_kwargs: model
    loader_utils.process_weights_after_loading = lambda *_args: events.append(
        "post_load"
    )
    torch_utils = types.ModuleType("vllm.utils.torch_utils")
    torch_utils.set_default_torch_dtype = lambda _dtype: contextlib.nullcontext()
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.utils",
        loader_utils,
    )
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)
    monkeypatch.setattr(
        model_loader,
        "gms_use_mem_pool",
        lambda *_args: contextlib.nullcontext(),
    )
    monkeypatch.setattr(model_loader, "get_mx_load_context", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        model_loader,
        "_normalize_mla_derived_tensors",
        lambda *_args: events.append("normalize")
        or model_loader._MLANormalizationStats(
            private_bytes=48,
            source_allocation_ids=frozenset({"dequant"}),
        ),
    )

    def prepare(*_args):
        events.append("prepare")
        gms_client.mappings.pop(2)
        return stats

    monkeypatch.setattr(model_loader, "prepare_gms_write", prepare)
    retained = []

    def rebind(*_args, retain_gms_tensors):
        events.append("rebind")
        retain_gms_tensors.append("owner")
        retained.extend(retain_gms_tensors)
        return 12

    monkeypatch.setattr(model_loader, "rebind_nonparameter_tensors", rebind)

    loaded = model_loader._load_write_mode_impl(
        gms_client,
        object(),
        model_config,
        SimpleNamespace(load_weights=lambda *_args: events.append("load")),
        target_device,
    )

    assert loaded is model
    assert events == ["load", "post_load", "normalize", "prepare", "rebind"]
    assert retained == ["owner"]
    assert model_loader.get_imported_weights_bytes() == 100
    assert model_loader.get_model_memory_usage_offset_bytes() == 60


def test_writer_memory_offset_reaches_worker_profile(
    monkeypatch,
    gms_worker_module,
):
    mla = _GLMLikeMLA(build_derived=False)
    temporary_source = mla.kv_b_proj.weight.detach().clone()
    model = torch.nn.Module()
    model.mla = mla
    manager = _PruningManager(mla.kv_b_proj.weight, temporary_source)
    loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    loader_utils.initialize_model = lambda **_kwargs: model

    def process_weights(*_args):
        projected = temporary_source.T.view(2, 2, 3)
        w_uk, w_uv = projected.split([1, 2], dim=-1)
        mla.W_UV = w_uv.transpose(0, 1)
        mla.W_UV_alias = mla.W_UV
        mla.W_UK_T = w_uk.permute(1, 2, 0)

    loader_utils.process_weights_after_loading = process_weights
    torch_utils = types.ModuleType("vllm.utils.torch_utils")
    torch_utils.set_default_torch_dtype = lambda _dtype: contextlib.nullcontext()
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.utils",
        loader_utils,
    )
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)
    monkeypatch.setattr(
        model_loader,
        "gms_use_mem_pool",
        lambda *_args: contextlib.nullcontext(),
    )
    monkeypatch.setattr(model_loader, "get_mx_load_context", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(
        torch_module,
        "_iter_module_tensors",
        lambda _model: iter(_mla_entries(mla)),
    )

    loaded = model_loader._load_write_mode_impl(
        manager,
        object(),
        SimpleNamespace(dtype=torch.float32),
        SimpleNamespace(load_weights=lambda *_args: None),
        torch.device("cpu"),
    )

    assert loaded is model
    assert {mapping.allocation_id for mapping in manager.mappings.values()} == {
        "weight"
    }
    assert model_loader.get_imported_weights_bytes() == 100
    assert model_loader.get_model_memory_usage_offset_bytes() == 48
    _assert_worker_profiles_loader_memory(
        gms_worker_module,
        imported_weights_bytes=100,
        memory_usage_offset_bytes=48,
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
