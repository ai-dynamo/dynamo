# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the vLLM first writer leaves in GMS memory before it publishes.

Publication remaps every kept allocation read-only, so a non-Parameter tensor
still inside one faults on its next write. These tests drive the writer's
prepare step (``_load_write_mode``) with CPU tensors standing in for the GMS
pool: one "allocation" is a single storage that the model's tensors view.
"""

import gc
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS or not HAS_TORCH:
    pytest.skip("GMS and Torch are required", allow_module_level=True)

import torch
from gpu_memory_service.client.torch import allocator, module
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.integrations.common import utils as common_utils
from gpu_memory_service.integrations.vllm import model_loader

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.none,
]

_ARENA_BYTES = 1 << 16


class _Manager:
    """The parts of GMSClientMemoryManager the writer's prepare step uses."""

    tag = "weights"
    device = 0
    granted_lock_type = GrantedLockType.RW
    is_unmapped = False

    def __init__(self, *storages: torch.UntypedStorage):
        self.mappings = {
            storage.data_ptr(): SimpleNamespace(
                va=storage.data_ptr(),
                aligned_size=storage.nbytes(),
                allocation_id=f"weights-{index}",
                handle=1,
            )
            for index, storage in enumerate(storages)
        }
        self.total_bytes = sum(storage.nbytes() for storage in storages)
        self.metadata = {}

    def metadata_put(self, key, allocation_id, offset_bytes, value):
        self.metadata[key] = (allocation_id, offset_bytes)

    def destroy_mapping(self, va):
        self.total_bytes -= self.mappings.pop(va).aligned_size

    def close(self, *, best_effort=False):
        pass

    def in_gms(self, tensor: torch.Tensor) -> bool:
        ptr = tensor.data_ptr()
        return any(va <= ptr < va + m.aligned_size for va, m in self.mappings.items())


def _view(storage, dtype, byte_offset, shape):
    # set_ instead of slicing a base tensor, which would keep the base alive.
    item = torch.empty(0, dtype=dtype).element_size()
    stride = []
    step = 1
    for size in reversed(shape):
        stride.insert(0, step)
        step *= size
    return torch.empty(0, dtype=dtype).set_(
        storage, byte_offset // item, tuple(shape), tuple(stride)
    )


@pytest.fixture
def write_with(monkeypatch):
    """Run the vLLM writer's prepare step on a model ``build(storage)`` returns.

    ``allocations=2`` gives ``build`` a second storage, in its own allocation.
    """
    monkeypatch.setattr(model_loader, "_pending_gms_client", None)
    monkeypatch.setattr(model_loader, "_pending_retained_gms_tensors", [])
    monkeypatch.setattr(model_loader, "_last_imported_weights_bytes", 0)
    monkeypatch.setattr(model_loader, "_last_model_memory_usage_offset_bytes", 0)
    monkeypatch.setattr(model_loader, "_published_gms_tensor_owners", [], raising=False)
    monkeypatch.setattr(module, "_is_accelerator_tensor", lambda t: True)
    device = SimpleNamespace(synchronize=lambda index: None, empty_cache=lambda: None)
    monkeypatch.setattr(model_loader, "torch_device", lambda: device)
    monkeypatch.setattr(common_utils, "torch_device", lambda: device)
    monkeypatch.setattr(model_loader, "get_mx_load_context", lambda *args: None)
    monkeypatch.setattr(model_loader, "gms_use_mem_pool", lambda *args: nullcontext())

    def run(build, allocations=1):
        storages = [
            torch.zeros(_ARENA_BYTES, dtype=torch.uint8).untyped_storage()
            for _ in range(allocations)
        ]
        manager = _Manager(*storages)

        class Pool:
            pass

        monkeypatch.setattr(
            allocator,
            "_tag_states",
            {
                "weights": SimpleNamespace(
                    manager=manager, mem_pool=Pool(), callback_failure=None
                )
            },
        )
        loader_utils = ModuleType("vllm.model_executor.model_loader.utils")
        loader_utils.initialize_model = lambda **kwargs: build(*storages)
        loader_utils.process_weights_after_loading = lambda *args: None
        torch_utils = ModuleType("vllm.utils.torch_utils")
        torch_utils.set_default_torch_dtype = lambda dtype: nullcontext()
        monkeypatch.setitem(sys.modules, loader_utils.__name__, loader_utils)
        monkeypatch.setitem(sys.modules, torch_utils.__name__, torch_utils)

        model = model_loader._load_write_mode(
            manager,
            SimpleNamespace(),
            SimpleNamespace(dtype=torch.float32),
            SimpleNamespace(load_weights=lambda model, model_config: None),
            torch.device("cpu"),
        )
        return manager, model, storages[0]

    return run


class _SparseMLAIndexGroup:
    """Stand-in for vLLM's SparseMLAIndexGroup: a plain object, not a module."""

    def __init__(self, physical_topk_indices: torch.Tensor):
        self.physical_topk_indices = physical_topk_indices


class _AttentionImpl:
    """Attention impls are plain objects held by the attention layer."""

    def __init__(self, storage, backend):
        # A kernel counter that the attention backend keeps in a module global
        # and creates during model construction, so inside the GMS pool.
        backend.kv_counter = _view(storage, torch.uint8, 0x2200, (512,))
        self._kv_counter_buffer = backend.kv_counter
        topk = _view(storage, torch.int32, 0x4000, (8, 64))
        topk.copy_(torch.arange(8 * 64, dtype=torch.int32).view(8, 64))
        self.index_group = _SparseMLAIndexGroup(topk)


def test_writable_buffer_held_outside_the_module_tree_leaves_gms(write_with):
    """A kernel-written buffer in a kept allocation but outside the module tree
    must not stay in memory that publication remaps read-only (FlashInfer's
    multi-CTA KV counter, SparseMLAIndexGroup.physical_topk_indices)."""
    backend = SimpleNamespace()

    def build(storage):
        model = torch.nn.Module()
        # The weight keeps the allocation: it is registered, so not pruned.
        model.weight = torch.nn.Parameter(
            _view(storage, torch.float32, 0, (1024,)), requires_grad=False
        )
        model.attn = torch.nn.Module()
        model.attn.register_buffer(
            "_k_scale", _view(storage, torch.float32, 0x2000, (1,))
        )
        model.attn.impl = _AttentionImpl(storage, backend)
        return model

    manager, model, storage = write_with(build)

    counter = backend.kv_counter
    topk = model.attn.impl.index_group.physical_topk_indices
    assert manager.in_gms(model.weight)
    assert "weight" in manager.metadata
    # Neither out-of-tree buffer may remain in it ...
    assert [manager.in_gms(counter), manager.in_gms(topk)] == [False, False]
    # ... and every holder still sees one writable tensor with the old values.
    assert model.attn.impl._kv_counter_buffer is counter
    assert topk.flatten().tolist() == list(range(8 * 64))
    counter.add_(1)
    assert torch.equal(
        model.attn.impl._kv_counter_buffer, torch.ones(512, dtype=torch.uint8)
    )
    assert not manager.in_gms(model.attn._k_scale)


def test_parameter_held_outside_the_module_tree_fails_the_write(write_with):
    """Only non-Parameters are copied out, so a Parameter that no module holds
    stays in GMS, and nothing registers it. In an allocation of its own it
    would be committed without metadata. A reader maps only the allocations
    its metadata references, so it could not remap that layout."""
    kernel = SimpleNamespace()

    def build(storage, own):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(
            _view(storage, torch.float32, 0, (1024,)), requires_grad=False
        )
        kernel.table = torch.nn.Parameter(
            _view(own, torch.float32, 0, (1024,)), requires_grad=False
        )
        return model

    with pytest.raises(RuntimeError, match="keeps 1 allocation"):
        write_with(build, allocations=2)
    assert model_loader._pending_gms_client is None


def test_module_buffers_frozen_by_gc_still_leave_gms(write_with):
    """gc.freeze() hides objects from the heap scan (Dynamo's benchmark GC
    policy freezes periodically); tensors that modules hold are still found."""

    def build(storage):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(
            _view(storage, torch.float32, 0, (1024,)), requires_grad=False
        )
        model.register_buffer("_k_scale", _view(storage, torch.float32, 0x2000, (1,)))
        model.scales = [_view(storage, torch.float32, 0x2100, (4,))]
        gc.freeze()
        return model

    try:
        manager, model, storage = write_with(build)
    finally:
        gc.unfreeze()

    assert manager.in_gms(model.weight)
    assert not manager.in_gms(model._k_scale)
    assert not manager.in_gms(model.scales[0])


def test_one_tensor_under_many_names_stays_one_copy(write_with):
    """GLM-5.2's topk_indices_buffer is one tensor reachable under 121 module
    names (the model, and every layer's attention and indexer). The indexer
    writes it and sparse attention reads it, so the writer must move it out
    once and keep every name on the same memory."""
    layers = 60

    def build(storage):
        model = torch.nn.Module()
        model.embed = torch.nn.Parameter(
            _view(storage, torch.float32, 0, (1024,)), requires_grad=False
        )
        shared = _view(storage, torch.int32, 0x8000, (16, 64))
        model.topk_indices_buffer = shared
        model.layers = torch.nn.ModuleList()
        for _ in range(layers):
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.topk_indices_buffer = shared
            layer.self_attn.indexer = torch.nn.Module()
            layer.self_attn.indexer.topk_indices_buffer = shared
            model.layers.append(layer)
        return model

    manager, model, storage = write_with(build)

    holders = [model] + [
        owner
        for layer in model.layers
        for owner in (layer.self_attn, layer.self_attn.indexer)
    ]
    names = [owner.topk_indices_buffer for owner in holders]
    assert len(names) == 2 * layers + 1
    assert not any(manager.in_gms(tensor) for tensor in names)
    assert len({tensor.data_ptr() for tensor in names}) == 1
    # What the indexer writes, the attention reads.
    model.layers[7].self_attn.indexer.topk_indices_buffer.fill_(5)
    assert int(model.layers[42].self_attn.topk_indices_buffer.sum()) == 5 * 16 * 64
    assert int(model.topk_indices_buffer.sum()) == 5 * 16 * 64
    # One copy's worth of private memory, not one per name.
    assert model_loader.get_model_memory_usage_offset_bytes() == 16 * 64 * 4
