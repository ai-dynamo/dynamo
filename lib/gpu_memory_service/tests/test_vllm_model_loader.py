# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight-pool ownership in the vLLM first writer."""

import gc
from types import SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS or not HAS_TORCH:
    pytest.skip("GMS and Torch are required", allow_module_level=True)

import torch
from gpu_memory_service.client.torch import allocator, module
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.integrations.vllm import model_loader

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.none,
]


def test_publication_keeps_initial_value_storage_alive(monkeypatch):
    owner = torch.ones(8).untyped_storage()
    manager = SimpleNamespace(tag="weights")
    monkeypatch.setattr(
        allocator,
        "_tag_states",
        {"weights": SimpleNamespace(manager=manager, callback_failure=None)},
    )
    monkeypatch.setattr(model_loader, "_pending_gms_client", manager)
    monkeypatch.setattr(model_loader, "_pending_retained_gms_tensors", [owner])
    monkeypatch.setattr(model_loader, "_published_gms_tensor_owners", [])

    def publish(actual):
        assert actual is manager
        assert owner.nbytes() == 32

    monkeypatch.setattr(model_loader, "publish_gms_write", publish)
    assert model_loader.publish_pending_gms_write()
    assert model_loader._published_gms_tensor_owners == [owner]


def test_pool_finalization_preserves_live_mapping_and_surfaces_free_failure(
    monkeypatch,
):
    events = []
    manager = SimpleNamespace(tag="weights", mappings={1: "live"})

    class Pool:
        def __del__(self):
            events.append("torch_released_cached_blocks")

    state = SimpleNamespace(
        manager=manager, mem_pool=Pool(), callback_failure=RuntimeError("free failed")
    )
    monkeypatch.setattr(allocator, "_tag_states", {"weights": state})
    with pytest.raises(RuntimeError, match="free callback failed") as error:
        allocator.release_gms_mem_pool(manager)
    assert str(error.value.__cause__) == "free failed"
    assert events == ["torch_released_cached_blocks"]
    assert manager.mappings == {1: "live"}
    assert state.mem_pool is None


def _count_gc_collections(monkeypatch) -> list:
    calls = []
    collect = gc.collect

    def counted(*args):
        calls.append(args)
        return collect(*args)

    monkeypatch.setattr(gc, "collect", counted)
    return calls


@pytest.mark.parametrize("in_cycle", [False, True])
def test_pool_release_collects_again_only_for_a_pool_in_a_cycle(monkeypatch, in_cycle):
    events = []

    class Pool:
        def __del__(self):
            events.append("released")

    pool = Pool()
    if in_cycle:
        pool.cycle = pool
    manager = SimpleNamespace(tag="weights")
    state = SimpleNamespace(manager=manager, mem_pool=pool, callback_failure=None)
    del pool
    monkeypatch.setattr(allocator, "_tag_states", {"weights": state})
    calls = _count_gc_collections(monkeypatch)

    allocator.release_gms_mem_pool(manager)

    assert events == ["released"]
    assert len(calls) == (2 if in_cycle else 1)


def test_writer_copy_out_collects_once_and_logs_what_it_copied(monkeypatch, caplog):
    """The heap scan reuses the pool release's full garbage collection (each
    one walks about a million objects in an engine), and one INFO line reports
    the copies by module-tree membership."""
    mib = 1 << 20
    floats_per_mib = mib // 4
    storage = torch.arange(4 * floats_per_mib, dtype=torch.float32).untyped_storage()

    def mib_view(index: int, count: int = 1) -> torch.Tensor:
        # set_ instead of slicing: a slice would keep its base tensor alive.
        return torch.empty(0).set_(
            storage, index * floats_per_mib, (count * floats_per_mib,), (1,)
        )

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(mib_view(0, 2))
    model.register_buffer("scale", mib_view(2))
    model.scale_head = model.scale[:8]
    workspace = mib_view(3)

    unused_va = 1 << 12
    manager = SimpleNamespace(
        tag="weights",
        device=0,
        total_bytes=storage.nbytes() + (1 << 30),
        mappings={
            storage.data_ptr(): SimpleNamespace(
                va=storage.data_ptr(), aligned_size=storage.nbytes()
            ),
            unused_va: SimpleNamespace(va=unused_va, aligned_size=1 << 30),
        },
    )

    class Pool:
        def __del__(self):
            # Torch releases the pool's cached (unused) segment.
            manager.mappings.pop(unused_va)
            manager.total_bytes -= 1 << 30

    state = SimpleNamespace(manager=manager, mem_pool=Pool(), callback_failure=None)
    monkeypatch.setattr(allocator, "_tag_states", {"weights": state})
    monkeypatch.setattr(module, "_is_accelerator_tensor", lambda t: True)
    device = SimpleNamespace(synchronize=lambda index: None, empty_cache=lambda: None)
    monkeypatch.setattr(model_loader, "torch_device", lambda: device)
    calls = _count_gc_collections(monkeypatch)
    impls = [int(t._cdata) for t in (model.scale, model.scale_head, workspace)]

    with caplog.at_level("INFO", logger=model_loader.logger.name):
        stats, copied_bytes, owners = model_loader._release_pool_and_copy_out(
            manager, model
        )

    assert len(calls) == 1
    # TensorImpls are kept; non-Parameters moved to private copies with aliases.
    assert [int(t._cdata) for t in (model.scale, model.scale_head, workspace)] == (
        impls
    )
    assert int(model.weight.untyped_storage()._cdata) == int(storage._cdata)
    for tensor in (model.scale, workspace):
        assert int(tensor.untyped_storage()._cdata) != int(storage._cdata)
    assert model.scale_head.data_ptr() == model.scale.data_ptr()
    assert workspace[0].item() == 3 * floats_per_mib
    assert [int(owner._cdata) for owner in owners] == [int(storage._cdata)]
    assert copied_bytes == 2 * mib
    assert (stats.pruned_count, stats.pruned_bytes) == (1, 1 << 30)
    # A set: another test's logging setup can attach a second capture handler.
    assert {
        record.getMessage()
        for record in caplog.records
        if record.name == model_loader.logger.name
    } == {
        "[GMS] Write mode: copied 3 non-Parameter tensors (2.00 MiB) out of GMS: "
        "in-tree 2 (1.00 MiB), out-of-tree 1 (1.00 MiB); pruned 1 unused "
        "allocations (1.00 GiB)"
    }


def test_native_free_failure_is_preserved_for_the_publication_barrier(monkeypatch):
    def fail(va):
        raise RuntimeError("sidecar release failed")

    manager = SimpleNamespace(
        tag="weights",
        granted_lock_type=GrantedLockType.RW,
        mappings={4096: object()},
        destroy_scratch_mapping=lambda va: False,
        destroy_mapping=fail,
    )
    state = SimpleNamespace(manager=manager, mem_pool=None, callback_failure=None)
    monkeypatch.setattr(allocator, "_tag_states", {"weights": state})
    allocator._gms_free(4096, 64, 0, 0)
    with pytest.raises(RuntimeError, match="free callback failed") as error:
        allocator.raise_if_gms_allocator_failed(manager)
    assert str(error.value.__cause__) == "sidecar release failed"
