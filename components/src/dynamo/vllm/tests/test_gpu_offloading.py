# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU retained-cache configuration, source isolation and real local D2D copies.

The CUDA tests run only on allocated GPUs. They compare GPU tensors directly;
only scalar verdicts return to the host. Remote KVCR/NIXL tests are separate.
"""

import os
import subprocess
import sys
import threading
import uuid
from types import SimpleNamespace

import pytest
import torch
from kvbm.vllm_integration import gpu_memory
from kvbm.vllm_integration.gpu_memory import GPUCacheRegion, GPUCacheWorker, import_ipc
from kvbm.vllm_integration.gpu_offloading import (
    GPUOffloadingConnector,
    GPUOffloadingSpec,
    GPURetainedLoadStoreSpec,
    GPUTieringManager,
)
from vllm.distributed.kv_events import BlockRemoved
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
]


def make_config(**changes):
    extra = {
        "gpu_bytes_to_use": 8 * 384,
        "offload_prompt_only": False,
        "self_describing_kv_events": True,
        "router_hint_transfer_domain": "test-gpu-layout-v1",
        "secondary_tiers": [{"type": "GPUKVCRSecondaryTierManager"}],
    }
    extra.update(changes)
    return SimpleNamespace(
        extra_config=extra,
        engine_id=f"gpu-cache-test-{uuid.uuid4()}",
        groups=(SimpleNamespace(tokens_per_block=64),),
        cache=SimpleNamespace(tokens_per_hash=64, blocks_per_chunk=1),
        parallel=SimpleNamespace(world_size=1),
        canonical_layout=False,
        replicated_layout=False,
        worker_kv_bytes_per_block=384,
        enable_kv_cache_events=True,
    )


@pytest.mark.gpu_0
@pytest.mark.parametrize(
    "changes",
    [
        {"gpu_bytes_to_use": 0},
        {"gpu_bytes_to_use": True},
        {"gpu_bytes_to_use": 383},
        {"cpu_bytes_to_use": 4096},
        {"canonical_layout": True},
        {"router_hint_transfer_domain": ""},
        {"secondary_tiers": [{"type": "kvcr"}]},
        {"secondary_tiers": []},
        {"self_describing_kv_events": False},
    ],
)
def test_gpu_spec_rejects_cpu_or_uncertified_configuration(changes):
    with pytest.raises(ValueError):
        GPUOffloadingSpec(make_config(**changes))


@pytest.mark.gpu_0
def test_gpu_spec_rejects_uncertified_parallelism():
    config = make_config()
    config.parallel.world_size = 2
    with pytest.raises(ValueError, match="TP1"):
        GPUOffloadingSpec(config)


@pytest.mark.gpu_0
def test_gpu_inventory_has_its_own_medium(monkeypatch):
    event = BlockRemoved(block_hashes=[123], medium="CPU")
    monkeypatch.setattr(OffloadingConnector, "take_events", lambda self: iter([event]))
    connector = GPUOffloadingConnector.__new__(GPUOffloadingConnector)
    result = list(connector.take_events())
    assert result[0].medium == "gpu_direct"
    assert event.medium == "CPU"


@pytest.mark.gpu_0
def test_gpu_reset_does_not_release_native_owned_slots():
    manager = GPUTieringManager.__new__(GPUTieringManager)
    manager._gpu_lock = threading.RLock()
    manager._jobs = {}
    manager.primary_tier = SimpleNamespace(
        _num_allocated_chunks=2,
        _free_list=[],
        _num_evictable_cache_chunks=1,
    )
    manager.secondary_tiers = [SimpleNamespace(_hazard_ops=set())]
    with pytest.raises(RuntimeError, match="own its slots"):
        manager.reset_cache()


@pytest.mark.gpu_0
def test_ipc_rejects_different_geometry_before_touching_cuda():
    with pytest.raises(ValueError, match="layout"):
        import_ipc({"schema": 1, "shape": [9, 384]}, (8, 384))


@pytest.mark.gpu_0
def test_uniprocess_worker_uses_owned_gpu_storage_without_self_ipc(monkeypatch):
    engine_id = f"uniprocess-test-{uuid.uuid4()}"
    tensor = object()
    owner = SimpleNamespace(shape=(8, 384), tensor=tensor, _closed=False)
    monkeypatch.setitem(
        gpu_memory._OWNED_GPU_REGIONS, gpu_memory.region_path(engine_id), owner
    )
    assert gpu_memory.open_region(engine_id, (8, 384)) is tensor
    with pytest.raises(ValueError, match="layout"):
        gpu_memory.open_region(engine_id, (9, 384))
    owner._closed = True
    with pytest.raises(ValueError, match="closed"):
        gpu_memory.open_region(engine_id, (8, 384))


@pytest.mark.gpu_1
def test_cuda_ipc_owner_and_importer_share_payload_without_host_copy():
    engine_id = f"ipc-test-{uuid.uuid4()}"
    region = GPUCacheRegion(engine_id, 4, 256)
    try:
        # The child imports only IPC metadata and changes the GPU allocation.
        script = """import torch
from kvbm.vllm_integration.gpu_memory import open_region
region = open_region(__import__('sys').argv[1], (4,256))
region.fill_(37)
torch.cuda.synchronize()
"""
        result = subprocess.run(
            [sys.executable, "-c", script, engine_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(os.environ),
        )
        assert result.returncode == 0, result.stderr
        assert bool(torch.all(region.tensor == 37).item())
    finally:
        region.cleanup()


@pytest.mark.gpu_1
def test_mixed_bank_d2d_roundtrip_preserves_permuted_slots_and_guards():
    config = make_config()
    spec = GPUOffloadingSpec(config)
    region = GPUCacheRegion(config.engine_id, spec.num_chunks, spec.row_bytes)
    originals = [
        (torch.arange(6 * width, device="cuda", dtype=torch.int64) % 113)
        .to(torch.int8)
        .reshape(6, width)
        for width in (128, 256)
    ]
    expected = [tensor.clone() for tensor in originals]
    caches = SimpleNamespace(
        tensors=[
            SimpleNamespace(tensor=t, page_size_bytes=t.shape[1]) for t in originals
        ],
        group_data_refs=[
            [SimpleNamespace(tensor_idx=0, page_size_bytes=128)],
            [SimpleNamespace(tensor_idx=1, page_size_bytes=256)],
        ],
    )
    worker = GPUCacheWorker(spec, caches)
    try:
        source = GPULoadStoreSpec([3, 1, 2, 0], [2, 2], [0, 0])
        slots = GPURetainedLoadStoreSpec([5, 1, 6, 4])
        assert worker.submit_store(1, source, slots)
        worker.wait({1})
        assert [result.job_id for result in worker.get_finished()] == [1]
        assert bool(torch.all(region.tensor[0] == 0).item())
        originals[0][4].zero_()
        originals[0][0].zero_()
        originals[1][5].zero_()
        originals[1][1].zero_()
        target = GPULoadStoreSpec([4, 0, 5, 1], [2, 2], [0, 0])
        assert worker.submit_load(2, slots, target)
        worker.wait({2})
        assert [result.job_id for result in worker.get_finished()] == [2]
        assert torch.equal(originals[0][4], expected[0][3])
        assert torch.equal(originals[0][0], expected[0][1])
        assert torch.equal(originals[1][5], expected[1][2])
        assert torch.equal(originals[1][1], expected[1][0])
        assert torch.equal(originals[0][2], expected[0][2])
        assert torch.equal(originals[1][3], expected[1][3])
        assert bool(torch.all(region.tensor[0] == 0).item())
    finally:
        worker.shutdown()
        region.cleanup()
