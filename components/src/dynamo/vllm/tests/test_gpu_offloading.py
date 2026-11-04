# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU retained-cache configuration, source isolation and real local D2D copies.

The CUDA tests run only on allocated GPUs. They compare GPU tensors directly;
only scalar verdicts return to the host. Remote KVCR/NIXL tests are separate.
"""

import base64
import os
import subprocess
import sys
import uuid
import weakref
from types import SimpleNamespace

import pytest
import torch
from kvbm.vllm_integration import gpu_memory, gpu_mnnvl
from kvbm.vllm_integration.gpu_memory import GPUCacheRegion, GPUCacheWorker, import_ipc
from kvbm.vllm_integration.gpu_offloading import GPURetainedLoadStoreSpec
from vllm.v1.kv_offload.base import GPULoadStoreSpec

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
]


@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_ipc_rejects_different_geometry_before_touching_cuda():
    with pytest.raises(ValueError, match="layout"):
        import_ipc({"schema": 1, "shape": [9, 384]}, (8, 384))


@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_uniprocess_worker_uses_owned_gpu_storage_without_self_ipc(monkeypatch):
    engine_id = f"uniprocess-test-{uuid.uuid4()}"
    tensor = object()
    owner = SimpleNamespace(
        gpu_transport="cuda_ipc",
        shape=(8, 384),
        tensor=tensor,
        _closed=False,
        consume_local_export=lambda: None,
    )
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
def test_local_cuda_owner_balances_unused_ipc_export_once(monkeypatch):
    engine_id = f"local-ipc-refcount-{uuid.uuid4()}"
    region = GPUCacheRegion(engine_id, 4, 256)
    released = []
    original = torch.storage.TypedStorage._release_ipc_counter

    def release(*args, **kwargs):
        original(*args, **kwargs)
        released.append((args, kwargs))

    monkeypatch.setattr(
        torch.storage.TypedStorage, "_release_ipc_counter", staticmethod(release)
    )
    try:
        alias = gpu_memory.open_region(engine_id, region.shape)
        assert alias is region.tensor
        assert gpu_memory.open_region(engine_id, region.shape) is region.tensor
        assert len(released) == 1
        assert not (region.path / "ipc.json").exists()
        region.tensor.fill_(29)
        assert bool(torch.all(region.tensor == 29).item())
        region.cleanup()
        assert region.tensor is None
        # A local worker alias retains its own storage after the closed native
        # endpoint drops the primary owner. No payload crosses host memory.
        alias.fill_(31)
        assert bool(torch.all(alias == 31).item())
    finally:
        region.cleanup()


@pytest.mark.gpu_1
@pytest.mark.timeout(120)
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
del region
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
        assert "shared CUDA tensors released" not in result.stderr
        assert "could not unlink" not in result.stderr
        assert bool(torch.all(region.tensor == 37).item())
    finally:
        region.cleanup()


@pytest.mark.gpu_1
def test_mixed_bank_d2d_roundtrip_preserves_permuted_slots_and_guards():
    config = SimpleNamespace(
        engine_id=f"mixed-bank-{uuid.uuid4()}", parallel=SimpleNamespace(world_size=1)
    )
    spec = SimpleNamespace(
        config=config,
        num_chunks=8,
        row_bytes=384,
        blocks_per_chunk=1,
        gpu_transport="cuda_ipc",
    )
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
        handler_refs = [weakref.ref(handler) for handler in worker.handlers]
        view_ref = weakref.ref(worker.handlers[0].dst_tensors[0])
        worker.shutdown()
        # Correct teardown must release CUDA views without a garbage-collection
        # pass; a retained bound-method cycle otherwise outlives IPC globals.
        assert all(reference() is None for reference in handler_refs)
        assert view_ref() is None
    finally:
        worker.shutdown()
        region.cleanup()


@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_local_fabric_descriptor_mismatch_fails_before_importing_any_gpu_handle(
    monkeypatch,
):
    identity = gpu_mnnvl.FabricIdentity(
        "GPU-00000000-0000-0000-0000-000000000001", "1" * 32, 0
    )
    base = "test-gpu-layout-v1"
    domain = gpu_mnnvl.effective_domain(base, identity.domain)
    monkeypatch.setattr(gpu_memory.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(gpu_mnnvl, "observe_fabric", lambda device: identity)
    monkeypatch.setattr(gpu_memory, "_process_start_ticks", lambda pid: 17)
    monkeypatch.setattr(
        gpu_mnnvl,
        "FabricAllocation",
        lambda *args, **kwargs: pytest.fail(
            "Invalid descriptor must not import a handle"
        ),
    )
    original = {
        "schema": "dynamo-gpu-fabric-v1",
        "gpu_transport": "mnnvl",
        "engine_id": "fixture",
        "owner_pid": os.getpid(),
        "owner_start_ticks": 17,
        "gpu_uuid": identity.gpu_uuid,
        "fabric_domain": identity.domain,
        "base_domain": base,
        "compatibility_domain": domain,
        "shape": [2, 8],
        "stride": [8, 1],
        "logical_bytes": 16,
        "allocation_bytes": 16,
        "fabric_handle": base64.b64encode(b"x" * 64).decode(),
    }
    for change in (
        {"schema": 1},
        {"gpu_transport": "cuda_ipc"},
        {"engine_id": "other"},
        {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000002"},
        {"fabric_domain": "mnnvl:" + "2" * 32 + ":0"},
        {"base_domain": "other-layout"},
        {"compatibility_domain": "other-domain"},
        {"shape": [True, 8]},
        {"stride": [8, True]},
        {"owner_start_ticks": 18},
        {"logical_bytes": True},
        {"allocation_bytes": True},
        {"fabric_handle": base64.b64encode(b"x" * 63).decode()},
        {"unexpected": "value"},
    ):
        with pytest.raises(ValueError):
            gpu_memory.import_fabric(
                {**original, **change}, "fixture", (2, 8), base, domain, identity.domain
            )
