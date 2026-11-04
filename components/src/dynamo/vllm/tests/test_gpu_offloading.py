# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU retained-cache configuration, source isolation and real local D2D copies.

The CUDA tests run only on allocated GPUs. They compare GPU tensors directly;
only scalar verdicts return to the host. Remote KVCR/NIXL tests are separate.
"""

import os
import subprocess
import sys
import uuid
from types import SimpleNamespace

import pytest
import torch
from kvbm.vllm_integration import gpu_memory
from kvbm.vllm_integration.gpu_memory import GPUCacheRegion, import_ipc

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
