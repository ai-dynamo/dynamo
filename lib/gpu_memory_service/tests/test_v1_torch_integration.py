# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading

import pytest
import torch
from _deps import HAS_CUDA
from _v1_fakes import V1FakeVMM
from gpu_memory_service.v1.client.memory_manager import (
    LocalMapping,
    SnapshotMemoryManager,
)
from gpu_memory_service.v1.client.torch.allocator import (
    SnapshotTorchPool,
    _AllocatorCallbacks,
)
from gpu_memory_service.v1.client.torch.module import normalize_model_storages
from gpu_memory_service.v1.errors import FatalGMSError
from gpu_memory_service.v1.server.allocations import AllocationStore


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.none
@pytest.mark.gpu_0
def test_normalization_preserves_tensor_identity_views_and_mixed_storage() -> None:
    model = torch.nn.Module()
    parameter_backing = torch.arange(16, dtype=torch.float32)
    model.weight = torch.nn.Parameter(parameter_backing.view(4, 4))
    model.register_buffer("parameter_view", parameter_backing[2:12:2])
    runtime = torch.arange(12, dtype=torch.float32)
    model.register_buffer("runtime", runtime)
    model.runtime_alias = runtime
    model.runtime_view = runtime[2:10:2]

    parameter_storage = model.weight.untyped_storage()
    runtime_storage = runtime.untyped_storage()
    parameter_mapping = LocalMapping(
        "parameter",
        parameter_storage.nbytes(),
        parameter_storage.nbytes(),
        parameter_storage.data_ptr(),
        parameter_storage.nbytes(),
    )
    runtime_mapping = LocalMapping(
        "runtime",
        runtime_storage.nbytes(),
        runtime_storage.nbytes(),
        runtime_storage.data_ptr(),
        runtime_storage.nbytes(),
    )
    identities = (
        id(model.weight),
        id(model.parameter_view),
        id(model.runtime),
        id(model.runtime_view),
    )
    layouts = (
        (model.runtime.storage_offset(), model.runtime.stride()),
        (model.runtime_view.storage_offset(), model.runtime_view.stride()),
    )
    old_parameter_storage = int(parameter_storage._cdata)
    old_runtime_storage = int(runtime_storage._cdata)

    mappings = (parameter_mapping, runtime_mapping)
    normalize_model_storages(model, mappings)

    assert identities == (
        id(model.weight),
        id(model.parameter_view),
        id(model.runtime),
        id(model.runtime_view),
    )
    assert model.runtime is model.runtime_alias
    assert int(model.weight.untyped_storage()._cdata) == old_parameter_storage
    assert int(model.parameter_view.untyped_storage()._cdata) == old_parameter_storage
    assert int(model.runtime.untyped_storage()._cdata) != old_runtime_storage
    assert (
        model.runtime.untyped_storage()._cdata
        == model.runtime_view.untyped_storage()._cdata
    )
    assert model.runtime_view.data_ptr() == (
        model.runtime.data_ptr()
        + model.runtime_view.storage_offset() * model.runtime_view.element_size()
    )
    assert layouts == (
        (model.runtime.storage_offset(), model.runtime.stride()),
        (model.runtime_view.storage_offset(), model.runtime_view.stride()),
    )
    model.runtime_view.fill_(23)
    assert model.runtime.tolist()[2:10:2] == [23.0] * 4

    # Normalization does not prune mappings. The allocator removes the now
    # inactive runtime mapping when the temporary MemPool is destroyed.
    assert mappings == (parameter_mapping, runtime_mapping)


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.none
@pytest.mark.gpu_0
def test_void_free_callback_failure_surfaces_during_pool_destruction(
    monkeypatch,
) -> None:
    vmm = V1FakeVMM()
    store = AllocationStore("GPU-0", vmm, 0)
    monkeypatch.setattr(SnapshotMemoryManager, "_gpu_identity", lambda self: "GPU-0")
    manager = SnapshotMemoryManager(store, vmm, 0)
    base = manager.allocate(64)

    pool = SnapshotTorchPool.__new__(SnapshotTorchPool)
    pool._torch = torch
    pool._manager = manager
    pool._allocator = _AllocatorCallbacks(manager)
    pool._condition = threading.Condition()
    pool._active_scope = False
    pool._finalized = False
    pool.device = 0
    pool.model_load = object()
    pool.native_workspace = object()

    pool._allocator.free(base, 63, 0, 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    with pytest.raises(FatalGMSError, match="allocator free"):
        pool.finalize_model_load(torch.nn.Module())
    assert not vmm.server_handles
    assert not vmm.imports
    assert not vmm.reservations


@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.gpu_1
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required")
def test_real_cuda_normalization_releases_nonparameter_mapping() -> None:
    """Use a subprocess because Torch allocator callbacks are process-global."""
    code = textwrap.dedent(
        """
        import os
        import tempfile
        import threading

        import torch

        from gpu_memory_service.common.vmm import get_vmm
        from gpu_memory_service.v1.client.memory_manager import SnapshotMemoryManager
        from gpu_memory_service.v1.client.rpc import AllocationClient
        from gpu_memory_service.v1.client.torch import SnapshotTorchPool
        from gpu_memory_service.v1.errors import GMSError
        from gpu_memory_service.v1.server.allocations import AllocationStore
        from gpu_memory_service.v1.server.rpc import AllocationRPCServer

        torch.cuda.set_device(0)
        vmm = get_vmm()
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "gms-v1.sock")
            store = AllocationStore(gpu_uuid, vmm, 0)
            with AllocationRPCServer(path, store) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                client = AllocationClient(path)
                try:
                    manager = SnapshotMemoryManager(client, vmm, 0)
                    pool = SnapshotTorchPool(manager)
                    with pool.model_load_pool():
                        parameter_backing = torch.arange(
                            4 * 1024 * 1024,
                            device="cuda",
                            dtype=torch.float32,
                        )
                        runtime = torch.arange(
                            16 * 1024 * 1024,
                            device="cuda",
                            dtype=torch.uint8,
                        )
                        model = torch.nn.Module()
                        model.weight = torch.nn.Parameter(
                            parameter_backing.view(-1, 1024),
                            requires_grad=False,
                        )
                        model.empty = torch.nn.Parameter(
                            torch.empty(0, device="cuda"),
                            requires_grad=False,
                        )
                        model.register_buffer("runtime", runtime)
                        model.runtime_alias = runtime
                        model.runtime_view = runtime[1024:-1024:3]
                        assert model.empty.untyped_storage().nbytes() == 0

                    parameter_mapping = next(
                        mapping
                        for mapping in manager.mappings
                        if mapping.base <= model.weight.data_ptr() < mapping.end
                    )
                    runtime_mapping = next(
                        mapping
                        for mapping in manager.mappings
                        if mapping.base <= model.runtime.data_ptr() < mapping.end
                    )
                    assert parameter_mapping.base != runtime_mapping.base
                    object_identities = (
                        id(model.weight),
                        id(model.runtime),
                        id(model.runtime_view),
                    )
                    parameter_tensor_impl = int(model.weight._cdata)
                    pool.finalize_model_load(model)

                    assert object_identities == (
                        id(model.weight),
                        id(model.runtime),
                        id(model.runtime_view),
                    )
                    assert parameter_tensor_impl == int(model.weight._cdata)
                    assert model.runtime is model.runtime_alias
                    assert (
                        model.runtime.untyped_storage()._cdata
                        == model.runtime_view.untyped_storage()._cdata
                    )
                    assert runtime_mapping.base not in {
                        mapping.base for mapping in manager.mappings
                    }
                    assert parameter_mapping.base in {
                        mapping.base for mapping in manager.mappings
                    }

                    with pool.native_workspace_pool():
                        workspace = torch.empty(
                            4096, dtype=torch.uint8, device="cuda"
                        )
                    assert not any(
                        mapping.base <= workspace.data_ptr() < mapping.end
                        for mapping in manager.mappings
                    )

                    before = tuple(
                        (mapping.base, mapping.allocation_id)
                        for mapping in manager.mappings
                    )
                    checkpoint_identities = (
                        id(model.weight),
                        id(model.runtime),
                        id(model.runtime_view),
                        int(model.weight._cdata),
                        int(model.runtime._cdata),
                        int(model.runtime_view._cdata),
                    )
                    pool.prepare_snapshot()
                    try:
                        client.export(parameter_mapping.allocation_id)
                    except GMSError as error:
                        assert "disconnected" in str(error)
                    else:
                        raise AssertionError("sleep left the RPC stream connected")
                    manager.wake()
                    assert before == tuple(
                        (mapping.base, mapping.allocation_id)
                        for mapping in manager.mappings
                    )
                    assert checkpoint_identities == (
                        id(model.weight),
                        id(model.runtime),
                        id(model.runtime_view),
                        int(model.weight._cdata),
                        int(model.runtime._cdata),
                        int(model.runtime_view._cdata),
                    )
                    manager.retire()
                finally:
                    client.close()
                    server.shutdown()
                    thread.join(timeout=10)
                    assert not thread.is_alive()
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
