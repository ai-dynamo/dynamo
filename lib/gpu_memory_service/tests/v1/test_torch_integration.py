# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch
from _deps import HAS_CUDA
from gpu_memory_service.core.client.memory_manager import LocalMapping
from gpu_memory_service.v1.parameter_storage import (
    copy_non_parameter_tensors_to_default_allocator,
)


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.none
@pytest.mark.gpu_0
def test_copy_out_preserves_tensorimpls_and_nonparameter_aliases() -> None:
    model = torch.nn.Module()
    source = torch.arange(64, dtype=torch.float32)
    source_storage = source.untyped_storage()
    model.weight = torch.nn.Parameter(source[:32])
    model.overlapping_weight = torch.nn.Parameter(source[16:40])
    model.strided_weight = torch.nn.Parameter(source[32:52:2])
    model.register_buffer("overlap", model.weight[8:24])
    model.overlap_alias = model.overlap[4:12]
    model.register_buffer(
        "empty_view",
        torch.empty(0, dtype=torch.float32).set_(
            source_storage,
            44,
            (0, 3),
            (5, 1),
        ),
    )
    model.register_buffer(
        "disjoint",
        torch.empty(0, dtype=torch.float32).set_(
            source_storage,
            48,
            (4,),
            (1,),
        ),
    )
    workspace = torch.empty(0, dtype=torch.float32).set_(
        source_storage,
        56,
        (4,),
        (1,),
    )
    del source

    mapping = LocalMapping(
        "mixed",
        source_storage.nbytes(),
        source_storage.nbytes(),
        source_storage.data_ptr(),
        source_storage.nbytes(),
    )
    tensor_impls = {
        name: int(tensor._cdata)
        for name, tensor in (
            ("weight", model.weight),
            ("overlapping_weight", model.overlapping_weight),
            ("strided_weight", model.strided_weight),
            ("overlap", model.overlap),
            ("overlap_alias", model.overlap_alias),
            ("empty_view", model.empty_view),
            ("disjoint", model.disjoint),
            ("workspace", workspace),
        )
    }
    original_storage = int(source_storage._cdata)
    empty_layout = (
        model.empty_view.storage_offset(),
        model.empty_view.shape,
        model.empty_view.stride(),
    )

    accounting = copy_non_parameter_tensors_to_default_allocator(model, (mapping,))

    assert {
        name: int(tensor._cdata)
        for name, tensor in (
            ("weight", model.weight),
            ("overlapping_weight", model.overlapping_weight),
            ("strided_weight", model.strided_weight),
            ("overlap", model.overlap),
            ("overlap_alias", model.overlap_alias),
            ("empty_view", model.empty_view),
            ("disjoint", model.disjoint),
            ("workspace", workspace),
        )
    } == tensor_impls
    assert int(model.weight.untyped_storage()._cdata) == original_storage
    assert int(model.overlapping_weight.untyped_storage()._cdata) == original_storage
    assert int(model.strided_weight.untyped_storage()._cdata) == original_storage
    assert int(model.empty_view.untyped_storage()._cdata) != original_storage
    assert (
        model.empty_view.storage_offset(),
        model.empty_view.shape,
        model.empty_view.stride(),
    ) == empty_layout
    assert int(model.overlap.untyped_storage()._cdata) != original_storage
    assert (
        model.overlap.untyped_storage()._cdata
        == model.overlap_alias.untyped_storage()._cdata
    )
    assert (
        model.disjoint.untyped_storage()._cdata
        != model.overlap.untyped_storage()._cdata
    )
    assert int(workspace.untyped_storage()._cdata) != original_storage
    assert workspace.untyped_storage()._cdata != model.disjoint.untyped_storage()._cdata
    assert model.weight.tolist() == list(map(float, range(32)))
    assert model.overlap.tolist() == list(map(float, range(8, 24)))
    assert model.disjoint.tolist() == [48.0, 49.0, 50.0, 51.0]
    assert workspace.tolist() == [56.0, 57.0, 58.0, 59.0]

    with torch.inference_mode():
        model.overlap_alias.fill_(23)
    assert model.overlap.tolist()[4:12] == [23.0] * 8
    assert model.weight.tolist()[12:20] == list(map(float, range(12, 20)))
    assert accounting == (51 * 4, (16 + 4 + 4) * 4)


@pytest.mark.post_merge
@pytest.mark.integration
@pytest.mark.gpu_1
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA is required")
def test_real_cuda_allocator_and_dual_domain_lifecycle() -> None:
    """Use a subprocess because Torch allocator callbacks are process-global."""
    code = textwrap.dedent(
        """
        import logging
        import os
        import sys
        import tempfile
        import threading
        import types

        import torch

        from gpu_memory_service.common.vmm import get_vmm
        from gpu_memory_service.core.server.gms import GMSServerMemoryManager
        from gpu_memory_service.core.server.rpc import GMSRPCServer

        sleep_mode_backend = types.ModuleType(
            "vllm.device_allocator.sleep_mode_backend"
        )
        class SleepModeBackend:
            def __init__(self):
                self._state = "RUNNING"
        class SleepModeBackendFactory:
            @classmethod
            def register_backend(cls, *args):
                pass
        sleep_mode_backend.SleepModeBackend = SleepModeBackend
        sleep_mode_backend.SleepModeBackendFactory = SleepModeBackendFactory
        vllm = types.ModuleType("vllm")
        vllm.__path__ = []
        device_allocator = types.ModuleType("vllm.device_allocator")
        device_allocator.__path__ = []
        sys.modules["vllm"] = vllm
        logger = types.ModuleType("vllm.logger")
        logger.init_logger = logging.getLogger
        sys.modules["vllm.logger"] = logger
        sys.modules["vllm.device_allocator"] = device_allocator
        sys.modules[
            "vllm.device_allocator.sleep_mode_backend"
        ] = sleep_mode_backend

        from gpu_memory_service.v1.integrations.vllm import backend as backend_module

        torch.cuda.set_device(0)
        vmm = get_vmm()
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        with tempfile.TemporaryDirectory() as directory:
            weights_path = os.path.join(directory, "weights.sock")
            kv_path = os.path.join(directory, "kv_cache.sock")
            weights_gms = GMSServerMemoryManager(gpu_uuid, vmm, 0)
            kv_gms = GMSServerMemoryManager(gpu_uuid, vmm, 0)
            with (
                GMSRPCServer(weights_path, weights_gms) as weights_server,
                GMSRPCServer(kv_path, kv_gms) as kv_server,
            ):
                threads = [
                    threading.Thread(
                        target=server.serve_forever,
                        daemon=True,
                    )
                    for server in (weights_server, kv_server)
                ]
                for thread in threads:
                    thread.start()
                try:
                    backend_module.get_socket_path = lambda device, domain: (
                        weights_path if domain == "weights" else kv_path
                    )
                    backend_module.get_vmm = lambda: vmm
                    backend = backend_module.GMSV1SleepModeBackend()
                    manager = backend._weights
                    model = None
                    with backend.capture_weights(lambda: model):
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
                        del parameter_backing
                        model.register_buffer("runtime", runtime)
                        model.runtime_alias = runtime
                        model.runtime_view = runtime[1024:-1024:3]
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
                        object_identities = (
                            id(model.weight),
                            id(model.runtime),
                            id(model.runtime_view),
                        )
                        tensor_impls = (
                            int(model.weight._cdata),
                            int(model.runtime._cdata),
                            int(model.runtime_view._cdata),
                        )

                    assert object_identities == (
                        id(model.weight),
                        id(model.runtime),
                        id(model.runtime_view),
                    )
                    assert tensor_impls == (
                        int(model.weight._cdata),
                        int(model.runtime._cdata),
                        int(model.runtime_view._cdata),
                    )
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

                    before = tuple(
                        (mapping.base, mapping.allocation_id)
                        for mapping in manager.mappings
                    )
                    with backend.capture_kv_cache():
                        kv = torch.empty(1024, device="cuda")
                    backend.suspend()
                    backend.resume()
                    assert before == tuple(
                        (mapping.base, mapping.allocation_id)
                        for mapping in manager.mappings
                    )
                    assert kv.data_ptr() in {
                        mapping.base for mapping in backend._kv_cache.mappings
                    }
                finally:
                    weights_server.shutdown()
                    kv_server.shutdown()
                    for thread in threads:
                        thread.join(timeout=10)
                        assert not thread.is_alive()
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
