# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-GPU check that one V1 allocation can span two adjacent slabs."""

from __future__ import annotations

import threading

import pytest
from _deps import HAS_CUDA, HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_CUDA:
    pytest.skip("a CUDA GPU is required", allow_module_level=True)

import torch
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
from gpu_memory_service.v1 import device as device_identity
from gpu_memory_service.v1.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.none,
    pytest.mark.gpu_1,
]


class _CudaBytes:
    def __init__(self, va: int, size: int):
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (va, False),
            "version": 3,
        }


def _bytes_at(va: int, size: int) -> torch.Tensor:
    return torch.as_tensor(_CudaBytes(va, size), device="cuda")


def _pattern(size: int, seed: int) -> torch.Tensor:
    return ((torch.arange(size, device="cuda") + seed) % 251).to(torch.uint8)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "commit", [True, False], ids=["committed-remap", "reallocate-remap"]
)
def test_allocation_spanning_two_slabs_survives_remap(tmp_path, commit) -> None:
    init_vmm(VMMDeviceType.CUDA)
    vmm = get_vmm()
    torch.cuda.set_device(0)
    path = str(tmp_path / "gms.sock")
    manager = GMSServerMemoryManager(device_identity.get_device_uuid(0), vmm, 0)
    server = GMSRPCServer(path, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    granularity = int(vmm.get_allocation_granularity(0))
    client = GMSClientMemoryManager(path, vmm, 0, slab_size=2 * granularity)
    try:
        client.connect(RequestedLockType.RW)
        client.create_mapping(granularity)
        # The first slab has one granule left, so this allocation takes it
        # and the first granule of a second slab mapped right after it.
        size = 2 * granularity
        va = client.create_mapping(size)
        assert len(client.mappings) == 2
        boundary = client.mappings[1].base
        assert va < boundary < va + size

        data = _bytes_at(va, size)
        data.copy_(_pattern(size, 1))
        torch.cuda.synchronize()
        assert torch.equal(data, _pattern(size, 1))
        del data

        if commit:
            client.commit()
        client.unmap_all_vas()
        client.disconnect()
        if commit:
            client.connect(RequestedLockType.RO)
        else:
            client.connect(RequestedLockType.RW)
            client.reallocate_all_handles()
        client.remap_all_vas()
        assert client.mappings[1].base == boundary

        data = _bytes_at(va, size)
        if commit:
            assert torch.equal(data, _pattern(size, 1))
        else:
            data.copy_(_pattern(size, 2))
            torch.cuda.synchronize()
            assert torch.equal(data, _pattern(size, 2))
        del data
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=5)
