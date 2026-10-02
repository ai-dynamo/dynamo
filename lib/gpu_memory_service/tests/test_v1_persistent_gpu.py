# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Verify retained physical bytes across a real CUDA client's SIGKILL."""

import ctypes
import signal
import subprocess
import sys
import threading
import time

import pytest
from _deps import HAS_CUDA, HAS_GMS

if not HAS_GMS or not HAS_CUDA:
    pytest.skip("requires GMS and CUDA", allow_module_level=True)

from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.vmm.cuda_utils import CudaVMM
from gpu_memory_service.v1.client.mapping import reserve_and_install_mapping
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.device import get_device_uuid
from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_1]

_WRITE_AND_CRASH = """
import ctypes, os, signal, sys
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.vmm.cuda_utils import CudaVMM
from gpu_memory_service.v1.client.mapping import reserve_and_install_mapping
from gpu_memory_service.v1.client.session import _GMSClientSession
vmm = CudaVMM()
vmm.ensure_initialized()
vmm.runtime_set_device(0)
size = vmm.get_allocation_granularity(0)
session = _GMSClientSession(sys.argv[1], RequestedLockType.RW, process_fence=True)
session.allocate('retained', size)
mapping, handle = reserve_and_install_mapping(
    vmm, session.export('retained'), 'retained', size, size, size, size,
    0, session.lock_type)
stream = vmm.stream_create_nonblocking()
data = ctypes.create_string_buffer(bytes(range(256)))
vmm.host_register(ctypes.addressof(data), 256)
vmm.memcpy_h2d_async(mapping.base, ctypes.addressof(data), 256, stream)
vmm.stream_synchronize(stream)
session.retain_allocations()
if sys.argv[2] == 'pending':
    import torch
    pending = ctypes.create_string_buffer(b'\\xff' * 256)
    vmm.host_register(ctypes.addressof(pending), 256)
    with torch.cuda.stream(torch.cuda.ExternalStream(int(stream))):
        torch.cuda._sleep(2_000_000_000)
        vmm.memcpy_h2d_async(mapping.base, ctypes.addressof(pending), 256, stream)
os.kill(os.getpid(), signal.SIGKILL)
"""


@pytest.mark.timeout(30)
@pytest.mark.parametrize("phase", ["completed", "pending"])
def test_physical_kv_bytes_survive_client_sigkill(tmp_path, phase):
    vmm = CudaVMM()
    vmm.ensure_initialized()
    vmm.runtime_set_device(0)
    manager = GMSServerMemoryManager(get_device_uuid(0), vmm, 0, allow_retention=True)
    path = str(tmp_path / "kv.sock")
    with GMSRPCServer(path, manager) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            crashed = subprocess.run(
                [sys.executable, "-c", _WRITE_AND_CRASH, path, phase],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            assert crashed.returncode == -signal.SIGKILL, crashed.stderr
            replacement = _GMSClientSession(
                path, RequestedLockType.RW, process_fence=True
            )
            try:
                inventory = replacement.list_allocations().allocations
                assert len(inventory) == 1
                size = inventory[0].aligned_size
                mapping, handle = reserve_and_install_mapping(
                    vmm,
                    replacement.export("retained"),
                    "retained",
                    size,
                    size,
                    size,
                    size,
                    0,
                    replacement.lock_type,
                )
                output = ctypes.create_string_buffer(256)
                stream = vmm.stream_create_nonblocking()
                vmm.host_register(ctypes.addressof(output), 256)
                try:
                    if phase == "pending":
                        # A replacement must be able to overwrite its mapping
                        # without a predecessor's queued DMA writing afterward.
                        output.raw = bytes(range(256))
                        vmm.memcpy_h2d_async(
                            mapping.base, ctypes.addressof(output), 256, stream
                        )
                        vmm.stream_synchronize(stream)
                        time.sleep(3)
                    vmm.memcpy_d2h_async(
                        ctypes.addressof(output), mapping.base, 256, stream
                    )
                    vmm.stream_synchronize(stream)
                    assert output.raw == bytes(range(256))
                finally:
                    vmm.host_unregister(ctypes.addressof(output))
                    vmm.stream_destroy(stream)
                    vmm.unmap(mapping.base, size)
                    vmm.release(handle)
                    vmm.address_free(mapping.base, size)
            finally:
                replacement.close(quiesced=True)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()
            manager._clear_allocations()
