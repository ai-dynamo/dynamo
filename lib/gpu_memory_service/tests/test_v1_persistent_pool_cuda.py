# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import multiprocessing
import os
import signal
import threading
from contextlib import contextmanager

import pytest
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.persistent_pool import PersistentPoolKey
from gpu_memory_service.common.vmm import get_vmm
from gpu_memory_service.v1.client.persistent_pool import V1PersistentPoolBackend
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_1,
    pytest.mark.timeout(90),
]

_KEY = PersistentPoolKey("crash-test", "kv")


@contextmanager
def _mapping(backend, size, device):
    vmm = get_vmm()
    handle = vmm.import_shareable_handle_close_fd(backend.export(_KEY))
    va = None
    mapped = False
    stream = None
    try:
        va = vmm.address_reserve(size, vmm.get_allocation_granularity(device))
        vmm.map(va, size, handle)
        mapped = True
        vmm.set_access(va, size, device, GrantedLockType.RW)
        stream = vmm.stream_create_nonblocking()
        yield vmm, va, stream
    finally:
        if stream is not None:
            vmm.stream_destroy(stream)
        if mapped:
            vmm.unmap(va, size)
        if va is not None:
            vmm.address_free(va, size)
        vmm.release(handle)


def _payload(size):
    return bytes(range(256)) * (size // 256)


def _write_then_crash(path, device, result):
    torch.cuda.set_device(device)
    torch.empty(1, device=f"cuda:{device}")  # Establish this process's CUDA context.
    session = _GMSClientSession(path, RequestedLockType.RW_PERSISTENT)
    backend = V1PersistentPoolBackend(session)
    size = get_vmm().get_allocation_granularity(device)
    allocation = backend.claim(_KEY, size)
    with _mapping(backend, size, device) as (vmm, va, stream):
        host = ctypes.create_string_buffer(_payload(size))
        vmm.memcpy_h2d_async(va, ctypes.addressof(host), size, stream)
        vmm.stream_synchronize(stream)
        result.send((allocation.allocation_id, size))
        # No unmap, unclaim, socket close, or Python cleanup runs in the owner.
        os.kill(os.getpid(), signal.SIGKILL)


def test_v1_gpu_bytes_survive_killed_client(tmp_path):
    device = int(os.environ.get("GMS_TEST_CUDA_DEVICE", "0"))
    torch.cuda.set_device(device)
    torch.empty(1, device=f"cuda:{device}")
    path = str(tmp_path / "persistent.sock")
    vmm = get_vmm()
    manager = GMSServerMemoryManager(
        str(torch.cuda.get_device_properties(device).uuid), vmm, device
    )
    server = GMSRPCServer(path, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    context = multiprocessing.get_context("spawn")
    received, sent = context.Pipe(duplex=False)
    writer = context.Process(target=_write_then_crash, args=(path, device, sent))
    thread.start()
    try:
        writer.start()
        sent.close()
        assert received.poll(45), "writer did not finish a synchronized GPU write"
        allocation_id, size = received.recv()
        writer.join(timeout=15)
        assert writer.exitcode == -signal.SIGKILL
        session = _GMSClientSession(
            path, RequestedLockType.RW_PERSISTENT, expected_identity=manager.identity
        )
        try:
            backend = V1PersistentPoolBackend(session)
            recovered = backend.claim(_KEY, size)
            assert recovered.reattached
            assert recovered.allocation_id == allocation_id
            with _mapping(backend, size, device) as (vmm, va, stream):
                host = ctypes.create_string_buffer(size)
                vmm.memcpy_d2h_async(ctypes.addressof(host), va, size, stream)
                vmm.stream_synchronize(stream)
                assert host.raw == _payload(size)
            assert backend.destroy(_KEY)
            assert backend.inventory(include_unclaimed=True) == []
        finally:
            session.close()
    finally:
        if writer.is_alive():
            writer.kill()
            writer.join(timeout=5)
        received.close()
        sent.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        manager.persistent.clear_all()
        assert not thread.is_alive()
