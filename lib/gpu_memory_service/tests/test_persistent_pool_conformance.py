# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import threading
from contextlib import ExitStack

import gpu_memory_service.common.vmm as vmm_module
import pytest
from _fake_vmm import FakeVMM
from gpu_memory_service.client.persistent_pool import V0PersistentPoolBackend
from gpu_memory_service.client.session import _GMSClientSession as V0Session
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.persistent_pool import PersistentPoolKey
from gpu_memory_service.server.rpc import GMSRPCServer as V0Server
from gpu_memory_service.v1.client.persistent_pool import V1PersistentPoolBackend
from gpu_memory_service.v1.client.session import _GMSClientSession as V1Session
from gpu_memory_service.v1.server.rpc import GMSRPCServer as V1Server
from gpu_memory_service.v1.server.rpc import GMSServerMemoryManager

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.timeout(15),
]


@pytest.fixture(params=["v0", "v1"])
def pools(request, tmp_path, monkeypatch):
    """Same black-box ownership tests through both real Unix-socket transports."""
    vmm = FakeVMM(granularity=64)
    monkeypatch.setattr(vmm_module, "_vmm_instance", vmm)
    monkeypatch.setenv("GMS_PERSISTENT_CLAIM_RETRY_SECS", "0")
    path = str(tmp_path / "pool.sock")
    if request.param == "v0":
        server = V0Server(path)
        store = server._gms._persistent
        loop = asyncio.new_event_loop()
        serving = loop.create_task(server.serve())

        def run():
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(serving)
            except asyncio.CancelledError:
                pass
            finally:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
                loop.close()

        thread = threading.Thread(target=run)

        def connect():
            session = V0Session(path, RequestedLockType.RW_PERSISTENT, timeout_ms=2000)
            return session, V0PersistentPoolBackend(session)

        def stop():
            loop.call_soon_threadsafe(serving.cancel)

    else:
        manager = GMSServerMemoryManager("GPU-0", vmm, 0)
        store = manager.persistent
        server = V1Server(path, manager)
        thread = threading.Thread(target=server.serve_forever)

        def connect():
            session = V1Session(
                path, RequestedLockType.RW_PERSISTENT, connect_timeout=2
            )
            return session, V1PersistentPoolBackend(session)

        def stop():
            server.shutdown()
            server.server_close()

    thread.start()
    try:
        with ExitStack() as clients:

            def attach():
                session, backend = connect()
                clients.callback(session.close)
                return backend

            yield attach, vmm
    finally:
        stop()
        thread.join(timeout=5)
        assert not thread.is_alive()
        store.clear_all()


@pytest.mark.parametrize("shared", [False, True])
def test_claim_lifecycle_contract(pools, shared):
    attach, _vmm = pools
    owner, peer = attach(), attach()
    key = PersistentPoolKey("engine", "kv")
    created = owner.claim(key, 128, shared=shared)
    repeated = owner.claim(key, 128, shared=shared)
    assert not created.reattached and repeated.reattached
    assert repeated.allocation_id == created.allocation_id
    with pytest.raises(RuntimeError):
        owner.claim(key, 128, shared=not shared)
    with pytest.raises(RuntimeError):
        owner.claim(key, 256, shared=shared)
    with pytest.raises(RuntimeError):
        peer.export(key)
    with pytest.raises(RuntimeError):
        peer.destroy(key)
    assert peer.inventory() == []
    assert (
        peer.inventory(include_unclaimed=True)[0].allocation_id == created.allocation_id
    )
    fd = owner.export(key)
    os.fstat(fd)
    os.close(fd)
    if shared:
        assert peer.claim(key, 64, shared=True).aligned_size == 128
        with pytest.raises(RuntimeError):
            owner.destroy(key)
        assert peer.unclaim(key)
    else:
        with pytest.raises(RuntimeError):
            peer.claim(key, 128)
    assert owner.unclaim(key)
    assert not owner.unclaim(key)
    assert peer.claim(key, 128, shared=shared).allocation_id == created.allocation_id
    assert peer.destroy(key)
    assert not peer.destroy(key)
    assert peer.inventory(include_unclaimed=True) == []


def test_allocation_exhaustion_preserves_inventory(pools, monkeypatch):
    attach, vmm = pools
    backend = attach()
    monkeypatch.setattr(vmm, "create_tolerate_oom", lambda *_: (False, 0))
    with pytest.raises(MemoryError):
        backend.claim(PersistentPoolKey("engine", "kv"), 64)
    assert backend.inventory(include_unclaimed=True) == []
    assert not vmm.server_handles
