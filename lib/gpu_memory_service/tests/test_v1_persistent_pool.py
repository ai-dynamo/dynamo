# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.persistent_pool import PersistentPoolKey
from gpu_memory_service.v1.client.persistent_pool import V1PersistentPoolBackend
from gpu_memory_service.v1.client.session import GMSV1RemoteError, _GMSClientSession
from gpu_memory_service.v1.protocol import (
    ERROR_CLAIM_CONFLICT,
    ERROR_INVALID_REQUEST,
    ERROR_NOT_CLAIMED,
    AbortRequest,
    AllocateRequest,
    CommitRequest,
    ExportRequest,
    FreeRequest,
    ListAllocationsRequest,
    SuccessResponse,
)
from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.none,
    pytest.mark.gpu_0,
    pytest.mark.timeout(10),
]


@pytest.fixture
def v1_server(tmp_path):
    path = str(tmp_path / "gms-v1.sock")
    vmm = FakeVMM(granularity=64)
    manager = GMSServerMemoryManager("GPU-0", vmm, 0)
    server = GMSRPCServer(path, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield path, manager, vmm
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        manager.persistent.clear_all()


def _connect(path: str) -> tuple[_GMSClientSession, V1PersistentPoolBackend]:
    session = _GMSClientSession(
        path,
        RequestedLockType.RW_PERSISTENT,
        connect_timeout=1,
    )
    return session, V1PersistentPoolBackend(session)


def test_shared_claim_export_inventory_and_destroy(v1_server, monkeypatch) -> None:
    path, manager, vmm = v1_server
    key = PersistentPoolKey("engine-a", "kv:rank0")
    first_session, first = _connect(path)
    second_session, second = _connect(path)
    observer_session, observer = _connect(path)
    try:
        created = first.claim(key, 64, shared=True)
        attached = second.claim(key, 64, shared=True)

        assert not created.reattached
        assert attached.reattached
        assert attached.allocation_id == created.allocation_id
        monkeypatch.setenv("GMS_PERSISTENT_CLAIM_RETRY_SECS", "0")
        with pytest.raises(GMSV1RemoteError) as mode_conflict:
            first.claim(key, 64, shared=False)
        assert mode_conflict.value.code == ERROR_INVALID_REQUEST

        assert manager.persistent.shared_claim_count(key.engine_id, key.tag) == 2
        assert first.inventory() == [created]
        assert observer.inventory() == []
        assert observer.inventory(include_unclaimed=True) == [created]
        assert manager.session_snapshot().rw_sessions == 0
        assert manager.session_snapshot().ro_sessions == 0

        fd = second.export(key)
        try:
            os.fstat(fd)
        finally:
            os.close(fd)
        with pytest.raises(GMSV1RemoteError) as denied:
            observer.export(key)
        assert denied.value.code == ERROR_NOT_CLAIMED

        assert first.unclaim(key)
        assert second.destroy(key)
        assert observer.inventory(include_unclaimed=True) == []
        assert not vmm.server_handles
    finally:
        observer_session.close()
        second_session.close()
        first_session.close()


def test_disconnect_preserves_backing_and_conflicts_are_typed(
    v1_server,
    monkeypatch,
) -> None:
    path, manager, _vmm = v1_server
    key = PersistentPoolKey("engine-a", "kv:rank0")
    first_session, first = _connect(path)
    second_session, second = _connect(path)
    monkeypatch.setenv("GMS_PERSISTENT_CLAIM_RETRY_SECS", "0")
    created = first.claim(key, 64)
    try:
        with pytest.raises(GMSV1RemoteError) as conflict:
            second.claim(key, 64)
        assert conflict.value.code == ERROR_CLAIM_CONFLICT
    finally:
        first_session.close()

    monkeypatch.setenv("GMS_PERSISTENT_CLAIM_RETRY_SECS", "1")
    attached = second.claim(key, 64)
    try:
        assert attached.reattached
        assert attached.allocation_id == created.allocation_id
        assert manager.persistent.allocation_count == 1
    finally:
        assert second.destroy(key)
        second_session.close()


def test_transactional_epoch_reset_does_not_clear_persistent_pool(v1_server) -> None:
    path, manager, _vmm = v1_server
    key = PersistentPoolKey("engine-a", "kv:rank0")
    persistent_session, persistent = _connect(path)
    transactional = _GMSClientSession(
        path,
        RequestedLockType.RW,
        connect_timeout=1,
    )
    try:
        claimed = persistent.claim(key, 64)
        transactional.allocate("ephemeral", 64)
        transactional.close()

        assert manager.allocation_snapshot() == ()
        assert persistent.inventory() == [claimed]
        fd = persistent.export(key)
        os.close(fd)
        assert persistent.destroy(key)
    finally:
        transactional.close()
        persistent_session.close()


def test_backend_rejects_transactional_session(v1_server) -> None:
    path, _manager, _vmm = v1_server
    session = _GMSClientSession(
        path,
        RequestedLockType.RW,
        connect_timeout=1,
    )
    try:
        with pytest.raises(ValueError, match="RW_PERSISTENT"):
            V1PersistentPoolBackend(session)
        with pytest.raises(GMSV1RemoteError, match="RW_PERSISTENT"):
            session.claim_persistent(
                "engine-a",
                "kv:rank0",
                64,
            )
    finally:
        session.close()


@pytest.mark.parametrize(
    "operation",
    [
        AllocateRequest("other", 64),
        ExportRequest("uncommitted"),
        ListAllocationsRequest(),
        FreeRequest("uncommitted"),
        CommitRequest(),
        AbortRequest(),
    ],
)
def test_persistent_session_cannot_access_transactional_epoch(v1_server, operation):
    path, manager, _vmm = v1_server
    writer = _GMSClientSession(path, RequestedLockType.RW)
    persistent_session, _backend = _connect(path)
    try:
        writer.allocate("uncommitted", 64)
        with pytest.raises(GMSV1RemoteError, match="transactional operation"):
            persistent_session._call(operation, SuccessResponse)
        assert manager.allocation_snapshot() == (("uncommitted", 64),)
        # Rejection must not close or commit the legitimate writer's epoch.
        os.close(writer.export("uncommitted"))
    finally:
        persistent_session.close()
        writer.close()


@pytest.mark.parametrize("shared", [False, True])
def test_repeated_claim_and_permanent_conflicts_do_not_retry(
    v1_server, monkeypatch, shared
):
    path, manager, _vmm = v1_server
    session, backend = _connect(path)
    key = PersistentPoolKey("engine", "kv")
    # Any backoff here would hide a permanent error behind failover latency.
    monkeypatch.setattr(
        "gpu_memory_service.common.persistent_pool.time.sleep",
        lambda _: pytest.fail("permanent conflict must not retry"),
    )
    try:
        first = backend.claim(key, 64, shared=shared)
        repeated = backend.claim(key, 64, shared=shared)
        assert repeated.reattached
        assert repeated.allocation_id == first.allocation_id
        for size, mode in [(128, shared), (64, not shared)]:
            with pytest.raises(GMSV1RemoteError) as failure:
                backend.claim(key, size, shared=mode)
            assert failure.value.code == ERROR_INVALID_REQUEST
        assert backend.unclaim(key)
        assert not backend.unclaim(key)
        assert manager.persistent.active_claim_count == 0
        assert backend.destroy(key)
    finally:
        session.close()
