# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import socket
import threading
import time

import pytest
from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.core.client.session import GMSClientSession
from gpu_memory_service.core.errors import FatalGMSError, GMSError
from gpu_memory_service.core.protocol import send_message
from gpu_memory_service.core.server.allocations import GMSAllocationManager
from gpu_memory_service.core.server.gms import GMS
from gpu_memory_service.core.server.rpc import GMSRPCServer
from gpu_memory_service.v1.memory_manager import (
    EphemeralKVCacheMemoryManager,
    PersistentParameterMemoryManager,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


def _start(path):
    vmm = FakeVMM(granularity=64)
    allocations = GMSAllocationManager(vmm, 0)
    gms = GMS("GPU-0", allocations)
    server = GMSRPCServer(path, gms)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return vmm, allocations, gms, server, thread


def _stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)
    assert not thread.is_alive()


def _wait_for(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for GMS session transition")


def _open_in_thread(path, lock_type):
    connected = threading.Event()
    result = []

    def connect():
        result.append(GMSClientSession(path, lock_type))
        connected.set()

    thread = threading.Thread(target=connect, daemon=True)
    thread.start()
    return result, connected, thread


def test_rw_abort_clears_epoch_and_releases_socket_lock(tmp_path) -> None:
    path = str(tmp_path / "gms-v1.sock")
    vmm, allocations, gms, server, thread = _start(path)
    writer = GMSClientSession(path, RequestedLockType.RW)
    writer.allocate("uncommitted", 64)

    waiting, connected, waiter = _open_in_thread(path, RequestedLockType.RW)
    _wait_for(lambda: gms.snapshot().waiting_writers == 1)
    assert not connected.is_set()

    writer.close()
    assert connected.wait(5)
    next_writer = waiting.pop()
    try:
        assert allocations.allocation_count == 0
        assert not vmm.server_handles
        assert next_writer.lock_type == GrantedLockType.RW
    finally:
        next_writer.close()
        waiter.join(timeout=5)
        _stop(server, thread)


def test_commit_holds_same_socket_ro_and_rw_or_ro_respects_writer_priority(
    tmp_path,
) -> None:
    path = str(tmp_path / "gms-v1.sock")
    vmm, allocations, gms, server, thread = _start(path)
    writer = GMSClientSession(path, RequestedLockType.RW)
    writer.allocate("committed", 64)
    original_handles = set(vmm.server_handles)
    reader_result, reader_connected, reader_thread = _open_in_thread(
        path, RequestedLockType.RW_OR_RO
    )
    assert not reader_connected.wait(0.05)

    writer.commit()
    assert writer.lock_type == GrantedLockType.RO
    assert reader_connected.wait(5)
    reader = reader_result.pop()
    assert gms.snapshot().ro_session_count == 2

    next_result, next_connected, next_thread = _open_in_thread(
        path, RequestedLockType.RW
    )
    _wait_for(lambda: gms.snapshot().waiting_writers == 1)
    late_reader_result, late_reader_connected, late_reader_thread = _open_in_thread(
        path, RequestedLockType.RW_OR_RO
    )
    assert not late_reader_connected.wait(0.05)
    reader.close()
    assert not next_connected.wait(0.05)
    assert not late_reader_connected.is_set()
    assert allocations.allocation_count == 1
    assert set(vmm.server_handles) == original_handles

    writer.close()
    assert next_connected.wait(5)
    assert not late_reader_connected.is_set()
    next_writer = next_result.pop()
    try:
        assert allocations.allocation_count == 0
        assert not vmm.server_handles
        with pytest.raises(GMSError, match="unknown allocation ID"):
            next_writer.export("committed")
        next_writer.allocate("replacement", 64)
        next_writer.commit()
        assert next_writer.lock_type == GrantedLockType.RO
        assert late_reader_connected.wait(5)
        old_snapshot = late_reader_result.pop()
        try:
            with pytest.raises(GMSError, match="unknown allocation ID"):
                old_snapshot.export("committed")
        finally:
            old_snapshot.close()
    finally:
        next_writer.close()
        reader_thread.join(timeout=5)
        next_thread.join(timeout=5)
        late_reader_thread.join(timeout=5)
        _stop(server, thread)


def test_exports_use_fresh_transient_server_fds(tmp_path) -> None:
    path = str(tmp_path / "gms-v1.sock")
    vmm, allocations, _gms, server, thread = _start(path)
    with pytest.raises(FatalGMSError, match="incarnation|physical GPU"):
        GMSClientSession(
            path,
            RequestedLockType.RO,
            ("stale-server", "GPU-0"),
        )
    writer = GMSClientSession(path, RequestedLockType.RW)
    try:
        initial_fd_count = len(os.listdir("/proc/self/fd"))
        writer.allocate("allocation", 64)
        assert len(os.listdir("/proc/self/fd")) == initial_fd_count
        received_fds = [writer.export("allocation"), writer.export("allocation")]
        assert allocations.allocation_count == 1
        _wait_for(lambda: len(os.listdir("/proc/self/fd")) == initial_fd_count + 2)
        for received_fd in received_fds:
            os.fstat(received_fd)
            os.close(received_fd)
        assert len(os.listdir("/proc/self/fd")) == initial_fd_count
    finally:
        writer.close()
        _stop(server, thread)


def test_disconnected_queued_writer_preserves_committed_epoch(tmp_path) -> None:
    path = str(tmp_path / "gms-v1.sock")
    _vmm, allocations, gms, server, thread = _start(path)
    writer = GMSClientSession(path, RequestedLockType.RW)
    writer.allocate("committed", 64)
    writer.commit()
    reader = GMSClientSession(path, RequestedLockType.RO)

    dead_writer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead_writer.connect(path)
    send_message(
        dead_writer,
        ["handshake", [RequestedLockType.RW.value, None]],
    )
    _wait_for(lambda: gms.snapshot().waiting_writers == 1)
    dead_writer.close()
    _wait_for(lambda: gms.snapshot().waiting_writers == 0)

    reader.close()
    writer.close()
    verifier = GMSClientSession(path, RequestedLockType.RO)
    try:
        fd = verifier.export("committed")
        os.close(fd)
        assert allocations.allocation_count == 1
    finally:
        verifier.close()
        _stop(server, thread)


def test_kv_epoch_is_exclusive_and_wake_recreates_backing_at_saved_vas(
    tmp_path, monkeypatch
) -> None:
    path = str(tmp_path / "gms-v1.sock")
    vmm, allocations, gms, server, thread = _start(path)
    monkeypatch.setattr(
        EphemeralKVCacheMemoryManager,
        "_gpu_identity",
        lambda self: "GPU-0",
    )
    manager = EphemeralKVCacheMemoryManager(path, vmm, 0)
    first = manager.allocate(65)
    second = manager.allocate(33)
    before = {mapping.base: mapping.allocation_id for mapping in manager.mappings}
    first_epoch_handles = set(vmm.server_handles)
    assert set(vmm.mapped) == {first, second}
    assert set(vmm.access.values()) == {GrantedLockType.RW}

    waiting, connected, waiter = _open_in_thread(path, RequestedLockType.RW)
    _wait_for(lambda: gms.snapshot().waiting_writers == 1)
    assert not connected.is_set()
    manager.sleep()
    assert connected.wait(5)
    next_writer = waiting.pop()
    assert set(vmm.reservations) == {first, second}
    assert not vmm.mapped
    assert allocations.allocation_count == 0
    assert not vmm.server_handles
    next_writer.close()
    waiter.join(timeout=5)
    assert not waiter.is_alive()

    manager.wake()
    assert {
        mapping.base: mapping.allocation_id for mapping in manager.mappings
    } == before
    assert set(vmm.mapped) == {first, second}
    assert set(vmm.access.values()) == {GrantedLockType.RW}
    assert vmm.server_handles.isdisjoint(first_epoch_handles)
    assert allocations.allocation_count == 2

    manager.sleep()
    _wait_for(lambda: allocations.allocation_count == 0)
    _stop(server, thread)


def test_snapshot_wake_rejects_another_server_incarnation(
    tmp_path, monkeypatch
) -> None:
    path = str(tmp_path / "gms-v1.sock")
    vmm, allocations, gms, server, thread = _start(path)
    monkeypatch.setattr(
        PersistentParameterMemoryManager,
        "_gpu_identity",
        lambda self: "GPU-0",
    )
    manager = PersistentParameterMemoryManager(path, vmm, 0)
    manager.allocate(64)
    manager.commit()
    manager.sleep()
    _wait_for(lambda: gms.snapshot().ro_session_count == 0)

    gms.server_nonce = "replacement"
    try:
        with pytest.raises(FatalGMSError, match="incarnation"):
            manager.wake()
        assert gms.snapshot().ro_session_count == 0
        assert not vmm.imports
        assert allocations.allocation_count == 1
    finally:
        _stop(server, thread)
