# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import socket
import threading
from time import monotonic

import pytest
from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.v1.protocol import HandshakeRequest, send_message
from gpu_memory_service.v1.server import GMSRPCServer, GMSServerMemoryManager
from gpu_memory_service.v1.session import _GMSClientSession

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


def _serve(path: str, vmm: FakeVMM):
    manager = GMSServerMemoryManager("GPU-0", vmm, 0)
    server = GMSRPCServer(path, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return manager, server, thread


def _stop(server: GMSRPCServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)
    assert not thread.is_alive()


def _connect_in_thread(path: str, lock_type: RequestedLockType):
    connected = threading.Event()
    result: list[_GMSClientSession] = []

    def connect() -> None:
        result.append(_GMSClientSession(path, lock_type))
        connected.set()

    thread = threading.Thread(target=connect, daemon=True)
    thread.start()
    return result, connected, thread


@pytest.mark.timeout(10)
def test_client_waits_for_server_startup_and_deadlines_absent_socket(tmp_path) -> None:
    path = str(tmp_path / "gms.sock")
    result, connected, client_thread = _connect_in_thread(path, RequestedLockType.RW)
    assert not connected.wait(0.1)
    assert client_thread.is_alive()

    _manager, server, server_thread = _serve(path, FakeVMM(granularity=64))
    try:
        assert connected.wait(5)
        result.pop().close()
        client_thread.join(timeout=5)
        assert not client_thread.is_alive()
    finally:
        _stop(server, server_thread)

    started_at = monotonic()
    with pytest.raises(ConnectionError, match="GMS sidecar socket"):
        _GMSClientSession(
            str(tmp_path / "absent.sock"),
            RequestedLockType.RO,
            connect_timeout=0.05,
        )
    elapsed = monotonic() - started_at
    assert 0.05 <= elapsed < 0.5


@pytest.mark.timeout(10)
def test_rw_close_waits_for_epoch_release(tmp_path, monkeypatch) -> None:
    path = str(tmp_path / "gms.sock")
    vmm = FakeVMM(granularity=64)
    _manager, server, server_thread = _serve(path, vmm)
    writer = _GMSClientSession(path, RequestedLockType.RW)
    writer.allocate("ephemeral", 64)

    release_started = threading.Event()
    release_allowed = threading.Event()
    close_returned = threading.Event()
    original_release = vmm.release

    def blocked_release(handle: int) -> None:
        release_started.set()
        assert release_allowed.wait(5)
        original_release(handle)

    def close_writer() -> None:
        writer.close()
        close_returned.set()

    monkeypatch.setattr(vmm, "release", blocked_release)
    close_thread = threading.Thread(target=close_writer, daemon=True)
    close_thread.start()

    assert release_started.wait(5)
    assert not close_returned.is_set()
    release_allowed.set()
    assert close_returned.wait(5)
    close_thread.join(timeout=5)
    assert not close_thread.is_alive()
    assert not vmm.server_handles
    _stop(server, server_thread)


@pytest.mark.timeout(10)
def test_sessions_commit_share_prioritize_writer_and_release_on_disconnect(
    tmp_path,
    monkeypatch,
) -> None:
    path = str(tmp_path / "gms.sock")
    vmm = FakeVMM(granularity=64)
    manager, server, server_thread = _serve(path, vmm)
    first_writer = _GMSClientSession(path, RequestedLockType.RW)
    first_writer.allocate("aborted", 64)

    writer_waiting = threading.Event()
    can_grant_rw = manager._sessions._can_grant_rw

    def observe_blocked_writer() -> bool:
        granted = can_grant_rw()
        if not granted:
            writer_waiting.set()
        return granted

    monkeypatch.setattr(manager._sessions, "_can_grant_rw", observe_blocked_writer)
    replacement_result, replacement_connected, replacement_thread = _connect_in_thread(
        path, RequestedLockType.RW
    )
    assert writer_waiting.wait(5)
    first_writer.close()
    assert replacement_connected.wait(5)
    replacement = replacement_result.pop()
    with pytest.raises(RuntimeError, match="unknown allocation ID"):
        replacement.export("aborted")

    replacement.allocate("committed", 64)
    replacement.commit()
    assert replacement.lock_type is GrantedLockType.RO
    reader = _GMSClientSession(path, RequestedLockType.RO)
    fd = reader.export("committed")
    os.close(fd)

    writer_waiting.clear()
    disconnected_writer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    disconnected_writer.connect(path)
    send_message(disconnected_writer, HandshakeRequest(RequestedLockType.RW))
    assert writer_waiting.wait(5)
    disconnected_writer.close()
    reader_result, reader_connected, disconnected_thread = _connect_in_thread(
        path, RequestedLockType.RO
    )
    assert reader_connected.wait(5)
    reader_result.pop().close()

    writer_waiting.clear()
    next_writer_result, next_writer_connected, next_writer_thread = _connect_in_thread(
        path, RequestedLockType.RW
    )
    assert writer_waiting.wait(5)
    late_reader_result, late_reader_connected, late_reader_thread = _connect_in_thread(
        path, RequestedLockType.RW_OR_RO
    )
    assert not late_reader_connected.wait(0.05)

    reader.close()
    assert not next_writer_connected.wait(0.05)
    replacement.close()
    assert next_writer_connected.wait(5)
    assert not late_reader_connected.is_set()

    next_writer = next_writer_result.pop()
    next_writer.allocate("replacement", 64)
    next_writer.commit()
    assert late_reader_connected.wait(5)
    late_reader = late_reader_result.pop()
    fd = late_reader.export("replacement")
    os.close(fd)

    late_reader.close()
    next_writer.close()
    for thread in (
        replacement_thread,
        disconnected_thread,
        next_writer_thread,
        late_reader_thread,
    ):
        thread.join(timeout=5)
        assert not thread.is_alive()
    _stop(server, server_thread)
