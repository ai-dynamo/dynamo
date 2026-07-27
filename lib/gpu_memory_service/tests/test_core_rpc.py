# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import socket
import threading

import pytest
from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.core.client.session import _GMSClientSession
from gpu_memory_service.core.protocol import HandshakeRequest, send_message
from gpu_memory_service.core.server.gms import GMSServerMemoryManager
from gpu_memory_service.core.server.rpc import GMSRPCServer

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
def test_socket_sessions_commit_share_prioritize_writer_and_release_on_disconnect(
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
    dead_writer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead_writer.connect(path)
    send_message(dead_writer, HandshakeRequest(RequestedLockType.RW))
    assert writer_waiting.wait(5)
    dead_writer.close()
    cancellation_reader_result, cancellation_reader_connected, cancellation_thread = (
        _connect_in_thread(path, RequestedLockType.RO)
    )
    assert cancellation_reader_connected.wait(5)
    cancellation_reader_result.pop().close()

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
        cancellation_thread,
        next_writer_thread,
        late_reader_thread,
    ):
        thread.join(timeout=5)
        assert not thread.is_alive()
    _stop(server, server_thread)
