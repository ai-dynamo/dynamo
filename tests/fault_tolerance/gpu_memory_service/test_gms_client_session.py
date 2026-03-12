# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.client.memory_manager import LocalMapping
from gpu_memory_service.client.rpc import _GMSRPCTransport
from gpu_memory_service.client.session import _GMSClientSession
from gpu_memory_service.common.protocol import wire
from gpu_memory_service.common.protocol.messages import (
    CommitResponse,
    ErrorResponse,
    HandshakeResponse,
)
from gpu_memory_service.common.types import GrantedLockType, RequestedLockType

pytestmark = [
    pytest.mark.unit,
    pytest.mark.fault_tolerance,
]


def test_client_session_timeout_closes_transport(monkeypatch):
    closed = False

    monkeypatch.setattr(_GMSRPCTransport, "connect", lambda self: None)
    monkeypatch.setattr(
        _GMSRPCTransport,
        "handshake",
        lambda self, lock_type, timeout_ms: HandshakeResponse(
            success=False,
            committed=False,
        ),
    )

    def fake_close(self) -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(_GMSRPCTransport, "close", fake_close)

    with pytest.raises(TimeoutError, match="Timeout waiting for lock"):
        _GMSClientSession("/tmp/gms-test.sock", RequestedLockType.RO, 1000)

    assert closed


def test_client_session_handshake_failure_closes_transport(monkeypatch):
    closed = False

    monkeypatch.setattr(_GMSRPCTransport, "connect", lambda self: None)
    monkeypatch.setattr(
        _GMSRPCTransport,
        "handshake",
        lambda self, lock_type, timeout_ms: (_ for _ in ()).throw(
            RuntimeError("handshake failed")
        ),
    )

    def fake_close(self) -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(_GMSRPCTransport, "close", fake_close)

    with pytest.raises(RuntimeError, match="handshake failed"):
        _GMSClientSession("/tmp/gms-test.sock", RequestedLockType.RO, 1000)

    assert closed


def test_client_session_records_granted_lock_and_committed(monkeypatch):
    monkeypatch.setattr(_GMSRPCTransport, "connect", lambda self: None)
    monkeypatch.setattr(
        _GMSRPCTransport,
        "handshake",
        lambda self, lock_type, timeout_ms: HandshakeResponse(
            success=True,
            committed=True,
            granted_lock_type=GrantedLockType.RO,
        ),
    )

    session = _GMSClientSession("/tmp/gms-test.sock", RequestedLockType.RW_OR_RO, None)

    assert session.committed
    assert session.lock_type == GrantedLockType.RO
    assert session.is_ready()


def test_client_session_commit_marks_committed_and_closes_transport(monkeypatch):
    closed = False

    monkeypatch.setattr(_GMSRPCTransport, "connect", lambda self: None)
    monkeypatch.setattr(
        _GMSRPCTransport,
        "handshake",
        lambda self, lock_type, timeout_ms: HandshakeResponse(
            success=True,
            committed=False,
            granted_lock_type=GrantedLockType.RW,
        ),
    )
    monkeypatch.setattr(
        _GMSRPCTransport,
        "request",
        lambda self, request, response_type: CommitResponse(success=True),
    )

    def fake_close(self) -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(_GMSRPCTransport, "close", fake_close)

    session = _GMSClientSession("/tmp/gms-test.sock", RequestedLockType.RW, None)
    assert not session.committed

    assert session.commit()
    assert session.committed
    assert closed


def test_memory_manager_connect_rejects_double_connect(monkeypatch):
    manager = object.__new__(GMSClientMemoryManager)
    manager.socket_path = "/tmp/gms-test.sock"
    manager.device = 0
    manager._client = object()
    manager._granted_lock_type = GrantedLockType.RO
    manager._mappings = {}
    manager._inverse_mapping = {}
    manager._unmapped = False
    manager._va_preserved = False
    manager._last_memory_layout_hash = ""
    manager.granularity = 4096

    with pytest.raises(RuntimeError, match="already connected"):
        manager.connect(RequestedLockType.RO)


def test_memory_manager_disconnect_rejects_live_mappings():
    closed = False

    class _DummySession:
        def close(self_inner) -> None:
            nonlocal closed
            closed = True

    manager = object.__new__(GMSClientMemoryManager)
    manager.socket_path = "/tmp/gms-test.sock"
    manager.device = 0
    manager._client = _DummySession()
    manager._granted_lock_type = GrantedLockType.RO
    manager._mappings = {
        0x1000: LocalMapping(
            allocation_id="alloc-1",
            va=0x1000,
            size=4096,
            aligned_size=4096,
            handle=1234,
            tag="weights",
        )
    }
    manager._inverse_mapping = {"alloc-1": 0x1000}
    manager._unmapped = False
    manager._va_preserved = False
    manager._last_memory_layout_hash = ""
    manager.granularity = 4096

    with pytest.raises(RuntimeError, match="unmapped first"):
        manager.disconnect()

    assert not closed


def test_reallocate_all_handles_rolls_back_partial_failure():
    class _DummySession:
        def __init__(self) -> None:
            self.allocations = [
                ("alloc-new-1", 4096),
                RuntimeError("allocate failed"),
            ]
            self.freed: list[str] = []

        def allocate(self, size: int, tag: str):
            result = self.allocations.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        def free(self, allocation_id: str) -> bool:
            self.freed.append(allocation_id)
            return True

    manager = object.__new__(GMSClientMemoryManager)
    manager.socket_path = "/tmp/gms-test.sock"
    manager.device = 0
    manager._client = _DummySession()
    manager._granted_lock_type = GrantedLockType.RW
    manager._mappings = {
        0x1000: LocalMapping(
            allocation_id="alloc-old-1",
            va=0x1000,
            size=4096,
            aligned_size=4096,
            handle=0,
            tag="kv_cache",
        ),
        0x2000: LocalMapping(
            allocation_id="alloc-old-2",
            va=0x2000,
            size=4096,
            aligned_size=4096,
            handle=0,
            tag="kv_cache",
        ),
    }
    manager._inverse_mapping = {
        "alloc-old-1": 0x1000,
        "alloc-old-2": 0x2000,
    }
    manager._unmapped = True
    manager._va_preserved = True
    manager._last_memory_layout_hash = ""
    manager.granularity = 4096

    with pytest.raises(RuntimeError, match="allocate failed"):
        manager.reallocate_all_handles(tag="kv_cache")

    assert manager._client.freed == ["alloc-new-1"]
    assert manager._mappings[0x1000].allocation_id == "alloc-old-1"
    assert manager._mappings[0x2000].allocation_id == "alloc-old-2"
    assert manager._inverse_mapping == {
        "alloc-old-1": 0x1000,
        "alloc-old-2": 0x2000,
    }


def test_reallocate_all_handles_disconnects_on_rollback_failure():
    class _DummySession:
        def __init__(self) -> None:
            self.allocations = [
                ("alloc-new-1", 4096),
                RuntimeError("allocate failed"),
            ]
            self.closed = False

        def allocate(self, size: int, tag: str):
            result = self.allocations.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        def free(self, allocation_id: str) -> bool:
            return False

        def close(self) -> None:
            self.closed = True

    session = _DummySession()
    manager = object.__new__(GMSClientMemoryManager)
    manager.socket_path = "/tmp/gms-test.sock"
    manager.device = 0
    manager._client = session
    manager._granted_lock_type = GrantedLockType.RW
    manager._mappings = {
        0x1000: LocalMapping(
            allocation_id="alloc-old-1",
            va=0x1000,
            size=4096,
            aligned_size=4096,
            handle=0,
            tag="kv_cache",
        ),
        0x2000: LocalMapping(
            allocation_id="alloc-old-2",
            va=0x2000,
            size=4096,
            aligned_size=4096,
            handle=0,
            tag="kv_cache",
        ),
    }
    manager._inverse_mapping = {
        "alloc-old-1": 0x1000,
        "alloc-old-2": 0x2000,
    }
    manager._unmapped = True
    manager._va_preserved = True
    manager._last_memory_layout_hash = ""
    manager.granularity = 4096

    with pytest.raises(RuntimeError, match="session closed"):
        manager.reallocate_all_handles(tag="kv_cache")

    assert session.closed
    assert manager._client is None
    assert manager._granted_lock_type is None
    assert manager._mappings[0x1000].allocation_id == "alloc-old-1"
    assert manager._mappings[0x2000].allocation_id == "alloc-old-2"


def test_transport_failure_closes_socket_and_marks_disconnected(monkeypatch):
    class _DummySocket:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    transport = _GMSRPCTransport("/tmp/gms-test.sock")
    transport._socket = _DummySocket()

    monkeypatch.setattr(
        "gpu_memory_service.client.rpc.send_message_sync",
        lambda sock, request: None,
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.rpc.recv_message_sync",
        lambda sock, buffer: (_ for _ in ()).throw(BrokenPipeError("boom")),
    )

    with pytest.raises(ConnectionError, match="failed: boom"):
        transport.request(CommitResponse(success=True), HandshakeResponse)

    assert not transport.is_connected
    assert transport._socket is None


def test_request_with_fd_closes_fd_on_unexpected_response_type(monkeypatch):
    transport = _GMSRPCTransport("/tmp/gms-test.sock")
    closed_fds: list[int] = []

    monkeypatch.setattr(
        transport,
        "_send_recv",
        lambda request, error_prefix=None: (CommitResponse(success=True), 37),
    )
    monkeypatch.setattr("gpu_memory_service.client.rpc.os.close", closed_fds.append)

    with pytest.raises(RuntimeError, match="unexpected response type"):
        transport.request_with_fd(
            CommitResponse(success=True),
            HandshakeResponse,
        )

    assert closed_fds == [37]


def test_request_closes_fd_on_error_response(monkeypatch):
    class _DummySocket:
        def close(self) -> None:
            return None

    transport = _GMSRPCTransport("/tmp/gms-test.sock")
    transport._socket = _DummySocket()
    closed_fds: list[int] = []

    monkeypatch.setattr(
        "gpu_memory_service.client.rpc.send_message_sync",
        lambda sock, request: None,
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.rpc.recv_message_sync",
        lambda sock, buffer: (ErrorResponse(error="boom"), 41, bytearray()),
    )
    monkeypatch.setattr("gpu_memory_service.client.rpc.os.close", closed_fds.append)

    with pytest.raises(RuntimeError, match="error: boom"):
        transport.request(CommitResponse(success=True), HandshakeResponse)

    assert closed_fds == [41]


def test_request_closes_fd_on_unexpected_success_fd(monkeypatch):
    transport = _GMSRPCTransport("/tmp/gms-test.sock")
    closed_fds: list[int] = []

    monkeypatch.setattr(
        transport,
        "request_with_fd",
        lambda request, response_type: (CommitResponse(success=True), 43),
    )
    monkeypatch.setattr("gpu_memory_service.client.rpc.os.close", closed_fds.append)

    with pytest.raises(RuntimeError, match="unexpected FD"):
        transport.request(CommitResponse(success=True), CommitResponse)

    assert closed_fds == [43]


def test_recv_message_sync_closes_fd_on_decode_failure(monkeypatch):
    closed_fds: list[int] = []

    monkeypatch.setattr(
        wire.socket,
        "recv_fds",
        lambda sock, size, maxfds: (b"\x00\x00\x00\x01x", [53], 0, None),
    )
    monkeypatch.setattr(
        wire,
        "decode_message",
        lambda payload: (_ for _ in ()).throw(ValueError("bad frame")),
    )
    monkeypatch.setattr(
        "gpu_memory_service.common.protocol.wire.os.close",
        closed_fds.append,
    )

    with pytest.raises(ValueError, match="bad frame"):
        wire.recv_message_sync(object(), bytearray())

    assert closed_fds == [53]


def test_client_session_commit_tolerates_close_failure_after_success(monkeypatch):
    monkeypatch.setattr(_GMSRPCTransport, "connect", lambda self: None)
    monkeypatch.setattr(
        _GMSRPCTransport,
        "handshake",
        lambda self, lock_type, timeout_ms: HandshakeResponse(
            success=True,
            committed=False,
            granted_lock_type=GrantedLockType.RW,
        ),
    )
    monkeypatch.setattr(
        _GMSRPCTransport,
        "request",
        lambda self, request, response_type: CommitResponse(success=True),
    )
    monkeypatch.setattr(
        _GMSRPCTransport,
        "close",
        lambda self: (_ for _ in ()).throw(ConnectionError("close failed")),
    )

    session = _GMSClientSession("/tmp/gms-test.sock", RequestedLockType.RW, None)

    assert session.commit()
    assert session.committed
