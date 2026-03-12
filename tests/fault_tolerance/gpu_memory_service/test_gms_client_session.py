# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from gpu_memory_service.client.rpc import _GMSRPCTransport
from gpu_memory_service.client.session import _GMSClientSession
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
