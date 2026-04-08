# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from gpu_memory_service.client.rpc import _GMSRPCTransport
from gpu_memory_service.common.protocol import wire
from gpu_memory_service.common.protocol.messages import (
    CommitResponse,
    ErrorResponse,
    HandshakeResponse,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


class _DummySocket:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_transport_failure_closes_socket_and_marks_disconnected(monkeypatch):
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
