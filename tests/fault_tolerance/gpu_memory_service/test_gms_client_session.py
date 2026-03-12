# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from gpu_memory_service.client.rpc import _GMSRPCTransport
from gpu_memory_service.client.session import _GMSClientSession
from gpu_memory_service.common.protocol.messages import (
    CommitResponse,
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
