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
]


def _patch_handshake(
    monkeypatch,
    *,
    response: HandshakeResponse | None = None,
    error: Exception | None = None,
) -> dict[str, bool]:
    closed = {"value": False}

    monkeypatch.setattr(_GMSRPCTransport, "connect", lambda self: None)
    if error is None:
        monkeypatch.setattr(
            _GMSRPCTransport,
            "handshake",
            lambda self, lock_type, timeout_ms: response,
        )
    else:
        monkeypatch.setattr(
            _GMSRPCTransport,
            "handshake",
            lambda self, lock_type, timeout_ms: (_ for _ in ()).throw(error),
        )
    monkeypatch.setattr(
        _GMSRPCTransport,
        "close",
        lambda self: closed.__setitem__("value", True),
    )
    return closed


def test_client_session_timeout_closes_transport(monkeypatch):
    closed = _patch_handshake(
        monkeypatch,
        response=HandshakeResponse(success=False, committed=False),
    )

    with pytest.raises(TimeoutError, match="Timeout waiting for lock"):
        _GMSClientSession("/tmp/gms-test.sock", RequestedLockType.RO, 1000)

    assert closed["value"]


def test_client_session_handshake_failure_closes_transport(monkeypatch):
    closed = _patch_handshake(monkeypatch, error=RuntimeError("handshake failed"))

    with pytest.raises(RuntimeError, match="handshake failed"):
        _GMSClientSession("/tmp/gms-test.sock", RequestedLockType.RO, 1000)

    assert closed["value"]


def test_client_session_records_granted_lock_and_committed(monkeypatch):
    _patch_handshake(
        monkeypatch,
        response=HandshakeResponse(
            success=True,
            committed=True,
            granted_lock_type=GrantedLockType.RO,
        ),
    )

    session = _GMSClientSession("/tmp/gms-test.sock", RequestedLockType.RW_OR_RO, None)

    assert session.committed
    assert session.lock_type == GrantedLockType.RO
    assert session.is_ready()


def test_client_session_requires_granted_lock_type(monkeypatch):
    closed = _patch_handshake(
        monkeypatch,
        response=HandshakeResponse(
            success=True,
            committed=False,
            granted_lock_type=None,
        ),
    )

    with pytest.raises(RuntimeError, match="granted_lock_type"):
        _GMSClientSession("/tmp/gms-test.sock", RequestedLockType.RW_OR_RO, None)

    assert closed["value"]


def test_client_session_commit_marks_committed_and_closes_transport(monkeypatch):
    closed = _patch_handshake(
        monkeypatch,
        response=HandshakeResponse(
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

    session = _GMSClientSession("/tmp/gms-test.sock", RequestedLockType.RW, None)
    assert not session.committed

    assert session.commit()
    assert session.committed
    assert closed["value"]


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
