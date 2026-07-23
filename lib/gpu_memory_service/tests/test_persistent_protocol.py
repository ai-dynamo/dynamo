# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the persistent allocation wire contracts."""

import asyncio

import pytest
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.protocol.messages import (
    ClaimPersistentAllocationRequest,
    ClaimPersistentAllocationResponse,
    ErrorResponse,
    ExportPersistentAllocationRequest,
    ExportPersistentAllocationResponse,
    HandshakeRequest,
    ListPersistentAllocationsRequest,
    ListPersistentAllocationsResponse,
    PersistentAllocationInfo,
    ReleasePersistentAllocationRequest,
    ReleasePersistentAllocationResponse,
    decode_message,
    encode_message,
)
from gpu_memory_service.server import rpc as rpc_module
from gpu_memory_service.server.rpc import GMSRPCServer
from gpu_memory_service.server.session import GMSSessionManager, OperationNotAllowed


@pytest.mark.parametrize(
    "message",
    [
        HandshakeRequest(lock_type=RequestedLockType.RW_PERSISTENT),
        ClaimPersistentAllocationRequest(
            engine_id="engine-0",
            tag="kv",
            size=4096,
            shared=True,
        ),
        ClaimPersistentAllocationResponse(
            allocation_id="allocation-0",
            size=4096,
            aligned_size=65536,
            reattached=False,
        ),
        ReleasePersistentAllocationRequest(engine_id="engine-0", tag="kv"),
        ReleasePersistentAllocationResponse(released=True),
        ExportPersistentAllocationRequest(engine_id="engine-0", tag="kv"),
        ExportPersistentAllocationResponse(
            allocation_id="allocation-0",
            size=4096,
            aligned_size=65536,
        ),
        ListPersistentAllocationsRequest(
            engine_id="engine-0",
            include_unclaimed=True,
        ),
        ListPersistentAllocationsResponse(
            allocations=[
                PersistentAllocationInfo(
                    allocation_id="allocation-0",
                    engine_id="engine-0",
                    tag="kv",
                    size=4096,
                    aligned_size=65536,
                    claimed=True,
                )
            ]
        ),
        PersistentAllocationInfo(
            allocation_id="allocation-0",
            engine_id="engine-0",
            tag="kv",
            size=4096,
            aligned_size=65536,
            claimed=False,
        ),
    ],
)
def test_persistent_messages_round_trip(message):
    decoded = decode_message(encode_message(message))
    assert decoded == message
    assert type(decoded) is type(message)


def test_persistent_lock_request_fails_closed_until_implemented():
    async def exercise():
        sessions = GMSSessionManager()
        with pytest.raises(
            OperationNotAllowed,
            match="persistent allocation sessions are not implemented",
        ):
            await sessions.acquire_lock(
                RequestedLockType.RW_PERSISTENT,
                timeout_ms=0,
                session_id="session-0",
            )

    asyncio.run(exercise())


def test_persistent_lock_handshake_rejects_only_requesting_client(monkeypatch):
    class RejectingGMS:
        committed = False

        async def acquire_lock(self, mode, timeout_ms, session_id):
            raise OperationNotAllowed(
                "persistent allocation sessions are not implemented"
            )

    class Writer:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    responses = []

    async def fake_recv_message(reader, recv_buffer):
        return (
            HandshakeRequest(lock_type=RequestedLockType.RW_PERSISTENT),
            -1,
            recv_buffer,
        )

    async def fake_send_message(writer, response):
        responses.append(response)

    async def exercise():
        server = object.__new__(GMSRPCServer)
        server._gms = RejectingGMS()
        writer = Writer()
        monkeypatch.setattr(rpc_module, "recv_message", fake_recv_message)
        monkeypatch.setattr(rpc_module, "send_message", fake_send_message)

        conn = await server._do_handshake(object(), writer, "session-0")

        assert conn is None
        assert responses == [
            ErrorResponse(
                error="persistent allocation sessions are not implemented",
            )
        ]
        assert writer.closed

    asyncio.run(exercise())
