# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the persistent allocation wire contracts."""

import asyncio

import pytest
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.protocol.messages import (
    ClaimPersistentAllocationRequest,
    ClaimPersistentAllocationResponse,
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
