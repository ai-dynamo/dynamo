# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import socket
import struct

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

import msgspec
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.protocol import (
    ERROR_CLAIM_CONFLICT,
    ErrorResponse,
    Message,
    PersistentPoolErrorResponse,
    SuccessResponse,
    receive_message,
    send_message,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
    pytest.mark.timeout(10),
]


@pytest.mark.parametrize("out_of_memory", [False, True])
def test_ordinary_errors_remain_decodable_by_legacy_clients(out_of_memory):
    class LegacyErrorResponse(
        msgspec.Struct, tag="error_response", forbid_unknown_fields=True
    ):
        message: str
        out_of_memory: bool = False

    payload = msgspec.msgpack.encode(ErrorResponse("failure", out_of_memory))
    decoded = msgspec.msgpack.decode(payload, type=LegacyErrorResponse)
    assert decoded.message == "failure"
    assert decoded.out_of_memory is out_of_memory
    # Also retain compatibility in the opposite direction.
    payload = msgspec.msgpack.encode(LegacyErrorResponse("old", out_of_memory))
    assert msgspec.msgpack.decode(payload, type=Message) == ErrorResponse(
        "old", out_of_memory
    )


def test_persistent_errors_use_separate_wire_tag():
    error = PersistentPoolErrorResponse("busy", ERROR_CLAIM_CONFLICT)
    payload = msgspec.msgpack.encode(error)
    assert msgspec.msgpack.decode(payload)["type"] == "persistent_pool_error_response"
    assert msgspec.msgpack.decode(payload, type=Message) == error


def test_received_fd_is_cloexec_and_unexpected_fd_is_closed() -> None:
    sender, receiver = socket.socketpair()
    read_fd, write_fd = os.pipe()
    try:
        send_message(sender, SuccessResponse(), read_fd)
        message, received_fd = receive_message(receiver)
        assert isinstance(message, SuccessResponse)
        assert not os.get_inheritable(received_fd)

        with pytest.raises(RuntimeError, match="unexpected FD"):
            _GMSClientSession._decode(
                "test",
                message,
                received_fd,
                SuccessResponse,
            )
        with pytest.raises(OSError):
            os.fstat(received_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)
        sender.close()
        receiver.close()


def test_protocol_rejects_unknown_fields() -> None:
    payload = msgspec.msgpack.encode(
        {
            "type": "success_response",
            "unexpected": True,
        }
    )
    with pytest.raises(msgspec.ValidationError):
        msgspec.msgpack.decode(payload, type=Message)


@pytest.mark.parametrize("payload", [b"\x00\x00\x00\x02\xc1", b"\x00\x00\x00\x02"])
def test_receive_rejects_malformed_or_truncated_frames(payload: bytes) -> None:
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(payload)
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises((EOFError, RuntimeError)):
            receive_message(receiver)
    finally:
        sender.close()
        receiver.close()


def test_receive_rejects_multiple_fds() -> None:
    sender, receiver = socket.socketpair()
    first_read, first_write = os.pipe()
    second_read, second_write = os.pipe()
    payload = msgspec.msgpack.encode(SuccessResponse())
    frame = struct.pack("!I", len(payload)) + payload
    try:
        sender.sendmsg(
            [frame],
            [
                (
                    socket.SOL_SOCKET,
                    socket.SCM_RIGHTS,
                    struct.pack("2i", first_read, second_read),
                )
            ],
        )
        with pytest.raises(RuntimeError, match="multiple file descriptors"):
            receive_message(receiver)
    finally:
        for fd in (first_read, first_write, second_read, second_write):
            os.close(fd)
        sender.close()
        receiver.close()
