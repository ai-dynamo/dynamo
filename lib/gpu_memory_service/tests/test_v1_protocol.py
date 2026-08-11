# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import socket
import struct

import msgspec
import pytest
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.protocol import (
    Message,
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
