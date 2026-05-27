# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import socket

from gpu_memory_service.common.protocol.messages import GetAllocationStateResponse
from gpu_memory_service.common.protocol.wire import (
    recv_message_sync_with_fds,
    send_message_sync_with_fds,
)


def _close_all(fds: list[int]) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def test_sync_wire_roundtrips_multiple_fds() -> None:
    left, right = socket.socketpair()
    sent_read_fds: list[int] = []
    sent_write_fds: list[int] = []
    received_fds: list[int] = []
    try:
        for _ in range(3):
            read_fd, write_fd = os.pipe()
            sent_read_fds.append(read_fd)
            sent_write_fds.append(write_fd)

        send_message_sync_with_fds(
            left,
            GetAllocationStateResponse(allocation_count=3),
            sent_read_fds,
        )
        message, received_fds, remaining = recv_message_sync_with_fds(right)

        assert isinstance(message, GetAllocationStateResponse)
        assert message.allocation_count == 3
        assert len(received_fds) == len(sent_read_fds)
        assert remaining == bytearray()
        for fd in received_fds:
            os.fstat(fd)
    finally:
        _close_all(received_fds)
        _close_all(sent_read_fds)
        _close_all(sent_write_fds)
        left.close()
        right.close()
