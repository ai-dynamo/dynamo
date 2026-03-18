# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire protocol for length-prefixed messages with optional FD passing."""

import asyncio
import os
import socket
import struct
from typing import Optional, Tuple

from .messages import Message, decode_message, encode_message

HEADER_SIZE = 4  # 4-byte big-endian length prefix


def _frame_message(msg: Message) -> bytes:
    """Encode and frame a message with length prefix."""
    data = encode_message(msg)
    return struct.pack("!I", len(data)) + data


def _try_extract_message(
    recv_buffer: bytearray,
) -> Tuple[Optional[Message], bytearray, int]:
    """Try to extract a complete message from buffer.

    Returns (message, remaining_buffer, bytes_needed).
    """
    if len(recv_buffer) < HEADER_SIZE:
        return None, recv_buffer, HEADER_SIZE - len(recv_buffer)

    length = struct.unpack("!I", bytes(recv_buffer[:HEADER_SIZE]))[0]
    total_needed = HEADER_SIZE + length

    if len(recv_buffer) < total_needed:
        return None, recv_buffer, total_needed - len(recv_buffer)

    msg_data = bytes(recv_buffer[HEADER_SIZE:total_needed])
    remaining = bytearray(recv_buffer[total_needed:])
    return decode_message(msg_data), remaining, 0


# ==================== Async (for server) ====================


async def send_message(writer, msg: Message, fd: int = -1) -> None:
    """Send a length-prefixed message with optional FD via SCM_RIGHTS."""
    frame = _frame_message(msg)

    if fd >= 0:
        transport_sock = writer.get_extra_info("socket")
        if transport_sock is None:
            raise RuntimeError("Cannot get socket from transport for FD passing")

        def do_send_fd():
            raw_fd = transport_sock.fileno()
            dup_fd = os.dup(raw_fd)
            try:
                sock = socket.socket(fileno=dup_fd)
                try:
                    sock.setblocking(True)
                    socket.send_fds(sock, [frame], [fd])
                finally:
                    sock.detach()
            except Exception:
                os.close(dup_fd)
                raise

        await asyncio.get_running_loop().run_in_executor(None, do_send_fd)
    else:
        writer.write(frame)
        await writer.drain()


async def recv_message(
    reader, recv_buffer: Optional[bytearray] = None, raw_sock=None
) -> Tuple[Optional[Message], int, bytearray]:
    """Receive a length-prefixed message with optional FD.

    Returns (message, fd, remaining_buffer). fd is -1 if none sent.
    """
    if recv_buffer is None:
        recv_buffer = bytearray()

    # Check if complete message already in buffer
    msg, remaining, _ = _try_extract_message(recv_buffer)
    if msg is not None:
        return msg, -1, remaining

    loop = asyncio.get_running_loop()
    fd = -1

    # Receive more data
    if raw_sock is not None:
        raw_msg, fds, _flags, _addr = await loop.run_in_executor(
            None, lambda: socket.recv_fds(raw_sock, 65536, 1)
        )
        if not raw_msg:
            raise ConnectionResetError("Connection closed")
        recv_buffer.extend(raw_msg)
        fd = fds[0] if fds else -1
    else:
        chunk = await reader.read(65536)
        if not chunk:
            raise ConnectionResetError("Connection closed")
        recv_buffer.extend(chunk)

    # Try to extract message, read more if needed
    msg, remaining, bytes_needed = _try_extract_message(recv_buffer)
    while msg is None and bytes_needed > 0:
        if raw_sock is not None:
            # Continue reading from raw socket to avoid buffer inconsistency
            chunk = await loop.run_in_executor(
                None, lambda n=bytes_needed: raw_sock.recv(n)
            )
        else:
            chunk = await reader.read(bytes_needed)
        if not chunk:
            raise ConnectionResetError("Connection closed")
        remaining.extend(chunk)
        msg, remaining, bytes_needed = _try_extract_message(remaining)

    return msg, fd, remaining


# ==================== Sync (for client) ====================


def send_message_sync(sock, msg: Message, fd: int = -1) -> None:
    """Send a length-prefixed message with optional FD via SCM_RIGHTS."""
    frame = _frame_message(msg)
    if fd >= 0:
        socket.send_fds(sock, [frame], [fd])
    else:
        sock.sendall(frame)


def recv_message_sync(
    sock, recv_buffer: Optional[bytearray] = None
) -> Tuple[Optional[Message], int, bytearray]:
    """Receive a length-prefixed message with optional FD.

    Returns (message, fd, remaining_buffer). fd is -1 if none sent.
    """
    if recv_buffer is None:
        recv_buffer = bytearray()

    # Check if complete message already in buffer
    msg, remaining, _ = _try_extract_message(recv_buffer)
    if msg is not None:
        return msg, -1, remaining

    # Receive more data (with potential FD)
    raw_msg, fds, _flags, _addr = socket.recv_fds(sock, 65536, 1)
    if not raw_msg:
        raise ConnectionResetError("Connection closed")
    recv_buffer.extend(raw_msg)
    fd = fds[0] if fds else -1

    # Try to extract message, read more if needed
    msg, remaining, bytes_needed = _try_extract_message(recv_buffer)
    while msg is None and bytes_needed > 0:
        chunk = sock.recv(bytes_needed)
        if not chunk:
            raise ConnectionResetError("Connection closed")
        remaining.extend(chunk)
        msg, remaining, bytes_needed = _try_extract_message(remaining)

    return msg, fd, remaining
