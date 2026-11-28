# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire protocol for length-prefixed messages with optional FD passing."""

import array
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


async def _wait_writable(writer, fd: int) -> None:
    """Wait until fd is writable; raise if the transport closed meanwhile."""
    loop = asyncio.get_running_loop()
    writable = loop.create_future()

    def ready() -> None:
        if not writable.done():
            writable.set_result(None)

    loop.add_writer(fd, ready)
    try:
        await writable
    finally:
        loop.remove_writer(fd)
    if writer.transport.is_closing():
        raise ConnectionResetError("Transport closed while sending an FD frame")


async def send_message(writer, msg: Message, fd: int = -1) -> None:
    """Send a length-prefixed message with optional FD via SCM_RIGHTS.

    An FD frame bypasses the transport, so only one task may send on a writer
    at a time.
    """
    frame = _frame_message(msg)

    if fd >= 0:
        transport = writer.transport
        if transport.is_closing():
            raise ConnectionResetError("Transport is closing")
        transport_sock = writer.get_extra_info("socket")
        if transport_sock is None:
            raise RuntimeError("Cannot get socket from transport for FD passing")

        view = memoryview(frame)
        sent = 0
        # The loop cannot watch the transport's own descriptor for a second
        # writer, so wait and send on a duplicate. MSG_DONTWAIT keeps every
        # send non-blocking without touching the shared file status flags.
        with transport_sock.dup() as sock:
            try:
                # Bytes the transport still buffers from an earlier frame must
                # reach the socket first, or the two frames interleave.
                while transport.get_write_buffer_size():
                    await _wait_writable(writer, sock.fileno())
                while sent < len(frame):
                    try:
                        if sent == 0:
                            # SCM_RIGHTS rides the first bytes only.
                            sent = sock.sendmsg(
                                [view],
                                [
                                    (
                                        socket.SOL_SOCKET,
                                        socket.SCM_RIGHTS,
                                        array.array("i", [fd]),
                                    )
                                ],
                                socket.MSG_DONTWAIT,
                            )
                        else:
                            sent += sock.send(view[sent:], socket.MSG_DONTWAIT)
                    except BlockingIOError:
                        await _wait_writable(writer, sock.fileno())
            finally:
                if 0 < sent < len(frame):
                    # A partial frame cannot be rolled back.
                    transport.abort()
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
        for extra_fd in fds[1:]:
            os.close(extra_fd)
        if not raw_msg:
            if fds:
                os.close(fds[0])
            raise ConnectionResetError("Connection closed")
        recv_buffer.extend(raw_msg)
        fd = fds[0] if fds else -1
    else:
        chunk = await reader.read(65536)
        if not chunk:
            raise ConnectionResetError("Connection closed")
        recv_buffer.extend(chunk)

    # Try to extract message, read more if needed
    try:
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
    except Exception:
        if fd >= 0:
            os.close(fd)
        raise


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
    for extra_fd in fds[1:]:
        os.close(extra_fd)
    if not raw_msg:
        if fds:
            os.close(fds[0])
        raise ConnectionResetError("Connection closed")
    recv_buffer.extend(raw_msg)
    fd = fds[0] if fds else -1

    # Try to extract message, read more if needed
    try:
        msg, remaining, bytes_needed = _try_extract_message(recv_buffer)
        while msg is None and bytes_needed > 0:
            chunk = sock.recv(bytes_needed)
            if not chunk:
                raise ConnectionResetError("Connection closed")
            remaining.extend(chunk)
            msg, remaining, bytes_needed = _try_extract_message(remaining)
        return msg, fd, remaining
    except Exception:
        if fd >= 0:
            os.close(fd)
        raise
