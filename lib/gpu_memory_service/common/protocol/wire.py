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
MAX_FDS_PER_MESSAGE = 128


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
    fds = [fd] if fd >= 0 else []
    await send_message_with_fds(writer, msg, fds)


async def send_message_with_fds(
    writer,
    msg: Message,
    fds: list[int] | tuple[int, ...],
) -> None:
    """Send a length-prefixed message with optional FDs via SCM_RIGHTS."""
    frame = _frame_message(msg)
    if not fds:
        writer.write(frame)
        await writer.drain()
        return
    if len(fds) > MAX_FDS_PER_MESSAGE:
        raise ValueError(
            f"too many FDs for one message: {len(fds)} > {MAX_FDS_PER_MESSAGE}"
        )

    transport_sock = writer.get_extra_info("socket")
    if transport_sock is None:
        raise RuntimeError("Cannot get socket from transport for FD passing")

    def do_send_fds():
        raw_fd = transport_sock.fileno()
        dup_fd = os.dup(raw_fd)
        sock = None
        try:
            sock = socket.socket(fileno=dup_fd)
            sock.setblocking(True)
            socket.send_fds(sock, [frame], list(fds))
        except Exception:
            if sock is None:
                os.close(dup_fd)
            raise
        finally:
            if sock is not None:
                sock.close()

    await asyncio.get_running_loop().run_in_executor(None, do_send_fds)


async def recv_message(
    reader, recv_buffer: Optional[bytearray] = None, raw_sock=None
) -> Tuple[Optional[Message], int, bytearray]:
    """Receive a length-prefixed message with optional FD.

    Returns (message, fd, remaining_buffer). fd is -1 if none sent.
    Additional FDs, if any, are closed for compatibility.
    """
    msg, fds, remaining = await recv_message_with_fds(reader, recv_buffer, raw_sock)
    for extra_fd in fds[1:]:
        os.close(extra_fd)
    return msg, (fds[0] if fds else -1), remaining


async def recv_message_with_fds(
    reader, recv_buffer: Optional[bytearray] = None, raw_sock=None
) -> Tuple[Optional[Message], list[int], bytearray]:
    """Receive a length-prefixed message with optional FDs."""
    if recv_buffer is None:
        recv_buffer = bytearray()

    msg, remaining, _ = _try_extract_message(recv_buffer)
    if msg is not None:
        return msg, [], remaining

    loop = asyncio.get_running_loop()
    fds: list[int] = []
    if raw_sock is not None:
        raw_msg, got_fds, _flags, _addr = await loop.run_in_executor(
            None, lambda: socket.recv_fds(raw_sock, 65536, MAX_FDS_PER_MESSAGE)
        )
        if not raw_msg:
            for fd in got_fds:
                os.close(fd)
            raise ConnectionResetError("Connection closed")
        recv_buffer.extend(raw_msg)
        fds = list(got_fds)
    else:
        chunk = await reader.read(65536)
        if not chunk:
            raise ConnectionResetError("Connection closed")
        recv_buffer.extend(chunk)

    try:
        msg, remaining, bytes_needed = _try_extract_message(recv_buffer)
        while msg is None and bytes_needed > 0:
            if raw_sock is not None:
                chunk = await loop.run_in_executor(
                    None, lambda n=bytes_needed: raw_sock.recv(n)
                )
            else:
                chunk = await reader.read(bytes_needed)
            if not chunk:
                raise ConnectionResetError("Connection closed")
            remaining.extend(chunk)
            msg, remaining, bytes_needed = _try_extract_message(remaining)
        return msg, fds, remaining
    except Exception:
        for fd in fds:
            os.close(fd)
        raise


# ==================== Sync (for client) ====================


def send_message_sync(sock, msg: Message, fd: int = -1) -> None:
    """Send a length-prefixed message with optional FD via SCM_RIGHTS."""
    fds = [fd] if fd >= 0 else []
    send_message_sync_with_fds(sock, msg, fds)


def send_message_sync_with_fds(
    sock,
    msg: Message,
    fds: list[int] | tuple[int, ...],
) -> None:
    """Send a length-prefixed message with optional FDs via SCM_RIGHTS."""
    frame = _frame_message(msg)
    if fds:
        if len(fds) > MAX_FDS_PER_MESSAGE:
            raise ValueError(
                f"too many FDs for one message: {len(fds)} > {MAX_FDS_PER_MESSAGE}"
            )
        socket.send_fds(sock, [frame], list(fds))
    else:
        sock.sendall(frame)


def recv_message_sync(
    sock, recv_buffer: Optional[bytearray] = None
) -> Tuple[Optional[Message], int, bytearray]:
    """Receive a length-prefixed message with optional FD.

    Returns (message, fd, remaining_buffer). fd is -1 if none sent.
    Additional FDs, if any, are closed for compatibility.
    """
    msg, fds, remaining = recv_message_sync_with_fds(sock, recv_buffer)
    for extra_fd in fds[1:]:
        os.close(extra_fd)
    return msg, (fds[0] if fds else -1), remaining


def recv_message_sync_with_fds(
    sock, recv_buffer: Optional[bytearray] = None
) -> Tuple[Optional[Message], list[int], bytearray]:
    """Receive a length-prefixed message with optional FDs."""
    if recv_buffer is None:
        recv_buffer = bytearray()

    msg, remaining, _ = _try_extract_message(recv_buffer)
    if msg is not None:
        return msg, [], remaining

    raw_msg, got_fds, _flags, _addr = socket.recv_fds(sock, 65536, MAX_FDS_PER_MESSAGE)
    if not raw_msg:
        for fd in got_fds:
            os.close(fd)
        raise ConnectionResetError("Connection closed")
    recv_buffer.extend(raw_msg)
    fds = list(got_fds)

    try:
        msg, remaining, bytes_needed = _try_extract_message(recv_buffer)
        while msg is None and bytes_needed > 0:
            chunk = sock.recv(bytes_needed)
            if not chunk:
                raise ConnectionResetError("Connection closed")
            remaining.extend(chunk)
            msg, remaining, bytes_needed = _try_extract_message(remaining)
        return msg, fds, remaining
    except Exception:
        for fd in fds:
            os.close(fd)
        raise
