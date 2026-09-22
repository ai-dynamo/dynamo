# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded length-prefixed JSON framing for the KV control socket."""

from __future__ import annotations

import asyncio
import json
import socket
import struct

from gpu_memory_service.common.protocol.framing import (
    FrameProtocolError,
    truncated_frame,
    validate_frame_length,
)

_HEADER_SIZE = 4


def encode_frame(message: dict) -> bytes:
    body = json.dumps(message).encode("utf-8")
    validate_frame_length(len(body))
    return struct.pack("<I", len(body)) + body


async def _read_exactly(reader, size: int, part: str) -> bytes:
    try:
        return await reader.readexactly(size)
    except asyncio.IncompleteReadError as exc:
        raise truncated_frame(part, size, len(exc.partial)) from exc


async def read_frame(reader, *, allow_eof: bool = False) -> dict | None:
    try:
        header = await reader.readexactly(_HEADER_SIZE)
    except asyncio.IncompleteReadError as exc:
        if allow_eof and not exc.partial:
            return None
        raise truncated_frame("header", _HEADER_SIZE, len(exc.partial)) from exc
    length = struct.unpack("<I", header)[0]
    validate_frame_length(length)
    body = await _read_exactly(reader, length, "body")
    return json.loads(body.decode("utf-8"))


async def write_frame(writer, message: dict) -> None:
    writer.write(encode_frame(message))
    await writer.drain()


def recv_frame(sock: socket.socket) -> dict:
    def recv_exact(size: int, part: str) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(min(size - len(data), 64 << 10))
            if not chunk:
                raise truncated_frame(part, size, len(data))
            data.extend(chunk)
        return bytes(data)

    length = struct.unpack("<I", recv_exact(_HEADER_SIZE, "header"))[0]
    validate_frame_length(length)
    body = recv_exact(length, "body")
    return json.loads(body.decode("utf-8"))


__all__ = [
    "FrameProtocolError",
    "encode_frame",
    "read_frame",
    "recv_frame",
    "write_frame",
]
