# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded JSON framing and SCM_RIGHTS transfer for GMS V1."""

from __future__ import annotations

import json
import os
import socket
import struct

from ..errors import GMSError

MAX_FRAME = 1 << 20
_INT_SIZE = struct.calcsize("i")
_ANCILLARY_SIZE = socket.CMSG_SPACE(16 * _INT_SIZE)


def send_message(sock: socket.socket, value: object, fd: int = -1) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode()
    if len(payload) > MAX_FRAME:
        raise GMSError("V1 RPC frame is too large")
    frame = struct.pack("!I", len(payload)) + payload
    if fd < 0:
        sock.sendall(frame)
        return
    sent = sock.sendmsg(
        [frame],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))],
    )
    if sent <= 0:
        raise ConnectionError("V1 RPC sendmsg made no progress")
    if sent < len(frame):
        sock.sendall(frame[sent:])


def receive_message(sock: socket.socket) -> tuple[object, int]:
    received_fds: list[int] = []

    def read_exact(size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk, ancillary, flags, _ = sock.recvmsg(size - len(data), _ANCILLARY_SIZE)
            for level, kind, raw in ancillary:
                if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                    continue
                if len(raw) % _INT_SIZE:
                    raise GMSError("malformed V1 RPC file descriptor data")
                count = len(raw) // _INT_SIZE
                received_fds.extend(
                    struct.unpack(f"{count}i", raw[: count * _INT_SIZE])
                )
            if flags & socket.MSG_CTRUNC:
                raise GMSError("V1 RPC ancillary data was truncated")
            if not chunk:
                raise EOFError
            data.extend(chunk)
        return bytes(data)

    try:
        (length,) = struct.unpack("!I", read_exact(4))
        if length > MAX_FRAME:
            raise GMSError("V1 RPC frame is too large")
        value = json.loads(read_exact(length).decode())
        if len(received_fds) > 1:
            raise GMSError("V1 RPC received multiple file descriptors")
        return value, received_fds.pop() if received_fds else -1
    except Exception:
        for fd in received_fds:
            os.close(fd)
        raise
