# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed MessagePack framing and SCM_RIGHTS transfer for GMS V1."""

from __future__ import annotations

import os
import socket
import struct
from typing import TypeAlias

import msgspec
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType

from .errors import GMSError

MAX_FRAME = 1 << 20
_INT_SIZE = struct.calcsize("i")
_ANCILLARY_SIZE = socket.CMSG_SPACE(16 * _INT_SIZE)


class HandshakeRequest(msgspec.Struct, tag="handshake_request"):
    lock_type: RequestedLockType
    expected_identity: tuple[str, str] | None = None


class HandshakeResponse(msgspec.Struct, tag="handshake_response"):
    lock_type: GrantedLockType
    server_nonce: str
    gpu_uuid: str


class AllocateRequest(msgspec.Struct, tag="allocate_request"):
    allocation_id: str
    aligned_size: int


class ExportRequest(msgspec.Struct, tag="export_request"):
    allocation_id: str


class FreeRequest(msgspec.Struct, tag="free_request"):
    allocation_id: str


class CommitRequest(msgspec.Struct, tag="commit_request"):
    pass


class SuccessResponse(msgspec.Struct, tag="success_response"):
    pass


class ExportResponse(msgspec.Struct, tag="export_response"):
    pass


class ErrorResponse(msgspec.Struct, tag="error_response"):
    message: str
    out_of_memory: bool = False


Request: TypeAlias = AllocateRequest | ExportRequest | FreeRequest | CommitRequest
Message: TypeAlias = (
    HandshakeRequest
    | HandshakeResponse
    | Request
    | (SuccessResponse | ExportResponse | ErrorResponse)
)
REQUEST_TYPES = (AllocateRequest, ExportRequest, FreeRequest, CommitRequest)

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(Message)


def send_message(sock: socket.socket, message: Message, fd: int = -1) -> None:
    payload = _encoder.encode(message)
    if len(payload) > MAX_FRAME:
        raise GMSError("GMS RPC frame is too large")
    frame = struct.pack("!I", len(payload)) + payload
    if fd < 0:
        sock.sendall(frame)
        return
    sent = sock.sendmsg(
        [frame],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))],
    )
    if sent <= 0:
        raise ConnectionError("GMS RPC sendmsg made no progress")
    if sent < len(frame):
        sock.sendall(frame[sent:])


def receive_message(sock: socket.socket) -> tuple[Message, int]:
    received_fds: list[int] = []

    def read_exact(size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk, ancillary, flags, _ = sock.recvmsg(size - len(data), _ANCILLARY_SIZE)
            for level, kind, raw in ancillary:
                if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                    continue
                if len(raw) % _INT_SIZE:
                    raise GMSError("malformed GMS RPC file descriptor data")
                count = len(raw) // _INT_SIZE
                received_fds.extend(
                    struct.unpack(f"{count}i", raw[: count * _INT_SIZE])
                )
            if flags & socket.MSG_CTRUNC:
                raise GMSError("GMS RPC ancillary data was truncated")
            if not chunk:
                raise EOFError
            data.extend(chunk)
        return bytes(data)

    try:
        (length,) = struct.unpack("!I", read_exact(4))
        if length > MAX_FRAME:
            raise GMSError("GMS RPC frame is too large")
        try:
            message = _decoder.decode(read_exact(length))
        except msgspec.DecodeError as exc:
            raise GMSError("invalid GMS RPC message") from exc
        if len(received_fds) > 1:
            raise GMSError("GMS RPC received multiple file descriptors")
        return message, received_fds.pop() if received_fds else -1
    except Exception:
        for fd in received_fds:
            os.close(fd)
        raise
