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

MAX_FRAME = 1 << 20
_INT_SIZE = struct.calcsize("i")
_ANCILLARY_SIZE = socket.CMSG_SPACE(16 * _INT_SIZE)


class HandshakeRequest(
    msgspec.Struct, tag="handshake_request", forbid_unknown_fields=True
):
    lock_type: RequestedLockType
    expected_identity: tuple[str, str] | None = None


class HandshakeResponse(
    msgspec.Struct, tag="handshake_response", forbid_unknown_fields=True
):
    lock_type: GrantedLockType
    server_nonce: str
    gpu_uuid: str


class AllocateRequest(
    msgspec.Struct, tag="allocate_request", forbid_unknown_fields=True
):
    allocation_id: str
    aligned_size: int


class ExportRequest(msgspec.Struct, tag="export_request", forbid_unknown_fields=True):
    allocation_id: str


class FreeRequest(msgspec.Struct, tag="free_request", forbid_unknown_fields=True):
    allocation_id: str


class ListAllocationsRequest(
    msgspec.Struct, tag="list_allocations_request", forbid_unknown_fields=True
):
    pass


class CommitRequest(msgspec.Struct, tag="commit_request", forbid_unknown_fields=True):
    pass


class AbortRequest(msgspec.Struct, tag="abort_request", forbid_unknown_fields=True):
    pass


class PrepareCheckpointRequest(
    msgspec.Struct, tag="prepare_checkpoint_request", forbid_unknown_fields=True
):
    pass


class AbortCheckpointRequest(
    msgspec.Struct, tag="abort_checkpoint_request", forbid_unknown_fields=True
):
    token: str


class CompleteRestoreRequest(
    msgspec.Struct, tag="complete_restore_request", forbid_unknown_fields=True
):
    token: str


class GetCheckpointStateRequest(
    msgspec.Struct, tag="get_checkpoint_state_request", forbid_unknown_fields=True
):
    pass


class AllocationRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    allocation_id: str
    aligned_size: int


class SuccessResponse(
    msgspec.Struct, tag="success_response", forbid_unknown_fields=True
):
    pass


class ExportResponse(msgspec.Struct, tag="export_response", forbid_unknown_fields=True):
    pass


class ListAllocationsResponse(
    msgspec.Struct, tag="list_allocations_response", forbid_unknown_fields=True
):
    allocations: tuple[AllocationRecord, ...]


class CheckpointStateResponse(
    msgspec.Struct, tag="checkpoint_state_response", forbid_unknown_fields=True
):
    state: str
    token: str | None


class ErrorResponse(msgspec.Struct, tag="error_response", forbid_unknown_fields=True):
    message: str
    out_of_memory: bool = False


Request: TypeAlias = (
    AllocateRequest
    | ExportRequest
    | FreeRequest
    | ListAllocationsRequest
    | CommitRequest
    | AbortRequest
)
CheckpointControlRequest: TypeAlias = (
    PrepareCheckpointRequest
    | AbortCheckpointRequest
    | CompleteRestoreRequest
    | GetCheckpointStateRequest
)
Response: TypeAlias = (
    SuccessResponse
    | ExportResponse
    | ListAllocationsResponse
    | CheckpointStateResponse
    | ErrorResponse
)
Message: TypeAlias = (
    HandshakeRequest | HandshakeResponse | Request | CheckpointControlRequest | Response
)
REQUEST_TYPES = (
    AllocateRequest,
    ExportRequest,
    FreeRequest,
    ListAllocationsRequest,
    CommitRequest,
    AbortRequest,
)
CHECKPOINT_CONTROL_TYPES = (
    PrepareCheckpointRequest,
    AbortCheckpointRequest,
    CompleteRestoreRequest,
    GetCheckpointStateRequest,
)

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(Message)


def send_message(sock: socket.socket, message: Message, fd: int = -1) -> None:
    payload = _encoder.encode(message)
    if len(payload) > MAX_FRAME:
        raise RuntimeError("GMS RPC frame is too large")
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
    receive_flags = getattr(socket, "MSG_CMSG_CLOEXEC", 0)

    def read_exact(size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk, ancillary, flags, _ = sock.recvmsg(
                size - len(data),
                _ANCILLARY_SIZE,
                receive_flags,
            )
            for level, kind, raw in ancillary:
                if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                    continue
                if len(raw) % _INT_SIZE:
                    raise RuntimeError("malformed GMS RPC file descriptor data")
                count = len(raw) // _INT_SIZE
                for fd in struct.unpack(f"{count}i", raw[: count * _INT_SIZE]):
                    try:
                        os.set_inheritable(fd, False)
                    except Exception:
                        os.close(fd)
                        raise
                    received_fds.append(fd)
            if flags & socket.MSG_CTRUNC:
                raise RuntimeError("GMS RPC ancillary data was truncated")
            if not chunk:
                raise EOFError
            data.extend(chunk)
        return bytes(data)

    try:
        (length,) = struct.unpack("!I", read_exact(4))
        if length > MAX_FRAME:
            raise RuntimeError("GMS RPC frame is too large")
        try:
            message = _decoder.decode(read_exact(length))
        except msgspec.DecodeError as exc:
            raise RuntimeError("invalid GMS RPC message") from exc
        if len(received_fds) > 1:
            raise RuntimeError("GMS RPC received multiple file descriptors")
        return message, received_fds.pop() if received_fds else -1
    except Exception:
        for fd in received_fds:
            os.close(fd)
        raise
