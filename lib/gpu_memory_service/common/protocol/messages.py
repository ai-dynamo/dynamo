# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message types for GPU Memory Service RPC protocol."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import msgspec


class RequestedLockType(str, Enum):
    """Lock type requested by client."""

    RW = "rw"
    RO = "ro"
    RW_OR_RO = "rw_or_ro"


class GrantedLockType(str, Enum):
    """Lock type actually granted by server."""

    RW = "rw"
    RO = "ro"


class HandshakeRequest(msgspec.Struct, tag="handshake_request"):
    lock_type: RequestedLockType
    timeout_ms: Optional[int] = None


class HandshakeResponse(msgspec.Struct, tag="handshake_response"):
    success: bool
    committed: bool
    granted_lock_type: Optional[GrantedLockType] = None


class CommitRequest(msgspec.Struct, tag="commit_request"):
    pass


class CommitResponse(msgspec.Struct, tag="commit_response"):
    success: bool


class GetLockStateRequest(msgspec.Struct, tag="get_lock_state_request"):
    pass


class GetLockStateResponse(msgspec.Struct, tag="get_lock_state_response"):
    state: str  # "EMPTY", "RW", "COMMITTED", "RO"
    has_rw_session: bool
    ro_session_count: int
    waiting_writers: int
    committed: bool
    is_ready: bool


class GetAllocationStateRequest(msgspec.Struct, tag="get_allocation_state_request"):
    pass


class GetAllocationStateResponse(msgspec.Struct, tag="get_allocation_state_response"):
    allocation_count: int
    total_bytes: int


class AllocateRequest(msgspec.Struct, tag="allocate_request"):
    size: int
    tag: str = "default"


class AllocateResponse(msgspec.Struct, tag="allocate_response"):
    allocation_id: str
    size: int
    aligned_size: int


class ExportRequest(msgspec.Struct, tag="export_request"):
    allocation_id: str


class GetAllocationRequest(msgspec.Struct, tag="get_allocation_request"):
    allocation_id: str


class GetAllocationResponse(msgspec.Struct, tag="get_allocation_response"):
    allocation_id: str
    size: int
    aligned_size: int
    tag: str


class ListAllocationsRequest(msgspec.Struct, tag="list_allocations_request"):
    tag: Optional[str] = None


class ListAllocationsResponse(msgspec.Struct, tag="list_allocations_response"):
    allocations: List[Dict[str, Any]] = []


class FreeRequest(msgspec.Struct, tag="free_request"):
    allocation_id: str


class FreeResponse(msgspec.Struct, tag="free_response"):
    success: bool


class ClearAllRequest(msgspec.Struct, tag="clear_all_request"):
    pass


class ClearAllResponse(msgspec.Struct, tag="clear_all_response"):
    cleared_count: int


class ErrorResponse(msgspec.Struct, tag="error_response"):
    error: str
    code: int = 0


class MetadataPutRequest(msgspec.Struct, tag="metadata_put_request"):
    key: str
    allocation_id: str
    offset_bytes: int
    value: bytes


class MetadataPutResponse(msgspec.Struct, tag="metadata_put_response"):
    success: bool


class MetadataGetRequest(msgspec.Struct, tag="metadata_get_request"):
    key: str


class MetadataGetResponse(msgspec.Struct, tag="metadata_get_response"):
    found: bool
    allocation_id: Optional[str] = None
    offset_bytes: Optional[int] = None
    value: Optional[bytes] = None


class MetadataDeleteRequest(msgspec.Struct, tag="metadata_delete_request"):
    key: str


class MetadataDeleteResponse(msgspec.Struct, tag="metadata_delete_response"):
    deleted: bool


class MetadataListRequest(msgspec.Struct, tag="metadata_list_request"):
    prefix: str = ""


class MetadataListResponse(msgspec.Struct, tag="metadata_list_response"):
    keys: List[str] = []


class GetStateHashRequest(msgspec.Struct, tag="get_memory_layout_hash_request"):
    pass


class GetStateHashResponse(msgspec.Struct, tag="get_memory_layout_hash_response"):
    memory_layout_hash: str  # Hash of allocations + metadata, empty if not committed


Message = Union[
    HandshakeRequest,
    HandshakeResponse,
    CommitRequest,
    CommitResponse,
    GetLockStateRequest,
    GetLockStateResponse,
    GetAllocationStateRequest,
    GetAllocationStateResponse,
    AllocateRequest,
    AllocateResponse,
    ExportRequest,
    GetAllocationRequest,
    GetAllocationResponse,
    ListAllocationsRequest,
    ListAllocationsResponse,
    FreeRequest,
    FreeResponse,
    ClearAllRequest,
    ClearAllResponse,
    ErrorResponse,
    MetadataPutRequest,
    MetadataPutResponse,
    MetadataGetRequest,
    MetadataGetResponse,
    MetadataDeleteRequest,
    MetadataDeleteResponse,
    MetadataListRequest,
    MetadataListResponse,
    GetStateHashRequest,
    GetStateHashResponse,
]

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(Message)


def encode_message(msg: Message) -> bytes:
    return _encoder.encode(msg)


def decode_message(data: bytes) -> Message:
    return _decoder.decode(data)
