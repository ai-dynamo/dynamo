# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message types for GPU Memory Service RPC protocol."""

from enum import Enum
from typing import List, Optional, Union

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


class AllocateRequest(msgspec.Struct, tag="allocate_request"):
    size: int
    tag: str = "default"


class AllocateResponse(msgspec.Struct, tag="allocate_response"):
    allocation_id: str
    size: int
    aligned_size: int
    epoch_id: int
    layout_slot: int


class ExportAllocationRequest(msgspec.Struct, tag="export_allocation_request"):
    allocation_id: str


class ExportAllocationResponse(msgspec.Struct, tag="export_allocation_response"):
    allocation_id: str
    size: int
    aligned_size: int
    tag: str
    epoch_id: int
    layout_slot: int


class GetAllocationRequest(msgspec.Struct, tag="get_allocation_request"):
    allocation_id: str


class GetAllocationResponse(msgspec.Struct, tag="get_allocation_response"):
    allocation_id: str
    size: int
    aligned_size: int
    tag: str
    epoch_id: int
    layout_slot: int


class ListAllocationsRequest(msgspec.Struct, tag="list_allocations_request"):
    tag: Optional[str] = None


class ListAllocationsResponse(msgspec.Struct, tag="list_allocations_response"):
    allocations: List[GetAllocationResponse] = []


class FreeAllocationRequest(msgspec.Struct, tag="free_allocation_request"):
    allocation_id: str


class FreeAllocationResponse(msgspec.Struct, tag="free_allocation_response"):
    success: bool


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


class GetRuntimeStateRequest(msgspec.Struct, tag="get_runtime_state_request"):
    pass


class GetRuntimeStateResponse(msgspec.Struct, tag="get_runtime_state_response"):
    state: str
    has_rw_session: bool
    ro_session_count: int
    waiting_writers: int
    committed: bool
    is_ready: bool
    committed_epoch_id: Optional[int] = None
    active_rw_epoch_id: Optional[int] = None
    allocation_count: int = 0
    memory_layout_hash: str = ""


class GMSRuntimeEvent(msgspec.Struct):
    kind: str
    epoch_id: int
    allocation_count: int = 0


class GetEventHistoryRequest(msgspec.Struct, tag="get_event_history_request"):
    pass


class GetEventHistoryResponse(msgspec.Struct, tag="get_event_history_response"):
    events: List[GMSRuntimeEvent] = []


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
    ExportAllocationRequest,
    ExportAllocationResponse,
    GetAllocationRequest,
    GetAllocationResponse,
    ListAllocationsRequest,
    ListAllocationsResponse,
    FreeAllocationRequest,
    FreeAllocationResponse,
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
    GetRuntimeStateRequest,
    GetRuntimeStateResponse,
    GetEventHistoryRequest,
    GetEventHistoryResponse,
]

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(Message)


def encode_message(msg: Message) -> bytes:
    return _encoder.encode(msg)


def decode_message(data: bytes) -> Message:
    return _decoder.decode(data)
