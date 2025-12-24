"""RPC Protocol for Allocation Server.

Wire format: MessagePack over Unix Domain Socket

Key Design: Socket connection IS the lock.
- First message must be HandshakeRequest with lock_type
- Server blocks until lock is available
- Closing socket = releasing lock (no explicit release RPC)
- Commit() is the only lock-related RPC (signals weights are valid)

Message Types:
1. Connection Handshake (replaces all lock RPCs):
   - HandshakeRequest(lock_type, timeout_ms?) -> HandshakeResponse(success, committed)

2. Commit Operation:
   - CommitRequest() -> CommitResponse(success) [requires RW connection]

3. State Query:
   - GetStateRequest() -> GetStateResponse(...) [any connection]

4. Allocation Operations:
   - AllocateRequest(size, tag) -> AllocateResponse(...) [requires RW]
   - ExportRequest(allocation_id) -> FD via SCM_RIGHTS [requires RW or RO]
   - GetAllocationRequest(allocation_id) -> GetAllocationResponse(...) [any]
   - ListAllocationsRequest(tag?) -> ListAllocationsResponse(...) [any]
   - FreeRequest(allocation_id) -> FreeResponse(...) [requires RW]
   - ClearAllRequest() -> ClearAllResponse(...) [requires RW]

5. Embedded Artifact Registry Operations (served on the same socket):
   - RegistryPutRequest(key, allocation_id, offset_bytes, value) -> RegistryPutResponse [requires RW]
   - RegistryGetRequest(key) -> RegistryGetResponse [RO or RW]
   - RegistryDeleteRequest(key) -> RegistryDeleteResponse [requires RW]
   - RegistryListRequest(prefix) -> RegistryListResponse [RO or RW]
   - RegistryDeletePrefixRequest(prefix) -> RegistryDeletePrefixResponse [requires RW]
   - RegistryPruneAllocationRequest(allocation_id) -> RegistryPruneAllocationResponse [requires RW]
   - RegistryPruneMissingAllocationsRequest(valid_allocation_ids) -> RegistryPruneMissingAllocationsResponse [requires RW]
"""

import struct
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import msgpack

# Message type codes
MSG_HANDSHAKE_REQUEST = 0x01
MSG_HANDSHAKE_RESPONSE = 0x02
MSG_COMMIT_REQUEST = 0x03
MSG_COMMIT_RESPONSE = 0x04
MSG_GET_STATE_REQUEST = 0x05
MSG_GET_STATE_RESPONSE = 0x06
MSG_ALLOCATE_REQUEST = 0x10
MSG_ALLOCATE_RESPONSE = 0x11
MSG_EXPORT_REQUEST = 0x12
MSG_GET_ALLOCATION_REQUEST = 0x13
MSG_GET_ALLOCATION_RESPONSE = 0x14
MSG_LIST_ALLOCATIONS_REQUEST = 0x15
MSG_LIST_ALLOCATIONS_RESPONSE = 0x16
MSG_FREE_REQUEST = 0x17
MSG_FREE_RESPONSE = 0x18
MSG_CLEAR_ALL_REQUEST = 0x19
MSG_CLEAR_ALL_RESPONSE = 0x1A

# Embedded registry message types (0x30+ range)
MSG_REGISTRY_PUT_REQUEST = 0x30
MSG_REGISTRY_PUT_RESPONSE = 0x31
MSG_REGISTRY_GET_REQUEST = 0x32
MSG_REGISTRY_GET_RESPONSE = 0x33
MSG_REGISTRY_DELETE_REQUEST = 0x34
MSG_REGISTRY_DELETE_RESPONSE = 0x35
MSG_REGISTRY_LIST_REQUEST = 0x36
MSG_REGISTRY_LIST_RESPONSE = 0x37
MSG_REGISTRY_DELETE_PREFIX_REQUEST = 0x38
MSG_REGISTRY_DELETE_PREFIX_RESPONSE = 0x39
MSG_REGISTRY_PRUNE_ALLOCATION_REQUEST = 0x3A
MSG_REGISTRY_PRUNE_ALLOCATION_RESPONSE = 0x3B
MSG_REGISTRY_PRUNE_MISSING_REQUEST = 0x3C
MSG_REGISTRY_PRUNE_MISSING_RESPONSE = 0x3D

MSG_ERROR_RESPONSE = 0xFF


# ==================== Connection Handshake (Lock Acquisition) ====================


@dataclass
class HandshakeRequest:
    """First message on new connection - specifies lock type.

    The handshake IS lock acquisition:
    - lock_type="rw": Blocks until all RO connections close, resets commit state
    - lock_type="ro": Blocks until no RW connection exists

    After successful handshake, the connection IS the lock.
    """

    lock_type: Literal["rw", "ro"]
    timeout_ms: Optional[int] = None  # How long to wait for lock (None = forever)


@dataclass
class HandshakeResponse:
    """Response to handshake - connection is ready when this returns.

    success=False means timeout waiting for lock.
    committed tells readers if weights are valid.
    """

    success: bool
    committed: bool


# ==================== Commit Operation ====================


@dataclass
class CommitRequest:
    """Writer signals weights are complete and valid.

    Only valid for RW connection. After commit, even if writer crashes,
    readers will see committed=True and know weights are valid.
    """

    pass


@dataclass
class CommitResponse:
    """Response to commit request."""

    success: bool


# ==================== State Query ====================


@dataclass
class GetStateRequest:
    """Query server state (any connection type can call this)."""

    pass


@dataclass
class GetStateResponse:
    """Server state information."""

    has_rw_connection: bool  # True if writer is active
    ro_connection_count: int  # Number of active readers
    committed: bool  # Whether weights are valid
    is_ready: bool  # Convenience: (no RW) AND committed
    allocation_count: int  # Total allocations


# ==================== Allocation Operations ====================


@dataclass
class AllocateRequest:
    """Create new allocation (RW connection only).

    Creates physical memory with shareable handle.
    No VA mapping on server side - workers handle that.
    """

    size: int
    tag: str = "default"


@dataclass
class AllocateResponse:
    """Response to allocation request."""

    allocation_id: str
    size: int  # Original requested size
    aligned_size: int  # Actual size (aligned to granularity)


@dataclass
class ExportRequest:
    """Export allocation FD (RW or RO connection).

    Returns FD via SCM_RIGHTS ancillary data.
    Workers use this FD with cuMemImportFromShareableHandle().
    """

    allocation_id: str


# ExportResponse: FD sent via SCM_RIGHTS ancillary data, no message body


@dataclass
class GetAllocationRequest:
    """Get allocation info (any connection type)."""

    allocation_id: str


@dataclass
class GetAllocationResponse:
    """Allocation information."""

    allocation_id: str
    size: int
    aligned_size: int
    tag: str


@dataclass
class ListAllocationsRequest:
    """List allocations (any connection type).

    Optionally filter by tag.
    """

    tag: Optional[str] = None


@dataclass
class ListAllocationsResponse:
    """List of allocations."""

    allocations: List[Dict[str, Any]] = field(default_factory=list)
    # Each dict contains: allocation_id, size, aligned_size, tag


@dataclass
class FreeRequest:
    """Free single allocation (RW connection only)."""

    allocation_id: str


@dataclass
class FreeResponse:
    """Response to free request."""

    success: bool


@dataclass
class ClearAllRequest:
    """Clear all allocations (RW connection only).

    Used by loaders before loading a new model.
    """

    pass


@dataclass
class ClearAllResponse:
    """Response to clear all request."""

    cleared_count: int


@dataclass
class ErrorResponse:
    """Error response for any failed operation."""

    error: str
    code: int = 0


# ==================== Embedded Artifact Registry ====================


@dataclass
class RegistryPutRequest:
    """Put/update a registry entry (RW connection only)."""

    key: str
    allocation_id: str
    offset_bytes: int
    value: bytes


@dataclass
class RegistryPutResponse:
    success: bool


@dataclass
class RegistryGetRequest:
    """Get a registry entry (RO or RW connection)."""

    key: str


@dataclass
class RegistryGetResponse:
    found: bool
    allocation_id: Optional[str] = None
    offset_bytes: Optional[int] = None
    value: Optional[bytes] = None


@dataclass
class RegistryDeleteRequest:
    """Delete a registry entry (RW connection only)."""

    key: str


@dataclass
class RegistryDeleteResponse:
    deleted: bool


@dataclass
class RegistryListRequest:
    """List keys with a prefix (RO or RW connection)."""

    prefix: str = ""


@dataclass
class RegistryListResponse:
    keys: List[str] = field(default_factory=list)


@dataclass
class RegistryDeletePrefixRequest:
    """Delete all keys with a prefix (RW connection only)."""

    prefix: str


@dataclass
class RegistryDeletePrefixResponse:
    deleted_count: int


@dataclass
class RegistryPruneAllocationRequest:
    """Delete all entries referencing an allocation_id (RW connection only)."""

    allocation_id: str


@dataclass
class RegistryPruneAllocationResponse:
    deleted_count: int


@dataclass
class RegistryPruneMissingAllocationsRequest:
    """Delete all entries whose allocation_id is not present in valid_allocation_ids (RW connection only)."""

    valid_allocation_ids: List[str]


@dataclass
class RegistryPruneMissingAllocationsResponse:
    deleted_count: int


# ==================== Message Type Registry ====================

_MSG_TYPE_TO_CLASS = {
    MSG_HANDSHAKE_REQUEST: HandshakeRequest,
    MSG_HANDSHAKE_RESPONSE: HandshakeResponse,
    MSG_COMMIT_REQUEST: CommitRequest,
    MSG_COMMIT_RESPONSE: CommitResponse,
    MSG_GET_STATE_REQUEST: GetStateRequest,
    MSG_GET_STATE_RESPONSE: GetStateResponse,
    MSG_ALLOCATE_REQUEST: AllocateRequest,
    MSG_ALLOCATE_RESPONSE: AllocateResponse,
    MSG_EXPORT_REQUEST: ExportRequest,
    MSG_GET_ALLOCATION_REQUEST: GetAllocationRequest,
    MSG_GET_ALLOCATION_RESPONSE: GetAllocationResponse,
    MSG_LIST_ALLOCATIONS_REQUEST: ListAllocationsRequest,
    MSG_LIST_ALLOCATIONS_RESPONSE: ListAllocationsResponse,
    MSG_FREE_REQUEST: FreeRequest,
    MSG_FREE_RESPONSE: FreeResponse,
    MSG_CLEAR_ALL_REQUEST: ClearAllRequest,
    MSG_CLEAR_ALL_RESPONSE: ClearAllResponse,
    # Embedded registry
    MSG_REGISTRY_PUT_REQUEST: RegistryPutRequest,
    MSG_REGISTRY_PUT_RESPONSE: RegistryPutResponse,
    MSG_REGISTRY_GET_REQUEST: RegistryGetRequest,
    MSG_REGISTRY_GET_RESPONSE: RegistryGetResponse,
    MSG_REGISTRY_DELETE_REQUEST: RegistryDeleteRequest,
    MSG_REGISTRY_DELETE_RESPONSE: RegistryDeleteResponse,
    MSG_REGISTRY_LIST_REQUEST: RegistryListRequest,
    MSG_REGISTRY_LIST_RESPONSE: RegistryListResponse,
    MSG_REGISTRY_DELETE_PREFIX_REQUEST: RegistryDeletePrefixRequest,
    MSG_REGISTRY_DELETE_PREFIX_RESPONSE: RegistryDeletePrefixResponse,
    MSG_REGISTRY_PRUNE_ALLOCATION_REQUEST: RegistryPruneAllocationRequest,
    MSG_REGISTRY_PRUNE_ALLOCATION_RESPONSE: RegistryPruneAllocationResponse,
    MSG_REGISTRY_PRUNE_MISSING_REQUEST: RegistryPruneMissingAllocationsRequest,
    MSG_REGISTRY_PRUNE_MISSING_RESPONSE: RegistryPruneMissingAllocationsResponse,
    MSG_ERROR_RESPONSE: ErrorResponse,
}

_CLASS_TO_MSG_TYPE = {v: k for k, v in _MSG_TYPE_TO_CLASS.items()}


# ==================== Serialization ====================


def encode_message(msg: Any) -> bytes:
    """Encode a message to bytes (MessagePack).

    Format: [msg_type (1 byte)] + [msgpack payload]
    """
    msg_type = _CLASS_TO_MSG_TYPE.get(type(msg))
    if msg_type is None:
        raise ValueError(f"Unknown message type: {type(msg)}")

    # Convert dataclass to dict for msgpack
    if hasattr(msg, "__dataclass_fields__"):
        payload = asdict(msg)
    else:
        payload = {}

    data = msgpack.packb(payload, use_bin_type=True)
    return struct.pack("!B", msg_type) + data


def decode_message(data: bytes) -> Any:
    """Decode a message from bytes (MessagePack).

    Returns the appropriate message dataclass instance.
    """
    if len(data) < 1:
        raise ValueError("Empty message data")

    msg_type = struct.unpack("!B", data[:1])[0]
    msg_class = _MSG_TYPE_TO_CLASS.get(msg_type)
    if msg_class is None:
        raise ValueError(f"Unknown message type: {msg_type:#x}")

    if len(data) > 1:
        payload = msgpack.unpackb(data[1:], raw=False)
    else:
        payload = {}

    return msg_class(**payload)


# ==================== Wire Protocol Helpers ====================


def send_message(sock, msg: Any, fd: int = -1) -> None:
    """Send a message over socket with optional FD via SCM_RIGHTS.

    Args:
        sock: Socket to send on
        msg: Message dataclass to send
        fd: Optional file descriptor to send (-1 for none)
    """
    import socket as socket_module

    data = encode_message(msg)
    length = struct.pack("!I", len(data))
    full = length + data

    if fd >= 0:
        # Send with FD using SCM_RIGHTS
        ancdata = [
            (socket_module.SOL_SOCKET, socket_module.SCM_RIGHTS, struct.pack("i", fd))
        ]
        sock.sendmsg([full], ancdata)
    else:
        sock.sendall(full)


def recv_message(sock, recv_buffer: bytearray = None) -> Tuple[Any, int, bytearray]:
    """Receive a message from socket with optional FD.

    Args:
        sock: Socket to receive from
        recv_buffer: Optional buffer with leftover data from previous recv

    Returns:
        Tuple of (message, fd, remaining_buffer)
        fd is -1 if no FD was sent
    """
    import array
    import socket as socket_module

    if recv_buffer is None:
        recv_buffer = bytearray()

    # Check if we have a complete message in buffer already
    if len(recv_buffer) >= 4:
        length = struct.unpack("!I", bytes(recv_buffer[:4]))[0]
        if len(recv_buffer) >= 4 + length:
            # Complete message in buffer (no FD possible from buffer)
            msg_data = bytes(recv_buffer[4 : 4 + length])
            remaining = bytearray(recv_buffer[4 + length :])
            msg = decode_message(msg_data)
            return msg, -1, remaining

    # Need to receive more data
    ancillary_size = socket_module.CMSG_SPACE(struct.calcsize("i"))
    raw_msg, ancdata, flags, _ = sock.recvmsg(65540, ancillary_size)

    fd = -1
    for level, typ, anc_data in ancdata:
        if level == socket_module.SOL_SOCKET and typ == socket_module.SCM_RIGHTS:
            fds = array.array("i")
            fds.frombytes(anc_data[: struct.calcsize("i")])
            if fds:
                fd = fds[0]

    # Prepend any buffered data
    recv_buffer.extend(raw_msg)

    if len(recv_buffer) < 4:
        if len(recv_buffer) == 0:
            raise ConnectionResetError("Connection closed")
        return None, fd, recv_buffer

    length = struct.unpack("!I", bytes(recv_buffer[:4]))[0]

    # Receive more if needed
    while len(recv_buffer) < 4 + length:
        more = sock.recv(4 + length - len(recv_buffer))
        if not more:
            raise ConnectionResetError("Connection closed")
        recv_buffer.extend(more)

    msg_data = bytes(recv_buffer[4 : 4 + length])
    remaining = bytearray(recv_buffer[4 + length :])
    msg = decode_message(msg_data)

    return msg, fd, remaining
