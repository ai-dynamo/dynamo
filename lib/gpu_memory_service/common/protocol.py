"""RPC Protocol for Allocation Server.

Wire format: MessagePack over Unix Domain Socket

Key Design: Socket connection IS the lock.
- First message must be HandshakeRequest with lock_type
- Server blocks until lock is available (waits forever)
- Closing socket = releasing lock (no explicit release RPC)
- Commit() is the only lock-related RPC (signals weights are valid)

Message Types:
1. Connection Handshake (replaces all lock RPCs):
   - HandshakeRequest(lock_type) -> HandshakeResponse(success, committed)

2. Commit Operation:
   - CommitRequest() -> CommitResponse(success) [requires RW connection]

3. State Queries:
   - GetLockStateRequest() -> GetLockStateResponse(...) [any connection] (lock/session state)
   - GetAllocationStateRequest() -> GetAllocationStateResponse(...) [any connection] (allocation state)

4. Allocation Operations:
   - AllocateRequest(size, tag) -> AllocateResponse(...) [requires RW]
   - ExportRequest(allocation_id) -> FD via SCM_RIGHTS [requires RW or RO]
   - GetAllocationRequest(allocation_id) -> GetAllocationResponse(...) [any]
   - ListAllocationsRequest(tag?) -> ListAllocationsResponse(...) [any]
   - FreeRequest(allocation_id) -> FreeResponse(...) [requires RW]
   - ClearAllRequest() -> ClearAllResponse(...) [requires RW]

5. Embedded Metadata Store Operations (served on the same socket):
   - MetadataPutRequest(key, allocation_id, offset_bytes, value) -> MetadataPutResponse [requires RW]
   - MetadataGetRequest(key) -> MetadataGetResponse [RO or RW]
   - MetadataDeleteRequest(key) -> MetadataDeleteResponse [requires RW]
   - MetadataListRequest(prefix) -> MetadataListResponse [RO or RW]
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
MSG_GET_LOCK_STATE_REQUEST = 0x05
MSG_GET_LOCK_STATE_RESPONSE = 0x06
MSG_GET_ALLOCATION_STATE_REQUEST = 0x07
MSG_GET_ALLOCATION_STATE_RESPONSE = 0x08
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

# Embedded metadata store message types (0x30+ range)
MSG_METADATA_PUT_REQUEST = 0x30
MSG_METADATA_PUT_RESPONSE = 0x31
MSG_METADATA_GET_REQUEST = 0x32
MSG_METADATA_GET_RESPONSE = 0x33
MSG_METADATA_DELETE_REQUEST = 0x34
MSG_METADATA_DELETE_RESPONSE = 0x35
MSG_METADATA_LIST_REQUEST = 0x36
MSG_METADATA_LIST_RESPONSE = 0x37

MSG_ERROR_RESPONSE = 0xFF


# ==================== Connection Handshake (Lock Acquisition) ====================


@dataclass
class HandshakeRequest:
    """First message on new connection - specifies lock type.

    The handshake IS lock acquisition:
    - lock_type="rw": Blocks until all RO connections close, resets commit state
    - lock_type="ro": Blocks until no RW connection exists
    - lock_type="rw_or_ro": Try RW first; if another writer holds RW, fall back to RO

    After successful handshake, the connection IS the lock.
    """

    lock_type: Literal["rw", "ro", "rw_or_ro"]
    timeout_ms: Optional[int] = None  # How long to wait for lock (None = forever)


@dataclass
class HandshakeResponse:
    """Response to handshake - connection is ready when this returns.

    success=False means timeout waiting for lock.
    committed tells readers if weights are valid.
    granted_lock_type tells which lock was actually granted (for rw_or_ro mode).
    """

    success: bool
    committed: bool
    granted_lock_type: Optional[Literal["rw", "ro"]] = None  # Which lock was granted


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


# ==================== State Queries ====================


@dataclass
class GetLockStateRequest:
    """Query lock/session state (any connection type can call this)."""

    pass


@dataclass
class GetLockStateResponse:
    """Lock state information (from LockManager)."""

    state: str  # "EMPTY", "RW", "COMMITTED", "RO"
    has_rw_session: bool  # True if writer is active
    ro_session_count: int  # Number of active readers
    waiting_writers: int  # Number of writers waiting for the lock
    committed: bool  # Whether allocations are committed
    is_ready: bool  # Convenience: (no RW) AND committed


@dataclass
class GetAllocationStateRequest:
    """Query allocation state (any connection type can call this)."""

    pass


@dataclass
class GetAllocationStateResponse:
    """Allocation state information (from GMSServerMemoryManager)."""

    allocation_count: int  # Total number of allocations
    total_bytes: int  # Sum of aligned_sizes across all allocations


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


# ==================== Embedded Metadata Store ====================


@dataclass
class MetadataPutRequest:
    """Put/update a metadata entry (RW connection only)."""

    key: str
    allocation_id: str
    offset_bytes: int
    value: bytes


@dataclass
class MetadataPutResponse:
    success: bool


@dataclass
class MetadataGetRequest:
    """Get a metadata entry (RO or RW connection)."""

    key: str


@dataclass
class MetadataGetResponse:
    found: bool
    allocation_id: Optional[str] = None
    offset_bytes: Optional[int] = None
    value: Optional[bytes] = None


@dataclass
class MetadataDeleteRequest:
    """Delete a metadata entry (RW connection only)."""

    key: str


@dataclass
class MetadataDeleteResponse:
    deleted: bool


@dataclass
class MetadataListRequest:
    """List keys with a prefix (RO or RW connection)."""

    prefix: str = ""


@dataclass
class MetadataListResponse:
    keys: List[str] = field(default_factory=list)


# ==================== Message Type Lookup ====================

_MSG_TYPE_TO_CLASS = {
    MSG_HANDSHAKE_REQUEST: HandshakeRequest,
    MSG_HANDSHAKE_RESPONSE: HandshakeResponse,
    MSG_COMMIT_REQUEST: CommitRequest,
    MSG_COMMIT_RESPONSE: CommitResponse,
    MSG_GET_LOCK_STATE_REQUEST: GetLockStateRequest,
    MSG_GET_LOCK_STATE_RESPONSE: GetLockStateResponse,
    MSG_GET_ALLOCATION_STATE_REQUEST: GetAllocationStateRequest,
    MSG_GET_ALLOCATION_STATE_RESPONSE: GetAllocationStateResponse,
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
    # Embedded metadata store
    MSG_METADATA_PUT_REQUEST: MetadataPutRequest,
    MSG_METADATA_PUT_RESPONSE: MetadataPutResponse,
    MSG_METADATA_GET_REQUEST: MetadataGetRequest,
    MSG_METADATA_GET_RESPONSE: MetadataGetResponse,
    MSG_METADATA_DELETE_REQUEST: MetadataDeleteRequest,
    MSG_METADATA_DELETE_RESPONSE: MetadataDeleteResponse,
    MSG_METADATA_LIST_REQUEST: MetadataListRequest,
    MSG_METADATA_LIST_RESPONSE: MetadataListResponse,
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


async def send_message(writer, msg: Any, fd: int = -1) -> None:
    """Send a message over asyncio StreamWriter with optional FD via SCM_RIGHTS.

    For FD passing, we use the underlying socket directly since
    asyncio streams don't support ancillary data.

    Args:
        writer: asyncio.StreamWriter
        msg: Message dataclass to send
        fd: Optional file descriptor to send (-1 for none)
    """
    import asyncio
    import os
    import socket as socket_module

    data = encode_message(msg)
    length = struct.pack("!I", len(data))
    full = length + data

    if fd >= 0:
        # FD passing requires sendmsg on raw socket
        # The socket from get_extra_info may not properly support sendmsg in asyncio context
        # So we duplicate the fd and use a fresh socket object
        transport_sock = writer.get_extra_info("socket")
        if transport_sock is None:
            raise RuntimeError("Cannot get socket from transport for FD passing")

        ancdata = [
            (socket_module.SOL_SOCKET, socket_module.SCM_RIGHTS, struct.pack("i", fd))
        ]

        def do_sendmsg():
            # Create a socket from the fd for proper sendmsg support
            raw_fd = transport_sock.fileno()
            # Use dup to get a new fd we can safely use
            dup_fd = os.dup(raw_fd)
            try:
                # Create a socket from the duplicated fd
                sock = socket_module.socket(fileno=dup_fd)
                try:
                    # Set blocking mode for reliable send
                    sock.setblocking(True)
                    sock.sendmsg([full], ancdata)
                finally:
                    # Detach to avoid closing the original fd
                    sock.detach()
            except Exception:
                os.close(dup_fd)
                raise

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, do_sendmsg)
    else:
        writer.write(full)
        await writer.drain()


async def recv_message(
    reader, recv_buffer: bytearray = None, raw_sock=None
) -> Tuple[Any, int, bytearray]:
    """Receive a message from asyncio StreamReader with optional FD.

    For FD receiving, we need the raw socket for recvmsg.

    Args:
        reader: asyncio.StreamReader
        recv_buffer: Optional buffer with leftover data
        raw_sock: Raw socket for FD receiving (required for FD support)

    Returns:
        Tuple of (message, fd, remaining_buffer)
        fd is -1 if no FD was sent
    """
    import array
    import asyncio
    import socket as socket_module

    if recv_buffer is None:
        recv_buffer = bytearray()

    loop = asyncio.get_running_loop()
    fd = -1

    # Check if we have a complete message in buffer already
    if len(recv_buffer) >= 4:
        length = struct.unpack("!I", bytes(recv_buffer[:4]))[0]
        if len(recv_buffer) >= 4 + length:
            msg_data = bytes(recv_buffer[4 : 4 + length])
            remaining = bytearray(recv_buffer[4 + length :])
            msg = decode_message(msg_data)
            return msg, -1, remaining

    # Need to receive more data
    if raw_sock is not None:
        # Use recvmsg for potential FD receiving
        ancillary_size = socket_module.CMSG_SPACE(struct.calcsize("i"))

        def do_recvmsg():
            return raw_sock.recvmsg(65540, ancillary_size)

        raw_msg, ancdata, flags, _ = await loop.run_in_executor(None, do_recvmsg)

        for level, typ, anc_data in ancdata:
            if level == socket_module.SOL_SOCKET and typ == socket_module.SCM_RIGHTS:
                fds = array.array("i")
                fds.frombytes(anc_data[: struct.calcsize("i")])
                if fds:
                    fd = fds[0]

        recv_buffer.extend(raw_msg)
    else:
        # Use StreamReader (no FD support)
        chunk = await reader.read(65540)
        if not chunk:
            raise ConnectionResetError("Connection closed")
        recv_buffer.extend(chunk)

    if len(recv_buffer) < 4:
        if len(recv_buffer) == 0:
            raise ConnectionResetError("Connection closed")
        return None, fd, recv_buffer

    length = struct.unpack("!I", bytes(recv_buffer[:4]))[0]

    # Receive more if needed
    while len(recv_buffer) < 4 + length:
        chunk = await reader.read(4 + length - len(recv_buffer))
        if not chunk:
            raise ConnectionResetError("Connection closed")
        recv_buffer.extend(chunk)

    msg_data = bytes(recv_buffer[4 : 4 + length])
    remaining = bytearray(recv_buffer[4 + length :])
    msg = decode_message(msg_data)

    return msg, fd, remaining


# ==================== Synchronous Wire Protocol (for sync client) ====================


def send_message_sync(sock, msg: Any, fd: int = -1) -> None:
    """Send a message over socket with optional FD via SCM_RIGHTS.

    Synchronous version for the sync client.

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


def recv_message_sync(
    sock, recv_buffer: bytearray = None
) -> Tuple[Any, int, bytearray]:
    """Receive a message from socket with optional FD.

    Synchronous version for the sync client.

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

    recv_buffer.extend(raw_msg)

    if len(recv_buffer) < 4:
        if len(recv_buffer) == 0:
            raise ConnectionResetError("Connection closed")
        return None, fd, recv_buffer

    length = struct.unpack("!I", bytes(recv_buffer[:4]))[0]

    # Receive more if needed
    while len(recv_buffer) < 4 + length:
        chunk = sock.recv(4 + length - len(recv_buffer))
        if not chunk:
            raise ConnectionResetError("Connection closed")
        recv_buffer.extend(chunk)

    msg_data = bytes(recv_buffer[4 : 4 + length])
    remaining = bytearray(recv_buffer[4 + length :])
    msg = decode_message(msg_data)

    return msg, fd, remaining
