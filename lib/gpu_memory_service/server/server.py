"""Allocation Server - Physical memory owner with connection-based locking.

The Allocation Server:
- Creates physical GPU memory via CUDA VMM (cuMemCreate)
- Exports handles as POSIX FDs for workers to import
- Manages server-wide RW/RO lock (connection = lock)
- Tracks commit state separate from lock state

CRITICAL: Socket connection IS the lock. No separate lock acquisition RPCs.
- Connect with lock_type="rw" -> blocks until RW available, resets commit
- Connect with lock_type="ro" -> blocks until no RW, checks commit state
- Closing socket = releasing lock

Readiness = (no RW connection) AND (committed)
"""

import ctypes
import logging
import os
import select
import socket
import threading
import time
from ctypes import Structure, byref, c_int, c_size_t, c_ulonglong, c_void_p
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Set
from uuid import uuid4

from .protocol import (
    AllocateRequest,
    AllocateResponse,
    ClearAllRequest,
    ClearAllResponse,
    CommitRequest,
    CommitResponse,
    ErrorResponse,
    ExportRequest,
    FreeRequest,
    FreeResponse,
    GetAllocationRequest,
    GetAllocationResponse,
    GetStateRequest,
    GetStateResponse,
    HandshakeRequest,
    HandshakeResponse,
    ListAllocationsRequest,
    ListAllocationsResponse,
    RegistryDeletePrefixRequest,
    RegistryDeletePrefixResponse,
    RegistryDeleteRequest,
    RegistryDeleteResponse,
    RegistryGetRequest,
    RegistryGetResponse,
    RegistryListRequest,
    RegistryListResponse,
    RegistryPruneAllocationRequest,
    RegistryPruneAllocationResponse,
    RegistryPruneMissingAllocationsRequest,
    RegistryPruneMissingAllocationsResponse,
    RegistryPutRequest,
    RegistryPutResponse,
    recv_message,
    send_message,
)
from .registry import ArtifactRegistry

logger = logging.getLogger(__name__)


# Server state machine (GPU_MEMORY_SERVICE_PROPOSAL.md)
class AllocationServerState(str, Enum):
    EMPTY = "EMPTY"
    RW = "RW"
    COMMITTED = "COMMITTED"
    RO = "RO"


# CUDA constants
CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1

CUdeviceptr = c_ulonglong
CUmemGenericAllocationHandle = c_ulonglong


class CUmemLocation(Structure):
    _fields_ = [("type", c_int), ("id", c_int)]


class CUmemAllocationProp(Structure):
    _fields_ = [
        ("type", c_int),
        ("requestedHandleTypes", c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", c_void_p),
        ("allocFlags_compressionType", ctypes.c_uint),
        ("allocFlags_gpuDirectRDMACapable", ctypes.c_uint),
        ("allocFlags_usage", ctypes.c_uint),
        ("allocFlags_reserved", ctypes.c_uint * 4),
    ]


# Load CUDA library (lazy)
_cuda = None


def _get_cuda():
    global _cuda
    if _cuda is None:
        _cuda = ctypes.CDLL("libcuda.so.1")
    return _cuda


def _check(result: int, name: str):
    """Check CUDA result and raise on error."""
    if result != 0:
        cuda = _get_cuda()
        err = ctypes.c_char_p()
        cuda.cuGetErrorString(result, byref(err))
        raise RuntimeError(f"{name}: {err.value.decode() if err.value else result}")


def _ensure_cuda_initialized():
    """Ensure CUDA is initialized."""
    cuda = _get_cuda()
    result = cuda.cuInit(0)
    if result != 0:
        _check(result, "cuInit")


@dataclass
class AllocationInfo:
    """Information about a single allocation."""

    allocation_id: str
    size: int  # Requested size
    aligned_size: int  # Actual size (aligned to granularity)
    handle: int  # CUmemGenericAllocationHandle
    tag: str
    created_at: float


@dataclass
class ConnectionState:
    """State for a connected client."""

    socket: socket.socket
    lock_type: Optional[str] = None  # "rw", "ro", or None if pending handshake
    recv_buffer: bytearray = field(default_factory=bytearray)


class AllocationServer:
    """Physical memory owner with connection-based locking.

    CRITICAL: Socket connection IS the lock. No separate lock acquisition RPCs.
    - Connect with lock_type="rw" -> blocks until RW available, resets commit
    - Connect with lock_type="ro" -> blocks until no RW, checks commit state
    - Closing socket = releasing lock

    Readiness = (no RW connection) AND (committed)

    Durability: Allocations survive worker restarts (in-memory).
    """

    def __init__(self, socket_path: str, device: int = 0):
        """Initialize Allocation Server.

        Args:
            socket_path: Path for Unix domain socket
            device: CUDA device ID
        """
        self.socket_path = socket_path
        self.device = device

        # Connection state (connection = lock)
        self.rw_connection: Optional[socket.socket] = None
        self.ro_connections: Set[socket.socket] = set()

        # Unified global server state (GPU_MEMORY_SERVICE_PROPOSAL.md)
        self.state: AllocationServerState = AllocationServerState.EMPTY

        # Writer preference: while any writer is waiting, new RO handshakes block.
        self.waiting_writers: int = 0

        # Condition variable for blocking operations
        self._lock = threading.Lock()
        self._state_changed = threading.Condition(self._lock)

        # Allocations (no per-allocation state - just the handles)
        self.allocations: Dict[str, AllocationInfo] = {}

        # Embedded artifact registry (served on the same Unix socket)
        self.registry = ArtifactRegistry()

        # Connection tracking
        self._connections: Dict[int, ConnectionState] = {}  # fd -> state
        self._pending_handshakes: Set[int] = set()  # fds awaiting handshake

        # Server state
        self._running = False
        self._listen_sock: Optional[socket.socket] = None
        self._wakeup_r: Optional[int] = None
        self._wakeup_w: Optional[int] = None

        # Initialize CUDA
        _ensure_cuda_initialized()
        self.granularity = self._get_granularity()

        logger.info(
            f"AllocationServer initialized: device={device}, granularity={self.granularity}"
        )

    def _get_granularity(self) -> int:
        """Get VMM allocation granularity for device."""
        cuda = _get_cuda()
        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        gran = c_size_t()
        _check(
            cuda.cuMemGetAllocationGranularity(byref(gran), byref(prop), 0),
            "granularity",
        )
        return gran.value

    def _align(self, size: int) -> int:
        """Align size to granularity."""
        return ((size + self.granularity - 1) // self.granularity) * self.granularity

    # ==================== Connection Handling (Socket = Lock) ====================

    def handle_connect(
        self, sock: socket.socket, handshake: HandshakeRequest
    ) -> HandshakeResponse:
        """Handle new connection with lock type. Blocks until lock available.

        This is the ONLY way to acquire a lock - connection establishment IS lock acquisition.

        Since each connection runs in its own thread, blocking here doesn't
        block the main event loop from processing other connection closes.
        """
        deadline = None
        if handshake.timeout_ms is not None:
            deadline = time.time() + handshake.timeout_ms / 1000

        with self._state_changed:
            if handshake.lock_type == "rw":
                # Writer preference: mark a writer as waiting so new RO handshakes block.
                self.waiting_writers += 1
                try:
                    # Block until no readers AND no active writer.
                    while self.ro_connections or self.rw_connection is not None:
                        # Check for shutdown
                        if not self._running:
                            logger.info("RW handshake aborted due to shutdown")
                            committed = self.state in (
                                AllocationServerState.COMMITTED,
                                AllocationServerState.RO,
                            )
                            return HandshakeResponse(success=False, committed=committed)
                        if deadline is not None:
                            remaining = deadline - time.time()
                            if remaining <= 0:
                                logger.warning("RW handshake timeout waiting for lock")
                                committed = self.state in (
                                    AllocationServerState.COMMITTED,
                                    AllocationServerState.RO,
                                )
                                return HandshakeResponse(
                                    success=False, committed=committed
                                )
                            self._state_changed.wait(timeout=remaining)
                        else:
                            self._state_changed.wait()

                    # Acquire RW
                    self.rw_connection = sock
                    # RW acquisition invalidates any previously-committed weights.
                    self.state = AllocationServerState.RW
                    logger.info("RW connection established, state=RW (invalidated)")
                    return HandshakeResponse(success=True, committed=False)
                finally:
                    # Done waiting (either acquired or timed out)
                    self.waiting_writers -= 1
                    self._state_changed.notify_all()

            elif handshake.lock_type == "ro":
                # Block until READY and no writer is waiting (writer preference).
                while (
                    self.rw_connection is not None
                    or self.waiting_writers > 0
                    or self.state
                    not in (AllocationServerState.COMMITTED, AllocationServerState.RO)
                ):
                    # Check for shutdown
                    if not self._running:
                        logger.info("RO handshake aborted due to shutdown")
                        committed = self.state in (
                            AllocationServerState.COMMITTED,
                            AllocationServerState.RO,
                        )
                        return HandshakeResponse(success=False, committed=committed)
                    if deadline is not None:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            logger.warning(
                                "RO handshake timeout waiting for READY state"
                            )
                            committed = self.state in (
                                AllocationServerState.COMMITTED,
                                AllocationServerState.RO,
                            )
                            return HandshakeResponse(success=False, committed=committed)
                        self._state_changed.wait(timeout=remaining)
                    else:
                        self._state_changed.wait()

                self.ro_connections.add(sock)
                self.state = AllocationServerState.RO
                logger.info(
                    f"RO connection established, state=RO, ro_count={len(self.ro_connections)}"
                )
                return HandshakeResponse(success=True, committed=True)

            else:
                logger.error(f"Unknown lock_type: {handshake.lock_type}")
                committed = self.state in (
                    AllocationServerState.COMMITTED,
                    AllocationServerState.RO,
                )
                return HandshakeResponse(success=False, committed=committed)

    def on_connection_close(self, sock: socket.socket) -> None:
        """Handle connection close. Closing socket = releasing lock.

        This is called by the socket event loop when a connection drops.
        Handles both clean disconnects and crashes.
        """
        with self._state_changed:
            if self.rw_connection == sock:
                # Writer disconnected.
                if self.state == AllocationServerState.RW:
                    # Abort path (crash or disconnect without publish).
                    logger.warning(
                        "RW connection closed without commit; transitioning to EMPTY and clearing state"
                    )
                    self._clear_all_locked()
                    self.registry.clear()
                    self.state = AllocationServerState.EMPTY
                elif self.state == AllocationServerState.COMMITTED:
                    # Commit path: server closed RW socket after publish.
                    logger.info("RW connection closed after commit; state=COMMITTED")
                else:
                    logger.warning(
                        f"RW connection closed in unexpected state={self.state}"
                    )

                self.rw_connection = None
                self._state_changed.notify_all()  # Wake up waiting handshakes

            elif sock in self.ro_connections:
                # Reader disconnected
                self.ro_connections.discard(sock)
                logger.info(
                    f"RO connection closed, remaining ro_count={len(self.ro_connections)}"
                )

                if (
                    len(self.ro_connections) == 0
                    and self.state == AllocationServerState.RO
                ):
                    # Last reader leaving returns to COMMITTED.
                    self.state = AllocationServerState.COMMITTED

                if len(self.ro_connections) == 0:
                    # Wake up any waiting writers and/or blocked readers.
                    self._state_changed.notify_all()

    # ==================== Commit Operation ====================

    def handle_commit(self, sock: socket.socket) -> CommitResponse:
        """Writer signals that weights are complete and valid.

        Only valid for RW connection. After commit, even if writer crashes,
        readers will see committed=True and know weights are valid.
        """
        with self._lock:
            if self.rw_connection != sock or self.state != AllocationServerState.RW:
                # Enforce: mutating RPCs (including commit) require the active RW connection.
                raise ValueError("Only RW connection can commit")
            # Publish: RW -> COMMITTED. Connection will be closed by the server after response.
            self.state = AllocationServerState.COMMITTED
            logger.info(
                f"Weights committed (published), allocation_count={len(self.allocations)}"
            )
            return CommitResponse(success=True)

    def _require_rw_connection(self, sock: socket.socket, op: str) -> None:
        """Enforce that a request is coming from the active RW connection.

        This is a defense-in-depth check: even if a client bypasses the
        high-level client guardrails and sends raw RPCs, the server rejects
        all mutating operations unless issued on the RW connection.
        """
        with self._lock:
            if self.rw_connection != sock or self.state != AllocationServerState.RW:
                raise ValueError(f"RW connection required for {op}")

    def _clear_all_locked(self) -> int:
        """Clear all allocations (internal; expects self._lock held)."""
        count = len(self.allocations)
        cuda = _get_cuda()
        for info in self.allocations.values():
            cuda.cuMemRelease(CUmemGenericAllocationHandle(info.handle))
        self.allocations.clear()
        return count

    # ==================== State Queries ====================

    def is_ready(self) -> bool:
        """Check if server is ready for inference.

        Ready = state in {COMMITTED, RO}
        """
        with self._lock:
            return self.state in (
                AllocationServerState.COMMITTED,
                AllocationServerState.RO,
            )

    def get_state(self) -> GetStateResponse:
        """Get current server state."""
        with self._lock:
            committed = self.state in (
                AllocationServerState.COMMITTED,
                AllocationServerState.RO,
            )
            return GetStateResponse(
                has_rw_connection=self.rw_connection is not None,
                ro_connection_count=len(self.ro_connections),
                committed=committed,
                is_ready=committed,
                allocation_count=len(self.allocations),
            )

    # ==================== Allocation Operations ====================

    def allocate(
        self, sock: socket.socket, size: int, tag: str = "default"
    ) -> AllocateResponse:
        """Create physical memory allocation (no VA mapping).

        Requires: sock must be the RW connection.
        """
        with self._lock:
            if self.rw_connection != sock or self.state != AllocationServerState.RW:
                raise ValueError("Only RW connection can allocate")

        aligned_size = self._align(size)
        cuda = _get_cuda()

        # Set up allocation properties for shareable handle
        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

        # Create physical allocation (no VA mapping!)
        handle = CUmemGenericAllocationHandle()
        _check(
            cuda.cuMemCreate(
                byref(handle), c_size_t(aligned_size), byref(prop), c_ulonglong(0)
            ),
            "cuMemCreate",
        )

        allocation_id = str(uuid4())
        info = AllocationInfo(
            allocation_id=allocation_id,
            size=size,
            aligned_size=aligned_size,
            handle=handle.value,
            tag=tag,
            created_at=time.time(),
        )
        self.allocations[allocation_id] = info

        logger.debug(
            f"Allocated {allocation_id}: size={size}, aligned_size={aligned_size}, tag={tag}"
        )
        return AllocateResponse(
            allocation_id=allocation_id,
            size=size,
            aligned_size=aligned_size,
        )

    def export_fd(self, sock: socket.socket, allocation_id: str) -> int:
        """Export allocation as POSIX FD for SCM_RIGHTS transfer.

        Requires: sock must be RW or RO connection.
        Returns: File descriptor (caller must close after sending)
        """
        with self._lock:
            if self.rw_connection == sock:
                pass  # RW can export
            elif sock in self.ro_connections:
                pass  # RO can export
            else:
                raise ValueError("Only RW or RO connections can export FDs")

        info = self.allocations.get(allocation_id)
        if info is None:
            raise ValueError(f"Unknown allocation: {allocation_id}")

        cuda = _get_cuda()
        fd = c_int()
        _check(
            cuda.cuMemExportToShareableHandle(
                byref(fd),
                CUmemGenericAllocationHandle(info.handle),
                c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                c_ulonglong(0),
            ),
            "cuMemExportToShareableHandle",
        )
        return fd.value

    def get_allocation(self, allocation_id: str) -> GetAllocationResponse:
        """Get allocation info (any connection type can read)."""
        info = self.allocations.get(allocation_id)
        if info is None:
            raise ValueError(f"Unknown allocation: {allocation_id}")

        return GetAllocationResponse(
            allocation_id=info.allocation_id,
            size=info.size,
            aligned_size=info.aligned_size,
            tag=info.tag,
        )

    def list_allocations(self, tag: Optional[str] = None) -> ListAllocationsResponse:
        """List all allocations, optionally filtered by tag."""
        result = []
        for info in self.allocations.values():
            if tag is None or info.tag == tag:
                result.append(
                    {
                        "allocation_id": info.allocation_id,
                        "size": info.size,
                        "aligned_size": info.aligned_size,
                        "tag": info.tag,
                    }
                )
        return ListAllocationsResponse(allocations=result)

    def free(self, sock: socket.socket, allocation_id: str) -> FreeResponse:
        """Release physical memory for a single allocation.

        Requires: sock must be the RW connection.
        """
        with self._lock:
            if self.rw_connection != sock or self.state != AllocationServerState.RW:
                raise ValueError("Only RW connection can free")

        info = self.allocations.pop(allocation_id, None)
        if info is None:
            return FreeResponse(success=False)

        cuda = _get_cuda()
        cuda.cuMemRelease(CUmemGenericAllocationHandle(info.handle))
        logger.debug(f"Freed allocation: {allocation_id}")
        return FreeResponse(success=True)

    def clear_all(self, sock: socket.socket) -> ClearAllResponse:
        """Release ALL allocations.

        Requires: sock must be the RW connection.
        Used by loaders before loading a new model.
        """
        with self._lock:
            if self.rw_connection != sock or self.state != AllocationServerState.RW:
                raise ValueError("Only RW connection can clear")

            count = self._clear_all_locked()
        logger.info(f"Cleared {count} allocations")
        return ClearAllResponse(cleared_count=count)

    # ==================== Request Handling ====================

    def _handle_request(self, conn_state: ConnectionState) -> bool:
        """Handle a request from a connection.

        Returns True if connection should remain open, False to close.
        """
        sock = conn_state.socket

        try:
            msg, fd, conn_state.recv_buffer = recv_message(sock, conn_state.recv_buffer)
        except socket.timeout:
            return True
        except ConnectionResetError:
            return False
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return False

        if msg is None:
            return True  # Incomplete message, wait for more

        try:
            response, response_fd, close_after = self._dispatch_request(sock, msg)
            try:
                send_message(sock, response, response_fd)
            finally:
                # IMPORTANT: if we exported a POSIX FD to the client via SCM_RIGHTS,
                # we must close our copy after sendmsg() returns. Otherwise the
                # Allocation Server leaks file descriptors over time.
                if response_fd is not None and int(response_fd) >= 0:
                    try:
                        os.close(int(response_fd))
                    except OSError:
                        pass
            return not close_after
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            try:
                send_message(sock, ErrorResponse(error=str(e)))
            except Exception:
                pass
            return True

    def _dispatch_request(self, sock: socket.socket, msg) -> tuple:
        """Dispatch a request to the appropriate handler.

        Returns (response, fd, close_after_response).
        close_after_response is used for Commit(): publish + server closes RW socket.
        """
        if isinstance(msg, CommitRequest):
            self._require_rw_connection(sock, "Commit")
            return self.handle_commit(sock), -1, True

        elif isinstance(msg, GetStateRequest):
            return self.get_state(), -1, False

        elif isinstance(msg, AllocateRequest):
            self._require_rw_connection(sock, "Allocate")
            return self.allocate(sock, msg.size, msg.tag), -1, False

        elif isinstance(msg, ExportRequest):
            fd = self.export_fd(sock, msg.allocation_id)
            # Send empty success response with FD
            return (
                GetAllocationResponse(
                    allocation_id=msg.allocation_id,
                    size=self.allocations[msg.allocation_id].size,
                    aligned_size=self.allocations[msg.allocation_id].aligned_size,
                    tag=self.allocations[msg.allocation_id].tag,
                ),
                fd,
                False,
            )

        elif isinstance(msg, GetAllocationRequest):
            return self.get_allocation(msg.allocation_id), -1, False

        elif isinstance(msg, ListAllocationsRequest):
            return self.list_allocations(msg.tag), -1, False

        elif isinstance(msg, FreeRequest):
            self._require_rw_connection(sock, "Free")
            return self.free(sock, msg.allocation_id), -1, False

        elif isinstance(msg, ClearAllRequest):
            self._require_rw_connection(sock, "ClearAll")
            return self.clear_all(sock), -1, False

        # ==================== Embedded Artifact Registry RPCs ====================

        elif isinstance(msg, RegistryPutRequest):
            self._require_rw_connection(sock, "RegistryPut")
            self.registry.put(
                key=msg.key,
                allocation_id=msg.allocation_id,
                offset_bytes=msg.offset_bytes,
                value=msg.value,
            )
            return RegistryPutResponse(success=True), -1, False

        elif isinstance(msg, RegistryGetRequest):
            entry = self.registry.get(msg.key)
            if entry is None:
                return RegistryGetResponse(found=False), -1, False
            return (
                RegistryGetResponse(
                    found=True,
                    allocation_id=entry.allocation_id,
                    offset_bytes=entry.offset_bytes,
                    value=entry.value,
                ),
                -1,
                False,
            )

        elif isinstance(msg, RegistryDeleteRequest):
            self._require_rw_connection(sock, "RegistryDelete")
            deleted = self.registry.delete(msg.key)
            return RegistryDeleteResponse(deleted=deleted), -1, False

        elif isinstance(msg, RegistryListRequest):
            keys = self.registry.list_keys(msg.prefix)
            return RegistryListResponse(keys=keys), -1, False

        elif isinstance(msg, RegistryDeletePrefixRequest):
            self._require_rw_connection(sock, "RegistryDeletePrefix")
            deleted_count = self.registry.delete_prefix(msg.prefix)
            return RegistryDeletePrefixResponse(deleted_count=deleted_count), -1, False

        elif isinstance(msg, RegistryPruneAllocationRequest):
            self._require_rw_connection(sock, "RegistryPruneAllocation")
            deleted_count = self.registry.prune_allocation(msg.allocation_id)
            return (
                RegistryPruneAllocationResponse(deleted_count=deleted_count),
                -1,
                False,
            )

        elif isinstance(msg, RegistryPruneMissingAllocationsRequest):
            self._require_rw_connection(sock, "RegistryPruneMissingAllocations")
            deleted_count = self.registry.prune_missing_allocations(
                set(msg.valid_allocation_ids)
            )
            return (
                RegistryPruneMissingAllocationsResponse(deleted_count=deleted_count),
                -1,
                False,
            )

        else:
            raise ValueError(f"Unknown request type: {type(msg)}")

    # ==================== Server Main Loop ====================

    def start(self) -> None:
        """Start the server (bind socket, prepare for connections)."""
        # Remove existing socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Wakeup pipe to interrupt poll() on shutdown.
        if self._wakeup_r is None or self._wakeup_w is None:
            self._wakeup_r, self._wakeup_w = os.pipe()

        self._listen_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listen_sock.bind(self.socket_path)
        self._listen_sock.listen(16)
        self._listen_sock.setblocking(False)

        self._running = True
        logger.info(f"AllocationServer started at {self.socket_path}")

    def stop(self) -> None:
        """Stop the server and cleanup."""
        self._running = False

        # Wake all threads waiting on state changes so they can check _running and exit
        with self._state_changed:
            self._state_changed.notify_all()

        # Wake the accept poll loop so shutdown is fast (tests otherwise look hung).
        if self._wakeup_w is not None:
            try:
                os.write(self._wakeup_w, b"\x00")
            except Exception:
                pass

        # Release all GPU memory allocations before closing connections.
        # This is needed for graceful shutdown when state is COMMITTED/RO,
        # since on_connection_close() only releases memory in RW state.
        with self._lock:
            if self.allocations:
                count = self._clear_all_locked()
                self.registry.clear()
                logger.info(f"Released {count} GPU allocations during shutdown")
            self.state = AllocationServerState.EMPTY

        # Close all connections
        for fd, conn_state in list(self._connections.items()):
            self.on_connection_close(conn_state.socket)
            try:
                conn_state.socket.close()
            except Exception:
                pass
        self._connections.clear()
        self._pending_handshakes.clear()

        # Close listen socket
        if self._listen_sock:
            try:
                self._listen_sock.close()
            except Exception:
                pass
            self._listen_sock = None

        # Close wakeup pipe
        if self._wakeup_r is not None:
            try:
                os.close(self._wakeup_r)
            except Exception:
                pass
            self._wakeup_r = None
        if self._wakeup_w is not None:
            try:
                os.close(self._wakeup_w)
            except Exception:
                pass
            self._wakeup_w = None

        # Clean up socket file
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        logger.info("AllocationServer stopped")

    def serve_forever(self) -> None:
        """Run the server main loop (blocking).

        Uses a hybrid threading model:
        - Main thread runs the poll loop for accepting new connections only
        - Each connection's handshake and request handling runs in a separate thread
        - Handler threads are responsible for detecting closes and cleanup
        """
        self.start()

        poller = select.poll()
        poller.register(self._listen_sock.fileno(), select.POLLIN)
        if self._wakeup_r is not None:
            poller.register(self._wakeup_r, select.POLLIN)

        # Track connection handler threads
        handler_threads: Dict[int, threading.Thread] = {}

        while self._running:
            try:
                events = poller.poll(1000)  # 1 second timeout
            except InterruptedError:
                continue

            for fd, event in events:
                if not self._running or self._listen_sock is None:
                    break
                if self._wakeup_r is not None and fd == self._wakeup_r:
                    # Drain wakeup pipe.
                    try:
                        os.read(self._wakeup_r, 4096)
                    except Exception:
                        pass
                    continue
                if fd == self._listen_sock.fileno():
                    # New connection - accept and spawn handler thread
                    self._accept_connection_threaded(poller, handler_threads)
                # Note: We don't handle POLLHUP/POLLERR here anymore.
                # Handler threads detect closes via recv failures and handle cleanup.

        # Wait for handler threads to finish
        for thread in handler_threads.values():
            thread.join(timeout=1.0)

        self.stop()

    def _accept_connection_threaded(
        self, poller: select.poll, handler_threads: Dict[int, threading.Thread]
    ) -> None:
        """Accept a new connection and spawn a handler thread."""
        try:
            conn_sock, _ = self._listen_sock.accept()
            conn_sock.setblocking(True)
            fd = conn_sock.fileno()

            conn_state = ConnectionState(socket=conn_sock)
            self._connections[fd] = conn_state

            # Handler thread will manage all I/O and cleanup for this connection
            # No need to poll - thread detects closes via recv/send failures
            thread = threading.Thread(
                target=self._connection_handler_thread, args=(conn_state,), daemon=True
            )
            handler_threads[fd] = thread
            thread.start()

            logger.debug(f"Accepted connection fd={fd}, spawned handler thread")

        except Exception as e:
            logger.error(f"Error accepting connection: {e}")

    def _connection_handler_thread(self, conn_state: ConnectionState) -> None:
        """Handle a connection in its own thread.

        This allows the handshake to block waiting for locks without
        blocking the main event loop. The handler thread is responsible
        for all I/O and cleanup for this connection.
        """
        sock = conn_state.socket
        fd = sock.fileno()

        try:
            # Phase 1: Handshake (may block waiting for lock)
            if not self._do_handshake(conn_state):
                self._close_connection(fd)
                return

            # Phase 2: Request loop
            while self._running:
                try:
                    # Set a timeout so we can check _running periodically
                    sock.settimeout(1.0)
                    if not self._handle_request(conn_state):
                        break
                except socket.timeout:
                    continue
                except ConnectionResetError:
                    break
                except OSError as e:
                    # Handle "Bad file descriptor" and other socket errors
                    if e.errno == 9:  # EBADF
                        break
                    logger.error(f"Error in connection handler: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error in connection handler: {e}")
                    break

        except Exception as e:
            logger.error(f"Connection handler thread error: {e}")
        finally:
            self._close_connection(fd)

    def _do_handshake(self, conn_state: ConnectionState) -> bool:
        """Perform handshake (blocking wait for lock if needed).

        Returns True if handshake succeeded, False to close connection.
        """
        sock = conn_state.socket

        try:
            msg, _, conn_state.recv_buffer = recv_message(sock, conn_state.recv_buffer)
        except ConnectionResetError:
            return False
        except Exception as e:
            logger.error(f"Error receiving handshake: {e}")
            return False

        if msg is None:
            return False  # Incomplete message

        if not isinstance(msg, HandshakeRequest):
            logger.error(f"First message must be HandshakeRequest, got {type(msg)}")
            try:
                send_message(
                    sock, ErrorResponse(error="First message must be HandshakeRequest")
                )
            except Exception:
                pass
            return False

        # Handle handshake (may block waiting for lock)
        response = self.handle_connect(sock, msg)
        try:
            send_message(sock, response)
        except Exception as e:
            logger.error(f"Error sending handshake response: {e}")
            return False

        if not response.success:
            return False

        conn_state.lock_type = msg.lock_type
        logger.debug(
            f"Handshake complete fd={sock.fileno()}, lock_type={msg.lock_type}"
        )
        return True

    def _accept_connection(self, poller: select.poll) -> None:
        """Accept a new connection (legacy single-threaded mode)."""
        try:
            conn_sock, _ = self._listen_sock.accept()
            conn_sock.setblocking(True)
            fd = conn_sock.fileno()

            conn_state = ConnectionState(socket=conn_sock)
            self._connections[fd] = conn_state
            self._pending_handshakes.add(fd)

            poller.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
            logger.debug(f"Accepted connection fd={fd}")

        except Exception as e:
            logger.error(f"Error accepting connection: {e}")

    def _handle_handshake(
        self, conn_state: ConnectionState, poller: select.poll
    ) -> bool:
        """Handle connection handshake (lock acquisition).

        Returns True to keep connection, False to close.
        """
        sock = conn_state.socket
        fd = sock.fileno()

        try:
            msg, _, conn_state.recv_buffer = recv_message(sock, conn_state.recv_buffer)
        except ConnectionResetError:
            return False
        except Exception as e:
            logger.error(f"Error receiving handshake: {e}")
            return False

        if msg is None:
            return True  # Incomplete, wait for more

        if not isinstance(msg, HandshakeRequest):
            logger.error(f"First message must be HandshakeRequest, got {type(msg)}")
            send_message(
                sock, ErrorResponse(error="First message must be HandshakeRequest")
            )
            return False

        # Handle handshake (may block waiting for lock)
        response = self.handle_connect(sock, msg)
        send_message(sock, response)

        if not response.success:
            return False

        conn_state.lock_type = msg.lock_type
        self._pending_handshakes.discard(fd)
        logger.debug(f"Handshake complete fd={fd}, lock_type={msg.lock_type}")
        return True

    def _close_connection(self, fd: int) -> None:
        """Close a connection and release any held lock."""
        conn_state = self._connections.pop(fd, None)
        self._pending_handshakes.discard(fd)

        if conn_state:
            self.on_connection_close(conn_state.socket)
            try:
                conn_state.socket.close()
            except Exception:
                pass

        logger.debug(f"Closed connection fd={fd}")

    def serve_in_thread(self) -> threading.Thread:
        """Start server in a background thread.

        Returns the thread object. Call thread.join() to wait for shutdown.
        """
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread
