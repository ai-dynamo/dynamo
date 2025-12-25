"""Allocation Server Client - RPC client stub.

The client provides a simple interface for acquiring locks and performing
allocation operations. The socket connection IS the lock.

Usage:
    # Writer (acquires RW lock in constructor)
    with AllocationServerClient(socket_path, lock_type="rw") as client:
        alloc_id, aligned_size = client.allocate(size=1024*1024)
        fd = client.export(alloc_id)
        # ... write weights using fd ...
        client.commit()
    # Lock released on exit

    # Reader (acquires RO lock in constructor)
    client = AllocationServerClient(socket_path, lock_type="ro")
    if client.committed:  # Check if weights are valid
        allocations = client.list_allocations()
        for alloc in allocations:
            fd = client.export(alloc["allocation_id"])
            # ... import and map fd ...
    # Keep connection open during inference!
    # client.close() only when done with inference
"""

import logging
import socket
from typing import Dict, List, Optional, Tuple

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

logger = logging.getLogger(__name__)


class AllocationServerClient:
    """Client for Allocation Server RPC.

    CRITICAL: Socket connection IS the lock.
    - Constructor blocks until lock is acquired
    - close() releases the lock
    - committed property tells readers if weights are valid

    For writers (lock_type="rw"):
        - Use context manager (with statement) for automatic lock release
        - Call commit() after weights are written
        - Call clear_all() before loading new model

    For readers (lock_type="ro"):
        - Check committed property after construction
        - Keep connection open during inference lifetime
        - Only call close() when shutting down or allowing weight updates
    """

    def __init__(
        self,
        socket_path: str,
        lock_type: str = "ro",
        timeout_ms: Optional[int] = None,
    ):
        """Connect to Allocation Server and acquire lock.

        BLOCKS until lock is available.

        Args:
            socket_path: Path to server's Unix domain socket
            lock_type: "rw" for writer, "ro" for reader
            timeout_ms: Optional timeout for lock acquisition (None = forever)

        Raises:
            TimeoutError: If timeout_ms exceeded waiting for lock
            ConnectionError: If connection fails
        """
        self.socket_path = socket_path
        self.lock_type = lock_type
        self._socket: Optional[socket.socket] = None
        self._recv_buffer = bytearray()
        self._committed = False

        # Connect and acquire lock
        self._connect(timeout_ms)

    def _connect(self, timeout_ms: Optional[int]) -> None:
        """Connect to server and perform handshake (lock acquisition)."""
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            self._socket.connect(self.socket_path)
        except FileNotFoundError:
            raise ConnectionError(f"Server not running at {self.socket_path}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect: {e}")

        # Send handshake (this IS lock acquisition)
        request = HandshakeRequest(lock_type=self.lock_type, timeout_ms=timeout_ms)
        send_message(self._socket, request)

        # Receive response (may block waiting for lock)
        response, _, self._recv_buffer = recv_message(self._socket, self._recv_buffer)

        if isinstance(response, ErrorResponse):
            self._socket.close()
            self._socket = None
            raise ConnectionError(f"Handshake error: {response.error}")

        if not isinstance(response, HandshakeResponse):
            self._socket.close()
            self._socket = None
            raise ConnectionError(f"Unexpected response: {type(response)}")

        if not response.success:
            self._socket.close()
            self._socket = None
            raise TimeoutError("Timeout waiting for lock")

        self._committed = response.committed
        logger.info(
            f"Connected with {self.lock_type} lock, committed={self._committed}"
        )

    @property
    def committed(self) -> bool:
        """Check if weights are committed (valid).

        For readers: check this after construction to know if weights are valid.
        If False, a writer may have crashed before committing.
        """
        return self._committed

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._socket is not None

    def _send_recv(self, request, expect_fd: bool = False) -> Tuple[object, int]:
        """Send request and receive response.

        Returns (response, fd) where fd is -1 if no FD received.
        """
        if not self._socket:
            raise RuntimeError("Client not connected")

        send_message(self._socket, request)
        response, fd, self._recv_buffer = recv_message(self._socket, self._recv_buffer)

        if isinstance(response, ErrorResponse):
            raise RuntimeError(f"Server error: {response.error}")

        return response, fd

    # ==================== State Operations ====================

    def get_state(self) -> GetStateResponse:
        """Get server state."""
        response, _ = self._send_recv(GetStateRequest())
        return response

    def is_ready(self) -> bool:
        """Check if server is ready (no RW, committed).

        Note: For readers, you can also just check self.committed since
        you wouldn't have connected if there was an RW.
        """
        return self.committed

    # ==================== Commit Operation (RW only) ====================

    def commit(self) -> bool:
        """Signal that weights are complete and valid.

        Only valid for RW connections. After commit, even if this process
        crashes, readers will see committed=True.

        Returns True on success.
        """
        if self.lock_type != "rw":
            raise RuntimeError("Only RW connections can commit")

        try:
            response, _ = self._send_recv(CommitRequest())
            ok = isinstance(response, CommitResponse) and response.success
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            # The server closes the RW socket as part of Commit() (publish + release).
            # If we race with that close, verify readiness by attempting an RO connect.
            logger.debug(
                f"Commit saw socket error ({type(e).__name__}); verifying via RO connect"
            )
            self.close()
            try:
                ro = AllocationServerClient(
                    self.socket_path, lock_type="ro", timeout_ms=1000
                )
                ok = ro.committed
                ro.close()
            except TimeoutError:
                ok = False

        if ok:
            self._committed = True
            # Commit releases the RW lock; the server will close the connection. Close locally too.
            self.close()
            logger.info("Committed weights (published) and released RW connection")
            return True

        return False

    # ==================== Allocation Operations ====================

    def allocate(self, size: int, tag: str = "default") -> Tuple[str, int]:
        """Allocate physical memory (RW only).

        Args:
            size: Size in bytes
            tag: Tag for grouping

        Returns:
            Tuple of (allocation_id, aligned_size)
        """
        if self.lock_type != "rw":
            raise RuntimeError("Only RW connections can allocate")

        response, _ = self._send_recv(AllocateRequest(size=size, tag=tag))
        if not isinstance(response, AllocateResponse):
            raise RuntimeError(f"Unexpected response: {type(response)}")

        logger.debug(
            f"Allocated {response.allocation_id}: {size} -> {response.aligned_size}"
        )
        return response.allocation_id, response.aligned_size

    def export(self, allocation_id: str) -> int:
        """Export allocation as POSIX FD.

        The FD can be used with cuMemImportFromShareableHandle().
        Caller is responsible for closing the FD when done.

        Args:
            allocation_id: ID of allocation to export

        Returns:
            File descriptor
        """
        response, fd = self._send_recv(
            ExportRequest(allocation_id=allocation_id), expect_fd=True
        )
        if fd < 0:
            raise RuntimeError("No FD received from server")
        return fd

    def get_allocation(self, allocation_id: str) -> Dict:
        """Get allocation info.

        Returns dict with allocation_id, size, aligned_size, tag.
        """
        response, _ = self._send_recv(GetAllocationRequest(allocation_id=allocation_id))
        if not isinstance(response, GetAllocationResponse):
            raise RuntimeError(f"Unexpected response: {type(response)}")

        return {
            "allocation_id": response.allocation_id,
            "size": response.size,
            "aligned_size": response.aligned_size,
            "tag": response.tag,
        }

    def list_allocations(self, tag: Optional[str] = None) -> List[Dict]:
        """List all allocations.

        Args:
            tag: Optional tag to filter by

        Returns:
            List of allocation dicts
        """
        response, _ = self._send_recv(ListAllocationsRequest(tag=tag))
        if not isinstance(response, ListAllocationsResponse):
            raise RuntimeError(f"Unexpected response: {type(response)}")

        return response.allocations

    def free(self, allocation_id: str) -> bool:
        """Free a single allocation (RW only).

        Args:
            allocation_id: ID of allocation to free

        Returns:
            True on success
        """
        if self.lock_type != "rw":
            raise RuntimeError("Only RW connections can free")

        response, _ = self._send_recv(FreeRequest(allocation_id=allocation_id))
        if isinstance(response, FreeResponse):
            return response.success
        return False

    def clear_all(self) -> int:
        """Clear all allocations (RW only).

        Used before loading a new model.

        Returns:
            Number of allocations cleared
        """
        if self.lock_type != "rw":
            raise RuntimeError("Only RW connections can clear")

        response, _ = self._send_recv(ClearAllRequest())
        if not isinstance(response, ClearAllResponse):
            raise RuntimeError(f"Unexpected response: {type(response)}")

        logger.info(f"Cleared {response.cleared_count} allocations")
        return response.cleared_count

    # ==================== Embedded Artifact Registry ====================

    def registry_put(
        self, key: str, allocation_id: str, offset_bytes: int, value: bytes
    ) -> bool:
        """Put/update a registry entry (RW only)."""
        if self.lock_type != "rw":
            raise RuntimeError("Only RW connections can mutate the registry")
        response, _ = self._send_recv(
            RegistryPutRequest(
                key=key,
                allocation_id=allocation_id,
                offset_bytes=offset_bytes,
                value=value,
            )
        )
        if not isinstance(response, RegistryPutResponse):
            raise RuntimeError(f"Unexpected response: {type(response)}")
        return response.success

    def registry_get(self, key: str) -> Optional[tuple[str, int, bytes]]:
        """Get a registry entry (RO or RW).

        Returns (allocation_id, offset_bytes, value) or None if not found.
        """
        response, _ = self._send_recv(RegistryGetRequest(key=key))
        if not isinstance(response, RegistryGetResponse):
            raise RuntimeError(f"Unexpected response: {type(response)}")
        if not response.found:
            return None
        assert response.allocation_id is not None
        assert response.offset_bytes is not None
        assert response.value is not None
        return response.allocation_id, response.offset_bytes, response.value

    def registry_delete(self, key: str) -> bool:
        """Delete a registry entry (RW only)."""
        if self.lock_type != "rw":
            raise RuntimeError("Only RW connections can mutate the registry")
        response, _ = self._send_recv(RegistryDeleteRequest(key=key))
        if not isinstance(response, RegistryDeleteResponse):
            raise RuntimeError(f"Unexpected response: {type(response)}")
        return response.deleted

    def registry_list(self, prefix: str = "") -> List[str]:
        """List registry keys by prefix (RO or RW)."""
        response, _ = self._send_recv(RegistryListRequest(prefix=prefix))
        if not isinstance(response, RegistryListResponse):
            raise RuntimeError(f"Unexpected response: {type(response)}")
        return response.keys

    def registry_delete_prefix(self, prefix: str) -> int:
        """Delete all registry keys with prefix (RW only)."""
        if self.lock_type != "rw":
            raise RuntimeError("Only RW connections can mutate the registry")
        response, _ = self._send_recv(RegistryDeletePrefixRequest(prefix=prefix))
        if not isinstance(response, RegistryDeletePrefixResponse):
            raise RuntimeError(f"Unexpected response: {type(response)}")
        return response.deleted_count

    def registry_prune_allocation(self, allocation_id: str) -> int:
        """Delete all registry keys referencing allocation_id (RW only)."""
        if self.lock_type != "rw":
            raise RuntimeError("Only RW connections can mutate the registry")
        response, _ = self._send_recv(
            RegistryPruneAllocationRequest(allocation_id=allocation_id)
        )
        if not isinstance(response, RegistryPruneAllocationResponse):
            raise RuntimeError(f"Unexpected response: {type(response)}")
        return response.deleted_count

    def registry_prune_missing_allocations(
        self, valid_allocation_ids: List[str]
    ) -> int:
        """Delete all registry entries whose allocation_id is not present (RW only)."""
        if self.lock_type != "rw":
            raise RuntimeError("Only RW connections can mutate the registry")
        response, _ = self._send_recv(
            RegistryPruneMissingAllocationsRequest(
                valid_allocation_ids=valid_allocation_ids
            )
        )
        if not isinstance(response, RegistryPruneMissingAllocationsResponse):
            raise RuntimeError(f"Unexpected response: {type(response)}")
        return response.deleted_count

    # ==================== Connection Management ====================

    def close(self) -> None:
        """Close connection and release lock.

        For writers: Commit() releases the lock (server closes RW socket). Calling close() is safe and idempotent.
        For readers: Only call when shutting down or allowing weight updates.
        """
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
            logger.info(f"Closed {self.lock_type} connection")

    def __enter__(self) -> "AllocationServerClient":
        """Context manager entry (lock already acquired in __init__)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: close connection (release lock)."""
        self.close()
        return False

    def __del__(self):
        """Destructor: ensure connection is closed."""
        self.close()
