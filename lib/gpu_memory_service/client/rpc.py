# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal GPU Memory Service RPC client.

This module has no PyTorch dependency. `_GMSRPCClient` is an internal transport
layer used by `GMSClientMemoryManager`.
"""

import logging
import socket
from typing import List, Optional, Tuple, Type, TypeVar

from gpu_memory_service.common.protocol.messages import (
    AllocateRequest,
    AllocateResponse,
    CommitRequest,
    CommitResponse,
    ErrorResponse,
    ExportAllocationRequest,
    ExportAllocationResponse,
    FreeAllocationRequest,
    FreeAllocationResponse,
    GetAllocationRequest,
    GetAllocationResponse,
    GetAllocationStateRequest,
    GetAllocationStateResponse,
    GetLockStateRequest,
    GetLockStateResponse,
    GetStateHashRequest,
    GetStateHashResponse,
    HandshakeRequest,
    HandshakeResponse,
    ListAllocationsRequest,
    ListAllocationsResponse,
    MetadataDeleteRequest,
    MetadataDeleteResponse,
    MetadataGetRequest,
    MetadataGetResponse,
    MetadataListRequest,
    MetadataListResponse,
    MetadataPutRequest,
    MetadataPutResponse,
)
from gpu_memory_service.common.protocol.wire import recv_message_sync, send_message_sync
from gpu_memory_service.common.types import (
    GrantedLockType,
    RequestedLockType,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)


class _GMSRPCClient:
    """GPU Memory Service RPC Client.

    CRITICAL: Socket connection IS the lock.
    - Constructor blocks until lock is acquired
    - close() releases the lock
    - committed property tells readers if weights are valid

    For writers (lock_type=RequestedLockType.RW):
        - Use context manager (with statement) for automatic lock release
        - Call commit() after weights are written

    For readers (lock_type=RequestedLockType.RO):
        - Check committed property after construction
        - Keep connection open during inference lifetime
        - Only call close() when shutting down or allowing weight updates
    """

    def __init__(
        self,
        socket_path: str,
        lock_type: RequestedLockType = RequestedLockType.RO,
        timeout_ms: Optional[int] = None,
    ):
        """Connect to Allocation Server and acquire lock.

        Args:
            socket_path: Path to server's Unix domain socket
            lock_type: Requested lock type (RW, RO, or RW_OR_RO)
            timeout_ms: Timeout in milliseconds for lock acquisition.
                        None means wait indefinitely.

        Raises:
            TimeoutError: If timeout_ms expires waiting for lock

        Any transport, protocol, or server-side RPC error raises. The only
        expected constructor failure during normal contention is lock
        acquisition timeout.
        """
        self.socket_path = socket_path
        self._requested_lock_type = lock_type
        self._socket: Optional[socket.socket] = None
        self._recv_buffer = bytearray()
        self._committed = False
        self._granted_lock_type: Optional[GrantedLockType] = None

        # Connect and acquire lock
        self._connect(timeout_ms=timeout_ms)

    def _connect(self, timeout_ms: Optional[int]) -> None:
        """Connect to server and perform handshake (lock acquisition)."""
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._socket.connect(self.socket_path)
        except FileNotFoundError:
            self._socket.close()
            self._socket = None
            raise ConnectionError(
                f"GMS server not running at {self.socket_path}"
            ) from None
        except Exception as e:
            self._socket.close()
            self._socket = None
            raise ConnectionError(f"Failed to connect to GMS: {e}") from e

        # Handshake I/O is strict: transport and protocol failures raise.
        try:
            request = HandshakeRequest(
                lock_type=self._requested_lock_type,
                timeout_ms=timeout_ms,
            )
            send_message_sync(self._socket, request)

            # May block waiting for lock
            response, _, self._recv_buffer = recv_message_sync(
                self._socket, self._recv_buffer
            )
        except Exception as e:
            self._socket.close()
            self._socket = None
            raise ConnectionError(f"GMS handshake I/O failed: {e}") from e

        if isinstance(response, ErrorResponse):
            self._socket.close()
            self._socket = None
            raise RuntimeError(f"GMS handshake error: {response.error}")

        if not isinstance(response, HandshakeResponse):
            self._socket.close()
            self._socket = None
            raise RuntimeError(f"Unexpected GMS handshake response: {type(response)}")

        if not response.success:
            self._socket.close()
            self._socket = None
            raise TimeoutError("Timeout waiting for lock")

        self._committed = response.committed
        # Store granted lock type (may differ from requested for rw_or_ro mode)
        if response.granted_lock_type is not None:
            self._granted_lock_type = response.granted_lock_type
        elif self._requested_lock_type == RequestedLockType.RW:
            self._granted_lock_type = GrantedLockType.RW
        else:
            self._granted_lock_type = GrantedLockType.RO
        logger.info(
            f"Connected with {self._requested_lock_type.value} lock (granted={self._granted_lock_type.value}), "
            f"committed={self._committed}"
        )

    @property
    def committed(self) -> bool:
        """Check if weights are committed (valid)."""
        return self._committed

    @property
    def lock_type(self) -> Optional[GrantedLockType]:
        """Get the lock type actually granted by the server.

        For rw_or_ro mode, this tells you whether RW or RO was granted.
        """
        return self._granted_lock_type

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._socket is not None

    def _send_recv(self, request) -> Tuple[object, int]:
        """Send request and receive response. Returns (response, fd)."""
        if not self._socket:
            raise RuntimeError("Attempted GMS request on disconnected client")

        try:
            send_message_sync(self._socket, request)
            response, fd, self._recv_buffer = recv_message_sync(
                self._socket, self._recv_buffer
            )
        except Exception as e:
            raise ConnectionError(
                f"GMS request {type(request).__name__} failed: {e}"
            ) from e

        if isinstance(response, ErrorResponse):
            raise RuntimeError(f"GMS server error: {response.error}")

        return response, fd

    def _call(self, request, response_type: Type[T]) -> T:
        """Send request, validate response type, return typed response."""
        response, _ = self._send_recv(request)
        if not isinstance(response, response_type):
            raise RuntimeError(
                f"Unexpected GMS response type for {type(request).__name__}: "
                f"{type(response)}"
            )
        return response

    def get_lock_state(self) -> GetLockStateResponse:
        return self._call(GetLockStateRequest(), GetLockStateResponse)

    def get_allocation_state(self) -> GetAllocationStateResponse:
        return self._call(GetAllocationStateRequest(), GetAllocationStateResponse)

    def is_ready(self) -> bool:
        return self.committed

    def commit(self) -> bool:
        """Commit weights and release RW lock. Returns True on success."""
        response, _ = self._send_recv(CommitRequest())
        if not isinstance(response, CommitResponse) or not response.success:
            raise RuntimeError("GMS commit returned failure")

        self._committed = True
        self.close()
        logger.info("Committed weights and released RW connection")
        return True

    def allocate(self, size: int, tag: str = "default") -> Tuple[str, int]:
        """Returns (allocation_id, aligned_size)."""
        r = self._call(AllocateRequest(size=size, tag=tag), AllocateResponse)
        return r.allocation_id, r.aligned_size

    def export(self, allocation_id: str) -> int:
        """Export allocation as POSIX FD. Caller must close."""
        response, fd = self._send_recv(
            ExportAllocationRequest(allocation_id=allocation_id)
        )
        if not isinstance(response, ExportAllocationResponse):
            raise RuntimeError(
                "Unexpected GMS response type for "
                f"ExportAllocationRequest: {type(response)}"
            )
        if fd < 0:
            raise RuntimeError(
                f"GMS export returned no FD for allocation_id={allocation_id}"
            )
        return fd

    def get_allocation(self, allocation_id: str) -> GetAllocationResponse:
        return self._call(
            GetAllocationRequest(allocation_id=allocation_id), GetAllocationResponse
        )

    def list_allocations(
        self, tag: Optional[str] = None
    ) -> List[GetAllocationResponse]:
        return self._call(
            ListAllocationsRequest(tag=tag), ListAllocationsResponse
        ).allocations

    def free(self, allocation_id: str) -> bool:
        return self._call(
            FreeAllocationRequest(allocation_id=allocation_id),
            FreeAllocationResponse,
        ).success

    def metadata_put(
        self, key: str, allocation_id: str, offset_bytes: int, value: bytes
    ) -> bool:
        req = MetadataPutRequest(
            key=key, allocation_id=allocation_id, offset_bytes=offset_bytes, value=value
        )
        return self._call(req, MetadataPutResponse).success

    def metadata_get(self, key: str) -> Optional[tuple[str, int, bytes]]:
        """Returns (allocation_id, offset_bytes, value) or None if not found."""
        r = self._call(MetadataGetRequest(key=key), MetadataGetResponse)
        return (r.allocation_id, r.offset_bytes, r.value) if r.found else None

    def metadata_delete(self, key: str) -> bool:
        return self._call(
            MetadataDeleteRequest(key=key), MetadataDeleteResponse
        ).deleted

    def metadata_list(self, prefix: str = "") -> List[str]:
        return self._call(MetadataListRequest(prefix=prefix), MetadataListResponse).keys

    def get_memory_layout_hash(self) -> str:
        """Get state hash (hash of allocations + metadata). Empty if not committed."""
        return self._call(
            GetStateHashRequest(), GetStateHashResponse
        ).memory_layout_hash

    def close(self) -> None:
        """Close connection and release lock."""
        if self._socket:
            try:
                self._socket.close()
            except Exception as e:
                raise ConnectionError(f"Failed to close GMS client socket: {e}") from e
            self._socket = None
            lock_str = self.lock_type.value if self.lock_type else "unknown"
            logger.info(f"Closed {lock_str} connection")

    def __enter__(self) -> "_GMSRPCClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor: warn if connection not closed."""
        if self._socket:
            logger.warning("_GMSRPCClient not closed properly")
