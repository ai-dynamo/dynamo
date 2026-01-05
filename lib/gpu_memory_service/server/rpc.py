# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async Allocation RPC Server - Single-threaded event loop with explicit state machine.

State transitions are explicit and validated by the GlobalLockFSM class.
Operations are checked against state/mode permissions before execution.

State Machine (see locking.py for full diagram):
    EMPTY: No connections, not committed
    RW: Writer connected (exclusive)
    COMMITTED: No connections, committed (weights valid)
    RO: Reader(s) connected (shared)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from gpu_memory_service.common.protocol import (
    AllocateRequest,
    ClearAllRequest,
    CommitRequest,
    CommitResponse,
    ErrorResponse,
    ExportRequest,
    FreeRequest,
    GetAllocationRequest,
    GetAllocationStateRequest,
    GetLockStateRequest,
    HandshakeRequest,
    HandshakeResponse,
    ListAllocationsRequest,
    MetadataDeleteRequest,
    MetadataGetRequest,
    MetadataListRequest,
    MetadataPutRequest,
    recv_message,
    send_message,
)
from gpu_memory_service.common.types import (
    ConnectionMode,
    Operation,
    ServerState,
    StateEvent,
)

from .handler import RequestHandler
from .locking import Connection, GlobalLockFSM

logger = logging.getLogger(__name__)


class GMSRPCServer:
    """GPU Memory Service RPC Server.

    Async single-threaded server using GlobalLockFSM for explicit state transitions
    and operation validation. All state mutations happen through the state machine's
    transition() method.
    """

    def __init__(self, socket_path: str, device: int = 0):
        self.socket_path = socket_path
        self.device = device

        # Request handler (business logic)
        self._handler = RequestHandler(device)

        # State machine - handles all state transitions and permission checks
        self._sm = GlobalLockFSM(on_rw_abort=self._handler.on_rw_abort)
        self._waiting_writers: int = 0

        # Async waiting for lock acquisition
        self._condition = asyncio.Condition()
        self._shutdown = False

        # Session ID generation
        self._next_session_id: int = 0

        # Server state
        self._server: Optional[asyncio.Server] = None
        self._running: bool = False

        logger.info(f"GMSRPCServer initialized: device={device}")

    # ==================== State Properties ====================

    @property
    def state(self) -> ServerState:
        """Current server state (delegated to state machine)."""
        return self._sm.state

    @property
    def granularity(self) -> int:
        return self._handler.granularity

    def is_ready(self) -> bool:
        """Ready = committed and no RW connection."""
        return self._sm.committed and self._sm.rw_conn is None

    def _generate_session_id(self) -> str:
        self._next_session_id += 1
        return f"session_{self._next_session_id}"

    # ==================== Connection Lifecycle ====================

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a connection from accept to close."""
        session_id = self._generate_session_id()
        conn: Optional[Connection] = None

        try:
            conn = await self._do_handshake(reader, writer, session_id)
            if conn is None:
                return
            await self._request_loop(conn)
        except ConnectionResetError:
            logger.debug(f"Connection reset: {session_id}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Connection error: {session_id}, {e}")
        finally:
            await self._cleanup_connection(conn, session_id)

    async def _do_handshake(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        session_id: str,
    ) -> Optional[Connection]:
        """Perform handshake and acquire lock via state machine transition."""
        try:
            # Server never receives FDs from clients, so no need for raw_sock
            msg, _, recv_buffer = await recv_message(reader, bytearray())
        except Exception as e:
            logger.error(f"Handshake recv error: {e}")
            return None

        if not isinstance(msg, HandshakeRequest):
            await send_message(writer, ErrorResponse(error="Expected HandshakeRequest"))
            writer.close()
            return None

        try:
            mode = ConnectionMode(msg.lock_type)
        except ValueError:
            await send_message(
                writer, HandshakeResponse(success=False, committed=self._sm.committed)
            )
            writer.close()
            return None

        # Acquire lock (blocks until available or timeout)
        if not await self._acquire_lock(mode, msg.timeout_ms):
            await send_message(
                writer, HandshakeResponse(success=False, committed=self._sm.committed)
            )
            writer.close()
            return None

        conn = Connection(reader, writer, mode, session_id, recv_buffer)

        # State transition: connect
        event = (
            StateEvent.RW_CONNECT
            if mode == ConnectionMode.RW
            else StateEvent.RO_CONNECT
        )
        self._sm.transition(event, conn)

        await send_message(
            writer, HandshakeResponse(success=True, committed=self._sm.committed)
        )
        return conn

    async def _acquire_lock(
        self, mode: ConnectionMode, timeout_ms: Optional[int]
    ) -> bool:
        """Wait until lock can be acquired (uses state machine predicates)."""
        timeout = timeout_ms / 1000 if timeout_ms is not None else None

        if mode == ConnectionMode.RW:
            self._waiting_writers += 1
            try:
                async with self._condition:
                    try:
                        await asyncio.wait_for(
                            self._condition.wait_for(
                                lambda: self._shutdown or self._sm.can_acquire_rw()
                            ),
                            timeout=timeout,
                        )
                        return not self._shutdown
                    except asyncio.TimeoutError:
                        return False
            finally:
                self._waiting_writers -= 1
        else:
            async with self._condition:
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(
                            lambda: self._shutdown
                            or self._sm.can_acquire_ro(self._waiting_writers)
                        ),
                        timeout=timeout,
                    )
                    return not self._shutdown
                except asyncio.TimeoutError:
                    return False

    async def _cleanup_connection(
        self, conn: Optional[Connection], session_id: str
    ) -> None:
        """Clean up after connection closes via state machine transition."""
        if conn is None:
            return

        # State transition: disconnect
        if conn.mode == ConnectionMode.RW:
            if self._sm.rw_conn is conn and not self._sm.committed:
                # RW abort - state machine callback handles cleanup
                self._sm.transition(StateEvent.RW_ABORT, conn)
            elif self._sm.rw_conn is conn:
                # Already committed, no transition needed (commit already did it)
                pass
        else:
            if conn in self._sm.ro_conns:
                self._sm.transition(StateEvent.RO_DISCONNECT, conn)

        await conn.close()
        async with self._condition:
            self._condition.notify_all()

    # ==================== Request Handling ====================

    async def _request_loop(self, conn: Connection) -> None:
        """Process requests until close or commit."""
        while self._running:
            try:
                # Server never receives FDs from clients, so no need for raw_socket
                msg, _, conn.recv_buffer = await recv_message(
                    conn.reader, conn.recv_buffer
                )
            except ConnectionResetError:
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Recv error: {e}")
                return

            if msg is None:
                continue

            try:
                response, fd, should_close = await self._dispatch(conn, msg)
                if response is not None:
                    try:
                        await send_message(conn.writer, response, fd)
                    finally:
                        if fd >= 0:
                            os.close(fd)
                if should_close:
                    return
            except Exception as e:
                logger.error(f"Request error: {e}")
                await send_message(conn.writer, ErrorResponse(error=str(e)))

    async def _dispatch(self, conn: Connection, msg) -> tuple[object, int, bool]:
        """Dispatch request to handler. Returns (response, fd, should_close).

        Permission checks are delegated to the state machine.
        """
        if isinstance(msg, CommitRequest):
            self._sm.check_operation(Operation.COMMIT, conn)
            return await self._handle_commit(conn)

        if isinstance(msg, GetLockStateRequest):
            self._sm.check_operation(Operation.GET_LOCK_STATE, conn)
            return (
                self._handler.handle_get_lock_state(
                    self._sm.rw_conn is not None,
                    self._sm.ro_count,
                    self._waiting_writers,
                    self._sm.committed,
                ),
                -1,
                False,
            )

        if isinstance(msg, GetAllocationStateRequest):
            self._sm.check_operation(Operation.GET_ALLOCATION_STATE, conn)
            return self._handler.handle_get_allocation_state(), -1, False

        if isinstance(msg, AllocateRequest):
            self._sm.check_operation(Operation.ALLOCATE, conn)
            return self._handler.handle_allocate(msg), -1, False

        if isinstance(msg, ExportRequest):
            self._sm.check_operation(Operation.EXPORT, conn)
            response, fd = self._handler.handle_export(msg.allocation_id)
            return response, fd, False

        if isinstance(msg, GetAllocationRequest):
            self._sm.check_operation(Operation.GET_ALLOCATION, conn)
            return self._handler.handle_get_allocation(msg), -1, False

        if isinstance(msg, ListAllocationsRequest):
            self._sm.check_operation(Operation.LIST_ALLOCATIONS, conn)
            return self._handler.handle_list_allocations(msg), -1, False

        if isinstance(msg, FreeRequest):
            self._sm.check_operation(Operation.FREE, conn)
            return self._handler.handle_free(msg), -1, False

        if isinstance(msg, ClearAllRequest):
            self._sm.check_operation(Operation.CLEAR_ALL, conn)
            return self._handler.handle_clear_all(), -1, False

        if isinstance(msg, MetadataPutRequest):
            self._sm.check_operation(Operation.METADATA_PUT, conn)
            return self._handler.handle_metadata_put(msg), -1, False

        if isinstance(msg, MetadataGetRequest):
            self._sm.check_operation(Operation.METADATA_GET, conn)
            return self._handler.handle_metadata_get(msg), -1, False

        if isinstance(msg, MetadataDeleteRequest):
            self._sm.check_operation(Operation.METADATA_DELETE, conn)
            return self._handler.handle_metadata_delete(msg), -1, False

        if isinstance(msg, MetadataListRequest):
            self._sm.check_operation(Operation.METADATA_LIST, conn)
            return self._handler.handle_metadata_list(msg), -1, False

        raise ValueError(f"Unknown request: {type(msg)}")

    async def _handle_commit(self, conn: Connection) -> tuple[object, int, bool]:
        """Handle commit via state machine transition - atomic with disconnect."""
        # Permission already checked in _dispatch
        # State transition: commit
        self._sm.transition(StateEvent.RW_COMMIT, conn)

        await send_message(conn.writer, CommitResponse(success=True))
        await conn.close()

        async with self._condition:
            self._condition.notify_all()

        return None, -1, True

    # ==================== Server Lifecycle ====================

    async def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=self.socket_path
        )
        self._running = True
        logger.info(f"Server started: {self.socket_path}")

    async def stop(self) -> None:
        self._running = False
        self._shutdown = True
        async with self._condition:
            self._condition.notify_all()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Close connections (bypassing state machine - this is shutdown)
        if self._sm.rw_conn:
            await self._sm.rw_conn.close()

        for conn in list(self._sm.ro_conns):
            await conn.close()

        self._handler.on_shutdown()

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        logger.info("Server stopped")

    async def serve_forever(self) -> None:
        await self.start()
        try:
            while self._running:
                await asyncio.sleep(1)
        finally:
            await self.stop()
