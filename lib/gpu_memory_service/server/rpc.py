# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import ClassVar, Optional

from gpu_memory_service.common.protocol.messages import (
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
    GetStateHashRequest,
    HandshakeRequest,
    HandshakeResponse,
    ListAllocationsRequest,
    MetadataDeleteRequest,
    MetadataGetRequest,
    MetadataListRequest,
    MetadataPutRequest,
)
from gpu_memory_service.common.protocol.wire import recv_message, send_message
from gpu_memory_service.common.types import (
    GrantedLockType,
    RequestedLockType,
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

    @property
    def running(self) -> bool:
        """Whether the server is running."""
        return self._running

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
        except Exception:
            logger.exception(f"Connection error: {session_id}")
        finally:
            await self._cleanup_connection(conn)

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
        except Exception:
            logger.exception("Handshake recv error")
            return None

        if not isinstance(msg, HandshakeRequest):
            await send_message(writer, ErrorResponse(error="Expected HandshakeRequest"))
            writer.close()
            return None

        # Acquire lock (blocks until available or timeout)
        # Returns the actual granted mode (may differ from requested for rw_or_ro)
        granted_mode = await self._acquire_lock(msg.lock_type, msg.timeout_ms)
        if granted_mode is None:
            await send_message(
                writer, HandshakeResponse(success=False, committed=self._sm.committed)
            )
            writer.close()
            return None

        conn = Connection(reader, writer, granted_mode, session_id, recv_buffer)

        # State transition: connect
        event = (
            StateEvent.RW_CONNECT
            if granted_mode == GrantedLockType.RW
            else StateEvent.RO_CONNECT
        )
        self._sm.transition(event, conn)

        await send_message(
            writer,
            HandshakeResponse(
                success=True,
                committed=self._sm.committed,
                granted_lock_type=granted_mode,
            ),
        )
        return conn

    async def _acquire_lock(
        self, mode: RequestedLockType, timeout_ms: Optional[int]
    ) -> Optional[GrantedLockType]:
        """Wait until lock can be acquired (uses state machine predicates).

        Returns the granted lock type, or None if failed/timeout.
        For rw_or_ro mode, returns RW if available immediately, else waits for RO.
        """
        timeout = timeout_ms / 1000 if timeout_ms is not None else None

        if mode == RequestedLockType.RW:
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
                        return None if self._shutdown else GrantedLockType.RW
                    except asyncio.TimeoutError:
                        return None
            finally:
                self._waiting_writers -= 1

        elif mode == RequestedLockType.RO:
            async with self._condition:
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(
                            lambda: self._shutdown
                            or self._sm.can_acquire_ro(self._waiting_writers)
                        ),
                        timeout=timeout,
                    )
                    return None if self._shutdown else GrantedLockType.RO
                except asyncio.TimeoutError:
                    return None

        elif mode == RequestedLockType.RW_OR_RO:
            # Auto mode: try RW if available immediately AND no committed weights,
            # otherwise wait for RO (to import existing weights)
            async with self._condition:
                # Check if RW is available AND no committed weights exist
                # If weights are already committed, prefer RO to import them
                if self._sm.can_acquire_rw() and not self._sm.committed:
                    return GrantedLockType.RW

                # Either RW not available OR weights already committed - wait for RO
                if self._sm.committed:
                    logger.info(
                        "RW_OR_RO: Weights already committed, preferring RO to import"
                    )
                else:
                    logger.info(
                        "RW_OR_RO: RW not available (another writer active), "
                        "falling back to RO"
                    )
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(
                            lambda: self._shutdown
                            or self._sm.can_acquire_ro(self._waiting_writers)
                        ),
                        timeout=timeout,
                    )
                    return None if self._shutdown else GrantedLockType.RO
                except asyncio.TimeoutError:
                    return None
        return None

    async def _cleanup_connection(self, conn: Optional[Connection]) -> None:
        """Clean up after connection closes via state machine transition."""
        if conn is None:
            return

        # State transition: disconnect
        if conn.mode == GrantedLockType.RW:
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
            except Exception:
                logger.exception("Recv error")
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
                logger.exception("Request error")
                await send_message(conn.writer, ErrorResponse(error=str(e)))

    # Dispatch table: message type -> handler method name
    # Handlers take (msg) and return response. Special cases handled separately.
    _HANDLERS: ClassVar[dict[type, str]] = {
        AllocateRequest: "handle_allocate",
        GetAllocationRequest: "handle_get_allocation",
        ListAllocationsRequest: "handle_list_allocations",
        FreeRequest: "handle_free",
        MetadataPutRequest: "handle_metadata_put",
        MetadataGetRequest: "handle_metadata_get",
        MetadataDeleteRequest: "handle_metadata_delete",
        MetadataListRequest: "handle_metadata_list",
    }

    async def _dispatch(self, conn: Connection, msg) -> tuple[object, int, bool]:
        """Dispatch request to handler. Returns (response, fd, should_close)."""
        msg_type = type(msg)
        self._sm.check_operation(msg_type, conn)

        # Special cases
        if msg_type is CommitRequest:
            return await self._handle_commit(conn)

        if msg_type is GetLockStateRequest:
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

        if msg_type is GetAllocationStateRequest:
            return self._handler.handle_get_allocation_state(), -1, False

        if msg_type is ExportRequest:
            response, fd = self._handler.handle_export(msg.allocation_id)
            return response, fd, False

        if msg_type is ClearAllRequest:
            return self._handler.handle_clear_all(), -1, False

        if msg_type is GetStateHashRequest:
            return self._handler.handle_get_memory_layout_hash(), -1, False

        # Standard dispatch: handler takes msg, returns response
        handler_name = self._HANDLERS.get(msg_type)
        if handler_name:
            handler = getattr(self._handler, handler_name)
            return handler(msg), -1, False

        raise ValueError(f"Unknown request: {msg_type.__name__}")

    async def _handle_commit(self, conn: Connection) -> tuple[object, int, bool]:
        """Handle commit via state machine transition - atomic with disconnect."""
        # Compute state hash before transitioning
        self._handler.on_commit()
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
