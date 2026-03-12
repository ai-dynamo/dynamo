# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async Allocation RPC Server - Single-threaded event loop with explicit state machine.

State transitions are explicit and validated by the GMSLocalFSM class.
Operations are checked against state/mode permissions before operation.

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

from gpu_memory_service.common.protocol.messages import (
    AllocateRequest,
    CommitRequest,
    CommitResponse,
    ErrorResponse,
    ExportAllocationRequest,
    FreeAllocationRequest,
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
from gpu_memory_service.common.utils import fail

from .handler import RequestHandler
from .locking import Connection, GMSLocalFSM, InvalidTransition, OperationNotAllowed
from .memory_manager import AllocationNotFoundError

logger = logging.getLogger(__name__)


class GMSRPCServer:
    """GPU Memory Service RPC Server.

    Async single-threaded server using GMSLocalFSM for explicit state transitions
    and operation validation. All state mutations happen through the state machine's
    transition() method.
    """

    def __init__(
        self,
        socket_path: str,
        device: int = 0,
        *,
        allocation_retry_interval: float = 0.5,
        allocation_retry_timeout: Optional[float] = None,
    ):
        self.socket_path = socket_path
        self.device = device

        # Request handler (business logic)
        self._handler = RequestHandler(
            device,
            allocation_retry_interval=allocation_retry_interval,
            allocation_retry_timeout=allocation_retry_timeout,
        )

        # State machine - handles all state transitions and permission checks
        self._sm = GMSLocalFSM(on_rw_abort=self._handler.on_rw_abort)
        self._waiting_writers: int = 0

        # Async waiting for lock acquisition
        self._condition = asyncio.Condition()

        # Session ID generation
        self._next_session_id: int = 0

        # Server state
        self._server: Optional[asyncio.Server] = None

        logger.info(f"GMSRPCServer initialized: device={device}")

    # ==================== State Properties ====================

    @property
    def state(self) -> ServerState:
        """Current server state (delegated to state machine)."""
        return self._sm.state

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
        except (InvalidTransition, AssertionError) as exc:
            fail("fatal server error", exc_info=exc)
        except ConnectionResetError:
            logger.debug(f"Connection reset: {session_id}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            fail("fatal server error", exc_info=exc)
        finally:
            try:
                await self._cleanup_connection(conn)
            except Exception as exc:
                fail("fatal server error", exc_info=exc)

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
            try:
                await send_message(
                    writer, ErrorResponse(error="Expected HandshakeRequest")
                )
            except Exception:
                pass
            writer.close()
            return None

        # Acquire lock (blocks until available or timeout)
        # Returns the actual granted mode (may differ from requested for rw_or_ro)
        granted_mode = await self._acquire_lock(msg.lock_type, msg.timeout_ms)
        if granted_mode is None:
            try:
                await send_message(
                    writer,
                    HandshakeResponse(success=False, committed=self._sm.committed),
                )
            except Exception:
                pass
            writer.close()
            return None

        conn = Connection(
            reader=reader,
            writer=writer,
            mode=granted_mode,
            session_id=session_id,
            recv_buffer=recv_buffer,
        )
        rw_epoch_initialized = False
        fsm_transitioned = False

        try:
            if granted_mode == GrantedLockType.RW:
                # Intentional ordering: initialize epoch state via on_rw_connect()
                # before _sm.transition(StateEvent.RW_CONNECT, ...). If the FSM
                # transition fails, we must abort the partially initialized epoch.
                self._handler.on_rw_connect()
                rw_epoch_initialized = True

            event = (
                StateEvent.RW_CONNECT
                if granted_mode == GrantedLockType.RW
                else StateEvent.RO_CONNECT
            )
            self._sm.transition(event, conn)
            fsm_transitioned = True

            await send_message(
                writer,
                HandshakeResponse(
                    success=True,
                    committed=self._sm.committed,
                    granted_lock_type=granted_mode,
                ),
            )
        except (InvalidTransition, AssertionError):
            if fsm_transitioned:
                await self._cleanup_connection(conn)
            else:
                if rw_epoch_initialized:
                    self._handler.on_rw_abort()
                await conn.close()
            raise
        except Exception as e:
            logger.warning(
                "Handshake failed after acquiring %s for session %s: %s",
                granted_mode.value,
                session_id,
                e,
            )
            if fsm_transitioned:
                await self._cleanup_connection(conn)
            else:
                if rw_epoch_initialized:
                    self._handler.on_rw_abort()
                await conn.close()
            return None

        return conn

    async def _acquire_lock(
        self,
        mode: RequestedLockType,
        timeout_ms: Optional[int],
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
                            self._condition.wait_for(self._sm.can_acquire_rw),
                            timeout=timeout,
                        )
                        return GrantedLockType.RW
                    except asyncio.TimeoutError:
                        return None
            finally:
                self._waiting_writers -= 1

        elif mode == RequestedLockType.RO:
            async with self._condition:
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(
                            lambda: self._sm.can_acquire_ro(self._waiting_writers)
                        ),
                        timeout=timeout,
                    )
                    return GrantedLockType.RO
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
                            lambda: self._sm.can_acquire_ro(self._waiting_writers)
                        ),
                        timeout=timeout,
                    )
                    return GrantedLockType.RO
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
        while True:
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
                logger.warning("Recv error on session %s: %s", conn.session_id, e)
                return

            if msg is None:
                continue

            fd = -1
            try:
                response, fd, should_close = await self._dispatch(conn, msg)
            except ConnectionAbortedError as e:
                logger.warning(
                    "Connection lost during %s on session %s: %s",
                    type(msg).__name__,
                    conn.session_id,
                    e,
                )
                return
            except (
                OperationNotAllowed,
                ValueError,
                TimeoutError,
                AllocationNotFoundError,
            ) as e:
                logger.warning(
                    "Rejected %s from session %s: %s",
                    type(msg).__name__,
                    conn.session_id,
                    e,
                )
                try:
                    await send_message(conn.writer, ErrorResponse(error=str(e)))
                except Exception as send_error:
                    logger.warning(
                        "Failed to send ErrorResponse for %s on session %s: %s",
                        type(msg).__name__,
                        conn.session_id,
                        send_error,
                    )
                    return
                continue
            except (InvalidTransition, AssertionError) as exc:
                fail("fatal server error", exc_info=exc)
            except Exception as exc:
                fail("fatal server error", exc_info=exc)

            try:
                if response is not None:
                    await send_message(conn.writer, response, fd)
            except Exception as e:
                logger.warning(
                    "Response send failed for %s on session %s: %s",
                    type(msg).__name__,
                    conn.session_id,
                    e,
                )
                return
            finally:
                if fd >= 0:
                    os.close(fd)

            if should_close:
                return

    async def _dispatch(self, conn: Connection, msg) -> tuple[object, int, bool]:
        """Dispatch request to handler. Returns (response, fd, should_close)."""
        msg_type = type(msg)
        self._sm.check_operation(msg_type, conn)

        if msg_type is CommitRequest:
            self._handler.on_commit()
            self._sm.transition(StateEvent.RW_COMMIT, conn)
            return CommitResponse(success=True), -1, True
        elif msg_type is AllocateRequest:
            return (
                await self._handler.handle_allocate(
                    msg,
                    lambda: not conn.reader.at_eof()
                    and conn.reader.exception() is None
                    and not conn.writer.is_closing(),
                ),
                -1,
                False,
            )
        elif msg_type is GetLockStateRequest:
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
        elif msg_type is GetAllocationStateRequest:
            return self._handler.handle_get_allocation_state(), -1, False
        elif msg_type is ExportAllocationRequest:
            response, fd = self._handler.handle_export(msg, conn.mode)
            return response, fd, False
        elif msg_type is GetStateHashRequest:
            return self._handler.handle_get_memory_layout_hash(), -1, False
        elif msg_type is GetAllocationRequest:
            return self._handler.handle_get_allocation(msg, conn.mode), -1, False
        elif msg_type is ListAllocationsRequest:
            return self._handler.handle_list_allocations(msg, conn.mode), -1, False
        elif msg_type is MetadataGetRequest:
            return self._handler.handle_metadata_get(msg, conn.mode), -1, False
        elif msg_type is MetadataListRequest:
            return self._handler.handle_metadata_list(msg, conn.mode), -1, False
        elif msg_type is FreeAllocationRequest:
            return self._handler.handle_free(msg), -1, False
        elif msg_type is MetadataPutRequest:
            return self._handler.handle_metadata_put(msg), -1, False
        elif msg_type is MetadataDeleteRequest:
            return self._handler.handle_metadata_delete(msg), -1, False
        else:
            raise ValueError(f"Unknown request: {msg_type.__name__}")

    # ==================== Server Lifecycle ====================

    async def serve(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=self.socket_path
        )
        logger.info(f"Server started: {self.socket_path}")
        await self._server.serve_forever()
