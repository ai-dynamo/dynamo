# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async GMS RPC transport server."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from typing import Optional

from gpu_memory_service.common.protocol.messages import (
    ErrorResponse,
    HandshakeRequest,
    HandshakeResponse,
)
from gpu_memory_service.common.protocol.wire import recv_message, send_message
from gpu_memory_service.common.utils import fail

from .allocations import AllocationNotFoundError
from .gms import GMS
from .session import Connection, InvalidTransition, OperationNotAllowed

logger = logging.getLogger(__name__)


def _is_connection_alive(conn: Connection) -> bool:
    if conn.writer.is_closing():
        return False
    if conn.reader.at_eof() or conn.reader.exception() is not None:
        return False
    sock = getattr(conn.raw_socket, "_sock", conn.raw_socket)
    try:
        data = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
    except BlockingIOError:
        return True
    except OSError:
        return False
    return data != b""


class GMSRPCServer:
    """Unix-socket transport for the GPU Memory Service."""

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
        self._gms = GMS(
            device,
            allocation_retry_interval=allocation_retry_interval,
            allocation_retry_timeout=allocation_retry_timeout,
        )
        self._server: Optional[asyncio.Server] = None
        logger.info("GMSRPCServer initialized: device=%d", device)

    @property
    def state(self):
        return self._gms.state

    def is_ready(self) -> bool:
        return self._gms.is_ready()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        conn: Optional[Connection] = None
        session_id = self._gms.next_session_id()
        try:
            conn = await self._do_handshake(reader, writer, session_id)
            if conn is None:
                return
            await self._request_loop(conn)
        except (InvalidTransition, AssertionError) as exc:
            fail("fatal server error", exc_info=exc)
        except ConnectionResetError:
            logger.debug("Connection reset: %s", session_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            fail("fatal server error", exc_info=exc)
        finally:
            try:
                await self._gms.cleanup_connection(conn)
            except Exception as exc:
                fail("fatal server error", exc_info=exc)

    async def _do_handshake(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        session_id: str,
    ) -> Optional[Connection]:
        try:
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

        granted_mode = await self._gms.acquire_lock(
            msg.lock_type,
            msg.timeout_ms,
            session_id,
        )
        if granted_mode is None:
            try:
                await send_message(
                    writer,
                    HandshakeResponse(success=False, committed=self._gms.committed),
                )
            except Exception:
                pass
            writer.close()
            return None

        try:
            conn = Connection(
                reader=reader,
                writer=writer,
                mode=granted_mode,
                session_id=session_id,
                recv_buffer=recv_buffer,
            )
            self._gms.on_connect(conn)
        except Exception:
            await self._gms.cancel_connect(session_id, granted_mode)
            raise

        try:
            await send_message(
                writer,
                HandshakeResponse(
                    success=True,
                    committed=self._gms.committed,
                    granted_lock_type=granted_mode,
                ),
            )
        except Exception as exc:
            logger.warning(
                "Handshake failed after acquiring %s for session %s: %s",
                granted_mode.value,
                session_id,
                exc,
            )
            await self._gms.cleanup_connection(conn)
            return None

        return conn

    async def _request_loop(self, conn: Connection) -> None:
        while True:
            try:
                msg, _, conn.recv_buffer = await recv_message(
                    conn.reader, conn.recv_buffer
                )
            except ConnectionResetError:
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Recv error on session %s: %s", conn.session_id, exc)
                return

            if msg is None:
                continue

            fd = -1
            try:
                response, fd, should_close = await self._gms.handle_request(
                    conn,
                    msg,
                    lambda: _is_connection_alive(conn),
                )
            except ConnectionAbortedError as exc:
                logger.warning(
                    "Connection lost during %s on session %s: %s",
                    type(msg).__name__,
                    conn.session_id,
                    exc,
                )
                return
            except (
                OperationNotAllowed,
                ValueError,
                TimeoutError,
                AllocationNotFoundError,
            ) as exc:
                logger.warning(
                    "Rejected %s from session %s: %s",
                    type(msg).__name__,
                    conn.session_id,
                    exc,
                )
                try:
                    await send_message(conn.writer, ErrorResponse(error=str(exc)))
                except Exception as send_exc:
                    logger.warning(
                        "Failed to send ErrorResponse for %s on session %s: %s",
                        type(msg).__name__,
                        conn.session_id,
                        send_exc,
                    )
                    return
                continue
            except (InvalidTransition, AssertionError) as exc:
                fail("fatal server error", exc_info=exc)
            except Exception as exc:
                fail("fatal server error", exc_info=exc)

            try:
                await send_message(conn.writer, response, fd)
            except Exception as exc:
                logger.warning(
                    "Response send failed for %s on session %s: %s",
                    type(msg).__name__,
                    conn.session_id,
                    exc,
                )
                return
            finally:
                if fd >= 0:
                    os.close(fd)

            if should_close:
                return

    async def serve(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=self.socket_path,
        )
        logger.info("Server started: %s", self.socket_path)
        await self._server.serve_forever()
