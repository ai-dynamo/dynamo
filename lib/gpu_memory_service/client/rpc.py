# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal GPU Memory Service transport.

This module only owns Unix socket transport and typed request/response exchange.
Session semantics live in `gpu_memory_service.client.session`.
"""

from __future__ import annotations

import logging
import socket
from typing import Optional, Tuple, Type, TypeVar

from gpu_memory_service.common.protocol.messages import (
    ErrorResponse,
    HandshakeRequest,
    HandshakeResponse,
)
from gpu_memory_service.common.protocol.wire import recv_message_sync, send_message_sync
from gpu_memory_service.common.types import RequestedLockType

T = TypeVar("T")

logger = logging.getLogger(__name__)


class _GMSRPCTransport:
    """Raw GMS Unix socket transport."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._socket: Optional[socket.socket] = None
        self._recv_buffer = bytearray()

    @property
    def is_connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._socket.connect(self.socket_path)
        except FileNotFoundError:
            self._socket.close()
            self._socket = None
            raise ConnectionError(
                f"GMS server not running at {self.socket_path}"
            ) from None
        except Exception as exc:
            self._socket.close()
            self._socket = None
            raise ConnectionError(f"Failed to connect to GMS: {exc}") from exc

    def handshake(
        self,
        lock_type: RequestedLockType,
        timeout_ms: Optional[int],
    ) -> HandshakeResponse:
        response, _ = self.request_with_fd(
            HandshakeRequest(lock_type=lock_type, timeout_ms=timeout_ms),
            HandshakeResponse,
            error_prefix="GMS handshake",
        )
        return response

    def request(self, request, response_type: Type[T]) -> T:
        response, _ = self.request_with_fd(request, response_type)
        return response

    def request_with_fd(
        self,
        request,
        response_type: Type[T],
        *,
        error_prefix: Optional[str] = None,
    ) -> Tuple[T, int]:
        response, fd = self._send_recv(request, error_prefix=error_prefix)
        if not isinstance(response, response_type):
            prefix = error_prefix or f"GMS request {type(request).__name__}"
            raise RuntimeError(
                f"{prefix} returned unexpected response type: {type(response)}"
            )
        return response, fd

    def _send_recv(
        self, request, *, error_prefix: Optional[str] = None
    ) -> Tuple[object, int]:
        if self._socket is None:
            raise RuntimeError("Attempted GMS request on disconnected transport")

        prefix = error_prefix or f"GMS request {type(request).__name__}"
        try:
            send_message_sync(self._socket, request)
            response, fd, self._recv_buffer = recv_message_sync(
                self._socket, self._recv_buffer
            )
        except Exception as exc:
            raise ConnectionError(f"{prefix} failed: {exc}") from exc

        if isinstance(response, ErrorResponse):
            raise RuntimeError(f"{prefix} error: {response.error}")
        return response, fd

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        except Exception as exc:
            raise ConnectionError(
                f"Failed to close GMS transport socket: {exc}"
            ) from exc
        finally:
            self._socket = None

    def __enter__(self) -> "_GMSRPCTransport":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        if self._socket is not None:
            logger.warning("_GMSRPCTransport not closed properly")
