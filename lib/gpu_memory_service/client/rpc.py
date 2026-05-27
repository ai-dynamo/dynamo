# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal GPU Memory Service transport.

This module only owns Unix socket transport and typed request/response exchange.
Session semantics live in `gpu_memory_service.client.session`.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from typing import Optional, Tuple, Type, TypeVar

from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.protocol.messages import (
    ErrorResponse,
    HandshakeRequest,
    HandshakeResponse,
)
from gpu_memory_service.common.protocol.wire import (
    recv_message_sync,
    recv_message_sync_with_fds,
    send_message_sync,
)

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

    def connect(self, timeout_ms: int = 0) -> None:
        """Open the UDS socket, retrying while the server is not yet listening.

        timeout_ms=0 (default) fails fast like the pre-retry behavior. A
        positive value retries on FileNotFoundError / ConnectionRefusedError
        until the deadline.
        """
        deadline = time.monotonic() + timeout_ms / 1000
        logged_wait = False
        while True:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                self._socket.connect(self.socket_path)
                if logged_wait:
                    logger.info("Connected to GMS server at %s", self.socket_path)
                return
            except (FileNotFoundError, ConnectionRefusedError):
                self._socket.close()
                self._socket = None
                if timeout_ms <= 0 or time.monotonic() >= deadline:
                    raise ConnectionError(
                        f"GMS server not running at {self.socket_path}"
                    ) from None
                if not logged_wait:
                    logger.info("Waiting for GMS server at %s", self.socket_path)
                    logged_wait = True
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
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
        response, fd = self.request_with_fd(request, response_type)
        if fd >= 0:
            os.close(fd)
            raise RuntimeError(
                f"GMS request {type(request).__name__} returned an unexpected FD"
            )
        return response

    def request_with_fd(
        self,
        request,
        response_type: Type[T],
        *,
        error_prefix: Optional[str] = None,
    ) -> Tuple[T, int]:
        response, fds = self.request_with_fds(
            request, response_type, error_prefix=error_prefix
        )
        for extra_fd in fds[1:]:
            os.close(extra_fd)
        return response, (fds[0] if fds else -1)

    def request_with_fds(
        self,
        request,
        response_type: Type[T],
        *,
        error_prefix: Optional[str] = None,
    ) -> Tuple[T, list[int]]:
        response, fds = self._send_recv_fds(request, error_prefix=error_prefix)
        if not isinstance(response, response_type):
            prefix = error_prefix or f"GMS request {type(request).__name__}"
            for fd in fds:
                os.close(fd)
            raise RuntimeError(
                f"{prefix} returned unexpected response type: {type(response)}"
            )
        return response, fds

    def _send_recv(
        self, request, *, error_prefix: Optional[str] = None
    ) -> Tuple[object, int]:
        response, fds = self._send_recv_fds(request, error_prefix=error_prefix)
        for extra_fd in fds[1:]:
            os.close(extra_fd)
        return response, (fds[0] if fds else -1)

    def _send_recv_fds(
        self, request, *, error_prefix: Optional[str] = None
    ) -> Tuple[object, list[int]]:
        if self._socket is None:
            raise RuntimeError("Attempted GMS request on disconnected transport")

        prefix = error_prefix or f"GMS request {type(request).__name__}"
        try:
            send_message_sync(self._socket, request)
            response, fds, self._recv_buffer = recv_message_sync_with_fds(
                self._socket, self._recv_buffer
            )
        except Exception as exc:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
            raise ConnectionError(f"{prefix} failed: {exc}") from exc

        if isinstance(response, ErrorResponse):
            for fd in fds:
                os.close(fd)
            raise RuntimeError(f"{prefix} error: {response.error}")
        return response, fds

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
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
            logger.warning("_GMSRPCTransport not closed properly")
