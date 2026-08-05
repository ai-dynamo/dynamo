# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private typed GMS client session and socket lease."""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import TypeVar

from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.v1.protocol import (
    AbortRequest,
    AllocateRequest,
    CommitRequest,
    ErrorResponse,
    ExportRequest,
    ExportResponse,
    FreeRequest,
    HandshakeRequest,
    HandshakeResponse,
    Message,
    SuccessResponse,
    receive_message,
    send_message,
)

T = TypeVar("T")

_STARTUP_CONNECT_RETRY_INTERVAL = 0.1


class _GMSClientSession:
    """One connected, handshaken GMS socket session."""

    def __init__(
        self,
        path: str,
        lock_type: RequestedLockType,
        expected_identity: tuple[str, str] | None = None,
        connect_timeout: float | None = 30.0,
        admission_timeout: float | None = None,
    ):
        if connect_timeout is not None and connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")
        if admission_timeout is not None and admission_timeout <= 0:
            raise ValueError("admission_timeout must be positive")
        self._lock = threading.RLock()
        self._socket: socket.socket | None = None
        deadline = (
            None if connect_timeout is None else time.monotonic() + connect_timeout
        )
        try:
            # Only startup socket availability is retried; handshake failures propagate.
            while True:
                self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    self._socket.connect(path)
                    break
                except (FileNotFoundError, ConnectionRefusedError) as cause:
                    self._socket.close()
                    self._socket = None
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        raise ConnectionError(
                            f"Timed out waiting for GMS sidecar socket at {path}"
                        ) from cause
                    time.sleep(
                        _STARTUP_CONNECT_RETRY_INTERVAL
                        if remaining is None
                        else min(_STARTUP_CONNECT_RETRY_INTERVAL, remaining)
                    )
            send_message(self._socket, HandshakeRequest(lock_type, expected_identity))
            if admission_timeout is not None:
                self._socket.settimeout(admission_timeout)
            try:
                response, received_fd = receive_message(self._socket)
            except TimeoutError as cause:
                raise ConnectionError(
                    "Timed out waiting for GMS lock admission"
                ) from cause
            finally:
                self._socket.settimeout(None)
            handshake = self._decode(
                "handshake",
                response,
                received_fd,
                HandshakeResponse,
            )
            self._granted_lock_type = handshake.lock_type
            self._identity = (handshake.server_nonce, handshake.gpu_uuid)
        except (Exception, KeyboardInterrupt):
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            raise

    @property
    def identity(self) -> tuple[str, str]:
        return self._identity

    @property
    def lock_type(self) -> GrantedLockType:
        return self._granted_lock_type

    def allocate(self, allocation_id: str, aligned_size: int) -> None:
        self._call(
            AllocateRequest(allocation_id, aligned_size),
            SuccessResponse,
        )

    def export(self, allocation_id: str) -> int:
        _response, fd = self._call(
            ExportRequest(allocation_id),
            ExportResponse,
            expect_fd=True,
        )
        return fd

    def free(self, allocation_id: str) -> None:
        self._call(FreeRequest(allocation_id), SuccessResponse)

    def commit(self) -> None:
        self._call(CommitRequest(), SuccessResponse)
        self._granted_lock_type = GrantedLockType.RO

    def close(self) -> None:
        with self._lock:
            try:
                if (
                    self._socket is not None
                    and self._granted_lock_type is GrantedLockType.RW
                ):
                    self._call(AbortRequest(), SuccessResponse)
            finally:
                if self._socket is not None:
                    self._socket.close()
                    self._socket = None

    def _call(
        self,
        request: Message,
        response_type: type[T],
        *,
        expect_fd: bool = False,
    ) -> T | tuple[T, int]:
        operation = type(request).__name__
        with self._lock:
            if self._socket is None:
                raise RuntimeError("GMS session is disconnected")
            try:
                send_message(self._socket, request)
                response, received_fd = receive_message(self._socket)
            except (EOFError, OSError) as cause:
                self._socket.close()
                self._socket = None
                raise ConnectionError(f"GMS {operation} failed") from cause
            decoded = self._decode(
                operation,
                response,
                received_fd,
                response_type,
                expect_fd=expect_fd,
            )
            if expect_fd:
                return decoded, received_fd
            return decoded

    @staticmethod
    def _decode(
        operation: str,
        response: Message,
        received_fd: int,
        response_type: type[T],
        *,
        expect_fd: bool = False,
    ) -> T:
        try:
            if isinstance(response, ErrorResponse):
                if response.out_of_memory:
                    raise MemoryError(response.message)
                raise RuntimeError(response.message)  # noqa: TRY004
            if not isinstance(response, response_type):
                raise RuntimeError(  # noqa: TRY004
                    f"GMS {operation} returned {type(response).__name__}, "
                    f"expected {response_type.__name__}"
                )
            if expect_fd and received_fd < 0:
                raise RuntimeError(f"GMS {operation} did not return an FD")
            if not expect_fd and received_fd >= 0:
                raise RuntimeError(f"GMS {operation} returned an unexpected FD")
            return response
        except Exception:
            if received_fd >= 0:
                os.close(received_fd)
            raise
