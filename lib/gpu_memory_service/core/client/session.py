# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connected GMS client session and socket lease."""

from __future__ import annotations

import os
import socket
import threading

from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType

from ..errors import FatalGMSError, GMSError
from ..protocol import receive_message, send_message


class GMSClientSession:
    """One connected, handshaken GMS socket session."""

    def __init__(
        self,
        path: str,
        lock_type: RequestedLockType,
        expected_identity: tuple[str, str] | None = None,
    ):
        self.path = path
        self._lock = threading.Lock()
        self._socket: socket.socket | None = socket.socket(
            socket.AF_UNIX, socket.SOCK_STREAM
        )
        try:
            self._socket.connect(path)
            send_message(
                self._socket,
                [
                    "handshake",
                    [
                        lock_type.value,
                        list(expected_identity) if expected_identity else None,
                    ],
                ],
            )
            response, received_fd = receive_message(self._socket)
            result = self._decode("handshake", response, received_fd, False)
            self._granted_lock_type, self._identity = self._parse_handshake(result)
        except Exception:
            self._socket.close()
            self._socket = None
            raise

    @property
    def identity(self) -> tuple[str, str]:
        return self._identity

    @property
    def lock_type(self) -> GrantedLockType:
        return self._granted_lock_type

    @property
    def is_connected(self) -> bool:
        return self._socket is not None

    def allocate(self, allocation_id: str, aligned_size: int) -> None:
        self._call("allocate", [allocation_id, aligned_size])

    def export(self, allocation_id: str) -> int:
        result = self._call("export", [allocation_id], expect_fd=True)
        if not isinstance(result, int):
            raise GMSError("GMS export returned an invalid file descriptor")
        return result

    def free(self, allocation_id: str) -> None:
        self._call("free", [allocation_id])

    def commit(self) -> None:
        self._call("commit", [])
        self._granted_lock_type = GrantedLockType.RO

    def close(self) -> None:
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None

    def _call(
        self, method: str, params: list[object], *, expect_fd: bool = False
    ) -> object:
        with self._lock:
            if self._socket is None:
                raise GMSError("GMS session is disconnected")
            try:
                send_message(self._socket, [method, params])
                response, received_fd = receive_message(self._socket)
            except (EOFError, OSError) as cause:
                self._socket.close()
                self._socket = None
                raise ConnectionError(f"GMS {method} failed") from cause
            return self._decode(method, response, received_fd, expect_fd)

    @staticmethod
    def _decode(
        method: str, response: object, received_fd: int, expect_fd: bool
    ) -> object:
        try:
            if (
                not isinstance(response, list)
                or len(response) not in (2, 3)
                or type(response[0]) is not bool
            ):
                raise GMSError("invalid GMS RPC response")
            if not response[0]:
                if len(response) != 3:
                    raise GMSError("invalid GMS RPC error response")
                error_type, message = response[1:]
                if not isinstance(error_type, str) or not isinstance(message, str):
                    raise GMSError("invalid GMS RPC error response")
                error = f"{error_type}: {message}"
                if error_type == "FatalGMSError":
                    raise FatalGMSError(error)
                raise GMSError(error)
            if len(response) != 2:
                raise GMSError("invalid GMS RPC success response")
            if expect_fd and received_fd < 0:
                raise GMSError(f"{method} did not return an FD")
            if not expect_fd and received_fd >= 0:
                raise GMSError(f"{method} returned an unexpected FD")
            return received_fd if expect_fd else response[1]
        except Exception:
            if received_fd >= 0:
                os.close(received_fd)
            raise

    @staticmethod
    def _parse_handshake(
        value: object,
    ) -> tuple[GrantedLockType, tuple[str, str]]:
        if (
            not isinstance(value, list)
            or len(value) != 3
            or not all(isinstance(item, str) and item for item in value)
        ):
            raise GMSError("invalid GMS handshake response")
        try:
            granted = GrantedLockType(value[0])
        except ValueError:
            raise GMSError("invalid GMS granted lock type") from None
        return granted, (value[1], value[2])

    def __enter__(self) -> "GMSClientSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
