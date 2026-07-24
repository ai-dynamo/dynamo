# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialized GMS V1 transport with sidecar-incarnation enforcement."""

from __future__ import annotations

import os
import socket
import threading

from ..common.protocol import receive_message, send_message
from ..errors import FatalGMSError, GMSError


class AllocationClient:
    """Tiny allocation facade that safely replays stable mutations."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._fatal: FatalGMSError | None = None
        self._socket: socket.socket | None = self._connect()
        try:
            self._identity = self._hello_on(self._socket)
        except Exception:
            self._socket.close()
            self._socket = None
            raise

    def hello(self) -> tuple[str, str]:
        result = self._call("hello", [])
        identity = self._parse_identity(result)
        if identity != self._identity:
            raise self._latch("GMS V1 sidecar incarnation or physical GPU changed")
        return identity

    def allocate(self, allocation_id: str, aligned_size: int) -> None:
        self._call("allocate", [allocation_id, aligned_size])

    def export(self, allocation_id: str) -> int:
        result = self._call("export", [allocation_id], expect_fd=True)
        if not isinstance(result, int):
            raise GMSError("V1 export returned an invalid file descriptor")
        return result

    def free(self, allocation_id: str) -> None:
        self._call("free", [allocation_id])

    def disconnect(self) -> None:
        """Close the live stream before CRIU capture."""
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None

    def reconnect(self) -> tuple[str, str]:
        """Open a fresh stream and verify the original sidecar identity."""
        self._check()
        with self._lock:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            self._socket = self._connect()
            try:
                identity = self._hello_on(self._socket)
            except Exception:
                self._socket.close()
                self._socket = None
                raise
            if identity != self._identity:
                self._socket.close()
                self._socket = None
                raise self._latch("GMS V1 sidecar incarnation or physical GPU changed")
            return identity

    def close(self) -> None:
        self.disconnect()

    def _call(
        self, method: str, params: list[object], *, expect_fd: bool = False
    ) -> object:
        self._check()
        with self._lock:
            if self._socket is None:
                raise GMSError("GMS V1 client is disconnected")
            for attempt in range(2):
                try:
                    send_message(self._socket, [method, params])
                    response, received_fd = receive_message(self._socket)
                    return self._decode(method, response, received_fd, expect_fd)
                except (EOFError, OSError):
                    self._socket.close()
                    self._socket = None
                    if attempt:
                        raise
                    self._socket = self._connect()
                    try:
                        identity = self._hello_on(self._socket)
                    except Exception:
                        self._socket.close()
                        self._socket = None
                        raise
                    if identity != self._identity:
                        self._socket.close()
                        self._socket = None
                        raise self._latch(
                            "GMS V1 sidecar incarnation or physical GPU changed"
                        )
        raise GMSError("V1 RPC retry did not complete")

    def _hello_on(self, sock: socket.socket) -> tuple[str, str]:
        send_message(sock, ["hello", []])
        response, received_fd = receive_message(sock)
        result = self._decode("hello", response, received_fd, False)
        return self._parse_identity(result)

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
                raise GMSError("invalid V1 RPC response")
            if not response[0]:
                if len(response) != 3:
                    raise GMSError("invalid V1 RPC error response")
                error_type, message = response[1:]
                if not isinstance(error_type, str) or not isinstance(message, str):
                    raise GMSError("invalid V1 RPC error response")
                error = f"{error_type}: {message}"
                if error_type == "FatalGMSError":
                    raise FatalGMSError(error)
                raise GMSError(error)
            if len(response) != 2:
                raise GMSError("invalid V1 RPC success response")
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
    def _parse_identity(value: object) -> tuple[str, str]:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(item, str) and item for item in value)
        ):
            raise GMSError("invalid V1 hello response")
        return value[0], value[1]

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.path)
        except Exception:
            sock.close()
            raise
        return sock

    def _check(self) -> None:
        if self._fatal is not None:
            raise self._fatal

    def _latch(self, message: str) -> FatalGMSError:
        if self._fatal is None:
            self._fatal = FatalGMSError(message)
        return self._fatal
