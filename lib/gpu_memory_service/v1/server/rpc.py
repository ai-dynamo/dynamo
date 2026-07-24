# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unix-domain RPC server for the GMS V1 allocation store."""

from __future__ import annotations

import os
import socketserver
from pathlib import Path

from ..common.protocol import receive_message, send_message
from ..errors import GMSError
from .allocations import AllocationStore


class _AllocationRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        while True:
            try:
                request, received_fd = receive_message(self.request)
            except EOFError:
                return
            except Exception:
                return
            if received_fd >= 0:
                os.close(received_fd)
                return
            export_fd = -1
            try:
                result, export_fd = self.server.dispatch(request)  # type: ignore[attr-defined]
                send_message(self.request, [True, result], export_fd)
            except Exception as exc:
                try:
                    send_message(self.request, [False, type(exc).__name__, str(exc)])
                except Exception:
                    return
            finally:
                if export_fd >= 0:
                    os.close(export_fd)


class AllocationRPCServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(
        self,
        path: str,
        allocations: AllocationStore,
    ):
        self.path = path
        self.allocations = allocations
        super().__init__(path, _AllocationRequestHandler)
        os.chmod(path, 0o600)

    def server_close(self) -> None:
        super().server_close()
        Path(self.path).unlink(missing_ok=True)

    def dispatch(self, request: object) -> tuple[object, int]:
        if not isinstance(request, list) or len(request) != 2:
            raise GMSError("invalid V1 RPC request")
        method, params = request
        if not isinstance(method, str) or not isinstance(params, list):
            raise GMSError("invalid V1 RPC request")

        if method == "hello":
            self._expect(params, 0)
            return list(self.allocations.hello()), -1
        if method == "allocate":
            self._expect(params, 2)
            allocation_id = self._string(params[0], "allocation ID")
            if type(params[1]) is not int:
                raise GMSError("allocation size must be an integer")
            self.allocations.allocate(allocation_id, params[1])
            return None, -1
        if method == "export":
            self._expect(params, 1)
            return None, self.allocations.export(
                self._string(params[0], "allocation ID")
            )
        if method == "free":
            self._expect(params, 1)
            self.allocations.free(self._string(params[0], "allocation ID"))
            return None, -1
        raise GMSError(f"unknown V1 RPC method {method!r}")

    @staticmethod
    def _expect(params: list[object], count: int) -> None:
        if len(params) != count:
            raise GMSError("invalid V1 RPC parameters")

    @staticmethod
    def _string(value: object, name: str) -> str:
        if not isinstance(value, str) or not value:
            raise GMSError(f"{name} must be a non-empty string")
        return value
