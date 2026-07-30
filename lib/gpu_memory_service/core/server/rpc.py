# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unix-domain transport for the typed GMS V1 protocol."""

from __future__ import annotations

import os
import socket
import socketserver
from pathlib import Path

from gpu_memory_service.core.protocol import (
    REQUEST_TYPES,
    ErrorResponse,
    HandshakeRequest,
    HandshakeResponse,
    Message,
    receive_message,
    send_message,
)
from gpu_memory_service.core.server.gms import GMSServerMemoryManager
from gpu_memory_service.core.server.lease import socket_is_alive
from gpu_memory_service.core.server.sessions import ServerSession


class _GMSRequestHandler(socketserver.BaseRequestHandler):
    server: GMSRPCServer

    def handle(self) -> None:
        session: ServerSession | None = None
        try:
            request = self._receive()
            if not isinstance(request, HandshakeRequest):
                raise RuntimeError("expected GMS handshake")
            manager = self.server.manager
            if (
                request.expected_identity is not None
                and request.expected_identity != manager.identity
            ):
                send_message(
                    self.request,
                    ErrorResponse("GMS server incarnation or physical GPU changed"),
                )
                return
            session = manager.acquire(
                request.lock_type,
                lambda: not socket_is_alive(self.request),
            )
            if session is None:
                return
            nonce, gpu_uuid = manager.identity
            send_message(
                self.request,
                HandshakeResponse(session.mode, nonce, gpu_uuid),
            )

            while True:
                try:
                    request = self._receive()
                except EOFError:
                    return
                export_fd = -1
                try:
                    if not isinstance(request, REQUEST_TYPES):
                        raise RuntimeError(
                            "handshake is valid only as the first message"
                        )
                    response, export_fd = manager.handle_request(session, request)
                    send_message(self.request, response, export_fd)
                except Exception as exc:
                    try:
                        send_message(
                            self.request,
                            ErrorResponse(
                                str(exc),
                                out_of_memory=isinstance(exc, MemoryError),
                            ),
                        )
                    except Exception:
                        return
                finally:
                    if export_fd >= 0:
                        os.close(export_fd)
        except Exception:
            return
        finally:
            if session is not None:
                self.server.manager.close(session)

    def _receive(self) -> Message:
        request, received_fd = receive_message(self.request)
        if received_fd >= 0:
            os.close(received_fd)
            raise RuntimeError("GMS clients must not send file descriptors")
        return request


class GMSRPCServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, manager: GMSServerMemoryManager):
        self.path = path
        self.manager = manager
        self._prepare_socket_path()
        super().__init__(path, _GMSRequestHandler)
        os.chmod(path, 0o600)

    def _prepare_socket_path(self) -> None:
        if not os.path.exists(self.path):
            return

        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(self.path)
        except OSError:
            if os.path.exists(self.path):
                os.unlink(self.path)
            return
        finally:
            probe.close()

        raise RuntimeError(f"GMS already running at {self.path}")

    def server_close(self) -> None:
        super().server_close()
        Path(self.path).unlink(missing_ok=True)
