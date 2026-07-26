# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unix-domain RPC server for baseline GMS sessions."""

from __future__ import annotations

import os
import socket
import socketserver
from pathlib import Path

from gpu_memory_service.common.locks import RequestedLockType

from ..errors import GMSError
from ..protocol import receive_message, send_message
from .gms import GMS
from .lease import socket_is_alive
from .sessions import ServerSession


class _GMSRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        session: ServerSession | None = None
        try:
            request = self._receive()
            method, params = self._parse(request)
            if method != "handshake" or len(params) != 2:
                raise GMSError("expected GMS handshake")
            try:
                requested = RequestedLockType(params[0])
            except (TypeError, ValueError):
                raise GMSError("invalid requested lock type") from None
            expected_identity = params[1]
            if expected_identity is not None:
                if (
                    not isinstance(expected_identity, list)
                    or len(expected_identity) != 2
                    or not all(
                        isinstance(item, str) and item for item in expected_identity
                    )
                ):
                    raise GMSError("invalid expected server identity")
                if tuple(expected_identity) != self.server.gms.identity:  # type: ignore[attr-defined]
                    send_message(
                        self.request,
                        [
                            False,
                            "FatalGMSError",
                            "GMS server incarnation or physical GPU changed",
                        ],
                    )
                    return
            session = self.server.gms.acquire(  # type: ignore[attr-defined]
                requested,
                lambda: not socket_is_alive(self.request),
            )
            if session is None:
                return
            nonce, gpu_uuid = self.server.gms.identity  # type: ignore[attr-defined]
            send_message(
                self.request,
                [True, [session.mode.value, nonce, gpu_uuid]],
            )

            while True:
                try:
                    request = self._receive()
                except EOFError:
                    return
                export_fd = -1
                try:
                    method, params = self._parse(request)
                    result, export_fd = self.server.gms.dispatch(  # type: ignore[attr-defined]
                        session, method, params
                    )
                    send_message(self.request, [True, result], export_fd)
                except Exception as exc:
                    try:
                        send_message(
                            self.request,
                            [False, type(exc).__name__, str(exc)],
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
                self.server.gms.close(session)  # type: ignore[attr-defined]

    def _receive(self) -> object:
        request, received_fd = receive_message(self.request)
        if received_fd >= 0:
            os.close(received_fd)
            raise GMSError("GMS clients must not send file descriptors")
        return request

    @staticmethod
    def _parse(request: object) -> tuple[str, list[object]]:
        if not isinstance(request, list) or len(request) != 2:
            raise GMSError("invalid GMS RPC request")
        method, params = request
        if not isinstance(method, str) or not isinstance(params, list):
            raise GMSError("invalid GMS RPC request")
        return method, params


class GMSRPCServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, gms: GMS):
        self.path = path
        self.gms = gms
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
