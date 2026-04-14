# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Socket-level GMS helpers for the cross-component test suite."""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from typing import TYPE_CHECKING

from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.utils import get_socket_path

if TYPE_CHECKING:
    from gpu_memory_service.common.protocol.messages import (
        GetEventHistoryResponse,
        GetRuntimeStateResponse,
        ListAllocationsResponse,
    )

_SERVER_START_TIMEOUT_SECONDS = 30.0
_SERVER_STOP_TIMEOUT_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.1


def _request_gms(
    socket_path: str,
    request,
    response_type,
    *,
    lock_type: RequestedLockType | None = None,
    timeout_ms: int | None = None,
):
    """Send one raw request over a Unix socket, with optional lock handshake."""

    from gpu_memory_service.common.protocol.messages import (
        ErrorResponse,
        HandshakeRequest,
        HandshakeResponse,
    )
    from gpu_memory_service.common.protocol.wire import (
        recv_message_sync,
        send_message_sync,
    )

    recv_buffer = bytearray()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(socket_path)
        if lock_type is not None:
            send_message_sync(
                sock,
                HandshakeRequest(lock_type=lock_type, timeout_ms=timeout_ms),
            )
            handshake, fd, recv_buffer = recv_message_sync(sock, recv_buffer)
            if fd >= 0:
                os.close(fd)
                raise RuntimeError("GMS handshake returned an unexpected FD")
            if isinstance(handshake, ErrorResponse):
                raise RuntimeError(f"GMS handshake error: {handshake.error}")
            if not isinstance(handshake, HandshakeResponse):
                raise RuntimeError(
                    f"Unexpected handshake response: {type(handshake).__name__}"
                )
            if not handshake.success:
                raise TimeoutError("Timeout waiting for GMS lock")

        send_message_sync(sock, request)
        response, fd, recv_buffer = recv_message_sync(sock, recv_buffer)
        if isinstance(response, ErrorResponse):
            raise RuntimeError(f"GMS request error: {response.error}")
        if not isinstance(response, response_type):
            raise RuntimeError(f"Unexpected response type: {type(response).__name__}")
        if fd >= 0:
            os.close(fd)
            raise RuntimeError(
                f"GMS request {type(request).__name__} returned an unexpected FD"
            )
        return response
    finally:
        sock.close()


def list_allocations(socket_path: str) -> ListAllocationsResponse:
    from gpu_memory_service.common.protocol.messages import (
        ListAllocationsRequest,
        ListAllocationsResponse,
    )

    return _request_gms(
        socket_path,
        ListAllocationsRequest(),
        ListAllocationsResponse,
        lock_type=RequestedLockType.RO,
    )


class GMSServer:
    """In-process GMS server wrapper."""

    def __init__(self, device: int, tag: str = "weights"):
        from gpu_memory_service.server.rpc import GMSRPCServer

        self.socket_path = get_socket_path(device, tag)
        self.server = GMSRPCServer(self.socket_path, device=device)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._exception: BaseException | None = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._task = loop.create_task(self.server.serve())
        try:
            loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            self._exception = exc
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

    def __enter__(self):
        if os.path.exists(self.socket_path):
            try:
                self.get_runtime_state()
            except OSError:
                if os.path.exists(self.socket_path):
                    os.unlink(self.socket_path)
            else:
                raise RuntimeError(f"GMS already active at {self.socket_path}")
        self._thread.start()
        deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
        last_probe_error: OSError | None = None
        while True:
            if self._exception is not None:
                raise self._exception
            if self.server._server is not None and os.path.exists(self.socket_path):
                try:
                    self.get_runtime_state()
                    return self
                except OSError as exc:
                    last_probe_error = exc
            if time.monotonic() > deadline:
                timeout_error = TimeoutError(
                    f"GMS socket did not appear at {self.socket_path}"
                )
                if last_probe_error is not None:
                    raise timeout_error from last_probe_error
                raise timeout_error
            time.sleep(_POLL_INTERVAL_SECONDS)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._loop is not None:

            def cancel() -> None:
                if self.server._server is not None:
                    self.server._server.close()
                if self._task is not None:
                    self._task.cancel()

            self._loop.call_soon_threadsafe(cancel)
        self._thread.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError(
                f"GMS server thread failed to stop for {self.socket_path}"
            )
        if self._exception is not None:
            raise self._exception
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def get_runtime_state(self) -> GetRuntimeStateResponse:
        from gpu_memory_service.common.protocol.messages import (
            GetRuntimeStateRequest,
            GetRuntimeStateResponse,
        )

        return _request_gms(
            self.socket_path,
            GetRuntimeStateRequest(),
            GetRuntimeStateResponse,
        )

    def get_event_history(self) -> GetEventHistoryResponse:
        from gpu_memory_service.common.protocol.messages import (
            GetEventHistoryRequest,
            GetEventHistoryResponse,
        )

        return _request_gms(
            self.socket_path,
            GetEventHistoryRequest(),
            GetEventHistoryResponse,
        )
