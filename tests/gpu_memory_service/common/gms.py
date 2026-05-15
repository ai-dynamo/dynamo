# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Socket-level GMS helpers for the cross-component test suite."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
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
    """Subprocess GMS server wrapper.

    GMSRPCServer opens CUDA contexts on the host device. The NVIDIA driver
    tracks those contexts by PID, not by Python thread, so hosting the server
    in pytest pinned ~2.4 GiB of device memory (plus ~150 /dev/nvidia* fds)
    to the pytest PID for the entire session. Running the server in a
    subprocess confines that state: subprocess exit atomically closes every
    fd and drops every context, giving downstream tests a clean GPU.
    """

    def __init__(self, device: int, tag: str = "weights"):
        self.device = device
        self.socket_path = get_socket_path(device, tag)
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "GMSServer":
        self._reclaim_stale_socket()
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.gpu_memory_service.common._gms_server_runner",
                self.socket_path,
                "--device",
                str(self.device),
            ],
            start_new_session=True,
        )
        self._wait_until_serving()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._stop_subprocess()
        finally:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

    def _reclaim_stale_socket(self) -> None:
        """Detect and clear a socket file left behind by a crashed prior run.

        If a live server already owns the socket, refuse to start; two servers
        on the same path would race the test. If nothing answers, the socket is
        a stale artifact and we unlink it so Popen can bind cleanly.
        """
        if not os.path.exists(self.socket_path):
            return
        try:
            self.get_runtime_state()
        except OSError:
            os.unlink(self.socket_path)
            return
        raise RuntimeError(f"GMS already active at {self.socket_path}")

    def _wait_until_serving(self) -> None:
        """Block until the subprocess's RPC socket answers, or raise.

        Three terminating conditions, all explicit:
          - subprocess exits early      -> RuntimeError with its returncode
          - socket answers a probe RPC  -> return (success)
          - deadline elapses            -> TimeoutError (chained to last probe error)
        """
        assert self._proc is not None
        deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
        last_probe_error: OSError | None = None
        while True:
            rc = self._proc.poll()
            if rc is not None:
                raise RuntimeError(
                    f"GMS server subprocess exited early (rc={rc}) "
                    f"before opening {self.socket_path}"
                )
            if os.path.exists(self.socket_path):
                try:
                    self.get_runtime_state()
                    return
                except OSError as exc:
                    last_probe_error = exc
            if time.monotonic() > deadline:
                msg = f"GMS socket did not appear at {self.socket_path}"
                raise TimeoutError(msg) from last_probe_error
            time.sleep(_POLL_INTERVAL_SECONDS)

    def _stop_subprocess(self) -> None:
        """SIGTERM, wait, SIGKILL if still alive. Tolerant of an already-dead
        subprocess (shadow_failover tests may have SIGKILL'd it mid-test)."""
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=_SERVER_STOP_TIMEOUT_SECONDS)

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
