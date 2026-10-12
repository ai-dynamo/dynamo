# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TCP connection counter for SSRF tests."""

from __future__ import annotations

import socket
import socketserver
import threading
from contextlib import contextmanager
from typing import Iterator


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        assert isinstance(server, ConnectionCanary)
        server.record_connection()


class ConnectionCanary(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        self._count = 0
        self._lock = threading.Lock()
        self._connected = threading.Event()
        super().__init__(("127.0.0.1", 0), _Handler)

    def record_connection(self) -> None:
        with self._lock:
            self._count += 1
        self._connected.set()

    @property
    def connection_count(self) -> int:
        with self._lock:
            return self._count

    def blocked_url(self, path: str = "/private-image.png") -> str:
        """Return an HTTPS loopback URL that the media policy must reject."""
        host, port = self.server_address
        return f"https://{host}:{port}{path}"

    def touch(self) -> None:
        with socket.create_connection(self.server_address, timeout=5):
            pass

    def await_connection(self, timeout: float = 5.0) -> int:
        self._connected.wait(timeout)
        return self.connection_count

    def assert_no_connection(self, timeout: float = 0.25) -> None:
        """Wait briefly so delayed/background egress cannot escape detection."""
        if self._connected.wait(timeout):
            raise AssertionError(
                f"blocked destination received {self.connection_count} connection(s)"
            )


@contextmanager
def running_canary() -> Iterator[ConnectionCanary]:
    server = ConnectionCanary()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
