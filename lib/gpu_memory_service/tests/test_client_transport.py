# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS client transport behavior."""

from __future__ import annotations

import socket
import threading
import time

import pytest
from gpu_memory_service.client.rpc import _GMSRPCTransport


def test_transport_connect_waits_for_socket(tmp_path):
    socket_path = str(tmp_path / "gms.sock")
    accepted = threading.Event()

    def serve_once() -> None:
        time.sleep(0.1)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(socket_path)
            server.listen(1)
            conn, _ = server.accept()
            with conn:
                accepted.set()

    thread = threading.Thread(target=serve_once)
    thread.start()
    try:
        with _GMSRPCTransport(socket_path) as transport:
            transport.connect(timeout_ms=1_000)
            assert transport.is_connected
        assert accepted.wait(timeout=1.0)
    finally:
        thread.join(timeout=1.0)


def test_transport_connect_times_out_for_missing_socket(tmp_path):
    start = time.monotonic()
    with pytest.raises(ConnectionError):
        _GMSRPCTransport(str(tmp_path / "missing.sock")).connect(timeout_ms=50)
    assert time.monotonic() - start < 1.0
