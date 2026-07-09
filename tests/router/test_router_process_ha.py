# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import socket
import socketserver
import threading
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.router.router_process import (
    FrontendRouterProcess,
    ValkeyPrimaryProxy,
    ValkeySentinelProcess,
)
from tests.utils.managed_process import ManagedProcess


class _TaggedHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while payload := self.rfile.readline():
            self.wfile.write(self.server.tag + b":" + payload)  # type: ignore[attr-defined]
            self.wfile.flush()


class _TaggedServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, tag: bytes):
        self.tag = tag
        super().__init__(("127.0.0.1", 0), _TaggedHandler)


def _round_trip(sock: socket.socket, payload: bytes) -> bytes:
    sock.sendall(payload + b"\n")
    response = bytearray()
    while not response.endswith(b"\n"):
        response.extend(sock.recv(1024))
    return bytes(response).rstrip(b"\n")


def test_frontend_process_leaves_json_authority_to_frontend_config(monkeypatch):
    captured: dict[str, object] = {}

    def capture_process_init(_self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ManagedProcess, "__init__", capture_process_init)
    config = json.dumps(
        {
            "allow_insecure_plaintext": True,
            "urls": ["valkey://127.0.0.1:6379"],
            "index_scope": "shared",
            "worker_events": True,
            "authoritative_admission": True,
        }
    )

    FrontendRouterProcess(
        SimpleNamespace(node=SimpleNamespace(name="json-authority")),
        block_size=16,
        frontend_port=8000,
        namespace="test",
        router_valkey_config=config,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command.count("--router-valkey-config") == 1
    assert "--router-valkey-authoritative-admission" not in command
    assert "--router-queue-threshold" not in command
    assert "--no-router-track-prefill-tokens" not in command
    assert "--router-host-cache-hit-weight" not in command
    assert "--router-disk-cache-hit-weight" not in command


def test_valkey_primary_proxy_switches_only_new_connections():
    first = _TaggedServer(b"first")
    second = _TaggedServer(b"second")
    server_threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (first, second)
    ]
    for thread in server_threads:
        thread.start()

    proxy = ValkeyPrimaryProxy(
        listen_port=0,
        target_port=first.server_address[1],
    )
    try:
        with proxy:
            existing = socket.create_connection((proxy.listen_host, proxy.listen_port))
            try:
                assert _round_trip(existing, b"before") == b"first:before"
                assert proxy.switch_target(second.server_address[1]) == (
                    "127.0.0.1",
                    first.server_address[1],
                )
                assert _round_trip(existing, b"after") == b"first:after"

                with socket.create_connection(
                    (proxy.listen_host, proxy.listen_port)
                ) as new_connection:
                    assert _round_trip(new_connection, b"new") == b"second:new"
            finally:
                existing.close()
        assert not proxy.running
        assert proxy.active_tunnels == 0
    finally:
        for server in (first, second):
            server.shutdown()
            server.server_close()
        for thread in server_threads:
            thread.join(timeout=2)


def test_three_valkey_sentinels_form_quorum_and_cleanup(
    request, tmp_path: Path, unused_tcp_port_factory
):
    server = Path(
        os.environ.get(
            "VALKEY_SERVER",
            shutil.which("valkey-server") or "/nonexistent/valkey-server",
        )
    )
    cli = Path(os.environ.get("VALKEY_CLI", str(server.with_name("valkey-cli"))))
    if not server.is_file() or not cli.is_file():
        pytest.skip("local valkey-server and valkey-cli are required")

    primary_port = unused_tcp_port_factory()
    tokenizer_primary_port = unused_tcp_port_factory()
    sentinel_ports = [unused_tcp_port_factory() for _ in range(3)]
    primary_dir = tmp_path / "primary"
    primary_dir.mkdir()
    primary = ManagedProcess(
        command=[
            str(server),
            "--port",
            str(primary_port),
            "--bind",
            "127.0.0.1",
            "--protected-mode",
            "no",
            "--dir",
            str(primary_dir),
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        timeout=10,
        display_output=False,
        health_check_ports=[primary_port],
        log_dir=request.node.name,
        terminate_all_matching_process_names=False,
        display_name=f"sentinel-test-primary-{primary_port}",
    )
    tokenizer_primary_dir = tmp_path / "tokenizer-primary"
    tokenizer_primary_dir.mkdir()
    tokenizer_primary = ManagedProcess(
        command=[
            str(server),
            "--port",
            str(tokenizer_primary_port),
            "--bind",
            "127.0.0.1",
            "--protected-mode",
            "no",
            "--dir",
            str(tokenizer_primary_dir),
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        timeout=10,
        display_output=False,
        health_check_ports=[tokenizer_primary_port],
        log_dir=request.node.name,
        terminate_all_matching_process_names=False,
        display_name=f"sentinel-test-tokenizer-primary-{tokenizer_primary_port}",
    )

    process_handles = []
    with ExitStack() as stack:
        running_primary = stack.enter_context(primary)
        assert running_primary.proc is not None
        process_handles.append(running_primary.proc)
        running_tokenizer_primary = stack.enter_context(tokenizer_primary)
        assert running_tokenizer_primary.proc is not None
        process_handles.append(running_tokenizer_primary.proc)

        sentinels = []
        for sentinel_port in sentinel_ports:
            sentinel = stack.enter_context(
                ValkeySentinelProcess(
                    request,
                    port=sentinel_port,
                    data_dir=tmp_path / f"sentinel-{sentinel_port}",
                    server=str(server),
                    cli=str(cli),
                    master_port=primary_port,
                    master_name="dynkv-test",
                    additional_masters=(("tokenizer-test", tokenizer_primary_port),),
                )
            )
            assert sentinel.proc is not None
            process_handles.append(sentinel.proc)
            sentinels.append(sentinel)

        sentinels[0].wait_for_quorum(sentinel_count=3, timeout=10)
        sentinels[0].wait_for_quorum(
            sentinel_count=3, master_name="tokenizer-test", timeout=10
        )
        assert all(
            sentinel.get_master_addr() == ("127.0.0.1", primary_port)
            for sentinel in sentinels
        )
        assert all(
            sentinel.get_master_addr(master_name="tokenizer-test")
            == ("127.0.0.1", tokenizer_primary_port)
            for sentinel in sentinels
        )

    assert all(process.poll() is not None for process in process_handles)
