# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import socket

import pytest

from dynamo.trtllm.kv_p2p.source_rpc_server import (
    REMOTE_G2_DIRECT_CONTROL_PORT_ENV,
    RemoteG2DirectControlServer,
)
from dynamo.trtllm.kv_p2p.target_rpc_client import (
    RemoteG2DirectControlClient,
    build_target_rpc_client,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
]


class _FakeStreamClient:
    def __init__(self, response: dict):
        self.response = response
        self.direct_calls: list[tuple[dict, int]] = []

    async def wait_for_instances(self):
        return [42]

    async def direct(self, payload: dict, instance_id: int):
        self.direct_calls.append((payload, instance_id))

        async def _stream():
            yield self.response

        return _stream()


class _FakeEndpoint:
    def __init__(self, client: _FakeStreamClient):
        self._client = client

    async def client(self):
        return self._client


class _FakeRuntime:
    def __init__(self, client: _FakeStreamClient):
        self.client = client
        self.endpoint_names: list[str] = []

    def endpoint(self, name: str):
        self.endpoint_names.append(name)
        return _FakeEndpoint(self.client)


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def direct_control_server():
    server = RemoteG2DirectControlServer(
        worker_id=42,
        dp_rank=0,
        tp_rank=3,
        process_generation="proc-abc",
        source_generation=7,
        timeout_ms=1000,
    ).start()
    try:
        yield server
    finally:
        server.close()


def test_direct_identify_returns_source_identity(direct_control_server):
    client = RemoteG2DirectControlClient(timeout_ms=1000)

    identity = client.identify(
        direct_control_server.address, expected_worker_id=42
    )

    assert identity == {
        "worker_id": "42",
        "dp_rank": 0,
        "tp_rank": 3,
        "process_generation": "proc-abc",
        "source_generation": "7",
        "protocol_version": 1,
    }


def test_direct_identify_rejects_wrong_worker(direct_control_server):
    client = RemoteG2DirectControlClient(timeout_ms=1000)

    identity = client.identify(
        direct_control_server.address, expected_worker_id=43
    )

    assert identity is None


def test_direct_control_rejects_unknown_method(direct_control_server):
    client = RemoteG2DirectControlClient(timeout_ms=1000)

    response = client.request(
        direct_control_server.address, "resolve_and_lease", {"plan": {}}
    )

    assert response is not None
    assert response["ok"] is False
    assert "unknown method" in response["error"]


def test_multiple_direct_control_servers_use_distinct_ports():
    server_a = RemoteG2DirectControlServer(
        worker_id=1,
        process_generation="proc-a",
        source_generation=10,
        timeout_ms=1000,
    ).start()
    server_b = RemoteG2DirectControlServer(
        worker_id=2,
        process_generation="proc-b",
        source_generation=20,
        timeout_ms=1000,
    ).start()
    try:
        client = RemoteG2DirectControlClient(timeout_ms=1000)

        assert server_a.address != server_b.address
        assert client.identify(server_a.address, expected_worker_id=1)[
            "source_generation"
        ] == "10"
        assert client.identify(server_b.address, expected_worker_id=2)[
            "source_generation"
        ] == "20"
    finally:
        server_a.close()
        server_b.close()


def test_direct_control_port_can_come_from_env(monkeypatch):
    port = _unused_tcp_port()
    monkeypatch.setenv(REMOTE_G2_DIRECT_CONTROL_PORT_ENV, str(port))
    server = RemoteG2DirectControlServer(
        worker_id=55,
        process_generation="proc-env",
        source_generation=9,
        timeout_ms=1000,
    ).start()
    try:
        assert server.address == f"tcp://127.0.0.1:{port}"
        assert RemoteG2DirectControlClient(timeout_ms=1000).identify(
            server.address, expected_worker_id=55
        )["source_generation"] == "9"
    finally:
        server.close()


@pytest.mark.parametrize("bad_port", ["not-a-port", "-1", "65536"])
def test_direct_control_rejects_invalid_env_port(monkeypatch, bad_port):
    monkeypatch.setenv(REMOTE_G2_DIRECT_CONTROL_PORT_ENV, bad_port)

    with pytest.raises(ValueError):
        RemoteG2DirectControlServer(
            worker_id=55,
            process_generation="proc-env",
            source_generation=9,
        )


def test_target_fetches_control_info_then_validates_direct_identify(
    direct_control_server,
):
    response = {
        "ok": True,
        "result": {
            "address": direct_control_server.address,
            "identity": direct_control_server.identity,
        },
    }
    fake_client = _FakeStreamClient(response)
    runtime = _FakeRuntime(fake_client)
    target = build_target_rpc_client(runtime, "ns", "worker")

    control_info = asyncio.run(target.get_direct_control_info(42))

    assert control_info == {
        "address": direct_control_server.address,
        "identity": direct_control_server.identity,
    }
    assert runtime.endpoint_names == ["ns.worker.remote-g2-control-info"]
    assert fake_client.direct_calls == [({}, 42)]


def test_target_rejects_control_info_for_wrong_direct_worker(
    direct_control_server,
):
    response = {
        "ok": True,
        "result": {
            "address": direct_control_server.address,
            "identity": direct_control_server.identity,
        },
    }
    target = build_target_rpc_client(
        _FakeRuntime(_FakeStreamClient(response)), "ns", "worker"
    )

    assert asyncio.run(target.get_direct_control_info(43)) is None
