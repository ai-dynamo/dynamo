# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
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


class _FakeReqWrapper:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def request(self, method: str, payload: dict):
        self.calls.append((method, payload))
        return {"ok": True, "result": {"method": method, "payload": payload}}


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


def test_direct_control_rejects_unknown_method(direct_control_server):
    client = RemoteG2DirectControlClient(timeout_ms=1000)

    response = client.request(
        direct_control_server.address, "identify", {"plan": {}}
    )

    assert response is not None
    assert response["ok"] is False
    assert "unknown method" in response["error"]


def test_direct_resolve_forwards_to_engine():
    req_wrapper = _FakeReqWrapper()
    server = RemoteG2DirectControlServer(
        worker_id=42,
        process_generation="proc-abc",
        source_generation=7,
        req_wrapper=req_wrapper,
        timeout_ms=1000,
    ).start()
    try:
        plan = {"plan_id": "p1", "kv_block_hashes": [1, 2, 3]}
        response = RemoteG2DirectControlClient(timeout_ms=1000).request(
            server.address, "resolve_and_lease", {"plan": plan}
        )

        assert response == {
            "ok": True,
            "result": {
                "method": "resolve_and_lease",
                "payload": {"plan": plan},
            },
        }
        assert req_wrapper.calls == [("resolve_and_lease", {"plan": plan})]
    finally:
        server.close()


def test_direct_metadata_forwards_peer_identity_to_engine():
    req_wrapper = _FakeReqWrapper()
    server = RemoteG2DirectControlServer(
        worker_id=42,
        process_generation="proc-abc",
        source_generation=7,
        req_wrapper=req_wrapper,
        timeout_ms=1000,
    ).start()
    try:
        response = RemoteG2DirectControlClient(timeout_ms=1000).request(
            server.address,
            "get_metadata",
            {
                "peer_name": "target-agent",
                "peer_connection_info": "target-conn",
                "ignored": "not-forwarded",
            },
        )

        assert response is not None
        assert response["ok"] is True
        assert req_wrapper.calls == [
            (
                "get_metadata",
                {
                    "peer_name": "target-agent",
                    "peer_connection_info": "target-conn",
                },
            )
        ]
    finally:
        server.close()


def test_direct_release_forwards_to_engine():
    req_wrapper = _FakeReqWrapper()
    server = RemoteG2DirectControlServer(
        worker_id=42,
        process_generation="proc-abc",
        source_generation=7,
        req_wrapper=req_wrapper,
        timeout_ms=1000,
    ).start()
    try:
        response = RemoteG2DirectControlClient(timeout_ms=1000).request(
            server.address,
            "release_lease",
            {"lease_id": "lease-1", "reason": "done"},
        )

        assert response is not None
        assert response["ok"] is True
        assert req_wrapper.calls == [
            ("release_lease", {"lease_id": "lease-1", "reason": "done"})
        ]
    finally:
        server.close()


def test_direct_request_rejects_wrong_source_worker_before_forwarding(caplog):
    req_wrapper = _FakeReqWrapper()
    server = RemoteG2DirectControlServer(
        worker_id=42,
        process_generation="proc-abc",
        source_generation=7,
        req_wrapper=req_wrapper,
        timeout_ms=1000,
    ).start()
    try:
        caplog.set_level(logging.WARNING)
        response = RemoteG2DirectControlClient(timeout_ms=1000).request(
            server.address,
            "resolve_and_lease",
            {"plan": {"plan_id": "p1"}},
            expected_identity={"worker_id": 43},
        )

        assert response == {
            "ok": False,
            "error": "wrong_source_worker",
            "expected_worker_id": "43",
            "actual_worker_id": "42",
        }
        assert req_wrapper.calls == []
        assert "wrong source worker" in caplog.text
    finally:
        server.close()


def test_direct_request_rejects_stale_generation_before_forwarding(caplog):
    req_wrapper = _FakeReqWrapper()
    server = RemoteG2DirectControlServer(
        worker_id=42,
        process_generation="proc-abc",
        source_generation=7,
        req_wrapper=req_wrapper,
        timeout_ms=1000,
    ).start()
    try:
        caplog.set_level(logging.WARNING)
        response = RemoteG2DirectControlClient(timeout_ms=1000).request(
            server.address,
            "resolve_and_lease",
            {"plan": {"plan_id": "p1"}},
            expected_identity={
                "worker_id": 42,
                "process_generation": "stale-proc",
            },
        )

        assert response == {
            "ok": False,
            "error": "wrong_source_identity",
            "field": "process_generation",
            "expected": "stale-proc",
            "actual": "proc-abc",
        }
        assert req_wrapper.calls == []
        assert "stale source identity" in caplog.text
    finally:
        server.close()


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
        assert server_a.address != server_b.address
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


def test_target_fetches_control_info_without_direct_identify(
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


def test_target_uses_direct_control_for_side_effecting_rpcs():
    req_wrapper = _FakeReqWrapper()
    server = RemoteG2DirectControlServer(
        worker_id=42,
        process_generation="proc-abc",
        source_generation=7,
        req_wrapper=req_wrapper,
        timeout_ms=1000,
    ).start()
    try:
        response = {
            "ok": True,
            "result": {
                "address": server.address,
                "identity": server.identity,
            },
        }
        fake_client = _FakeStreamClient(response)
        target = build_target_rpc_client(_FakeRuntime(fake_client), "ns", "worker")
        plan = {"plan_id": "p1", "kv_block_hashes": [10, 11]}

        async def _run_calls():
            return (
                await target.resolve_and_lease(plan, 42),
                await target.get_source_metadata(
                    42,
                    peer_name="target-agent",
                    peer_connection_info="target-conn",
                ),
                await target.release_lease("lease-1", 42, reason="done"),
            )

        resolve_response, metadata, release_response = asyncio.run(_run_calls())

        assert resolve_response == {
            "ok": True,
            "result": {
                "method": "resolve_and_lease",
                "payload": {"plan": plan},
            },
        }
        assert metadata == {
            "method": "get_metadata",
            "payload": {
                "peer_name": "target-agent",
                "peer_connection_info": "target-conn",
            },
        }
        assert release_response == {
            "ok": True,
            "result": {
                "method": "release_lease",
                "payload": {"lease_id": "lease-1", "reason": "done"},
            },
        }
        assert fake_client.direct_calls == [({}, 42)]
        assert req_wrapper.calls == [
            ("resolve_and_lease", {"plan": plan}),
            (
                "get_metadata",
                {
                    "peer_name": "target-agent",
                    "peer_connection_info": "target-conn",
                },
            ),
            ("release_lease", {"lease_id": "lease-1", "reason": "done"}),
        ]
    finally:
        server.close()


def test_target_rejects_wrong_source_worker_without_extra_identify():
    req_wrapper = _FakeReqWrapper()
    server = RemoteG2DirectControlServer(
        worker_id=42,
        process_generation="proc-abc",
        source_generation=7,
        req_wrapper=req_wrapper,
        timeout_ms=1000,
    ).start()
    try:
        response = {
            "ok": True,
            "result": {
                "address": server.address,
                "identity": server.identity,
            },
        }
        target = build_target_rpc_client(
            _FakeRuntime(_FakeStreamClient(response)), "ns", "worker"
        )

        resolve_response = asyncio.run(
            target.resolve_and_lease({"plan_id": "p1"}, 43)
        )

        assert resolve_response == {
            "ok": False,
            "error": "wrong_source_worker",
            "expected_worker_id": "43",
            "actual_worker_id": "42",
        }
        assert req_wrapper.calls == []
    finally:
        server.close()
