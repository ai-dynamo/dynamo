# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

pytest.importorskip("dynamo._core", reason="dynamo Rust Python bindings are required")

from dynamo.runtime import DistributedRuntime  # noqa: E402
from dynamo.sglang.engine_routes import resolve_configured_engine_routes  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.e2e,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


_NO_BODY = object()


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    body: object = _NO_BODY,
) -> httpx.Response:
    last_error: Exception | None = None
    for _ in range(30):
        try:
            kwargs = {} if body is _NO_BODY else {"json": body}
            return await client.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            last_error = exc
            await asyncio.sleep(0.1)

    raise AssertionError(f"system server did not accept connections: {last_error}")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_engine_routes_http_contract(
    monkeypatch: pytest.MonkeyPatch, dynamo_dynamic_ports
):
    system_port = dynamo_dynamic_ports.system_ports[0]
    monkeypatch.setenv("DYN_SYSTEM_PORT", str(system_port))

    control_calls: list[dict[str, Any]] = []
    configured_call_count = 0

    async def sleep_control(body: dict[str, Any]) -> dict[str, Any]:
        control_calls.append(body)
        return {"status": "ok", "control": "sleep", "body": body}

    async def configured_route(body: dict[str, Any]) -> dict[str, Any]:
        nonlocal configured_call_count
        configured_call_count += 1
        return {"body": body}

    def get_server_info() -> dict[str, Any]:
        return {
            "tp_size": 4,
            "topology": {"tp_rank_ids": (0, 1, 2, 3)},
            "active_batch_ids": {7, 9},
        }

    engine = SimpleNamespace(
        loop=asyncio.get_running_loop(),
        tokenizer_manager=SimpleNamespace(),
        get_server_info=get_server_info,
    )
    handler = dict(
        resolve_configured_engine_routes(engine, ["server_info=get_server_info"])
    )["server_info"]

    # Keep this local-only test independent of ambient CI NATS_SERVER settings.
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
    )
    runtime.register_engine_route("control/sleep", sleep_control)
    runtime.register_engine_route("configured", configured_route)
    runtime.register_engine_route("server_info", handler)

    try:
        base_url = f"http://127.0.0.1:{system_port}/engine"
        async with httpx.AsyncClient(timeout=5.0) as client:

            async def request(method, path, body=_NO_BODY):
                return await _request_with_retry(
                    client, method, f"{base_url}/{path}", body
                )

            response = await request("POST", "control/sleep", {"level": 1})
            assert response.status_code == 200
            assert response.json() == {
                "status": "ok",
                "control": "sleep",
                "body": {"level": 1},
            }
            assert control_calls == [{"level": 1}]

            for method in (
                "GET",
                "POST",
                "PUT",
                "PATCH",
                "DELETE",
                "OPTIONS",
                "HEAD",
                "TRACE",
            ):
                call_count_before_request = configured_call_count
                response = await request(method, "configured")
                assert response.status_code == 200
                assert configured_call_count == call_count_before_request + 1
                if method != "HEAD":
                    assert response.json() == {"body": {}}

            response = await request("PUT", "configured", {"value": 3})
            assert response.status_code == 200
            assert response.json() == {"body": {"value": 3}}

            response = await request("POST", "configured", {"method": "dangerous"})
            assert response.status_code == 200
            assert response.json() == {"body": {"method": "dangerous"}}

            for unconfigured in ("call_tokenizer_manager", "dangerous"):
                response = await request("POST", unconfigured, {"method": unconfigured})
                assert response.status_code == 404
                assert response.json()["error"] == "Route not found"

            response = await request("POST", "server_info")
            assert response.status_code == 200
            state = response.json()
            assert "result" not in state
            assert state["tp_size"] == 4
            assert state["topology"]["tp_rank_ids"] == [0, 1, 2, 3]
            assert set(state["active_batch_ids"]) == {7, 9}
    finally:
        runtime.shutdown()
