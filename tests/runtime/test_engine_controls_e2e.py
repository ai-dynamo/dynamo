# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
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

    async def sleep_control(body: dict[str, Any]) -> dict[str, Any]:
        control_calls.append(body)
        return {"status": "ok", "control": "sleep", "body": body}

    configured_calls: list[dict[str, Any]] = []

    async def configured_route(body: dict[str, Any]) -> dict[str, Any]:
        configured_calls.append(body)
        return {"body": body}

    @dataclasses.dataclass
    class PhaseConfig:
        backend: str
        max_bs: int | None
        bs: list[int] | None
        tc_compiler: str
        full_prefill_max_req: int | None = None

    @dataclasses.dataclass
    class CudaGraphConfig:
        decode: PhaseConfig
        prefill: PhaseConfig

    def get_server_info() -> dict[str, Any]:
        return {
            "dp_rank": 0,
            "tp_size": 4,
            "pp_size": 2,
            "dp_size": 8,
            "cuda_graph_config": CudaGraphConfig(
                decode=PhaseConfig("full", 32, [1, 2, 4, 8], "eager"),
                prefill=PhaseConfig(
                    "breakable", None, [1], "eager", full_prefill_max_req=4
                ),
            ),
            "tp_rank_ids": (0, 1, 2, 3),
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
            response = await _request_with_retry(
                client,
                "POST",
                f"{base_url}/control/sleep",
                {"level": 1},
            )
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
                response = await _request_with_retry(
                    client, method, f"{base_url}/configured"
                )
                assert response.status_code == 200
                assert configured_calls[-1] == {}
                if method != "HEAD":
                    assert response.json() == {"body": {}}

            response = await _request_with_retry(
                client,
                "PUT",
                f"{base_url}/configured",
                {"value": 3},
            )
            assert response.status_code == 200
            assert response.json() == {"body": {"value": 3}}
            assert configured_calls[-1] == {"value": 3}

            response = await _request_with_retry(
                client,
                "POST",
                f"{base_url}/configured",
                {"method": "dangerous"},
            )
            assert response.status_code == 200
            assert response.json() == {"body": {"method": "dangerous"}}

            for unconfigured in ("call_tokenizer_manager", "dangerous"):
                response = await _request_with_retry(
                    client,
                    "POST",
                    f"{base_url}/{unconfigured}",
                    {"method": unconfigured},
                )
                assert response.status_code == 404
                assert response.json()["error"] == "Route not found"

            response = await _request_with_retry(
                client, "POST", f"{base_url}/server_info"
            )
            assert response.status_code == 200
            state = response.json()
            assert "result" not in state
            assert state["tp_size"] == 4
            assert state["pp_size"] == 2
            assert state["dp_size"] == 8
            assert state["cuda_graph_config"] == {
                "decode": {
                    "backend": "full",
                    "max_bs": 32,
                    "bs": [1, 2, 4, 8],
                    "tc_compiler": "eager",
                    "full_prefill_max_req": None,
                },
                "prefill": {
                    "backend": "breakable",
                    "max_bs": None,
                    "bs": [1],
                    "tc_compiler": "eager",
                    "full_prefill_max_req": 4,
                },
            }
            assert state["tp_rank_ids"] == [0, 1, 2, 3]
            assert set(state["active_batch_ids"]) == {7, 9}
    finally:
        runtime.shutdown()
