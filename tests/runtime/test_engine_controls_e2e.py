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

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.e2e,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


async def _post_with_retry(
    client: httpx.AsyncClient, url: str, body: dict[str, Any]
) -> httpx.Response:
    last_error: Exception | None = None
    for _ in range(30):
        try:
            return await client.post(url, json=body)
        except httpx.ConnectError as exc:
            last_error = exc
            await asyncio.sleep(0.1)

    raise AssertionError(f"system server did not accept connections: {last_error}")


@pytest.mark.asyncio
async def test_engine_control_route_invokes_registered_callback(
    monkeypatch: pytest.MonkeyPatch, dynamo_dynamic_ports
):
    system_port = dynamo_dynamic_ports.system_ports[0]
    monkeypatch.setenv("DYN_SYSTEM_PORT", str(system_port))

    # Keep this local-only test independent of ambient CI NATS_SERVER settings.
    runtime = DistributedRuntime(
        asyncio.get_running_loop(),
        "mem",
        "tcp",
        event_plane="zmq",
    )
    calls: list[dict[str, Any]] = []

    async def sleep_control(body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        return {"status": "ok", "control": "sleep", "body": body}

    runtime.register_engine_route("control/sleep", sleep_control)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await _post_with_retry(
                client,
                f"http://127.0.0.1:{system_port}/engine/control/sleep",
                {"level": 1},
            )

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "control": "sleep",
            "body": {"level": 1},
        }
        assert calls == [{"level": 1}]
    finally:
        runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.sglang
@pytest.mark.timeout(30)
async def test_tokenizer_manager_internal_state_route_serializes_nested_dataclass(
    monkeypatch: pytest.MonkeyPatch, dynamo_dynamic_ports
):
    handler_base = pytest.importorskip(
        "dynamo.sglang.request_handlers.handler_base",
        reason="SGLang is required for the tokenizer manager route",
    )

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

    async def get_internal_state() -> list[dict[str, Any]]:
        return [
            {
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
        ]

    handler = handler_base.RLMixin()
    handler.engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            get_internal_state=get_internal_state,
        )
    )

    system_port = dynamo_dynamic_ports.system_ports[0]
    monkeypatch.setenv("DYN_SYSTEM_PORT", str(system_port))
    runtime = DistributedRuntime(
        asyncio.get_running_loop(),
        "mem",
        "tcp",
        event_plane="zmq",
    )
    runtime.register_engine_route(
        "call_tokenizer_manager", handler.call_tokenizer_manager
    )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await _post_with_retry(
                client,
                f"http://127.0.0.1:{system_port}/engine/call_tokenizer_manager",
                {"method": "get_internal_state"},
            )

        assert response.status_code == 200
        state = response.json()["result"][0]
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
