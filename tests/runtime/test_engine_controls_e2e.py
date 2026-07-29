# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
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
