# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "registration", "endpoint"])
async def test_encoder_health_follows_registration_and_shutdown(monkeypatch, failure):
    """No engine canary exists: registration owns encoder process health."""
    from dynamo.sglang import init_multimodal

    registration_started = asyncio.Event()
    allow_registration = asyncio.Event()
    stop_serving = asyncio.Event()
    healthy = asyncio.Event()
    health = []

    def set_health_status(value):
        health.append(value)
        if value:
            healthy.set()

    async def register(*args, **kwargs):
        registration_started.set()
        await allow_registration.wait()
        if failure == "registration":
            raise RuntimeError("registration failed")

    async def serve(*args, **kwargs):
        await stop_serving.wait()
        if failure == "endpoint":
            raise RuntimeError("endpoint failed")

    client = SimpleNamespace(wait_for_instances=AsyncMock())
    endpoint = SimpleNamespace(
        client=AsyncMock(return_value=client), serve_endpoint=serve
    )
    runtime = SimpleNamespace(
        endpoint=lambda _: endpoint, set_health_status=set_health_status
    )
    server_args = SimpleNamespace(served_model_name="test-model")
    config = SimpleNamespace(
        server_args=server_args,
        dynamo_args=SimpleNamespace(
            namespace="test",
            component="encode",
            endpoint="generate",
            multimodal_embedding_cache_capacity_gb=0,
        ),
        use_resolved_server_args=lambda args: args,
    )
    handler = SimpleNamespace(
        encoder=SimpleNamespace(server_args=server_args),
        _embedding_cache=None,
        generate=AsyncMock(),
        cleanup=Mock(),
    )
    monkeypatch.setattr(
        init_multimodal, "MultimodalEncodeWorkerHandler", lambda *a: handler
    )
    monkeypatch.setattr(init_multimodal, "publish_server_args", Mock())
    monkeypatch.setattr(init_multimodal, "register_model_taint_route", Mock())
    monkeypatch.setattr(init_multimodal, "register_model_with_readiness_gate", register)

    task = asyncio.create_task(
        init_multimodal.init_multimodal_encode_worker(
            runtime,
            config,
            asyncio.Event(),
            [],
        )
    )
    try:
        await registration_started.wait()
        assert health == []
        if failure == "endpoint":
            # Endpoint failure must cancel pending registration, preventing a
            # late success from marking a shut-down worker healthy.
            stop_serving.set()
        else:
            allow_registration.set()
            if failure is None:
                await healthy.wait()
                assert health == [True]
                stop_serving.set()
        if failure:
            with pytest.raises(RuntimeError, match=f"{failure} failed"):
                await task
            assert health == [False]
        else:
            await task
            assert health == [True, False]
        handler.cleanup.assert_called_once()
    finally:
        allow_registration.set()
        stop_serving.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
