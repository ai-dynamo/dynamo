# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang canary payloads and worker health lifecycle tests."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo._core import WorkerDraining
from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.sglang.drain import WorkerDrain
from dynamo.sglang.health_check import (
    SglangDisaggHealthCheckPayload,
    SglangEmbeddingHealthCheckPayload,
    SglangHealthCheckPayload,
)
from dynamo.sglang.protocol import EmbeddingRequest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.fault_tolerance,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "payload_type", [SglangHealthCheckPayload, SglangDisaggHealthCheckPayload]
)
def test_generation_payload_has_marker(monkeypatch, payload_type):
    monkeypatch.delenv("DYN_HEALTH_CHECK_PAYLOAD", raising=False)
    assert payload_type().to_dict()[HEALTH_CHECK_KEY] is True


def test_embedding_payload_uses_engine_bos_token(monkeypatch):
    monkeypatch.delenv("DYN_HEALTH_CHECK_PAYLOAD", raising=False)
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(tokenizer=SimpleNamespace(bos_token_id=42))
    )
    payload = SglangEmbeddingHealthCheckPayload(
        "embedding-model", engine, use_text_input=False
    ).to_dict()

    request = EmbeddingRequest(**payload)
    assert request.model == "embedding-model"
    assert request.input == [42]


def test_embedding_env_override(monkeypatch):
    override = {"model": "embedding-model", "input": "custom probe"}
    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", json.dumps(override))

    payload = SglangEmbeddingHealthCheckPayload("embedding-model").to_dict()

    assert payload == override
    assert EmbeddingRequest(**payload).input == "custom probe"


@pytest.mark.parametrize(
    "payload_type", [SglangHealthCheckPayload, SglangDisaggHealthCheckPayload]
)
def test_generation_env_override_preserves_marker(monkeypatch, payload_type):
    override = {
        "token_ids": [42],
        "sampling_options": {"temperature": 0.0},
        "stop_conditions": {"max_tokens": 1},
        HEALTH_CHECK_KEY: False,
    }
    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", json.dumps(override))
    assert payload_type().to_dict() == {**override, HEALTH_CHECK_KEY: True}


@pytest.mark.asyncio
async def test_aggregate_canary_stays_admitted_during_drain(monkeypatch):
    monkeypatch.delenv("DYN_HEALTH_CHECK_PAYLOAD", raising=False)
    endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
    )
    ready, shutdown = asyncio.Event(), asyncio.Event()
    ready.set()
    drain = WorkerDrain(
        endpoint,
        lock=asyncio.Lock(),
        shutdown_event=shutdown,
        terminal_event=asyncio.Event(),
        ready_event=ready,
        is_paused=lambda: False,
        grace_period=0,
    )
    executed = []

    async def generate(request):
        executed.append(request)
        yield "healthy"

    tracked = drain.wrap(generate)
    await drain.drain({})
    with pytest.raises(WorkerDraining):
        await anext(tracked({"token_ids": [42]}))
    canary = SglangHealthCheckPayload().to_dict()
    assert [item async for item in tracked(canary)] == ["healthy"]
    assert executed == [canary]
    assert drain.withdrawn
    endpoint.unregister_endpoint_instance.assert_awaited_once()
    endpoint.register_endpoint_instance.assert_not_awaited()
    shutdown.set()
    with pytest.raises(WorkerDraining):
        await anext(tracked(canary))


@pytest.mark.asyncio
@pytest.mark.multimodal
@pytest.mark.profiled_vram_gib(0)
@pytest.mark.timeout(10)
@pytest.mark.parametrize("failure", [None, "registration", "endpoint"])
async def test_encoder_health_follows_registration_and_shutdown(monkeypatch, failure):
    """No engine canary exists: registration owns encoder process health."""
    from dynamo.sglang import init_multimodal

    registration_started = asyncio.Event()
    allow_registration = asyncio.Event()
    stop_serving = asyncio.Event()
    serving_stopped = asyncio.Event()
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

    async def serve_loop():
        try:
            await stop_serving.wait()
            if failure == "endpoint":
                raise RuntimeError("endpoint failed")
        finally:
            serving_stopped.set()

    def serve(*args, **kwargs):
        # DistributedRuntime returns a Future rather than a bare coroutine.
        return asyncio.ensure_future(serve_loop())

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
        assert serving_stopped.is_set()
        handler.cleanup.assert_called_once()
    finally:
        allow_registration.set()
        stop_serving.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
