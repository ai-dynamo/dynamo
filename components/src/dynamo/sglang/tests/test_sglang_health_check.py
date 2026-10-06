# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sglang health-check payload and lifecycle tests.

Asserts the canary HEALTH_CHECK_KEY marker is layered onto the disagg payload
(which the prefill handler reads), absent from the decode/agg payload (which
no handler reads), survives DYN_HEALTH_CHECK_PAYLOAD env overrides, and tracks
multimodal encoder registration and shutdown.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.sglang.health_check import (
    SglangDisaggHealthCheckPayload,
    SglangHealthCheckPayload,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_disagg_payload_has_marker():
    assert SglangDisaggHealthCheckPayload().to_dict()[HEALTH_CHECK_KEY] is True


def test_decode_payload_has_no_marker():
    # Decode/agg handler doesn't read the marker; payload stays unmarked.
    assert HEALTH_CHECK_KEY not in SglangHealthCheckPayload().to_dict()


def test_disagg_env_override_preserves_marker(monkeypatch):
    """DYN_HEALTH_CHECK_PAYLOAD must not drop the canary marker."""
    monkeypatch.setenv(
        "DYN_HEALTH_CHECK_PAYLOAD",
        json.dumps(
            {
                "token_ids": [1],
                "sampling_options": {"temperature": 0.0},
                "stop_conditions": {"max_tokens": 1},
            }
        ),
    )
    assert SglangDisaggHealthCheckPayload().to_dict()[HEALTH_CHECK_KEY] is True


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
