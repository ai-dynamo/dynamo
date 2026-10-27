# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.control_api import _build_app
from dynamo.planner.environment.metrics_provider.prometheus_traffic_provider import (
    PrometheusTrafficProvider,
)
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.monitoring.traffic_metrics import Metrics
from dynamo.planner.monitoring.worker_info import WorkerInfo

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _config() -> PlannerConfig:
    return PlannerConfig.model_construct(
        namespace="base-ns",
        backend="vllm",
        mode="disagg",
        throughput_adjustment_interval_seconds=30,
        throughput_metrics_source="frontend",
    )


def _state_source() -> MagicMock:
    source = MagicMock()
    state = DeploymentState(model_name="Qwen/Qwen3")
    state.decode.info = WorkerInfo(
        component_name="backend",
        endpoint="generate",
    )
    source.deployment_state.return_value = state
    return source


def _provider(namespace_source=None):
    client_patch = patch(
        "dynamo.planner.environment.metrics_provider."
        "prometheus_traffic_provider.PrometheusAPIClient"
    )
    client_class = client_patch.start()
    provider = PrometheusTrafficProvider(
        config=_config(),
        state_source=_state_source(),
        metrics_state=Metrics(),
        namespace_source=namespace_source,
    )
    return provider, client_class.return_value, client_patch


@pytest.mark.asyncio
async def test_accept_length_uses_current_runtime_namespace():
    namespace_source = MagicMock()
    namespace_source.runtime_namespace.return_value = "base-ns-workerhash"
    provider, client, client_patch = _provider(namespace_source)
    client.get_avg_spec_decode_accept_length.return_value = 2.5
    try:
        assert await provider.collect_accept_length("30s") == 2.5
    finally:
        client_patch.stop()

    client.get_avg_spec_decode_accept_length.assert_called_once_with(
        "30s",
        "vllm",
        "backend",
        "Qwen/Qwen3",
        namespace="base-ns-workerhash",
        endpoint_name="generate",
    )


def test_provider_passes_prometheus_request_timeout():
    config = _config()
    config.metric_pulling_prometheus_request_timeout_seconds = 3.5
    with patch(
        "dynamo.planner.environment.metrics_provider."
        "prometheus_traffic_provider.PrometheusAPIClient"
    ) as client_class:
        PrometheusTrafficProvider(
            config=config,
            state_source=_state_source(),
            metrics_state=Metrics(),
        )

        assert client_class.call_args.kwargs["request_timeout_seconds"] == 3.5


@pytest.mark.asyncio
async def test_accept_length_falls_back_to_configured_namespace():
    provider, client, client_patch = _provider()
    client.get_avg_spec_decode_accept_length.return_value = 2.5
    try:
        await provider.collect_accept_length("30s")
    finally:
        client_patch.stop()

    assert (
        client.get_avg_spec_decode_accept_length.call_args.kwargs["namespace"]
        == "base-ns"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_method",
    [
        "get_avg_time_to_first_token",
        "get_avg_inter_token_latency",
        "get_avg_request_count",
        "get_avg_input_sequence_tokens",
        "get_avg_output_sequence_tokens",
    ],
)
async def test_invalid_required_metric_is_checked_before_float_logging(missing_method):
    provider, client, client_patch = _provider()
    client.get_avg_time_to_first_token.return_value = 0.1
    client.get_avg_inter_token_latency.return_value = 0.01
    client.get_avg_request_count.return_value = 2.0
    client.get_avg_request_duration.return_value = 0.2
    client.get_avg_input_sequence_tokens.return_value = 100.0
    client.get_avg_output_sequence_tokens.return_value = 20.0
    client.get_avg_kv_hit_rate.return_value = None
    client.get_avg_spec_decode_accept_length.return_value = None
    getattr(client, missing_method).return_value = None
    try:
        observation = await provider.collect_traffic()
    finally:
        client_patch.stop()

    assert observation is None


@pytest.fixture
async def traffic_provider():
    provider, client, client_patch = _provider()
    client.get_avg_time_to_first_token.return_value = 0.1
    client.get_avg_inter_token_latency.return_value = 0.01
    client.get_avg_request_count.return_value = 2.0
    client.get_avg_request_duration.return_value = 0.2
    client.get_avg_input_sequence_tokens.return_value = 100.0
    client.get_avg_output_sequence_tokens.return_value = 20.0
    client.get_avg_kv_hit_rate.return_value = 0.5
    client.get_avg_spec_decode_accept_length.return_value = 2.5
    try:
        yield provider, client
    finally:
        if provider._collection_task is not None:
            await asyncio.gather(provider._collection_task, return_exceptions=True)
        client_patch.stop()


@pytest.fixture
async def blocked_query(traffic_provider):
    _, client = traffic_provider
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def query(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5), "Test did not release Prometheus query"
        return 0.5

    client.get_avg_kv_hit_rate.side_effect = query
    try:
        yield started, release
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize("full", [True, False])
async def test_control_api_responds_during_collection(
    traffic_provider, blocked_query, full
):
    provider, client = traffic_provider
    started, release = blocked_query
    controller = MagicMock()
    controller.get_min_endpoints = AsyncMock(return_value={"decode_min_endpoint": 1})
    controller.patch_min_endpoints = AsyncMock(return_value={"decode_min_endpoint": 2})
    async with TestClient(TestServer(_build_app(controller))) as http:
        collection = asyncio.create_task(
            provider.collect_traffic()
            if full
            else provider.collect_kv_hit_rate_observation(30)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            response = await asyncio.wait_for(http.get("/v1/min-endpoints"), timeout=2)
            assert response.status == 200
            assert await response.json() == {"decode_min_endpoint": 1}
            response = await asyncio.wait_for(
                http.patch("/v1/min-endpoints", json={"decode_min_endpoint": 2}),
                timeout=2,
            )
            assert response.status == 200
            assert await response.json() == {"decode_min_endpoint": 2}
            assert not collection.done()
            assert provider.metrics_state == Metrics()
            # Later queries must use the original snapshot, not live state.
            provider.config.namespace = "changed-ns"
            provider.state_source.deployment_state.return_value.decode.info.endpoint = (
                "changed-endpoint"
            )
            provider.config.throughput_adjustment_interval_seconds = 60
        finally:
            release.set()
            observation = await collection

    assert observation.duration_s == 30
    assert observation.kv_hit_rate == 0.5
    assert observation.accept_length == 2.5
    assert provider.metrics_state.kv_hit_rate == 0.5
    if full:
        assert provider.metrics_state.ttft == 100
        assert provider.metrics_state.itl == 10
        assert observation.num_req == 2
    client.get_avg_spec_decode_accept_length.assert_called_once_with(
        "30s",
        "vllm",
        "backend",
        "Qwen/Qwen3",
        namespace="base-ns",
        endpoint_name="generate",
    )


@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize("full", [True, False])
async def test_cancelled_collection_does_not_publish_or_overlap(
    traffic_provider, blocked_query, full
):
    provider, client = traffic_provider
    started, release = blocked_query
    collection = asyncio.create_task(
        provider.collect_traffic()
        if full
        else provider.collect_kv_hit_rate_observation(30)
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    collection.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(collection, timeout=2)

    # A new caller must wait without starting a second query. Cancelling that
    # waiter must not cancel the thread-backed task either.
    waiter = asyncio.create_task(provider.collect_kv_hit_rate_observation(30))
    await asyncio.sleep(0)
    assert not waiter.done()
    assert client.get_avg_kv_hit_rate.call_count == 1
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    await provider._collection_task
    assert provider.metrics_state == Metrics()

    observation = await provider.collect_kv_hit_rate_observation(30)
    assert observation.kv_hit_rate == 0.5
    assert client.get_avg_kv_hit_rate.call_count == 2


@pytest.mark.asyncio
async def test_query_failure_preserves_metrics_and_allows_retry(traffic_provider):
    provider, client = traffic_provider
    client.get_avg_request_duration.side_effect = RuntimeError("query failed")
    with pytest.raises(RuntimeError, match="query failed"):
        await provider.collect_traffic()
    assert provider.metrics_state == Metrics()
    client.get_avg_request_duration.side_effect = None
    assert await provider.collect_traffic() is not None


@pytest.mark.asyncio
async def test_idle_normalization_preserves_load_metrics(traffic_provider):
    provider, client = traffic_provider
    provider.metrics_state.p_load = 0.4
    provider.metrics_state.d_load = 0.6
    client.get_avg_request_count.return_value = 0.0
    client.get_avg_time_to_first_token.return_value = float("nan")
    client.get_avg_inter_token_latency.return_value = float("nan")
    client.get_avg_input_sequence_tokens.return_value = float("nan")
    client.get_avg_output_sequence_tokens.return_value = float("nan")
    client.get_avg_request_duration.return_value = float("nan")
    observation = await provider.collect_traffic()
    assert observation.num_req == observation.isl == observation.osl == 0
    assert provider.metrics_state.ttft == provider.metrics_state.itl == 0
    assert provider.metrics_state.p_load == 0.4
    assert provider.metrics_state.d_load == 0.6


@pytest.mark.asyncio
@pytest.mark.parametrize("model_source", ["global", "decode", "prefill"])
async def test_query_context_uses_one_deployment_snapshot(
    traffic_provider, model_source
):
    provider, client = traffic_provider
    state = _state_source().deployment_state()
    if model_source != "global":
        state.model_name = None
        if model_source == "decode":
            state.decode.info.model_name = "Qwen/Qwen3"
        else:
            state.prefill.info = WorkerInfo(model_name="Qwen/Qwen3")
    other = DeploymentState(model_name="other-model")
    other.decode.info = WorkerInfo(component_name="other", endpoint="other-endpoint")
    provider.state_source.deployment_state.side_effect = [state, other]

    assert await provider.collect_accept_length("30s") == 2.5

    provider.state_source.deployment_state.assert_called_once_with()
    client.get_avg_spec_decode_accept_length.assert_called_once_with(
        "30s",
        "vllm",
        "backend",
        "Qwen/Qwen3",
        namespace="base-ns",
        endpoint_name="generate",
    )
    client.get_avg_kv_hit_rate.assert_not_called()
    client.get_avg_time_to_first_token.assert_not_called()
    assert provider.metrics_state == Metrics()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_accept_length_waits_for_collection(
    traffic_provider, blocked_query, cancel_waiter
):
    provider, client = traffic_provider
    started, release = blocked_query
    collection = asyncio.create_task(provider.collect_traffic())
    await asyncio.wait_for(started.wait(), timeout=2)
    waiter = asyncio.create_task(provider.collect_accept_length("10s"))
    try:
        await asyncio.sleep(0)
        assert not waiter.done()
        client.get_avg_spec_decode_accept_length.assert_not_called()
        if cancel_waiter:
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
    finally:
        release.set()
        await collection
        await asyncio.gather(waiter, return_exceptions=True)

    if cancel_waiter:
        assert await provider.collect_accept_length("10s") == 2.5
    else:
        assert waiter.result() == 2.5
    assert client.get_avg_spec_decode_accept_length.call_count == 2
    assert client.get_avg_kv_hit_rate.call_count == 1


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_direct_accept_length_is_responsive_and_cancellation_safe(
    traffic_provider, blocked_query
):
    provider, client = traffic_provider
    started, release = blocked_query
    client.get_avg_spec_decode_accept_length.side_effect = (
        client.get_avg_kv_hit_rate.side_effect
    )
    controller = MagicMock()
    controller.get_min_endpoints = AsyncMock(return_value={"decode_min_endpoint": 1})
    async with TestClient(TestServer(_build_app(controller))) as http:
        collection = asyncio.create_task(provider.collect_accept_length("30s"))
        waiter = None
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            response = await asyncio.wait_for(http.get("/v1/min-endpoints"), timeout=2)
            assert response.status == 200
            assert not collection.done()
            collection.cancel()
            with pytest.raises(asyncio.CancelledError):
                await collection
            waiter = asyncio.create_task(provider.collect_accept_length("10s"))
            await asyncio.sleep(0)
            assert not waiter.done()
            assert client.get_avg_spec_decode_accept_length.call_count == 1
        finally:
            release.set()
            await asyncio.gather(collection, return_exceptions=True)
            if waiter is not None:
                await asyncio.gather(waiter, return_exceptions=True)
        assert waiter.result() == 0.5
    assert client.get_avg_spec_decode_accept_length.call_count == 2
    client.get_avg_kv_hit_rate.assert_not_called()
    assert provider.metrics_state == Metrics()
