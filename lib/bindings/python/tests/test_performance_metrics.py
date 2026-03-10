# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import time

import pytest

from dynamo.llm import PerformanceMetricsRegistry, RequestMetricsFactory
from dynamo.runtime import DistributedRuntime


@pytest.mark.asyncio
async def test_request_metrics_lifecycle_smoke(runtime: DistributedRuntime):
    endpoint = runtime.endpoint("test.metrics.generate")
    registry = PerformanceMetricsRegistry(endpoint.metrics)

    factory = RequestMetricsFactory(registry, metric_prefix="test_request_metrics")
    request = factory.new_request(input_tokens=128)
    request.record_tokens(
        129, cached_tokens=32
    )  # first token: TTFT + net-new token metrics
    request.record_tokens(145)  # later update: ITL sampling
    request.success()

    cancelled = factory.new_request(input_tokens=64)
    cancelled.cancel()


@pytest.mark.asyncio
async def test_request_metrics_factory_overrides(runtime: DistributedRuntime):
    endpoint = runtime.endpoint("test.metrics.custom")
    registry = PerformanceMetricsRegistry(endpoint.metrics)

    factory = RequestMetricsFactory(
        registry,
        metric_prefix="custom_request_metrics",
        itl_sample_rate=0.2,
        ttft_quantiles=[0.5, 0.9],
        ttft_per_input_token_quantiles=[0.5],
        ttft_per_input_token_window_seconds=30.0,
        request_sample_period_seconds=2.0,
    )

    request = factory.new_request(input_tokens=100)
    request.record_tokens(101)
    request.success()


@pytest.mark.asyncio
async def test_performance_metrics_registry_basics(runtime: DistributedRuntime):
    endpoint = runtime.endpoint("test.metrics.registry")
    registry = PerformanceMetricsRegistry(endpoint.metrics)

    rate = registry.new_rate_metric(
        "requests",
        quantiles=[0.5, 0.9, 0.99],
        sample_period_seconds=1.0,
        window_seconds=30.0,
    )
    dist = registry.new_distribution_metric(
        "ttft_ms",
        quantiles=[0.5, 0.9, 0.99],
        window_seconds=30.0,
    )
    ratio = registry.new_ratio_metric(
        "kv_hit_rate",
        quantiles=[0.5, 0.9, 0.99],
        window_seconds=30.0,
    )

    rate.record_count(3)
    dist.record_value(42.0)
    ratio.record_ratio(8.0, 10.0)

    assert rate.name == "requests"
    assert dist.name == "ttft_ms"
    assert ratio.name == "kv_hit_rate"


@pytest.mark.asyncio
async def test_performance_metrics_pef_round_trip(runtime: DistributedRuntime):
    endpoint = runtime.endpoint("test.metrics.pef")
    registry = PerformanceMetricsRegistry(
        endpoint.metrics,
        publish_interval_seconds=1.0,
        metric_prefix="baseten_frontend",
    )

    rate = registry.new_rate_metric(
        "requests",
        quantiles=[0.5, 0.99],
        sample_period_seconds=1.0,
        window_seconds=30.0,
    )
    dist = registry.new_distribution_metric(
        "ttft_ms",
        quantiles=[0.5, 0.99],
        window_seconds=30.0,
    )

    factory = RequestMetricsFactory(registry, metric_prefix="request_metrics")

    rate.record_count(5)
    dist.record_value(42.0)
    req = factory.new_request(input_tokens=128)
    req.record_tokens(129, cached_tokens=32)
    req.record_tokens(145)
    req.success()

    # Wait for one publish tick (configured at 1s), then scrape PEF.
    time.sleep(1.1)
    expfmt = endpoint.metrics.prometheus_expfmt()

    assert "dynamo_component_baseten_frontend_requests_per_second" in expfmt
    assert 'dynamo_namespace="test"' in expfmt
    assert 'dynamo_component="metrics"' in expfmt
    assert 'dynamo_endpoint="pef"' in expfmt
    assert (
        'dynamo_component_baseten_frontend_requests_per_second_quantile{quantile="0.99"'
        in expfmt
    )
    assert "dynamo_component_baseten_frontend_ttft_ms_avg " in expfmt
    assert (
        'dynamo_component_baseten_frontend_request_metrics_input_tokens_per_second_quantile{quantile="0.1"'
        in expfmt
    )
    assert "dynamo_component_baseten_frontend_request_metrics_itl_ms_avg " in expfmt

    # At least one representative series should have non-zero published values.
    m_rate = re.search(
        r"^dynamo_component_baseten_frontend_requests_per_second(?:\{.*\})?\s+([0-9]+(?:\.[0-9]+)?)$",
        expfmt,
        re.MULTILINE,
    )
    assert m_rate and float(m_rate.group(1)) > 0.0

    m_ttft = re.search(
        r"^dynamo_component_baseten_frontend_ttft_ms_avg(?:\{.*\})?\s+([0-9]+(?:\.[0-9]+)?)$",
        expfmt,
        re.MULTILINE,
    )
    assert m_ttft and float(m_ttft.group(1)) > 0.0
