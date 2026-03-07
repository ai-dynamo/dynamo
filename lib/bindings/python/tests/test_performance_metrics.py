# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import PerformanceMetricsRegistry, RequestMetricsFactory
from dynamo.runtime import DistributedRuntime


@pytest.mark.asyncio
async def test_request_metrics_lifecycle_smoke(runtime: DistributedRuntime):
    endpoint = runtime.endpoint("test.metrics.generate")
    registry = PerformanceMetricsRegistry(endpoint.metrics)

    factory = RequestMetricsFactory(registry, metric_prefix="test_request_metrics")
    request = factory.new_request(input_tokens=128)
    request.record_tokens(1)  # first token: TTFT
    request.record_tokens(16)  # later update: ITL sampling
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
    request.record_tokens(1)
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
