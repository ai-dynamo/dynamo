# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
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


def test_accept_length_uses_current_runtime_namespace():
    namespace_source = MagicMock()
    namespace_source.runtime_namespace.return_value = "base-ns-workerhash"
    provider, client, client_patch = _provider(namespace_source)
    client.get_avg_spec_decode_accept_length.return_value = 2.5
    try:
        assert provider.collect_accept_length("30s") == 2.5
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


def test_accept_length_falls_back_to_configured_namespace():
    provider, client, client_patch = _provider()
    client.get_avg_spec_decode_accept_length.return_value = 2.5
    try:
        provider.collect_accept_length("30s")
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
