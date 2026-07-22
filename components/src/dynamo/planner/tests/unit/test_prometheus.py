# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from prometheus_api_client import PrometheusApiClientException

from dynamo import prometheus_names
from dynamo.planner.core.throughput_scaling import ThroughputScalingMixin
from dynamo.planner.monitoring.traffic_metrics import (
    FrontendMetric,
    FrontendMetricContainer,
    Metrics,
    PrometheusAPIClient,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _FixedPrefillCapacity:
    def __init__(self, engine_rps: float):
        self.engine_rps = engine_rps

    def find_engine_capacity_rps(self, **kwargs):
        return SimpleNamespace(rps=self.engine_rps, ttft_ms=100.0, eligible=True)


class _ThroughputScalingHarness(ThroughputScalingMixin):
    def __init__(self, engine_rps: float):
        self._config = SimpleNamespace(ttft_ms=200.0, min_endpoint=1)
        self._prefill_regression = _FixedPrefillCapacity(engine_rps)
        self._diag_throughput_reason = None
        self._diag_engine_rps_prefill = None


@pytest.fixture
def mock_prometheus_result():
    """Fixture providing mock prometheus result data for testing"""
    return [
        {
            "metric": {
                "container": "main",
                "dynamo_namespace": "different_namespace",
                "model": "different_model",
                "namespace": "dynamo-system",
            },
            "value": [1758857776.071, 10.5],
        },
        {
            "metric": {
                "container": "main",
                "dynamo_namespace": "target_namespace",
                "model": "target_model",
                "namespace": "dynamo-system",
            },
            "value": [1758857776.071, 42.7],
        },
        {
            "metric": {
                "container": "worker",
                "dynamo_namespace": "target_namespace",
                "model": "target_model",
                "namespace": "dynamo-system",
            },
            "value": [1758857776.071, 35.5],
        },
        {
            "metric": {
                "container": "sidecar",
                "dynamo_namespace": "target_namespace",
                "model": "target_model",
                "namespace": "dynamo-system",
            },
            "value": [30.0, 15.5],
        },
    ]


def test_frontend_metric_container_with_nan_value():
    test_data = {
        "metric": {
            "container": "main",
            "dynamo_namespace": "vllm-disagg-planner",
            "endpoint": "http",
            "instance": "10.244.2.163:8000",
            "job": "dynamo-system/dynamo-frontend",
            "model": "qwen/qwen3-0.6b",
            "namespace": "dynamo-system",
            "pod": "vllm-disagg-planner-frontend-865f84c49-6q7s5",
        },
        "value": [1758857776.071, "NaN"],
    }

    container = FrontendMetricContainer.model_validate(test_data)
    assert container.metric.container == "main"
    assert container.metric.dynamo_namespace == "vllm-disagg-planner"
    assert container.metric.endpoint == "http"
    assert container.metric.instance == "10.244.2.163:8000"
    assert container.metric.job == "dynamo-system/dynamo-frontend"
    assert container.metric.model == "qwen/qwen3-0.6b"
    assert container.metric.namespace == "dynamo-system"
    assert container.metric.pod == "vllm-disagg-planner-frontend-865f84c49-6q7s5"
    assert container.value[0] == 1758857776.071
    assert math.isnan(
        container.value[1]
    )  # becomes special float value that can't be asserted to itself

    test_data["value"][1] = 42.5  # type: ignore[index]
    container = FrontendMetricContainer.model_validate(test_data)
    assert container.value[1] == 42.5


def test_metrics_normalize_nan_averages_for_idle_window():
    metrics = Metrics(
        ttft=math.nan,
        itl=math.nan,
        num_req=0,
        isl=math.nan,
        osl=math.nan,
        request_duration=math.nan,
    )

    normalized = metrics.normalize_idle_nans()

    assert normalized == ["ttft", "itl", "isl", "osl", "request_duration"]
    assert metrics.is_valid()
    assert metrics.ttft == 0
    assert metrics.itl == 0
    assert metrics.isl == 0
    assert metrics.osl == 0
    assert metrics.request_duration == 0


def test_metrics_keep_nan_averages_invalid_with_traffic():
    metrics = Metrics(
        ttft=math.nan,
        itl=math.nan,
        num_req=1,
        isl=math.nan,
        osl=math.nan,
        request_duration=math.nan,
    )

    assert metrics.normalize_idle_nans() == []
    assert not metrics.is_valid()


def test_metrics_keep_missing_idle_averages_invalid():
    metrics = Metrics(
        ttft=None,
        itl=math.nan,
        num_req=0,
        isl=math.nan,
        osl=math.nan,
        request_duration=math.nan,
    )

    assert metrics.normalize_idle_nans() == [
        "itl",
        "isl",
        "osl",
        "request_duration",
    ]
    assert not metrics.is_valid()


def test_frontend_metric_with_partial_data():
    """Test FrontendMetric with partial data (optional fields)"""
    test_data = {
        "container": "main",
        "model": "qwen/qwen3-0.6b",
        "namespace": "dynamo-system",
    }

    metric = FrontendMetric.model_validate(test_data)

    # Assert provided fields
    assert metric.container == "main"
    assert metric.model == "qwen/qwen3-0.6b"
    assert metric.namespace == "dynamo-system"

    # Assert optional fields are None
    assert metric.dynamo_namespace is None
    assert metric.endpoint is None
    assert metric.instance is None
    assert metric.job is None
    assert metric.pod is None


def test_get_average_metric_none_result():
    """Test _get_average_metric when prometheus returns None"""
    # TODO: Replace hardcoded port with allocate_port() from tests.utils.port_utils
    #       for xdist-safe parallel execution.
    client = PrometheusAPIClient("http://localhost:9090", "test_namespace")

    with patch.object(client.prom, "custom_query") as mock_query:
        mock_query.return_value = None

        result = client._get_average_metric(
            full_metric_name="test_metric",
            interval="60s",
            operation_name="test operation",
            model_name="test_model",
        )

        assert result == 0


def test_get_average_metric_empty_result():
    """Test _get_average_metric when prometheus returns empty list"""
    client = PrometheusAPIClient("http://localhost:9090", "test_namespace")

    with patch.object(client.prom, "custom_query") as mock_query:
        mock_query.return_value = []

        result = client._get_average_metric(
            full_metric_name="test_metric",
            interval="60s",
            operation_name="test operation",
            model_name="test_model",
        )

        assert result == 0


def test_get_average_metric_no_matching_containers(mock_prometheus_result):
    """Test _get_average_metric with valid containers but no matches"""
    client = PrometheusAPIClient("http://localhost:9090", "test_namespace")

    with patch.object(client.prom, "custom_query") as mock_query:
        # Use only the first container which doesn't match target criteria
        mock_query.return_value = [mock_prometheus_result[0]]

        result = client._get_average_metric(
            full_metric_name="test_metric",
            interval="60s",
            operation_name="test operation",
            model_name="target_model",
        )

        assert result == 0


def test_get_average_metric_one_matching_container(mock_prometheus_result):
    """Test _get_average_metric with one matching container"""
    client = PrometheusAPIClient("http://localhost:9090", "target_namespace")

    with patch.object(client.prom, "custom_query") as mock_query:
        # Use first two containers - one doesn't match, one does
        mock_query.return_value = mock_prometheus_result[:2]

        result = client._get_average_metric(
            full_metric_name="test_metric",
            interval="60s",
            operation_name="test operation",
            model_name="target_model",
        )

        assert result == 42.7


def test_get_average_metric_frontend_path_filters_nan_values():
    """Frontend-source path: NaN samples (from 0/0 increase ratios on quiet
    windows) must be dropped, matching the router-source NaN handling.
    Without this, Metrics.is_valid() would silently fail on a quiet cluster."""
    client = PrometheusAPIClient("http://localhost:9090", "target_namespace")

    nan_then_valid = [
        {
            "metric": {
                "container": "main",
                "dynamo_namespace": "target_namespace",
                "model": "target_model",
            },
            "value": [0.0, "NaN"],
        },
        {
            "metric": {
                "container": "main",
                "dynamo_namespace": "target_namespace",
                "model": "target_model",
            },
            "value": [0.0, 12.0],
        },
    ]

    with patch.object(client.prom, "custom_query") as mock_query:
        mock_query.return_value = nan_then_valid
        result = client._get_average_metric(
            full_metric_name="test_metric",
            interval="60s",
            operation_name="test operation",
            model_name="target_model",
        )

    assert result == 12.0


def test_get_average_metric_frontend_path_returns_zero_when_only_nan():
    """If every matching container is NaN, the path must return 0
    (not propagate NaN), so downstream Metrics.is_valid() stays meaningful."""
    client = PrometheusAPIClient("http://localhost:9090", "target_namespace")

    only_nan = [
        {
            "metric": {
                "container": "main",
                "dynamo_namespace": "target_namespace",
                "model": "target_model",
            },
            "value": [0.0, "NaN"],
        }
    ]

    with patch.object(client.prom, "custom_query") as mock_query:
        mock_query.return_value = only_nan
        result = client._get_average_metric(
            full_metric_name="test_metric",
            interval="60s",
            operation_name="test operation",
            model_name="target_model",
        )

    assert result == 0


def test_get_average_metric_with_validation_error():
    """Test _get_average_metric with one valid container and one that fails validation"""
    client = PrometheusAPIClient("http://localhost:9090", "target_namespace")

    mock_result = [
        {
            "metric": {
                "container": "main",
                "dynamo_namespace": "target_namespace",
                "model": "target_model",
                "namespace": "dynamo-system",
            },
            "value": [1758857776.071, 25.5],
        },
        {
            # Invalid structure - missing required fields that will cause validation error
            "invalid_structure": "bad_data",
            "value": "not_a_tuple",
        },
    ]

    with patch.object(client.prom, "custom_query") as mock_query:
        mock_query.return_value = mock_result

        result = client._get_average_metric(
            full_metric_name="test_metric",
            interval="60s",
            operation_name="test operation",
            model_name="target_model",
        )

        assert result == 25.5


def test_get_average_metric_multiple_matching_containers(mock_prometheus_result):
    """Test _get_average_metric with multiple matching containers returns average"""
    client = PrometheusAPIClient("http://localhost:9090", "target_namespace")

    with patch.object(client.prom, "custom_query") as mock_query:
        # Use containers 1, 2, 3 which all match target criteria
        mock_query.return_value = mock_prometheus_result[1:]

        result = client._get_average_metric(
            full_metric_name="test_metric",
            interval="60s",
            operation_name="test operation",
            model_name="target_model",
        )

        # Average of 42.7, 35.5, and 15.5 (using value[1] from each container)
        expected = (42.7 + 35.5 + 15.5) / 3
        assert result == expected


def test_get_avg_request_count_uses_started_requests():
    """Frontend request count uses started requests, not completed responses."""
    client = PrometheusAPIClient("http://localhost:9090", "target_namespace")

    started = [
        {
            "metric": {
                "dynamo_namespace": "target_namespace",
                "model": "target_model",
            },
            "value": [1758857776.071, 150.0],
        },
        {
            "metric": {
                "dynamo_namespace": "other_namespace",
                "model": "target_model",
            },
            "value": [1758857776.071, 1000.0],
        },
    ]

    with patch.object(client.prom, "custom_query") as mock_query:
        mock_query.return_value = started

        result = client.get_avg_request_count("30s", "TARGET_MODEL")

    assert result == 150.0
    queries = [call.kwargs["query"] for call in mock_query.call_args_list]
    assert "dynamo_frontend_requests_started_total" in queries[0]
    assert "increase(" in queries[0]
    assert len(queries) == 1


def test_get_avg_request_count_falls_back_to_completed_when_started_missing():
    """Older frontend images without started counter still report completed count."""
    client = PrometheusAPIClient("http://localhost:9090", "target_namespace")

    completed = [
        {
            "metric": {
                "dynamo_namespace": "target_namespace",
                "model": "target_model",
            },
            "value": [1758857776.071, 73.0],
        }
    ]

    with patch.object(client.prom, "custom_query") as mock_query:
        mock_query.side_effect = [[], completed]

        result = client.get_avg_request_count("30s", "target_model")

    assert result == 73.0
    queries = [call.kwargs["query"] for call in mock_query.call_args_list]
    assert "dynamo_frontend_requests_started_total" in queries[0]
    assert "dynamo_frontend_requests_total" in queries[1]


def test_vllm_spec_decode_accept_length_query_derives_from_counters():
    client = PrometheusAPIClient("http://localhost:9090", "test-ns")
    client.prom = MagicMock()
    client.prom.custom_query.return_value = [{"value": [0, "2.5"]}]

    result = client.get_avg_spec_decode_accept_length(
        "60s", "vllm", "backend", "Qwen/Qwen3"
    )

    assert result == 2.5
    query = client.prom.custom_query.call_args.kwargs["query"]
    assert "vllm:spec_decode_num_accepted_tokens_total" in query
    assert "vllm:spec_decode_num_drafts_total" in query
    assert 'dynamo_namespace="test-ns"' in query
    assert 'dynamo_component="backend"' in query
    assert 'model="Qwen/Qwen3"' in query


def test_spec_decode_accept_length_query_can_use_worker_runtime_namespace():
    client = PrometheusAPIClient("http://localhost:9090", "base-ns")
    client.prom = MagicMock()
    client.prom.custom_query.return_value = [{"value": [0, "2.5"]}]

    result = client.get_avg_spec_decode_accept_length(
        "60s",
        "vllm",
        "backend",
        "Qwen/Qwen3",
        namespace="base-ns-workerhash",
        endpoint_name="serve",
    )

    assert result == 2.5
    query = client.prom.custom_query.call_args.kwargs["query"]
    assert 'dynamo_namespace="base-ns-workerhash"' in query
    assert 'dynamo_endpoint="serve"' in query


def test_sglang_spec_decode_accept_length_query_uses_gauge():
    client = PrometheusAPIClient("http://localhost:9090", "test-ns")
    client.prom = MagicMock()
    client.prom.custom_query.return_value = [{"value": [0, "1.8"]}]

    result = client.get_avg_spec_decode_accept_length(
        "30s", "sglang", "decode", "model"
    )

    assert result == 1.8
    query = client.prom.custom_query.call_args.kwargs["query"]
    assert "sglang:spec_accept_length" in query
    assert "avg_over_time" in query


def test_trtllm_spec_decode_accept_length_query_uses_gauge():
    client = PrometheusAPIClient("http://localhost:9090", "test-ns")
    client.prom = MagicMock()
    client.prom.custom_query.return_value = [{"value": [0, "2.0"]}]

    result = client.get_avg_spec_decode_accept_length(
        "30s", "trtllm", "tensorrt_llm", "model"
    )

    assert result == 2.0
    query = client.prom.custom_query.call_args.kwargs["query"]
    assert "trtllm_spec_decode_acceptance_length" in query
    assert "avg_over_time" in query


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-Inf"])
def test_spec_decode_accept_length_returns_none_for_invalid_values(value):
    client = PrometheusAPIClient("http://localhost:9090", "test-ns")
    client.prom = MagicMock()
    client.prom.custom_query.return_value = [{"value": [0, value]}]

    assert (
        client.get_avg_spec_decode_accept_length("30s", "vllm", "backend", "model")
        is None
    )


# ---------------------------------------------------------------------------
# Router metrics source tests
# ---------------------------------------------------------------------------


@pytest.fixture
def router_client():
    """PrometheusAPIClient configured with metrics_source='router'."""
    # TODO: Replace hardcoded port with allocate_port() from tests.utils.port_utils
    #       for xdist-safe parallel execution.
    client = PrometheusAPIClient(
        "http://localhost:9090", "test-fe-namespace", metrics_source="router"
    )
    client.prom = MagicMock()
    client.prom.custom_query.return_value = [{"value": [0, "42.0"]}]
    return client


class TestPrometheusAPIClientRouterSource:
    """Tests for PrometheusAPIClient when metrics_source='router'."""

    def test_get_avg_inter_token_latency_dispatches_to_router_histogram(
        self, router_client
    ):
        """get_avg_inter_token_latency with router source queries dynamo_component_router_* metric."""
        result = router_client.get_avg_inter_token_latency("60s", "mymodel")
        assert result == 42.0

        call_args = str(router_client.prom.custom_query.call_args)
        expected_metric = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.INTER_TOKEN_LATENCY_SECONDS}"
        assert expected_metric in call_args

    def test_get_avg_time_to_first_token_dispatches_to_router_histogram(
        self, router_client
    ):
        """get_avg_time_to_first_token with router source queries dynamo_component_router_* metric."""
        result = router_client.get_avg_time_to_first_token("60s", "mymodel")
        assert result == 42.0

        call_args = str(router_client.prom.custom_query.call_args)
        expected_metric = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.TIME_TO_FIRST_TOKEN_SECONDS}"
        assert expected_metric in call_args

    def test_get_avg_input_sequence_tokens_dispatches_to_router_histogram(
        self, router_client
    ):
        """get_avg_input_sequence_tokens with router source queries dynamo_component_router_* metric."""
        result = router_client.get_avg_input_sequence_tokens("60s", "mymodel")
        assert result == 42.0

        call_args = str(router_client.prom.custom_query.call_args)
        expected_metric = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.INPUT_SEQUENCE_TOKENS}"
        assert expected_metric in call_args

    def test_get_avg_output_sequence_tokens_dispatches_to_router_histogram(
        self, router_client
    ):
        """get_avg_output_sequence_tokens with router source queries dynamo_component_router_* metric."""
        result = router_client.get_avg_output_sequence_tokens("60s", "mymodel")
        assert result == 42.0

        call_args = str(router_client.prom.custom_query.call_args)
        expected_metric = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.OUTPUT_SEQUENCE_TOKENS}"
        assert expected_metric in call_args

    def test_get_avg_kv_hit_rate_dispatches_to_router_histogram(self, router_client):
        """get_avg_kv_hit_rate with router source queries dynamo_component_router_kv_hit_rate."""
        # Return a plausible 0.0-1.0 ratio rather than the default 42.0 fixture.
        router_client.prom.custom_query.return_value = [{"value": [0, "0.35"]}]
        result = router_client.get_avg_kv_hit_rate("60s", "mymodel")
        assert result == 0.35

        call_args = str(router_client.prom.custom_query.call_args)
        expected_metric = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.KV_HIT_RATE}"
        assert expected_metric in call_args

    def test_get_avg_kv_hit_rate_frontend_source_queries_router_histogram(self):
        """Frontend source can expose router component metrics, so kv_hit_rate
        should still query dynamo_component_router_kv_hit_rate."""
        client = PrometheusAPIClient(
            "http://localhost:9090", "test-fe-namespace", metrics_source="frontend"
        )
        client.prom = MagicMock()
        client.prom.custom_query.return_value = [{"value": [0, "0.42"]}]
        result = client.get_avg_kv_hit_rate("60s", "mymodel")
        assert result == 0.42

        call_args = str(client.prom.custom_query.call_args)
        expected_metric = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.KV_HIT_RATE}"
        assert expected_metric in call_args

    def test_get_avg_request_count_uses_router_requests_started_total(
        self, router_client
    ):
        """Router count prefers admitted requests for each upgraded router."""
        result = router_client.get_avg_request_count("60s", "mymodel")
        assert result == 42.0

        call_args = str(router_client.prom.custom_query.call_args)
        started_metric = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.REQUESTS_STARTED_TOTAL}"
        completed_metric = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.REQUESTS_TOTAL}"
        assert started_metric in call_args
        assert completed_metric in call_args
        assert "or ignoring(__name__)" in call_args
        assert router_client.prom.custom_query.call_count == 1

    def test_router_request_count_falls_back_when_coalesced_query_is_empty(
        self, router_client, caplog
    ):
        """An empty coalesced query retries the completed counter directly."""
        router_client.prom.custom_query.side_effect = [
            [],
            [{"value": [0, "17.0"]}],
        ]

        with caplog.at_level(logging.WARNING):
            result = router_client.get_avg_request_count("60s", "mymodel")

        assert result == 17.0
        queries = [
            call.kwargs["query"]
            for call in router_client.prom.custom_query.call_args_list
        ]
        assert "dynamo_component_router_requests_started_total" in queries[0]
        assert "dynamo_component_router_requests_total" in queries[1]
        assert any(
            "may underestimate demand" in record.message for record in caplog.records
        )

    def test_router_request_count_falls_back_on_expected_query_error(
        self, router_client, caplog
    ):
        router_client.prom.custom_query.side_effect = [
            PrometheusApiClientException("started query failed"),
            [{"value": [0, "19.0"]}],
        ]

        with caplog.at_level(logging.WARNING):
            result = router_client.get_avg_request_count("60s", "mymodel")

        assert result == 19.0
        assert router_client.prom.custom_query.call_count == 2
        assert any(
            "started query failed" in record.message for record in caplog.records
        )

    def test_router_request_count_returns_zero_when_completed_query_fails(
        self, router_client
    ):
        router_client.prom.custom_query.side_effect = [
            [],
            PrometheusApiClientException("completed query failed"),
        ]

        result = router_client.get_avg_request_count("60s", "mymodel")

        assert result == 0

    def test_router_request_count_reraises_malformed_response(
        self, router_client, caplog
    ):
        router_client.prom.custom_query.return_value = [{"unexpected": "shape"}]

        with caplog.at_level(logging.ERROR), pytest.raises(KeyError):
            router_client.get_avg_request_count("60s", "mymodel")

        assert router_client.prom.custom_query.call_count == 1
        assert any(
            "Unexpected error querying admitted router requests" in record.message
            for record in caplog.records
        )

    def test_router_request_count_falls_back_when_started_is_nan(self, router_client):
        router_client.prom.custom_query.side_effect = [
            [{"value": [0, "NaN"]}],
            [{"value": [0, "23.0"]}],
        ]

        result = router_client.get_avg_request_count("60s", "mymodel")

        assert result == 23.0

    def test_router_started_count_preserves_backpressured_scale_up_signal(
        self, router_client
    ):
        """Admitted demand drives the Planner replica calculation under backpressure."""
        interval_seconds = 60.0
        admitted_count = 120.0
        completed_count = 60.0
        engine_rps = 1.0
        router_client.prom.custom_query.return_value = [
            {"value": [0, str(admitted_count)]}
        ]

        observed_count = router_client.get_avg_request_count("60s", "mymodel")
        scaling = _ThroughputScalingHarness(engine_rps)
        admitted_replicas = scaling._compute_prefill_replicas(
            demand_rps=observed_count / interval_seconds,
            isl=1000,
            osl=100,
        )
        completed_replicas = scaling._compute_prefill_replicas(
            demand_rps=completed_count / interval_seconds,
            isl=1000,
            osl=100,
        )

        assert admitted_replicas == 2
        assert completed_replicas == 1
        assert router_client.prom.custom_query.call_count == 1

    def test_router_request_count_preserves_valid_zero(self, router_client):
        router_client.prom.custom_query.return_value = [{"value": [0, "0.0"]}]

        result = router_client.get_avg_request_count("60s", "mymodel")

        assert result == 0.0
        assert router_client.prom.custom_query.call_count == 1

    def test_dynamo_namespace_filter_in_router_histogram_query(self, router_client):
        """Router histogram query must filter by dynamo_namespace so each pool planner
        only reads its own LocalRouter's metrics, not the cluster-wide aggregate.
        dynamo_component_router_* metrics use MetricsHierarchy which injects dynamo_namespace
        with underscores. DYN_NAMESPACE dashes are normalized to underscores for the PromQL filter.
        """
        router_client.get_avg_inter_token_latency("60s", "mymodel")
        call_args = str(router_client.prom.custom_query.call_args)
        assert "dynamo_namespace" in call_args, (
            "dynamo_namespace filter missing from router histogram query — "
            "without it, all pool planners read the same cluster-wide aggregate"
        )
        # MetricsHierarchy injects underscores; DYN_NAMESPACE dashes are normalized
        assert "test_fe_namespace" in call_args

    def test_dynamo_namespace_filter_in_router_request_count_query(self, router_client):
        """Router request count query must filter by dynamo_namespace.
        dynamo_component_router_* get dynamo_namespace from MetricsHierarchy (underscores).
        """
        router_client.get_avg_request_count("60s", "mymodel")
        call_args = str(router_client.prom.custom_query.call_args)
        assert "dynamo_namespace" in call_args, (
            "dynamo_namespace filter missing from router request count query — "
            "without it, all pool planners read the same cluster-wide aggregate"
        )
        # MetricsHierarchy injects underscores; DYN_NAMESPACE dashes are normalized
        assert "test_fe_namespace" in call_args

    def test_router_histogram_returns_zero_on_empty_result(self, router_client):
        """_get_router_average_histogram returns 0 when Prometheus has no data."""
        router_client.prom.custom_query.return_value = []
        result = router_client.get_avg_inter_token_latency("60s", "mymodel")
        assert result == 0

    def test_router_request_count_returns_zero_on_empty_result(self, router_client):
        """Router request count returns 0 when neither counter has data."""
        router_client.prom.custom_query.return_value = []
        result = router_client.get_avg_request_count("60s", "mymodel")
        assert result == 0

    def test_router_request_count_returns_zero_when_both_counters_are_nan(
        self, router_client
    ):
        router_client.prom.custom_query.return_value = [{"value": [0, "NaN"]}]

        result = router_client.get_avg_request_count("60s", "mymodel")

        assert result == 0

    def test_router_histogram_returns_zero_on_nan(self, router_client):
        """_get_router_average_histogram returns 0 when value is NaN."""
        router_client.prom.custom_query.return_value = [{"value": [0, "NaN"]}]
        result = router_client.get_avg_inter_token_latency("60s", "mymodel")
        assert result == 0

    def test_warn_if_router_not_scraped_logs_warning_when_absent(
        self, router_client, caplog
    ):
        """warn_if_router_not_scraped logs a warning when absent() returns a result."""
        router_client.prom.custom_query.return_value = [{"value": [0, "1"]}]
        with caplog.at_level(logging.WARNING):
            router_client.warn_if_router_not_scraped()
        assert any(
            "No 'dynamo_component_router_requests_total'" in r.message
            for r in caplog.records
        )

    def test_warn_if_router_not_scraped_silent_when_present(
        self, router_client, caplog
    ):
        """warn_if_router_not_scraped is silent when the metric exists (absent() returns empty)."""
        router_client.prom.custom_query.return_value = []
        with caplog.at_level(logging.WARNING):
            router_client.warn_if_router_not_scraped()
        assert not any(
            "dynamo_component_router_requests_total" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Power-aware planner DCGM queries (regression suite for bug #1)
#
# These methods MUST live on PrometheusAPIClient — they were briefly placed
# on DirectRouterMetricsClient by mistake.  base.py resolves the AIC closed
# loop's DCGM queries on prometheus_traffic_client (a PrometheusAPIClient
# instance), so a misplaced method would silently disable that loop.
# NOTE: get_total_dgd_power is Phase 3 (future) support and is not yet called
# by base.py in Phase 1 — the projected-watts gauge is published from static
# config, not this query.  This suite pins its class placement and query shape
# now so the wiring is correct when the AIC loop lands.  Each test below
# exists to detect that class of regression.
# ---------------------------------------------------------------------------


class TestPowerAwareDcgmQueries:
    """Smoke tests pinning the DCGM power queries to PrometheusAPIClient."""

    def _client(self) -> PrometheusAPIClient:
        return PrometheusAPIClient("http://localhost:9090", "test_namespace")

    def test_get_total_dgd_power_is_method_on_prometheus_api_client(self):
        """The method must exist on PrometheusAPIClient — not on a sibling class."""
        client = self._client()
        assert hasattr(client, "get_total_dgd_power"), (
            "get_total_dgd_power missing from PrometheusAPIClient — was it "
            "accidentally placed on DirectRouterMetricsClient?"
        )
        assert callable(client.get_total_dgd_power)

    def test_get_total_dgd_power_returns_float_on_match(self):
        client = self._client()
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.return_value = [{"value": [0, "1234.5"]}]
            result = client.get_total_dgd_power(
                k8s_namespace="kube-namespace", dgd_name="my-dgd"
            )
            assert result == pytest.approx(1234.5)
            # Assert on the raw query string (positional arg), not repr() —
            # the escaped regex contains backslashes that repr would double.
            query_str = mock_query.call_args[0][0]
            assert "DCGM_FI_DEV_POWER_USAGE" in query_str
            # exported_namespace carries the K8s namespace, NOT the dynamo
            # logical namespace.  Bare `namespace` would label the DCGM
            # exporter pod itself (DCGM exporter runs in its own ns), so
            # using it would silently match nothing once attribution works.
            assert 'exported_namespace="kube-namespace"' in query_str
            # DGD identity is matched via the operator-stamped pod label,
            # not a pod-name regex.
            assert 'nvidia_com_dynamo_graph_deployment_name="my-dgd"' in query_str

    def test_get_total_dgd_power_label_is_exact_not_regex(self):
        """dgd_name is an exact label value, not a regex — dots are literal.

        A name like "my.dgd" must match only pods whose label value is
        exactly "my.dgd"; it must not be treated as a PromQL regex where
        "." matches any character.  Verifies the switch from exported_pod=~
        to nvidia_com_...="<exact>".
        """
        client = self._client()

        # Name with a dot: must appear as a plain string, no backslash escaping.
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.return_value = []
            client.get_total_dgd_power(k8s_namespace="ns", dgd_name="my.dgd")
            query_str = mock_query.call_args[0][0]
            assert 'nvidia_com_dynamo_graph_deployment_name="my.dgd"' in query_str
            assert "exported_pod" not in query_str

        # Plain alphanumeric name: no stray backslashes or regex syntax.
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.return_value = []
            client.get_total_dgd_power(k8s_namespace="ns", dgd_name="qwen3quickstart")
            query_str = mock_query.call_args[0][0]
            assert (
                'nvidia_com_dynamo_graph_deployment_name="qwen3quickstart"' in query_str
            )

    def test_get_total_dgd_power_returns_none_on_empty(self):
        client = self._client()
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.return_value = []
            result = client.get_total_dgd_power(
                k8s_namespace="kube-namespace", dgd_name="my-dgd"
            )
            assert result is None

    def test_get_total_dgd_power_returns_none_on_exception(self):
        client = self._client()
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.side_effect = RuntimeError("prom down")
            result = client.get_total_dgd_power(
                k8s_namespace="kube-namespace", dgd_name="my-dgd"
            )
            assert result is None

    def test_get_total_dgd_power_uses_dgd_label_not_pod_name(self):
        """DGD identity must be matched via the operator pod label, not pod name.

        The label ``nvidia_com_dynamo_graph_deployment_name`` is the
        DCGM-exporter-sanitized form of the operator label stamped on every
        workload pod. Neither ``exported_pod`` (name-regex) nor bare ``pod``
        must appear.
        """
        client = self._client()
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.return_value = []
            client.get_total_dgd_power(k8s_namespace="ns", dgd_name="my-dgd")
            query_str = mock_query.call_args[0][0]
            assert 'nvidia_com_dynamo_graph_deployment_name="my-dgd"' in query_str
            assert "exported_pod" not in query_str
            assert "{pod=" not in query_str
            assert ",pod=" not in query_str
