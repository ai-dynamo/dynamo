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
from unittest.mock import MagicMock, patch

import pytest

from dynamo import prometheus_names
from dynamo.planner.monitoring.traffic_metrics import (
    DirectRouterMetricsClient,
    FrontendMetric,
    FrontendMetricContainer,
    Metrics,
    PrometheusAPIClient,
    _WORKER_METRIC_NAMES,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


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

    def test_get_avg_kv_hit_rate_returns_none_for_frontend_source(self):
        """Frontend source doesn't publish an aggregate kv_hit_rate, so the
        client should short-circuit to None rather than issue a PromQL query."""
        client = PrometheusAPIClient(
            "http://localhost:9090", "test-fe-namespace", metrics_source="frontend"
        )
        client.prom = MagicMock()
        client.prom.custom_query.return_value = [{"value": [0, "42.0"]}]
        result = client.get_avg_kv_hit_rate("60s", "mymodel")
        assert result is None
        client.prom.custom_query.assert_not_called()

    def test_get_avg_request_count_uses_router_requests_total(self, router_client):
        """get_avg_request_count with router source queries dynamo_component_router_requests_total."""
        result = router_client.get_avg_request_count("60s", "mymodel")
        assert result == 42.0

        call_args = str(router_client.prom.custom_query.call_args)
        expected_metric = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.REQUESTS_TOTAL}"
        assert expected_metric in call_args

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
        """get_avg_request_count (router) returns 0 when Prometheus has no data."""
        router_client.prom.custom_query.return_value = []
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
# on DirectRouterMetricsClient by mistake, which would have silently disabled
# the AIC closed loop because base.py calls them on prometheus_traffic_client
# (a PrometheusAPIClient instance).  Each test below exists to detect that
# class of regression.
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

    def test_get_avg_per_gpu_power_by_component_is_method_on_prometheus_api_client(
        self,
    ):
        client = self._client()
        assert hasattr(client, "get_avg_per_gpu_power_by_component"), (
            "get_avg_per_gpu_power_by_component missing from PrometheusAPIClient — "
            "the AIC closed-loop EMA update path calls this method via base.py."
        )
        assert callable(client.get_avg_per_gpu_power_by_component)

    def test_get_total_dgd_power_returns_float_on_match(self):
        client = self._client()
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.return_value = [{"value": [0, "1234.5"]}]
            result = client.get_total_dgd_power(
                k8s_namespace="kube-namespace", dgd_name="my-dgd"
            )
            assert result == pytest.approx(1234.5)
            call_args = str(mock_query.call_args).replace("'", '"')
            assert "DCGM_FI_DEV_POWER_USAGE" in call_args
            # exported_namespace carries the K8s namespace, NOT the dynamo
            # logical namespace.  Bare `namespace` would label the DCGM
            # exporter pod itself (DCGM exporter runs in its own ns), so
            # using it would silently match nothing once attribution works.
            assert 'exported_namespace="kube-namespace"' in call_args
            # exported_pod (not bare `pod`) is the workload pod label.
            # The operator emits `<dgd>-<replica-idx>-<service-key-lc>-<hash>`
            # so the regex must accept the `-<digits>-` segment.
            assert 'exported_pod=~"^my-dgd-[0-9]+-.*"' in call_args
            # The old broken regex must not appear.
            assert "(prefill|decode|agg|frontend)" not in call_args

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

    def test_get_total_dgd_power_uses_exported_pod_not_bare_pod(self):
        """Regression guard for the original DCGM attribution bug.

        Bare ``pod`` labels the DCGM exporter pod itself; only ``exported_pod``
        is rewritten to the workload pod by the DCGM exporter.  Mixing them
        up is a silent failure mode (no exception, just zero results).
        """
        client = self._client()
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.return_value = []
            client.get_total_dgd_power(k8s_namespace="ns", dgd_name="dgd")
            # The query string must use exported_pod, not the bare pod label.
            query_str = mock_query.call_args[0][0]
            assert "exported_pod=~" in query_str
            # No bare `pod=~` pattern should leak in.
            assert "{pod=~" not in query_str
            assert ",pod=~" not in query_str

    def test_get_avg_per_gpu_power_by_component_returns_float_on_match(self):
        client = self._client()
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.return_value = [{"value": [0, "537.0"]}]
            result = client.get_avg_per_gpu_power_by_component(
                interval="60s",
                k8s_namespace="kube-namespace",
                dgd_name="my-dgd",
                component="prefill",
                service_key="VllmPrefillWorker",
            )
            assert result == pytest.approx(537.0)
            call_args = str(mock_query.call_args).replace("'", '"')
            assert "avg_over_time" in call_args
            # Selector must filter by exported_pod regex matching the
            # operator's `<dgd>-<replica-idx>-<service-key-lc>-<hash>` form.
            assert (
                'exported_pod=~"^my-dgd-[0-9]+-vllmprefillworker-.*"'
                in call_args
            )
            assert 'exported_namespace="kube-namespace"' in call_args

    def test_get_avg_per_gpu_power_by_component_lowercases_service_key(self):
        """The operator embeds service_key.lower() in the pod name; the regex must too."""
        client = self._client()
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.return_value = [{"value": [0, "100.0"]}]
            client.get_avg_per_gpu_power_by_component(
                interval="60s",
                k8s_namespace="ns",
                dgd_name="dgd",
                component="decode",
                service_key="VllmDecodeWorker",  # mixed case
            )
            call_args = str(mock_query.call_args).replace("'", '"')
            assert "vllmdecodeworker" in call_args
            # The original mixed case must NOT appear in the regex.
            assert "VllmDecodeWorker" not in call_args

    def test_get_avg_per_gpu_power_by_component_empty_service_key_returns_none(self):
        """Empty service_key cannot build a meaningful regex; short-circuit to None."""
        client = self._client()
        with patch.object(client.prom, "custom_query") as mock_query:
            result = client.get_avg_per_gpu_power_by_component(
                interval="60s",
                k8s_namespace="ns",
                dgd_name="dgd",
                component="prefill",
                service_key="",
            )
            assert result is None
            # The Prometheus client must not even be called — a regex with
            # empty service_key would degenerate to `^dgd-[0-9]+--.*` which
            # never matches and just wastes a roundtrip.
            mock_query.assert_not_called()

    def test_get_avg_per_gpu_power_by_component_returns_none_on_empty(self):
        client = self._client()
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.return_value = []
            result = client.get_avg_per_gpu_power_by_component(
                interval="60s",
                k8s_namespace="ns",
                dgd_name="dgd",
                component="decode",
                service_key="VllmWorker",
            )
            assert result is None

    def test_get_avg_per_gpu_power_by_component_agg_component_pod_regex(self):
        """Agg mode uses the decode worker's k8s_name as service_key."""
        client = self._client()
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.return_value = [{"value": [0, "400.0"]}]
            result = client.get_avg_per_gpu_power_by_component(
                interval="30s",
                k8s_namespace="kube-namespace",
                dgd_name="my-dgd",
                component="agg",
                service_key="VllmWorker",
            )
            assert result == pytest.approx(400.0)
            call_args = str(mock_query.call_args).replace("'", '"')
            assert (
                'exported_pod=~"^my-dgd-[0-9]+-vllmworker-.*"'
                in call_args
            )

    def test_get_avg_per_gpu_power_by_component_returns_none_on_exception(self):
        client = self._client()
        with patch.object(client.prom, "custom_query") as mock_query:
            mock_query.side_effect = RuntimeError("prom down")
            result = client.get_avg_per_gpu_power_by_component(
                interval="60s",
                k8s_namespace="ns",
                dgd_name="dgd",
                component="prefill",
                service_key="VllmPrefillWorker",
            )
            assert result is None


# ---------------------------------------------------------------------------
# Metrics.is_valid() — gate that controls whether the planner has enough data
# ---------------------------------------------------------------------------


def _valid_metrics(**overrides) -> Metrics:
    """Return a Metrics instance with all required fields set to sane values."""
    base = dict(
        ttft=0.1,
        itl=0.05,
        num_req=10.0,
        isl=256.0,
        osl=128.0,
        request_duration=1.0,
        p_load=None,
        d_load=None,
        kv_hit_rate=None,
    )
    base.update(overrides)
    return Metrics(**base)


class TestMetricsIsValid:
    def test_all_required_fields_present_returns_true(self):
        assert _valid_metrics().is_valid() is True

    @pytest.mark.parametrize(
        "field",
        ["ttft", "itl", "num_req", "isl", "osl", "request_duration"],
    )
    def test_none_required_field_returns_false(self, field):
        m = _valid_metrics(**{field: None})
        assert m.is_valid() is False, f"is_valid() should be False when {field}=None"

    @pytest.mark.parametrize(
        "field",
        ["ttft", "itl", "num_req", "isl", "osl", "request_duration"],
    )
    def test_nan_required_field_returns_false(self, field):
        m = _valid_metrics(**{field: float("nan")})
        assert m.is_valid() is False, f"is_valid() should be False when {field}=NaN"

    def test_optional_fields_none_does_not_affect_validity(self):
        """p_load, d_load, kv_hit_rate are optional — None must not invalidate."""
        m = _valid_metrics(p_load=None, d_load=None, kv_hit_rate=None)
        assert m.is_valid() is True

    def test_optional_fields_set_does_not_affect_validity(self):
        m = _valid_metrics(p_load=0.8, d_load=0.3, kv_hit_rate=0.5)
        assert m.is_valid() is True

    def test_zero_values_are_valid(self):
        """Zero is a legal observed value — is_valid() must not reject it."""
        m = _valid_metrics(ttft=0.0, itl=0.0, num_req=0.0, isl=0.0, osl=0.0, request_duration=0.0)
        assert m.is_valid() is True


# ---------------------------------------------------------------------------
# DirectRouterMetricsClient._parse_prometheus_text
# ---------------------------------------------------------------------------

# Derive the full metric names from constants so the test stays in sync with
# any rename in prometheus_names.py without manual string updates.
_PREFILL_TOKENS_METRIC = _WORKER_METRIC_NAMES["active_prefill_tokens"]
_DECODE_BLOCKS_METRIC = _WORKER_METRIC_NAMES["active_decode_blocks"]
_LAST_TTFT_METRIC = _WORKER_METRIC_NAMES["last_ttft"]
_LAST_ISL_METRIC = _WORKER_METRIC_NAMES["last_isl"]
_LAST_ITL_METRIC = _WORKER_METRIC_NAMES["last_itl"]


def _prom_text(*lines: str) -> str:
    """Join Prometheus text-exposition lines with a trailing newline."""
    return "\n".join(lines) + "\n"


def _gauge_block(metric_name: str, labels: dict, value: float) -> str:
    """Build a minimal valid Prometheus gauge block for one sample."""
    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
    return (
        f"# HELP {metric_name} help\n"
        f"# TYPE {metric_name} gauge\n"
        f"{metric_name}{{{label_str}}} {value}"
    )


@pytest.fixture
def direct_client() -> DirectRouterMetricsClient:
    return DirectRouterMetricsClient(
        router_metrics_url="http://localhost:9091/metrics",
        dynamo_namespace="test-ns",
    )


class TestDirectRouterMetricsClientParseText:
    def test_parse_single_prefill_worker(self, direct_client):
        text = _prom_text(
            _gauge_block(
                _PREFILL_TOKENS_METRIC,
                {"worker_type": "prefill", "worker_id": "w0"},
                42.0,
            )
        )
        result = direct_client._parse_prometheus_text(text)
        assert "prefill" in result
        assert "w0" in result["prefill"]
        assert result["prefill"]["w0"]["active_prefill_tokens"] == pytest.approx(42.0)

    def test_parse_multiple_workers_same_type(self, direct_client):
        text = _prom_text(
            _gauge_block(
                _PREFILL_TOKENS_METRIC,
                {"worker_type": "prefill", "worker_id": "w0"},
                10.0,
            ),
            _gauge_block(
                _PREFILL_TOKENS_METRIC,
                {"worker_type": "prefill", "worker_id": "w1"},
                20.0,
            ),
        )
        result = direct_client._parse_prometheus_text(text)
        assert len(result["prefill"]) == 2
        assert result["prefill"]["w0"]["active_prefill_tokens"] == pytest.approx(10.0)
        assert result["prefill"]["w1"]["active_prefill_tokens"] == pytest.approx(20.0)

    def test_parse_separates_prefill_and_decode_workers(self, direct_client):
        text = _prom_text(
            _gauge_block(
                _PREFILL_TOKENS_METRIC,
                {"worker_type": "prefill", "worker_id": "p0"},
                5.0,
            ),
            _gauge_block(
                _DECODE_BLOCKS_METRIC,
                {"worker_type": "decode", "worker_id": "d0"},
                99.0,
            ),
        )
        result = direct_client._parse_prometheus_text(text)
        assert "p0" in result.get("prefill", {})
        assert "d0" in result.get("decode", {})
        assert "active_prefill_tokens" in result["prefill"]["p0"]
        assert "active_decode_blocks" in result["decode"]["d0"]

    def test_parse_all_five_worker_metric_fields(self, direct_client):
        worker_labels = {"worker_type": "prefill", "worker_id": "w0"}
        lines = []
        for field, metric in _WORKER_METRIC_NAMES.items():
            lines.append(_gauge_block(metric, worker_labels, float(hash(field) % 100)))
        text = _prom_text(*lines)
        result = direct_client._parse_prometheus_text(text)
        parsed = result["prefill"]["w0"]
        assert set(parsed.keys()) == set(_WORKER_METRIC_NAMES.keys()), (
            "Not all five worker metric fields were parsed"
        )

    def test_parse_ignores_metrics_not_in_worker_metric_names(self, direct_client):
        text = _prom_text(
            "# HELP some_other_metric help",
            "# TYPE some_other_metric gauge",
            'some_other_metric{worker_type="prefill",worker_id="w0"} 999',
        )
        result = direct_client._parse_prometheus_text(text)
        assert result == {}, "Non-target metrics must be filtered out"

    def test_parse_empty_text_returns_empty_dict(self, direct_client):
        result = direct_client._parse_prometheus_text("")
        assert result == {}

    def test_parse_missing_worker_labels_uses_unknown(self, direct_client):
        """Samples with no worker_type/worker_id labels fall back to 'unknown'."""
        text = _prom_text(
            f"# HELP {_PREFILL_TOKENS_METRIC} help",
            f"# TYPE {_PREFILL_TOKENS_METRIC} gauge",
            f"{_PREFILL_TOKENS_METRIC} 7.0",
        )
        result = direct_client._parse_prometheus_text(text)
        assert "unknown" in result
        assert "unknown" in result["unknown"]


# ---------------------------------------------------------------------------
# DirectRouterMetricsClient.get_recent_and_averaged_metrics
# ---------------------------------------------------------------------------


def _make_sample(worker_type: str, worker_id: str, **metrics) -> dict:
    """Build a single sample buffer entry."""
    return {worker_type: {worker_id: metrics}}


class TestDirectRouterMetricsClientGetRecentAndAveraged:
    def test_empty_buffer_returns_none(self, direct_client):
        assert direct_client.get_recent_and_averaged_metrics("prefill") is None

    def test_single_sample_recent_matches_that_sample(self, direct_client):
        direct_client._sample_buffer = [
            _make_sample("prefill", "w0", active_prefill_tokens=50.0, last_ttft=0.1)
        ]
        result = direct_client.get_recent_and_averaged_metrics("prefill")
        assert result is not None
        recent, per_worker_avg, cluster_avg = result
        assert recent["w0"]["active_prefill_tokens"] == pytest.approx(50.0)
        assert recent["w0"]["last_ttft"] == pytest.approx(0.1)

    def test_single_sample_per_worker_avg_equals_recent(self, direct_client):
        direct_client._sample_buffer = [
            _make_sample("prefill", "w0", active_prefill_tokens=50.0)
        ]
        recent, per_worker_avg, cluster_avg = direct_client.get_recent_and_averaged_metrics("prefill")
        assert per_worker_avg["w0"]["active_prefill_tokens"] == pytest.approx(50.0)

    def test_multiple_samples_per_worker_averaged_over_time(self, direct_client):
        direct_client._sample_buffer = [
            _make_sample("prefill", "w0", active_prefill_tokens=10.0),
            _make_sample("prefill", "w0", active_prefill_tokens=20.0),
            _make_sample("prefill", "w0", active_prefill_tokens=30.0),
        ]
        _, per_worker_avg, cluster_avg = direct_client.get_recent_and_averaged_metrics("prefill")
        assert per_worker_avg["w0"]["active_prefill_tokens"] == pytest.approx(20.0)
        assert cluster_avg["active_prefill_tokens"] == pytest.approx(20.0)

    def test_cluster_avg_aggregates_across_workers(self, direct_client):
        """cluster_averaged must be the mean across all workers, not just one."""
        direct_client._sample_buffer = [
            {
                "prefill": {
                    "w0": {"active_prefill_tokens": 10.0},
                    "w1": {"active_prefill_tokens": 30.0},
                }
            }
        ]
        _, per_worker_avg, cluster_avg = direct_client.get_recent_and_averaged_metrics("prefill")
        assert cluster_avg["active_prefill_tokens"] == pytest.approx(20.0)

    def test_filters_by_worker_type(self, direct_client):
        """Requesting 'decode' must not return prefill worker data."""
        direct_client._sample_buffer = [
            {
                "prefill": {"p0": {"active_prefill_tokens": 10.0}},
                "decode": {"d0": {"active_decode_blocks": 5.0}},
            }
        ]
        result_decode = direct_client.get_recent_and_averaged_metrics("decode")
        assert result_decode is not None
        recent, _, _ = result_decode
        assert "d0" in recent
        assert "p0" not in recent

    def test_requested_type_absent_in_samples_returns_none(self, direct_client):
        """If the buffer has only prefill samples, querying 'decode' should return None."""
        direct_client._sample_buffer = [
            _make_sample("prefill", "w0", active_prefill_tokens=1.0)
        ]
        result = direct_client.get_recent_and_averaged_metrics("decode")
        assert result is None

    def test_recent_reflects_latest_sample_not_oldest(self, direct_client):
        direct_client._sample_buffer = [
            _make_sample("prefill", "w0", active_prefill_tokens=1.0),
            _make_sample("prefill", "w0", active_prefill_tokens=2.0),
            _make_sample("prefill", "w0", active_prefill_tokens=99.0),
        ]
        recent, _, _ = direct_client.get_recent_and_averaged_metrics("prefill")
        assert recent["w0"]["active_prefill_tokens"] == pytest.approx(99.0)

    def test_workers_appearing_only_in_some_samples_averaged_correctly(self, direct_client):
        """A worker that appears in only 2 of 3 samples must be averaged over 2, not 3."""
        direct_client._sample_buffer = [
            _make_sample("prefill", "w0", active_prefill_tokens=10.0),
            _make_sample("prefill", "w0", active_prefill_tokens=20.0),
            _make_sample("prefill", "w1", active_prefill_tokens=50.0),
        ]
        _, per_worker_avg, _ = direct_client.get_recent_and_averaged_metrics("prefill")
        assert per_worker_avg["w0"]["active_prefill_tokens"] == pytest.approx(15.0)
        assert per_worker_avg["w1"]["active_prefill_tokens"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# PrometheusAPIClient.get_avg_request_duration — router source known gap
# ---------------------------------------------------------------------------


class TestGetAvgRequestDurationRouterSource:
    """The router path queries dynamo_work_handler_request_duration_seconds, which
    is not yet registered on the LocalRouter.  The test pins the queried metric name
    so a future fix can be validated, and verifies the graceful 0-fallback while the
    metric is absent."""

    def test_router_source_queries_work_handler_metric(self):
        # The router path uses the dynamo_component_ prefix (not dynamo_work_handler_)
        # until RouterRequestMetrics exposes router_request_duration_seconds.  The TODO
        # in traffic_metrics.py documents the intended future rename.  This test pins the
        # current query so a prefix change doesn't go unnoticed.
        client = PrometheusAPIClient(
            "http://localhost:9090", "test-ns", metrics_source="router"
        )
        client.prom = MagicMock()
        client.prom.custom_query.return_value = [{"value": [0, "0.5"]}]

        client.get_avg_request_duration("60s", "mymodel")

        call_args = str(client.prom.custom_query.call_args)
        expected_metric = (
            f"{prometheus_names.name_prefix.COMPONENT}_"
            f"{prometheus_names.work_handler.REQUEST_DURATION_SECONDS}"
        )
        assert expected_metric in call_args, (
            f"Router source must query {expected_metric!r} — "
            "if this name has changed, update the TODO in traffic_metrics.py too"
        )

    def test_router_source_returns_zero_when_metric_absent(self):
        """While the metric isn't published, Prometheus returns []; planner gets 0."""
        client = PrometheusAPIClient(
            "http://localhost:9090", "test-ns", metrics_source="router"
        )
        client.prom = MagicMock()
        client.prom.custom_query.return_value = []

        result = client.get_avg_request_duration("60s", "mymodel")
        assert result == 0

    def test_frontend_source_queries_frontend_metric(self):
        client = PrometheusAPIClient(
            "http://localhost:9090", "test-ns", metrics_source="frontend"
        )
        client.prom = MagicMock()
        client.prom.custom_query.return_value = []

        client.get_avg_request_duration("60s", "mymodel")

        call_args = str(client.prom.custom_query.call_args)
        assert prometheus_names.frontend_service.REQUEST_DURATION_SECONDS in call_args


# ---------------------------------------------------------------------------
# PrometheusAPIClient.get_avg_kv_hit_rate — additional edge cases
# ---------------------------------------------------------------------------


class TestGetAvgKvHitRateEdgeCases:
    def test_router_source_returns_none_on_nan(self):
        """NaN from Prometheus must become None, not 0.0, to avoid dragging down EMA."""
        client = PrometheusAPIClient(
            "http://localhost:9090", "test-ns", metrics_source="router"
        )
        client.prom = MagicMock()
        client.prom.custom_query.return_value = [{"value": [0, "NaN"]}]
        result = client.get_avg_kv_hit_rate("60s", "mymodel")
        assert result is None

    def test_router_source_returns_none_on_exception(self):
        client = PrometheusAPIClient(
            "http://localhost:9090", "test-ns", metrics_source="router"
        )
        client.prom = MagicMock()
        client.prom.custom_query.side_effect = RuntimeError("prom down")
        result = client.get_avg_kv_hit_rate("60s", "mymodel")
        assert result is None

    def test_router_source_returns_float_on_valid_result(self):
        client = PrometheusAPIClient(
            "http://localhost:9090", "test-ns", metrics_source="router"
        )
        client.prom = MagicMock()
        client.prom.custom_query.return_value = [{"value": [0, "0.75"]}]
        result = client.get_avg_kv_hit_rate("60s", "mymodel")
        assert result == pytest.approx(0.75)

    def test_router_namespace_filter_uses_underscores(self):
        """DYN_NAMESPACE 'my-ns-pool' must appear as 'my_ns_pool' in the PromQL filter."""
        client = PrometheusAPIClient(
            "http://localhost:9090", "my-ns-pool", metrics_source="router"
        )
        client.prom = MagicMock()
        client.prom.custom_query.return_value = [{"value": [0, "0.5"]}]
        client.get_avg_kv_hit_rate("60s", "mymodel")
        call_args = str(client.prom.custom_query.call_args)
        assert "my_ns_pool" in call_args, (
            "Dashes in DYN_NAMESPACE must be normalized to underscores for the "
            "PromQL dynamo_namespace filter (MetricsHierarchy uses underscores)"
        )
