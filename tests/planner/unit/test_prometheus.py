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

import math
from unittest.mock import patch

import pytest

from dynamo.planner.utils.prometheus import (
    FrontendMetric,
    FrontendMetricContainer,
    PrometheusAPIClient,
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
