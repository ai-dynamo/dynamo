# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Prometheus utilities."""

from unittest.mock import Mock, patch

import pytest

from dynamo.common.utils.prometheus import get_prometheus_expfmt

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
]


class TestGetPrometheusExpfmt:
    """Test class for get_prometheus_expfmt function."""

    TRTLLM_SAMPLE_METRICS = """# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 123.0
# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 45.6
# HELP trtllm_request_latency_seconds Request latency in seconds
# TYPE trtllm_request_latency_seconds histogram
trtllm_request_latency_seconds_bucket{le="0.1"} 10.0
trtllm_request_latency_seconds_count 25.0
# HELP trtllm_num_requests_running Number of requests currently running
# TYPE trtllm_num_requests_running gauge
trtllm_num_requests_running 3.0
# HELP trtllm_tokens_per_second Tokens generated per second
# TYPE trtllm_tokens_per_second gauge
trtllm_tokens_per_second 245.7
"""

    def test_trtllm_use_case(self):
        """Test TensorRT-LLM use case: filter to include only trtllm_* metrics (after traffic)."""
        registry = Mock()

        with patch(
            "prometheus_client.generate_latest",
            return_value=self.TRTLLM_SAMPLE_METRICS.encode("utf-8"),
        ):
            result = get_prometheus_expfmt(
                registry,
                metric_prefix_filters=["trtllm_"],
            )

        # Should not contain excluded metrics (filtered out by metric_prefix_filters)
        assert "python_gc_objects_collected_total" not in result
        assert "process_cpu_seconds_total" not in result

        # All remaining metrics should have trtllm_ prefix (already present from TRT-LLM engine)
        assert "trtllm_request_latency_seconds" in result
        assert "trtllm_num_requests_running" in result
        assert "trtllm_tokens_per_second" in result

        # HELP/TYPE comments should have prefix
        assert "# HELP trtllm_request_latency_seconds" in result
        assert "# TYPE trtllm_num_requests_running" in result

        # Check specific content and structure preservation
        assert 'trtllm_request_latency_seconds_bucket{le="0.1"} 10.0' in result
        assert "trtllm_tokens_per_second 245.7" in result
        assert result.endswith("\n")

    def test_no_filtering_all_frameworks(self):
        """Test that without any filters, all metrics are returned."""
        registry = Mock()

        with patch(
            "prometheus_client.generate_latest",
            return_value=self.TRTLLM_SAMPLE_METRICS.encode("utf-8"),
        ):
            result = get_prometheus_expfmt(registry)

        # Should contain all metrics including excluded ones
        assert "python_gc_objects_collected_total" in result
        assert "process_cpu_seconds_total" in result
        assert "request_latency_seconds" in result
        assert "num_requests_running" in result
        assert result.endswith("\n")

    def test_empty_result_handling(self):
        """Test handling when all metrics are filtered out."""
        registry = Mock()

        with patch(
            "prometheus_client.generate_latest",
            return_value=self.TRTLLM_SAMPLE_METRICS.encode("utf-8"),
        ):
            result = get_prometheus_expfmt(
                registry,
                exclude_prefixes=["python_", "process_", "trtllm_"],
            )

        # Should return empty string with newline or just newline
        assert result == "\n" or result == ""

    def test_prefix_already_exists(self):
        """Test that prefix is not added if it already exists."""
        registry = Mock()

        # Metrics that already have trtllm_ prefix
        sample_metrics = """# HELP trtllm_request_success_total Count of successfully processed requests
# TYPE trtllm_request_success_total counter
trtllm_request_success_total{model_name="test",finished_reason="stop"} 10.0
# HELP trtllm_time_to_first_token_seconds Time to first token
# TYPE trtllm_time_to_first_token_seconds histogram
trtllm_time_to_first_token_seconds_count 5.0
"""

        with patch(
            "prometheus_client.generate_latest",
            return_value=sample_metrics.encode("utf-8"),
        ):
            result = get_prometheus_expfmt(
                registry,
                exclude_prefixes=["python_", "process_"],
                metric_prefix_filters=["trtllm_"],
            )

        # Should not double-add prefix
        assert "trtllm_trtllm_request_success_total" not in result
        assert "trtllm_request_success_total" in result
        assert "trtllm_time_to_first_token_seconds" in result
        assert result.endswith("\n")

    def test_error_handling(self):
        """Test error handling when registry fails."""
        bad_registry = Mock()

        with patch(
            "prometheus_client.generate_latest",
            side_effect=Exception("Registry error"),
        ):
            result = get_prometheus_expfmt(bad_registry)

        # Should return empty string on error
        assert result == ""
