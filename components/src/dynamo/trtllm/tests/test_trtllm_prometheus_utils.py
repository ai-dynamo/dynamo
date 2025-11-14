# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Prometheus utilities."""

from unittest.mock import Mock

import pytest

from dynamo.common.utils.prometheus import get_prometheus_expfmt

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
]


class TestGetPrometheusExpfmt:
    """Test class for get_prometheus_expfmt function."""

    @pytest.fixture
    def trtllm_registry(self):
        """Create a mock registry with TensorRT-LLM-style metrics (no existing prefixes)."""
        registry = Mock()

        sample_metrics = """# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 123.0
# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 45.6
# HELP request_latency_seconds Request latency in seconds
# TYPE request_latency_seconds histogram
request_latency_seconds_bucket{le="0.1"} 10.0
request_latency_seconds_count 25.0
# HELP num_requests_running Number of requests currently running
# TYPE num_requests_running gauge
num_requests_running 3.0
# HELP tokens_per_second Tokens generated per second
# TYPE tokens_per_second gauge
tokens_per_second 245.7
"""

        def mock_generate_latest(reg):
            return sample_metrics.encode("utf-8")

        import dynamo.common.utils.prometheus

        original_generate_latest = dynamo.common.utils.prometheus.generate_latest
        dynamo.common.utils.prometheus.generate_latest = mock_generate_latest

        yield registry

        dynamo.common.utils.prometheus.generate_latest = original_generate_latest

    def test_trtllm_use_case(self, trtllm_registry):
        """Test TensorRT-LLM use case: exclude python_/process_ and add trtllm_ prefix."""
        result = get_prometheus_expfmt(
            trtllm_registry,
            exclude_prefixes=["python_", "process_"],
            add_prefix="trtllm_",
        )

        # Should not contain excluded metrics
        assert "python_gc_objects_collected_total" not in result
        assert "process_cpu_seconds_total" not in result

        # All remaining metrics should have trtllm_ prefix
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

    def test_no_filtering_all_frameworks(self, trtllm_registry):
        """Test that without any filters, all metrics are returned."""
        result = get_prometheus_expfmt(trtllm_registry)

        # Should contain all metrics including excluded ones
        assert "python_gc_objects_collected_total" in result
        assert "process_cpu_seconds_total" in result
        assert "request_latency_seconds" in result
        assert "num_requests_running" in result
        assert result.endswith("\n")

    def test_empty_result_handling(self, trtllm_registry):
        """Test handling when all metrics are filtered out."""
        result = get_prometheus_expfmt(
            trtllm_registry,
            exclude_prefixes=["python_", "process_", "request_", "num_", "tokens_"],
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

        def mock_generate_latest(reg):
            return sample_metrics.encode("utf-8")

        import dynamo.common.utils.prometheus

        original_generate_latest = dynamo.common.utils.prometheus.generate_latest
        dynamo.common.utils.prometheus.generate_latest = mock_generate_latest

        try:
            result = get_prometheus_expfmt(
                registry,
                exclude_prefixes=["python_", "process_"],
                add_prefix="trtllm_",
            )

            # Should not double-add prefix
            assert "trtllm_trtllm_request_success_total" not in result
            assert "trtllm_request_success_total" in result
            assert "trtllm_time_to_first_token_seconds" in result
            assert result.endswith("\n")
        finally:
            dynamo.common.utils.prometheus.generate_latest = original_generate_latest

    def test_error_handling(self):
        """Test error handling when registry fails."""
        # Create a registry that raises an exception
        bad_registry = Mock()
        bad_registry.side_effect = Exception("Registry error")

        result = get_prometheus_expfmt(bad_registry)

        # Should return empty string on error
        assert result == ""
