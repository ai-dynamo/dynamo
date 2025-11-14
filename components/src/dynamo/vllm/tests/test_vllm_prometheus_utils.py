# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Prometheus utilities."""

from unittest.mock import Mock

import pytest

from dynamo.common.utils.prometheus import get_prometheus_expfmt

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
]


class TestGetPrometheusExpfmt:
    """Test class for get_prometheus_expfmt function."""

    @pytest.fixture
    def vllm_registry(self):
        """Create a mock registry with vLLM-style metrics."""
        registry = Mock()

        sample_metrics = """# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 123.0
# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 45.6
# HELP vllm:request_success_total Number of successfully finished requests
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="meta-llama/Llama-3.1-8B"} 150.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B"} 5.0
vllm:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B"} 165.0
"""

        def mock_generate_latest(reg):
            return sample_metrics.encode("utf-8")

        import dynamo.common.utils.prometheus

        original_generate_latest = dynamo.common.utils.prometheus.generate_latest
        dynamo.common.utils.prometheus.generate_latest = mock_generate_latest

        yield registry

        dynamo.common.utils.prometheus.generate_latest = original_generate_latest

    def test_vllm_use_case(self, vllm_registry):
        """Test vLLM use case: filter to vllm: metrics and exclude python_/process_."""
        result = get_prometheus_expfmt(
            vllm_registry,
            metric_prefix_filter="vllm:",
            exclude_prefixes=["python_", "process_"],
        )

        # Should only contain vllm: metrics
        assert "vllm:request_success_total" in result
        assert "vllm:time_to_first_token_seconds" in result
        assert "# HELP vllm:request_success_total" in result

        # Should not contain excluded metrics
        assert "python_gc_objects_collected_total" not in result
        assert "process_cpu_seconds_total" not in result

        # Check specific content
        assert 'finished_reason="stop"' in result
        assert 'model_name="meta-llama/Llama-3.1-8B"' in result
        assert result.endswith("\n")

    def test_error_handling(self):
        """Test error handling when registry fails."""
        # Create a registry that raises an exception
        bad_registry = Mock()
        bad_registry.side_effect = Exception("Registry error")

        result = get_prometheus_expfmt(bad_registry)

        # Should return empty string on error
        assert result == ""
