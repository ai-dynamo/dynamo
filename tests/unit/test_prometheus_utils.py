# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Prometheus utilities."""

from unittest.mock import Mock

import pytest

from dynamo.common.utils.prometheus import get_prometheus_expfmt

pytestmark = [
    pytest.mark.unit,
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

    @pytest.fixture
    def sglang_registry(self):
        """Create a mock registry with SGLang-style metrics."""
        registry = Mock()

        sample_metrics = """# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 123.0
# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 45.6
# HELP sglang:prompt_tokens_total Number of prefill tokens processed
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8128902.0
# HELP sglang:generation_tokens_total Number of generation tokens processed
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7557572.0
# HELP sglang:cache_hit_rate The cache hit rate
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0075
"""

        def mock_generate_latest(reg):
            return sample_metrics.encode("utf-8")

        import dynamo.common.utils.prometheus

        original_generate_latest = dynamo.common.utils.prometheus.generate_latest
        dynamo.common.utils.prometheus.generate_latest = mock_generate_latest

        yield registry

        dynamo.common.utils.prometheus.generate_latest = original_generate_latest

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

    def test_sglang_use_case(self, sglang_registry):
        """Test SGLang use case: filter to sglang: metrics and exclude python_/process_."""
        result = get_prometheus_expfmt(
            sglang_registry,
            metric_prefix_filter="sglang:",
            exclude_prefixes=["python_", "process_"],
        )

        # Should only contain sglang: metrics
        assert "sglang:prompt_tokens_total" in result
        assert "sglang:generation_tokens_total" in result
        assert "sglang:cache_hit_rate" in result
        assert "# HELP sglang:prompt_tokens_total" in result

        # Should not contain excluded metrics
        assert "python_gc_objects_collected_total" not in result
        assert "process_cpu_seconds_total" not in result

        # Check specific content
        assert 'model_name="meta-llama/Llama-3.1-8B-Instruct"' in result
        assert "8128902.0" in result  # prompt tokens value
        assert result.endswith("\n")

    def test_trtllm_use_case(self, trtllm_registry):
        """Test TensorRT-LLM use case: exclude python_/process_ and add trtllm: prefix."""
        result = get_prometheus_expfmt(
            trtllm_registry,
            exclude_prefixes=["python_", "process_"],
            add_prefix="trtllm:",
        )

        # Should not contain excluded metrics
        assert "python_gc_objects_collected_total" not in result
        assert "process_cpu_seconds_total" not in result

        # All remaining metrics should have trtllm: prefix
        assert "trtllm:request_latency_seconds" in result
        assert "trtllm:num_requests_running" in result
        assert "trtllm:tokens_per_second" in result

        # HELP/TYPE comments should have prefix
        assert "# HELP trtllm:request_latency_seconds" in result
        assert "# TYPE trtllm:num_requests_running" in result

        # Check specific content and structure preservation
        assert 'trtllm:request_latency_seconds_bucket{le="0.1"} 10.0' in result
        assert "trtllm:tokens_per_second 245.7" in result
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

    def test_error_handling(self):
        """Test error handling when registry fails."""
        # Create a registry that raises an exception
        bad_registry = Mock()
        bad_registry.side_effect = Exception("Registry error")

        result = get_prometheus_expfmt(bad_registry)

        # Should return empty string on error
        assert result == ""
