# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Prometheus utilities."""

from unittest.mock import Mock

import pytest

from dynamo.common.utils.prometheus import get_prometheus_expfmt

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
]


class TestGetPrometheusExpfmt:
    """Test class for get_prometheus_expfmt function."""

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

    def test_sglang_use_case(self, sglang_registry):
        """Test SGLang use case: filter to sglang: metrics and exclude python_/process_."""
        result = get_prometheus_expfmt(
            sglang_registry,
            metric_prefix_filters=["sglang:"],
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

    def test_error_handling(self):
        """Test error handling when registry fails."""
        # Create a registry that raises an exception
        bad_registry = Mock()
        bad_registry.side_effect = Exception("Registry error")

        result = get_prometheus_expfmt(bad_registry)

        # Should return empty string on error
        assert result == ""
