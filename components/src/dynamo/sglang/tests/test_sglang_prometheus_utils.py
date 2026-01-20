# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Prometheus utilities."""

from unittest.mock import Mock, patch

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

    SAMPLE_METRICS = """# HELP python_gc_objects_collected_total Objects collected during gc
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

    def test_sglang_use_case(self):
        """Test SGLang use case: filter to sglang: metrics and exclude python_/process_."""
        registry = Mock()

        with patch(
            "prometheus_client.generate_latest",
            return_value=self.SAMPLE_METRICS.encode("utf-8"),
        ):
            result = get_prometheus_expfmt(
                registry,
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
        bad_registry = Mock()

        with patch(
            "prometheus_client.generate_latest",
            side_effect=Exception("Registry error"),
        ):
            result = get_prometheus_expfmt(bad_registry)

        # Should return empty string on error
        assert result == ""
