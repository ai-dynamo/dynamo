# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Prometheus utilities."""

from unittest.mock import Mock, patch

import pytest

from dynamo.common.utils.prometheus import get_prometheus_expfmt

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
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
# HELP vllm:request_success_total Number of successfully finished requests
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="meta-llama/Llama-3.1-8B"} 150.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B"} 5.0
vllm:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B"} 165.0
"""

    def test_vllm_use_case(self):
        """Test vLLM use case: filter to vllm: metrics and exclude python_/process_."""
        registry = Mock()

        with patch(
            "prometheus_client.generate_latest",
            return_value=self.SAMPLE_METRICS.encode("utf-8"),
        ):
            result = get_prometheus_expfmt(
                registry,
                metric_prefix_filters=["vllm:"],
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
        bad_registry = Mock()

        with patch(
            "prometheus_client.generate_latest",
            side_effect=Exception("Registry error"),
        ):
            result = get_prometheus_expfmt(bad_registry)

        # Should return empty string on error
        assert result == ""


class TestSetupMetricsCollectionMultiprocFallback:
    """Pins the whitelist split in setup_metrics_collection's fallback branch.

    When MultiProcessCollector cannot merge into the global REGISTRY, two
    callbacks are registered: the in-memory one owns only ["vllm:"], the
    .db one owns the full engine_metric_prefixes (incl. extras).

    Why the split matters: an extra-prefix metric created via a standard
    metric type writes through to .db, so it exists in both sources. If the
    memory callback also whitelisted it, both callbacks would emit it and
    Prometheus would reject the scrape. But the .db callback must keep it,
    because metrics that live only in .db have no other way to surface.
    The tests below pin one side each.
    """

    def _run_fallback(self, monkeypatch, tmp_path, extra_prefixes=None):
        try:
            from prometheus_client import REGISTRY, multiprocess

            from dynamo.vllm import main as vllm_main
        except ImportError as e:
            pytest.skip(f"vLLM not available: {e}")

        recorded = []

        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
        if extra_prefixes is not None:
            monkeypatch.setenv("DYN_EXTRA_METRIC_PREFIXES", extra_prefixes)

        real_mpc = multiprocess.MultiProcessCollector

        def _mpc_with_conflict(registry):
            if registry is REGISTRY:
                raise ValueError("Duplicated timeseries")
            return real_mpc(registry)

        monkeypatch.setattr(multiprocess, "MultiProcessCollector", _mpc_with_conflict)
        monkeypatch.setattr(
            vllm_main,
            "register_engine_metrics_callback",
            lambda **kwargs: recorded.append(kwargs),
        )
        monkeypatch.setattr(
            vllm_main, "get_metrics_model_name", lambda config: "test-model"
        )

        config = Mock()
        config.engine_args.disable_log_stats = False
        vllm_main.setup_metrics_collection(config, Mock(), Mock())

        return [
            call["metric_prefix_filters"]
            for call in recorded
            if call.get("metric_prefix_filters") is not None
        ]

    def test_fallback_memory_callback_excludes_extra_prefixes(
        self, monkeypatch, tmp_path
    ):
        filters = self._run_fallback(
            monkeypatch, tmp_path, extra_prefixes="myconnector_"
        )

        memory_filters = [f for f in filters if "myconnector_" not in f]
        assert memory_filters == [["vllm:"]]

    def test_fallback_db_callback_owns_extra_prefixes(self, monkeypatch, tmp_path):
        filters = self._run_fallback(
            monkeypatch, tmp_path, extra_prefixes="myconnector_"
        )

        db_filters = [f for f in filters if "myconnector_" in f]
        assert len(db_filters) == 1
        assert {"vllm:", "lmcache:", "myconnector_"} <= set(db_filters[0])

    def test_fallback_without_extra_prefixes_keeps_builtin_whitelist(
        self, monkeypatch, tmp_path
    ):
        filters = self._run_fallback(monkeypatch, tmp_path)

        assert filters.count(["vllm:"]) == 1
        assert ["vllm:", "lmcache:"] in filters
