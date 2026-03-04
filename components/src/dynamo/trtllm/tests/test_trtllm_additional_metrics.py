# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AdditionalMetricsCollector and additional metrics integration."""

import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

from prometheus_client import CollectorRegistry, generate_latest


class TestAdditionalMetricsCollector(unittest.TestCase):
    """Unit tests for AdditionalMetricsCollector."""

    def setUp(self):
        """Create a fresh registry and collector for each test."""
        self.registry = CollectorRegistry()

        # Patch prometheus_client.Counter and Histogram to use our test registry
        with patch("dynamo.trtllm.metrics.Counter") as MockCounter, patch(
            "dynamo.trtllm.metrics.Histogram"
        ) as MockHistogram:
            from prometheus_client import Counter, Histogram

            def make_counter(name, documentation, labelnames=None, **kw):
                return Counter(
                    name,
                    documentation,
                    labelnames=labelnames or [],
                    registry=self.registry,
                )

            def make_histogram(
                name, documentation, labelnames=None, buckets=None, **kw
            ):
                kwargs = {"registry": self.registry}
                if buckets is not None:
                    kwargs["buckets"] = buckets
                return Histogram(
                    name, documentation, labelnames=labelnames or [], **kwargs
                )

            MockCounter.side_effect = make_counter
            MockHistogram.side_effect = make_histogram

            from dynamo.trtllm.metrics import AdditionalMetricsCollector

            self.collector = AdditionalMetricsCollector(
                labels={
                    "model_name": "test-model",
                    "disaggregation_mode": "prefill_and_decode",
                    "engine_type": "trtllm",
                },
            )

    def _get_metric_value(self, name, labels=None):
        """Get a metric value from the registry."""
        output = generate_latest(self.registry).decode()
        for line in output.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith(name):
                # Extract value (last token)
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[-1])
        return None

    def test_abort_counter(self):
        """Test abort tracking."""
        self.collector.record_request_abort()
        output = generate_latest(self.registry).decode()
        self.assertIn("num_aborted_requests_total", output)

    def test_request_type_counters(self):
        """Test request type counters."""
        self.collector.record_request_type_image()
        self.collector.record_request_type_structured_output()
        output = generate_latest(self.registry).decode()
        self.assertIn("request_type_image_total", output)
        self.assertIn("request_type_structured_output_total", output)

    def test_kv_transfer_success_counter(self):
        """Test KV transfer success counter."""
        self.collector.record_kv_transfer_success()
        output = generate_latest(self.registry).decode()
        self.assertIn("kv_transfer_success_total", output)

    def test_kv_transfer_perf_metrics(self):
        """Test KV transfer latency/bytes/speed from timing_metrics."""
        tm = MagicMock()
        tm.kv_cache_transfer_start = timedelta(seconds=1.0)
        tm.kv_cache_transfer_end = timedelta(seconds=1.05)
        tm.kv_cache_size = 1_000_000_000  # 1 GB

        result = self.collector.record_kv_transfer_perf(tm)
        self.assertTrue(result)
        output = generate_latest(self.registry).decode()
        self.assertIn("kv_transfer_latency_seconds", output)
        self.assertIn("kv_transfer_bytes_bucket", output)
        self.assertIn("kv_transfer_speed_gb_s", output)

    def test_kv_transfer_perf_skipped_when_no_transfer(self):
        """Test that KV transfer perf is not recorded when no transfer occurred."""
        tm = MagicMock()
        tm.kv_cache_transfer_start = timedelta(0)
        tm.kv_cache_transfer_end = timedelta(0)
        tm.kv_cache_size = 0

        result = self.collector.record_kv_transfer_perf(tm)
        self.assertFalse(result)
        # Histogram is defined but should have no observations
        sample = self.registry.get_sample_value("kv_transfer_latency_seconds_count")
        self.assertEqual(sample, 0.0)

    def test_kv_transfer_perf_return_values(self):
        """Verify record_kv_transfer_perf returns True on record, False on skip."""
        # Transfer occurred
        tm_ok = MagicMock()
        tm_ok.kv_cache_transfer_start = timedelta(seconds=1.0)
        tm_ok.kv_cache_transfer_end = timedelta(seconds=1.05)
        tm_ok.kv_cache_size = 5_000_000
        self.assertTrue(self.collector.record_kv_transfer_perf(tm_ok))

        # No transfer (zero times)
        tm_zero = MagicMock()
        tm_zero.kv_cache_transfer_start = timedelta(0)
        tm_zero.kv_cache_transfer_end = timedelta(0)
        tm_zero.kv_cache_size = 0
        self.assertFalse(self.collector.record_kv_transfer_perf(tm_zero))

        # Negative latency (end before start)
        tm_neg = MagicMock()
        tm_neg.kv_cache_transfer_start = timedelta(seconds=2.0)
        tm_neg.kv_cache_transfer_end = timedelta(seconds=1.0)
        tm_neg.kv_cache_size = 1000
        self.assertFalse(self.collector.record_kv_transfer_perf(tm_neg))

    def test_no_duplicate_metrics(self):
        """Test that removed duplicate metrics are not present."""
        output = generate_latest(self.registry).decode()
        # These metrics were removed as they duplicate frontend/runtime metrics
        self.assertNotIn("prompt_tokens_total", output)
        self.assertNotIn("generation_tokens_total", output)
        self.assertNotIn("gen_throughput", output)
        self.assertNotIn("kv_cache_hit_tokens_total", output)
        self.assertNotIn("handler_time_to_first_token_seconds", output)
        self.assertNotIn("handler_inter_token_latency_seconds", output)
        self.assertNotIn("handler_e2e_request_latency_seconds", output)
        # Phase timing metrics also removed (derivable from existing trtllm_* metrics)
        self.assertNotIn("request_prefill_time_seconds", output)
        self.assertNotIn("request_decode_time_seconds", output)
        self.assertNotIn("request_inference_time_seconds", output)
        # Config info metrics removed (overlap dynamo_frontend_model_* and
        # dynamo_component_model_load_time_seconds)
        self.assertNotIn("model_config_info", output)
        self.assertNotIn("parallel_config_info", output)
        self.assertNotIn("detailed_config_info", output)
        self.assertNotIn("cache_config_info", output)
        self.assertNotIn("engine_startup_time", output)
        # KV transfer perf metrics are now wired from request_perf_metrics.timing_metrics
        # (kv_transfer_latency_seconds, kv_transfer_bytes, kv_transfer_speed_gb_s)


class TestHandlerBaseMetricsInstrumentation(unittest.TestCase):
    """Test metrics instrumentation in handler_base.py generate_locally()."""

    def test_structured_output_detection_keys(self):
        """Verify guided decoding detection keys in generate_locally match _override_sampling_params."""
        import ast
        import inspect
        import textwrap

        from dynamo.trtllm.request_handlers.handler_base import HandlerBase

        # Extract detection keys from generate_locally: the tuple in
        #   any(guided.get(k) for k in ("json", ...))
        gen_source = textwrap.dedent(inspect.getsource(HandlerBase.generate_locally))
        gen_tree = ast.parse(gen_source)
        detection_keys = set()
        for node in ast.walk(gen_tree):
            # Find: any(guided.get(k) for k in (...))
            if isinstance(node, ast.Tuple) and all(
                isinstance(e, (ast.Constant, ast.Str)) for e in node.elts
            ):
                vals = {
                    e.value if isinstance(e, ast.Constant) else e.s for e in node.elts
                }
                if "json_object" in vals:  # identify the right tuple
                    detection_keys = vals

        self.assertTrue(
            detection_keys, "Could not extract detection keys from generate_locally"
        )

        # Extract canonical keys from _override_sampling_params:
        #   GuidedDecodingParams(json=..., regex=..., ...)
        override_source = textwrap.dedent(
            inspect.getsource(HandlerBase._override_sampling_params)
        )
        override_tree = ast.parse(override_source)
        canonical_keys = set()
        for node in ast.walk(override_tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "GuidedDecodingParams"
            ):
                canonical_keys = {kw.arg for kw in node.keywords}

        self.assertTrue(
            canonical_keys, "Could not extract keys from GuidedDecodingParams call"
        )

        # Detection should cover all canonical keys
        missing = canonical_keys - detection_keys
        self.assertFalse(
            missing,
            f"Keys in GuidedDecodingParams but not in metrics detection: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
