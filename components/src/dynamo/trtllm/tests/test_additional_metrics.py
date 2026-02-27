# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AdditionalMetricsCollector and unified metrics integration."""

import time
import unittest
from unittest.mock import MagicMock, patch

from prometheus_client import REGISTRY, CollectorRegistry, generate_latest


class TestAdditionalMetricsCollector(unittest.TestCase):
    """Unit tests for AdditionalMetricsCollector."""

    def setUp(self):
        """Create a fresh registry and collector for each test."""
        self.registry = CollectorRegistry()

        # Patch prometheus_client to use our test registry
        with patch("dynamo.trtllm.metrics.Counter") as MockCounter, \
             patch("dynamo.trtllm.metrics.Gauge") as MockGauge, \
             patch("dynamo.trtllm.metrics.Histogram") as MockHistogram:

            # Create real counters/gauges/histograms on test registry
            from prometheus_client import Counter, Gauge, Histogram

            def make_counter(name, documentation, labelnames=None, **kw):
                return Counter(name, documentation, labelnames=labelnames or [],
                               registry=self.registry)

            def make_gauge(name, documentation, labelnames=None, **kw):
                return Gauge(name, documentation, labelnames=labelnames or [],
                             registry=self.registry, **{k: v for k, v in kw.items()
                                                        if k == "multiprocess_mode"})

            def make_histogram(name, documentation, labelnames=None, buckets=None, **kw):
                kwargs = {"registry": self.registry}
                if buckets:
                    kwargs["buckets"] = buckets
                return Histogram(name, documentation, labelnames=labelnames or [],
                                 **kwargs)

            MockCounter.side_effect = make_counter
            MockGauge.side_effect = make_gauge
            MockHistogram.side_effect = make_histogram

            from dynamo.trtllm.metrics import AdditionalMetricsCollector
            self.collector = AdditionalMetricsCollector(
                labels={
                    "model_name": "test-model",
                    "disaggregation_mode": "prefill_and_decode",
                    "engine_type": "trtllm",
                },
                enable_handler_timing=False,
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

    def test_token_counters(self):
        """Test prompt and generation token counters."""
        self.collector.record_request_finish(prompt_tokens=100, gen_tokens=50)
        self.collector.record_request_finish(prompt_tokens=200, gen_tokens=100)

        output = generate_latest(self.registry).decode()
        self.assertIn("prompt_tokens_total", output)
        self.assertIn("generation_tokens_total", output)

    def test_abort_counter(self):
        """Test abort tracking."""
        self.collector.record_request_abort()
        output = generate_latest(self.registry).decode()
        self.assertIn("num_aborted_requests_total", output)

    def test_phase_timing_with_ttft(self):
        """Test phase timing records prefill and decode when TTFT is provided."""
        self.collector.record_phase_times(ttft=0.05, e2e=1.0)
        output = generate_latest(self.registry).decode()
        self.assertIn("request_prefill_time_seconds", output)
        self.assertIn("request_decode_time_seconds", output)

    def test_phase_timing_with_queue_time(self):
        """Test phase timing records inference when queue_time is provided."""
        self.collector.record_phase_times(ttft=0.05, e2e=1.0, queue_time=0.01)
        output = generate_latest(self.registry).decode()
        self.assertIn("request_inference_time_seconds", output)

    def test_phase_timing_without_queue_time(self):
        """Test that inference time is not recorded without queue_time."""
        self.collector.record_phase_times(ttft=0.05, e2e=1.0)
        output = generate_latest(self.registry).decode()
        # inference_time histogram exists but should have 0 observations
        # when no queue_time is provided
        lines = [l for l in output.splitlines()
                 if "request_inference_time_seconds_count" in l and not l.startswith("#")]
        if lines:
            count = float(lines[0].split()[-1])
            self.assertEqual(count, 0.0)

    def test_config_info_gauges(self):
        """Test config info gauges are set correctly."""
        self.collector.set_model_config(
            model="/path/to/model",
            served_model_name="my-model",
            dtype="auto",
            gpu_type="H100",
        )
        self.collector.set_parallel_config(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            gpu_count=4,
        )
        self.collector.set_engine_startup_time(45.5)

        output = generate_latest(self.registry).decode()
        self.assertIn("model_config_info", output)
        self.assertIn("parallel_config_info", output)
        self.assertIn("engine_startup_time", output)

    def test_kv_cache_hits(self):
        """Test KV cache hit tracking."""
        self.collector.record_kv_cache_hits(42)
        output = generate_latest(self.registry).decode()
        self.assertIn("kv_cache_hit_tokens_total", output)

    def test_kv_cache_hits_zero_ignored(self):
        """Test that zero cache hits are not recorded."""
        self.collector.record_kv_cache_hits(0)
        output = generate_latest(self.registry).decode()
        lines = [l for l in output.splitlines()
                 if "kv_cache_hit_tokens_total" in l and not l.startswith("#")]
        if lines:
            val = float(lines[0].split()[-1])
            self.assertEqual(val, 0.0)

    def test_request_type_counters(self):
        """Test request type counters."""
        self.collector.record_request_type_image()
        self.collector.record_request_type_structured_output()
        output = generate_latest(self.registry).decode()
        self.assertIn("request_type_image_total", output)
        self.assertIn("request_type_structured_output_total", output)

    def test_gen_throughput(self):
        """Test generation throughput gauge."""
        self.collector.set_gen_throughput(150.5)
        output = generate_latest(self.registry).decode()
        self.assertIn("gen_throughput", output)

    def test_no_handler_timing_by_default(self):
        """Test that handler timing metrics are not created by default."""
        self.assertIsNone(self.collector.handler_ttft)
        self.assertIsNone(self.collector.handler_itl)
        self.assertIsNone(self.collector.handler_e2e)

    def test_handler_timing_noop_when_disabled(self):
        """Test that handler timing recording is no-op when disabled."""
        # These should not raise
        self.collector.record_handler_ttft(0.05)
        self.collector.record_handler_itl(0.01)
        self.collector.record_handler_e2e(1.0)

    def test_kv_transfer_counters(self):
        """Test KV transfer success/failure counters."""
        self.collector.record_kv_transfer_success()
        self.collector.record_kv_transfer_failure()
        output = generate_latest(self.registry).decode()
        self.assertIn("kv_transfer_success_total", output)
        self.assertIn("kv_transfer_failure_total", output)


class TestAdditionalMetricsWithHandlerTiming(unittest.TestCase):
    """Test handler timing metrics when enabled."""

    def setUp(self):
        self.registry = CollectorRegistry()

        with patch("dynamo.trtllm.metrics.Counter") as MockCounter, \
             patch("dynamo.trtllm.metrics.Gauge") as MockGauge, \
             patch("dynamo.trtllm.metrics.Histogram") as MockHistogram:

            from prometheus_client import Counter, Gauge, Histogram

            def make_counter(name, documentation, labelnames=None, **kw):
                return Counter(name, documentation, labelnames=labelnames or [],
                               registry=self.registry)

            def make_gauge(name, documentation, labelnames=None, **kw):
                return Gauge(name, documentation, labelnames=labelnames or [],
                             registry=self.registry, **{k: v for k, v in kw.items()
                                                        if k == "multiprocess_mode"})

            def make_histogram(name, documentation, labelnames=None, buckets=None, **kw):
                kwargs = {"registry": self.registry}
                if buckets:
                    kwargs["buckets"] = buckets
                return Histogram(name, documentation, labelnames=labelnames or [],
                                 **kwargs)

            MockCounter.side_effect = make_counter
            MockGauge.side_effect = make_gauge
            MockHistogram.side_effect = make_histogram

            from dynamo.trtllm.metrics import AdditionalMetricsCollector
            self.collector = AdditionalMetricsCollector(
                labels={
                    "model_name": "test-model",
                    "disaggregation_mode": "prefill_and_decode",
                    "engine_type": "trtllm",
                },
                enable_handler_timing=True,
            )

    def test_handler_timing_created(self):
        """Test that handler timing metrics are created when enabled."""
        self.assertIsNotNone(self.collector.handler_ttft)
        self.assertIsNotNone(self.collector.handler_itl)
        self.assertIsNotNone(self.collector.handler_e2e)

    def test_handler_timing_recording(self):
        """Test handler timing metrics can be recorded."""
        self.collector.record_handler_ttft(0.05)
        self.collector.record_handler_itl(0.01)
        self.collector.record_handler_e2e(1.0)
        output = generate_latest(self.registry).decode()
        self.assertIn("handler_time_to_first_token_seconds", output)
        self.assertIn("handler_inter_token_latency_seconds", output)
        self.assertIn("handler_e2e_request_latency_seconds", output)


class TestBackwardsCompatAlias(unittest.TestCase):
    """Test the UnifiedMetricsCollector backwards compatibility alias."""

    def test_alias_exists(self):
        from dynamo.trtllm.metrics import AdditionalMetricsCollector, UnifiedMetricsCollector
        self.assertIs(UnifiedMetricsCollector, AdditionalMetricsCollector)


class TestHandlerBaseMetricsInstrumentation(unittest.TestCase):
    """Test metrics instrumentation in handler_base.py generate_locally()."""

    def test_single_chunk_ttft(self):
        """Regression: first-token time must be set before finish recording.

        When the engine returns all tokens in a single chunk (common for
        short outputs), TTFT must still be recorded correctly.
        """
        # This is a structural test - verify first-token tracking appears
        # before the finish recording block in the source
        import inspect
        from dynamo.trtllm.request_handlers.handler_base import HandlerBase

        source = inspect.getsource(HandlerBase.generate_locally)

        # Find positions of key blocks
        first_token_pos = source.find("_um_first_token_time is None")
        finish_recording_pos = source.find("Record unified (additional) metrics on request finish")

        self.assertGreater(first_token_pos, -1, "First token tracking not found")
        self.assertGreater(finish_recording_pos, -1, "Finish recording not found")
        self.assertLess(first_token_pos, finish_recording_pos,
                        "First token tracking must appear BEFORE finish recording")

    def test_choice_in_structured_output_detection(self):
        """Verify choice is included in structured output detection."""
        import inspect
        from dynamo.trtllm.request_handlers.handler_base import HandlerBase

        source = inspect.getsource(HandlerBase.generate_locally)
        self.assertIn('"choice"', source,
                      "choice should be in structured output detection")


class TestConfigValidation(unittest.TestCase):
    """Test DynamoTrtllmConfig validation."""

    def test_handler_timing_requires_unified_metrics(self):
        """--enable-handler-timing without --enable-unified-metrics should raise."""
        from dynamo.trtllm.backend_args import DynamoTrtllmConfig

        # Build a minimal config dict
        config = MagicMock(spec=DynamoTrtllmConfig)
        config.enable_handler_timing = True
        config.enable_unified_metrics = False
        config.disaggregation_mode = "aggregated"
        config.modality = "text"
        config.served_model_name = "test"

        # Call the actual validate method
        with self.assertRaises(ValueError, msg="--enable-handler-timing requires --enable-unified-metrics"):
            DynamoTrtllmConfig.validate(config)


if __name__ == "__main__":
    unittest.main()
