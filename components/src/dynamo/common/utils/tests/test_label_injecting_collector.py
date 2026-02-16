# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for LabelInjectingCollector.

Tests the custom Prometheus collector that injects labels into metrics without
modifying the source metrics.
"""

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary


class TestLabelInjectingCollector:
    """Test suite for LabelInjectingCollector"""

    def test_counter_label_injection(self):
        """Test injecting labels into Counter metrics"""
        from dynamo.common.utils.label_injecting_collector import (
            LabelInjectingCollector,
        )

        # Create source registry with a counter
        source_registry = CollectorRegistry()
        counter = Counter(
            "test_counter",
            "Test counter",
            registry=source_registry,
        )
        counter.inc(5)

        # Create collector that injects labels
        labels_to_inject = {
            "dynamo_namespace": "prod",
            "dynamo_component": "test-component",
        }
        collector = LabelInjectingCollector(source_registry, labels_to_inject)

        # Collect metrics
        metric_families = list(collector.collect())

        # Verify counter exists with injected labels
        assert len(metric_families) == 1
        mf = metric_families[0]
        assert mf.name == "test_counter"
        assert mf.type == "counter"

        # Verify samples have injected labels
        # Counter may have multiple samples: _total (value) and _created (timestamp)
        assert len(mf.samples) >= 1

        # Find the _total sample (the actual counter value)
        total_sample = next(s for s in mf.samples if s.name.endswith("_total"))
        assert total_sample.labels["dynamo_namespace"] == "prod"
        assert total_sample.labels["dynamo_component"] == "test-component"
        assert total_sample.value == 5

    def test_gauge_label_injection(self):
        """Test injecting labels into Gauge metrics"""
        from dynamo.common.utils.label_injecting_collector import (
            LabelInjectingCollector,
        )

        # Create source registry with a gauge
        source_registry = CollectorRegistry()
        gauge = Gauge(
            "test_gauge",
            "Test gauge",
            registry=source_registry,
        )
        gauge.set(42)

        # Create collector that injects labels
        labels_to_inject = {"dynamo_endpoint": "generate", "model": "llama-3-70b"}
        collector = LabelInjectingCollector(source_registry, labels_to_inject)

        # Collect metrics
        metric_families = list(collector.collect())

        # Verify gauge exists with injected labels
        assert len(metric_families) == 1
        mf = metric_families[0]
        assert mf.name == "test_gauge"
        assert mf.type == "gauge"

        # Verify samples have injected labels
        assert len(mf.samples) == 1
        sample = mf.samples[0]
        assert sample.labels["dynamo_endpoint"] == "generate"
        assert sample.labels["model"] == "llama-3-70b"
        assert sample.value == 42

    def test_histogram_preserves_le_label(self):
        """Test that histogram 'le' label is preserved and not overwritten"""
        from dynamo.common.utils.label_injecting_collector import (
            LabelInjectingCollector,
        )

        # Create source registry with a histogram
        source_registry = CollectorRegistry()
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            registry=source_registry,
        )
        histogram.observe(0.5)
        histogram.observe(1.5)
        histogram.observe(2.5)

        # Create collector that injects labels
        labels_to_inject = {
            "dynamo_namespace": "prod",
            "dynamo_component": "vllm-worker",
        }
        collector = LabelInjectingCollector(source_registry, labels_to_inject)

        # Collect metrics
        metric_families = list(collector.collect())

        # Find histogram metric family
        histogram_mf = next(mf for mf in metric_families if mf.name == "test_histogram")
        assert histogram_mf.type == "histogram"

        # Verify histogram buckets have 'le' label preserved
        bucket_samples = [s for s in histogram_mf.samples if s.name.endswith("_bucket")]
        assert len(bucket_samples) > 0

        for sample in bucket_samples:
            # Verify 'le' label exists (reserved for histogram buckets)
            assert "le" in sample.labels
            # Verify injected labels are present
            assert sample.labels["dynamo_namespace"] == "prod"
            assert sample.labels["dynamo_component"] == "vllm-worker"

        # Verify sum and count samples also have injected labels
        sum_sample = next(s for s in histogram_mf.samples if s.name.endswith("_sum"))
        assert sum_sample.labels["dynamo_namespace"] == "prod"
        assert sum_sample.labels["dynamo_component"] == "vllm-worker"

        count_sample = next(
            s for s in histogram_mf.samples if s.name.endswith("_count")
        )
        assert count_sample.labels["dynamo_namespace"] == "prod"
        assert count_sample.labels["dynamo_component"] == "vllm-worker"

    def test_summary_preserves_quantile_label(self):
        """Test that summary 'quantile' label is preserved"""
        from dynamo.common.utils.label_injecting_collector import (
            LabelInjectingCollector,
        )

        # Create source registry with a summary
        source_registry = CollectorRegistry()
        summary = Summary(
            "test_summary",
            "Test summary",
            registry=source_registry,
        )
        summary.observe(1.0)
        summary.observe(2.0)
        summary.observe(3.0)

        # Create collector that injects labels
        labels_to_inject = {"dynamo_endpoint": "generate", "rank": "0"}
        collector = LabelInjectingCollector(source_registry, labels_to_inject)

        # Collect metrics
        metric_families = list(collector.collect())

        # Find summary metric family
        summary_mf = next(mf for mf in metric_families if mf.name == "test_summary")
        assert summary_mf.type == "summary"

        # Verify summary quantiles have 'quantile' label preserved (if present)
        # Note: Not all summary implementations expose quantiles, so this is optional
        quantile_samples = [s for s in summary_mf.samples if "quantile" in s.labels]
        for sample in quantile_samples:
            # Verify 'quantile' label exists (reserved for summary quantiles)
            assert "quantile" in sample.labels
            # Verify injected labels are present
            assert sample.labels["dynamo_endpoint"] == "generate"
            assert sample.labels["rank"] == "0"

        # Verify sum and count samples have injected labels
        sum_sample = next(s for s in summary_mf.samples if s.name.endswith("_sum"))
        assert sum_sample.labels["dynamo_endpoint"] == "generate"
        assert sum_sample.labels["rank"] == "0"

        count_sample = next(s for s in summary_mf.samples if s.name.endswith("_count"))
        assert count_sample.labels["dynamo_endpoint"] == "generate"
        assert count_sample.labels["rank"] == "0"

    def test_multiple_labels_injection(self):
        """Test injecting multiple labels at once"""
        from dynamo.common.utils.label_injecting_collector import (
            LabelInjectingCollector,
        )

        # Create source registry with a counter
        source_registry = CollectorRegistry()
        counter = Counter("test_counter", "Test counter", registry=source_registry)
        counter.inc()

        # Create collector with multiple labels to inject
        labels_to_inject = {
            "dynamo_namespace": "prod",
            "dynamo_component": "vllm-worker",
            "dynamo_endpoint": "generate",
            "model": "llama-3-70b",
            "instance_id": "worker-0",
            "rank": "0",
        }
        collector = LabelInjectingCollector(source_registry, labels_to_inject)

        # Collect metrics
        metric_families = list(collector.collect())
        assert len(metric_families) == 1

        # Verify all labels are present
        sample = metric_families[0].samples[0]
        assert sample.labels["dynamo_namespace"] == "prod"
        assert sample.labels["dynamo_component"] == "vllm-worker"
        assert sample.labels["dynamo_endpoint"] == "generate"
        assert sample.labels["model"] == "llama-3-70b"
        assert sample.labels["instance_id"] == "worker-0"
        assert sample.labels["rank"] == "0"

    def test_merge_with_existing_labels(self):
        """Test injecting labels into metrics that already have labels"""
        from dynamo.common.utils.label_injecting_collector import (
            LabelInjectingCollector,
        )

        # Create source registry with a counter that has labels
        source_registry = CollectorRegistry()
        counter = Counter(
            "test_counter",
            "Test counter",
            labelnames=["status", "method"],
            registry=source_registry,
        )
        counter.labels(status="success", method="GET").inc(10)
        counter.labels(status="error", method="POST").inc(5)

        # Create collector that injects additional labels
        labels_to_inject = {
            "dynamo_namespace": "prod",
            "dynamo_component": "vllm-worker",
        }
        collector = LabelInjectingCollector(source_registry, labels_to_inject)

        # Collect metrics
        metric_families = list(collector.collect())
        assert len(metric_families) == 1

        # Verify both original and injected labels are present
        samples = metric_families[0].samples
        # Counter may have _total and _created samples for each label combination
        assert len(samples) >= 2

        # Filter to only _total samples (the actual counter values)
        total_samples = [s for s in samples if s.name.endswith("_total")]
        assert len(total_samples) == 2

        # First sample: status=success, method=GET
        sample1 = total_samples[0]
        assert sample1.labels["status"] == "success"
        assert sample1.labels["method"] == "GET"
        assert sample1.labels["dynamo_namespace"] == "prod"
        assert sample1.labels["dynamo_component"] == "vllm-worker"
        assert sample1.value == 10

        # Second sample: status=error, method=POST
        sample2 = total_samples[1]
        assert sample2.labels["status"] == "error"
        assert sample2.labels["method"] == "POST"
        assert sample2.labels["dynamo_namespace"] == "prod"
        assert sample2.labels["dynamo_component"] == "vllm-worker"
        assert sample2.value == 5

    def test_existing_label_not_overwritten(self):
        """Test that existing labels take precedence over injected labels"""
        from dynamo.common.utils.label_injecting_collector import (
            LabelInjectingCollector,
        )

        # Create source registry with a counter that has a 'model' label
        source_registry = CollectorRegistry()
        counter = Counter(
            "test_counter",
            "Test counter",
            labelnames=["model"],
            registry=source_registry,
        )
        counter.labels(model="original-model").inc()

        # Try to inject a 'model' label with different value
        labels_to_inject = {
            "model": "injected-model",  # This should NOT overwrite existing label
            "dynamo_namespace": "prod",
        }
        collector = LabelInjectingCollector(source_registry, labels_to_inject)

        # Collect metrics
        metric_families = list(collector.collect())
        sample = metric_families[0].samples[0]

        # Verify original label is preserved (not overwritten)
        assert sample.labels["model"] == "original-model"
        assert sample.labels["dynamo_namespace"] == "prod"

    def test_empty_labels_raises_error(self):
        """Test that empty labels dict raises ValueError"""
        from dynamo.common.utils.label_injecting_collector import (
            LabelInjectingCollector,
        )

        source_registry = CollectorRegistry()

        # Empty labels dict should raise ValueError
        with pytest.raises(ValueError, match="labels_to_inject cannot be empty"):
            LabelInjectingCollector(source_registry, {})

    def test_reserved_label_le_raises_error(self):
        """Test that trying to inject reserved label 'le' raises ValueError"""
        from dynamo.common.utils.label_injecting_collector import (
            LabelInjectingCollector,
        )

        source_registry = CollectorRegistry()

        # Trying to inject reserved 'le' label should raise ValueError
        with pytest.raises(ValueError, match="Cannot inject reserved label names"):
            LabelInjectingCollector(
                source_registry,
                {"le": "1.0", "dynamo_namespace": "prod"},
            )

    def test_reserved_label_quantile_raises_error(self):
        """Test that trying to inject reserved label 'quantile' raises ValueError"""
        from dynamo.common.utils.label_injecting_collector import (
            LabelInjectingCollector,
        )

        source_registry = CollectorRegistry()

        # Trying to inject reserved 'quantile' label should raise ValueError
        with pytest.raises(ValueError, match="Cannot inject reserved label names"):
            LabelInjectingCollector(
                source_registry,
                {"quantile": "0.99", "dynamo_namespace": "prod"},
            )

    def test_multiple_metrics_all_get_labels(self):
        """Test that all metrics in registry get injected labels"""
        from dynamo.common.utils.label_injecting_collector import (
            LabelInjectingCollector,
        )

        # Create source registry with multiple metrics
        source_registry = CollectorRegistry()
        counter = Counter("test_counter", "Test counter", registry=source_registry)
        gauge = Gauge("test_gauge", "Test gauge", registry=source_registry)
        histogram = Histogram(
            "test_histogram", "Test histogram", registry=source_registry
        )

        counter.inc(5)
        gauge.set(42)
        histogram.observe(1.5)

        # Create collector that injects labels
        labels_to_inject = {
            "dynamo_namespace": "prod",
            "dynamo_component": "vllm-worker",
        }
        collector = LabelInjectingCollector(source_registry, labels_to_inject)

        # Collect metrics
        metric_families = list(collector.collect())
        assert len(metric_families) == 3

        # Verify all metric families have injected labels in their samples
        for mf in metric_families:
            for sample in mf.samples:
                assert sample.labels["dynamo_namespace"] == "prod"
                assert sample.labels["dynamo_component"] == "vllm-worker"

    def test_timestamp_preservation(self):
        """Test that timestamps are preserved"""
        from dynamo.common.utils.label_injecting_collector import (
            LabelInjectingCollector,
        )

        # Create source registry with a gauge
        source_registry = CollectorRegistry()
        gauge = Gauge("test_gauge", "Test gauge", registry=source_registry)
        gauge.set(42)

        # Create collector
        labels_to_inject = {"dynamo_namespace": "prod"}
        collector = LabelInjectingCollector(source_registry, labels_to_inject)

        # Collect metrics
        metric_families = list(collector.collect())
        sample = metric_families[0].samples[0]

        # Verify timestamp attribute exists (may be None)
        assert sample.timestamp is None or isinstance(sample.timestamp, (float, int))
