# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Prometheus label injection via get_prometheus_expfmt.

Tests the complete flow of label injection through the exposition format generation:
get_prometheus_expfmt with inject_custom_labels -> verify labels in output text format.
"""

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from dynamo import prometheus_names
from dynamo.common.utils.prometheus import get_prometheus_expfmt

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


class TestPrometheusExpositionFormatInjection:
    """Integration tests for label injection through exposition format generation"""

    def test_inject_labels_into_counter_expfmt(self):
        """Test label injection produces correct exposition format for Counter"""
        # Create registry with a counter
        registry = CollectorRegistry()
        counter = Counter("requests_total", "Total requests", registry=registry)
        counter.inc(5)

        # Get exposition format with label injection (using prometheus_names constants)
        labels_to_inject = {
            prometheus_names.labels.NAMESPACE: "prod",
            prometheus_names.labels.COMPONENT: "vllm-worker",
            prometheus_names.labels.ENDPOINT: "generate",
        }
        expfmt = get_prometheus_expfmt(registry, inject_custom_labels=labels_to_inject)

        # Verify exposition format contains injected labels
        assert f'{prometheus_names.labels.NAMESPACE}="prod"' in expfmt
        assert f'{prometheus_names.labels.COMPONENT}="vllm-worker"' in expfmt
        assert f'{prometheus_names.labels.ENDPOINT}="generate"' in expfmt

        # Verify counter value is present
        assert "requests_total_total" in expfmt or "requests_total{" in expfmt

        # Verify HELP and TYPE comments are present
        assert "# HELP requests_total" in expfmt
        assert "# TYPE requests_total counter" in expfmt

    def test_inject_labels_into_gauge_expfmt(self):
        """Test label injection produces correct exposition format for Gauge"""
        # Create registry with a gauge
        registry = CollectorRegistry()
        gauge = Gauge("active_requests", "Active requests", registry=registry)
        gauge.set(10)

        # Get exposition format with label injection (using prometheus_names constants)
        labels_to_inject = {
            prometheus_names.labels.MODEL: "llama-3-70b",
            "instance_id": "worker-0",
        }
        expfmt = get_prometheus_expfmt(registry, inject_custom_labels=labels_to_inject)

        # Verify exposition format contains injected labels
        assert f'{prometheus_names.labels.MODEL}="llama-3-70b"' in expfmt
        assert 'instance_id="worker-0"' in expfmt

        # Verify gauge value
        assert "active_requests" in expfmt
        assert "10" in expfmt or "10.0" in expfmt

    def test_inject_labels_into_histogram_expfmt(self):
        """Test label injection preserves histogram structure and le label"""
        # Create registry with a histogram
        registry = CollectorRegistry()
        histogram = Histogram(
            "request_duration_seconds",
            "Request duration",
            registry=registry,
        )
        histogram.observe(0.5)
        histogram.observe(1.5)

        # Get exposition format with label injection (using prometheus_names constants)
        labels_to_inject = {
            prometheus_names.labels.NAMESPACE: "prod",
            prometheus_names.labels.ENDPOINT: "generate",
        }
        expfmt = get_prometheus_expfmt(registry, inject_custom_labels=labels_to_inject)

        # Verify injected labels are present
        assert f'{prometheus_names.labels.NAMESPACE}="prod"' in expfmt
        assert f'{prometheus_names.labels.ENDPOINT}="generate"' in expfmt

        # Verify histogram structure (buckets with 'le' label)
        assert "request_duration_seconds_bucket" in expfmt
        assert "le=" in expfmt  # Histogram buckets must have 'le' label

        # Verify sum and count
        assert "request_duration_seconds_sum" in expfmt
        assert "request_duration_seconds_count" in expfmt

    def test_inject_labels_with_prefix_filter(self):
        """Test label injection works with metric prefix filtering"""
        # Create registry with multiple metrics
        registry = CollectorRegistry()
        vllm_counter = Counter("vllm:requests", "vLLM requests", registry=registry)
        other_counter = Counter("python_gc_objects", "GC objects", registry=registry)

        vllm_counter.inc(5)
        other_counter.inc(100)

        # Get exposition format with filtering and label injection
        labels_to_inject = {
            prometheus_names.labels.NAMESPACE: "prod",
            prometheus_names.labels.MODEL: "llama-3-70b",
        }
        expfmt = get_prometheus_expfmt(
            registry,
            metric_prefix_filters=["vllm:"],
            inject_custom_labels=labels_to_inject,
        )

        # Verify vllm metric is present with injected labels
        assert "vllm:requests" in expfmt
        assert f'{prometheus_names.labels.NAMESPACE}="prod"' in expfmt
        assert f'{prometheus_names.labels.MODEL}="llama-3-70b"' in expfmt

        # Verify other metric is filtered out
        assert "python_gc_objects" not in expfmt

    def test_inject_labels_with_exclude_prefix(self):
        """Test label injection works with exclude prefixes"""
        # Create registry with multiple metrics
        registry = CollectorRegistry()
        app_counter = Counter("app_requests", "App requests", registry=registry)
        python_counter = Counter("python_gc_objects", "GC objects", registry=registry)

        app_counter.inc(5)
        python_counter.inc(100)

        # Get exposition format with exclude and label injection
        labels_to_inject = {prometheus_names.labels.COMPONENT: "test-component"}
        expfmt = get_prometheus_expfmt(
            registry,
            exclude_prefixes=["python_"],
            inject_custom_labels=labels_to_inject,
        )

        # Verify app metric is present with injected label
        assert "app_requests" in expfmt
        assert f'{prometheus_names.labels.COMPONENT}="test-component"' in expfmt

        # Verify python metric is excluded
        assert "python_gc_objects" not in expfmt

    def test_inject_labels_with_prefix_filter_trtllm(self):
        """Test label injection works with metric prefix filtering for trtllm"""
        # Create registry with a counter that has trtllm_ prefix
        registry = CollectorRegistry()
        counter = Counter("trtllm_requests", "TensorRT-LLM Requests", registry=registry)
        counter.inc(5)

        # Get exposition format filtering for trtllm metrics and inject labels
        labels_to_inject = {
            prometheus_names.labels.NAMESPACE: "prod",
            prometheus_names.labels.MODEL: "qwen-32b",
        }
        expfmt = get_prometheus_expfmt(
            registry,
            metric_prefix_filters=["trtllm_"],
            inject_custom_labels=labels_to_inject,
        )

        # Verify metric is present with injected labels
        assert "trtllm_requests" in expfmt
        assert f'{prometheus_names.labels.NAMESPACE}="prod"' in expfmt
        assert f'{prometheus_names.labels.MODEL}="qwen-32b"' in expfmt

    def test_inject_labels_with_existing_labels(self):
        """Test label injection merges with existing metric labels"""
        # Create registry with a counter that has labels
        registry = CollectorRegistry()
        counter = Counter(
            "requests",
            "Requests",
            labelnames=["status", "method"],
            registry=registry,
        )
        counter.labels(status="success", method="GET").inc(10)

        # Get exposition format with label injection
        labels_to_inject = {
            prometheus_names.labels.NAMESPACE: "prod",
            prometheus_names.labels.COMPONENT: "vllm-worker",
        }
        expfmt = get_prometheus_expfmt(registry, inject_custom_labels=labels_to_inject)

        # Verify both existing and injected labels are present
        assert 'status="success"' in expfmt
        assert 'method="GET"' in expfmt
        assert f'{prometheus_names.labels.NAMESPACE}="prod"' in expfmt
        assert f'{prometheus_names.labels.COMPONENT}="vllm-worker"' in expfmt

    def test_inject_multiple_labels(self):
        """Test injecting many labels at once"""
        # Create registry with a gauge
        registry = CollectorRegistry()
        gauge = Gauge("memory_usage_bytes", "Memory usage", registry=registry)
        gauge.set(1024)

        # Get exposition format with many injected labels
        labels_to_inject = {
            prometheus_names.labels.NAMESPACE: "prod",
            prometheus_names.labels.COMPONENT: "vllm-worker",
            prometheus_names.labels.ENDPOINT: "generate",
            prometheus_names.labels.MODEL: "llama-3-70b",
            "instance_id": "worker-0",
            "rank": "0",
            "gpu_id": "0",
        }
        expfmt = get_prometheus_expfmt(registry, inject_custom_labels=labels_to_inject)

        # Verify all labels are present
        for label_name, label_value in labels_to_inject.items():
            assert f'{label_name}="{label_value}"' in expfmt

    def test_inject_labels_none_is_noop(self):
        """Test that inject_custom_labels=None doesn't modify output"""
        # Create registry with a counter
        registry = CollectorRegistry()
        counter = Counter("requests", "Requests", registry=registry)
        counter.inc(5)

        # Get exposition format without label injection
        expfmt_without = get_prometheus_expfmt(registry, inject_custom_labels=None)

        # Get exposition format with empty dict (should raise error in collector)
        # But inject_custom_labels=None should work fine
        expfmt_with_none = get_prometheus_expfmt(registry, inject_custom_labels=None)

        # Both should be identical (no label injection)
        assert expfmt_without == expfmt_with_none
        assert "requests" in expfmt_without

    def test_inject_labels_align_with_rust_labels(self):
        """Test injecting labels that align with Rust auto-labels"""
        # Create registry with vllm metrics
        registry = CollectorRegistry()
        counter = Counter("vllm:requests_total", "Total requests", registry=registry)
        counter.inc(100)

        # Inject labels that match Rust auto-labels
        labels_to_inject = {
            prometheus_names.labels.NAMESPACE: "prod-inference",
            prometheus_names.labels.COMPONENT: "vllm-decode-worker",
            prometheus_names.labels.ENDPOINT: "generate",
        }
        expfmt = get_prometheus_expfmt(
            registry,
            metric_prefix_filters=["vllm:"],
            inject_custom_labels=labels_to_inject,
        )

        # Verify Rust-compatible labels are present
        assert f'{prometheus_names.labels.NAMESPACE}="prod-inference"' in expfmt
        assert f'{prometheus_names.labels.COMPONENT}="vllm-decode-worker"' in expfmt
        assert f'{prometheus_names.labels.ENDPOINT}="generate"' in expfmt

        # Verify metric is present
        assert "vllm:requests" in expfmt
