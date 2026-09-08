# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Prometheus label injection via get_prometheus_expfmt.

Tests the complete flow of label injection through the exposition format generation:
get_prometheus_expfmt with inject_custom_labels -> verify labels in output text format.
"""

import pytest
from prometheus_client import CollectorRegistry, Counter

from dynamo import prometheus_names
from dynamo.common.utils.prometheus import get_prometheus_expfmt

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


class TestPrometheusExpositionFormatInjection:
    """Integration tests for label injection through exposition format generation"""

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
