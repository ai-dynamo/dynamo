# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for framework-only helpers in
``dynamo.common.backend._internal_metrics``.

These cover behavior the PyO3 bridge depends on. Engine-facing
``register_global_registry`` is tested in ``test_metrics_helpers.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from dynamo.common.backend._internal_metrics import (
    make_component_metrics,
    register_engine_registry,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class _StubMetrics:
    def __init__(self, auto_labels: dict[str, str] | None = None) -> None:
        self.auto_labels = dict(auto_labels) if auto_labels is not None else {}
        self.callbacks: list[Callable[[], str]] = []

    def register_prometheus_expfmt_callback(self, cb: Callable[[], str]) -> None:
        self.callbacks.append(cb)


def test_make_component_metrics_rejects_empty_names():
    """Empty model_name or component_name would mask auto-injected labels
    via gather_with_labels's existing-label-wins rule."""
    with pytest.raises(ValueError):
        make_component_metrics("", "backend")
    with pytest.raises(ValueError):
        make_component_metrics("model", "")


def test_make_component_metrics_returns_gauges_and_registry():
    """Smoke test: gauges register with the returned registry so a scrape
    against that registry observes their values."""
    from prometheus_client import generate_latest

    gauges, registry = make_component_metrics("test-model", "test-component")
    gauges.set_total_blocks("0", 42)
    text = generate_latest(registry).decode()
    assert "dynamo_component_total_blocks" in text
    assert 'model="test-model"' in text
    assert 'dynamo_component="test-component"' in text


def test_register_engine_registry_routes_scrape_through_auto_labels(monkeypatch):
    """The registered callback must invoke gather_with_labels with the
    EngineMetrics handle's auto_labels — that's how vendor/component
    metrics inherit dynamo_namespace / dynamo_component / etc."""
    from prometheus_client import CollectorRegistry, Gauge

    reg = CollectorRegistry()
    g = Gauge("vllm:requests_running", "...", labelnames=["dp_rank"], registry=reg)
    g.labels(dp_rank="0").set(3)

    metrics = _StubMetrics(auto_labels={"dynamo_namespace": "n1", "model": "m1"})
    register_engine_registry(metrics, reg)

    assert len(metrics.callbacks) == 1
    text = metrics.callbacks[0]()
    assert 'dynamo_namespace="n1"' in text
    assert 'model="m1"' in text
    assert "vllm:requests_running" in text
