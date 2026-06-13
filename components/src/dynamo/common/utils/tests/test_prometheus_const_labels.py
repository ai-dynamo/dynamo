# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for globally configured Prometheus constant labels."""

import pytest
from prometheus_client import CollectorRegistry, Counter

from dynamo.common.utils.prometheus import (
    _global_const_labels_cached,
    global_const_labels,
    get_prometheus_expfmt,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_global_const_labels_parses_primary_env_var(monkeypatch):
    monkeypatch.setenv("DYNAMO_CONST_LABELS", "namespace:prod;model_version:v2")
    monkeypatch.delenv("MICHAEL_ADD_LABELS", raising=False)
    _global_const_labels_cached.cache_clear()

    assert global_const_labels() == {
        "namespace": "prod",
        "model_version": "v2",
    }


def test_global_const_labels_supports_legacy_alias(monkeypatch):
    monkeypatch.delenv("DYNAMO_CONST_LABELS", raising=False)
    monkeypatch.setenv("MICHAEL_ADD_LABELS", "namespace:staging;team:search")
    _global_const_labels_cached.cache_clear()

    assert global_const_labels() == {
        "namespace": "staging",
        "team": "search",
    }


def test_get_prometheus_expfmt_adds_global_const_labels(monkeypatch):
    monkeypatch.setenv("DYNAMO_CONST_LABELS", "namespace:prod;model_version:v2")
    monkeypatch.delenv("MICHAEL_ADD_LABELS", raising=False)
    _global_const_labels_cached.cache_clear()

    registry = CollectorRegistry()
    counter = Counter("router_requests_total", "Requests", registry=registry)
    counter.inc(1352)

    output = get_prometheus_expfmt(registry)

    assert 'router_requests_total' in output
    assert 'namespace="prod"' in output
    assert 'model_version="v2"' in output


def test_global_const_labels_do_not_override_existing_metric_labels(monkeypatch):
    monkeypatch.setenv("DYNAMO_CONST_LABELS", "namespace:prod;model_version:v2")
    monkeypatch.delenv("MICHAEL_ADD_LABELS", raising=False)
    _global_const_labels_cached.cache_clear()

    registry = CollectorRegistry()
    counter = Counter(
        "router_requests_total",
        "Requests",
        labelnames=["namespace", "status"],
        registry=registry,
    )
    counter.labels(namespace="runtime", status="200").inc(7)

    output = get_prometheus_expfmt(registry)

    assert 'namespace="runtime"' in output
    assert 'status="200"' in output
    assert 'model_version="v2"' in output
    assert 'namespace="prod"' not in output