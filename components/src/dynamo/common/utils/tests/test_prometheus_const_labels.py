# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for deployment-wide constant labels (DYN_METRICS_CONST_LABELS).

Covers the env-var parser and its integration with get_prometheus_expfmt(): env
labels are attached to every sample, never overwrite existing labels, and lose to
explicit inject_custom_labels.
"""

import pytest
from prometheus_client import CollectorRegistry, Counter

from dynamo.common.utils.prometheus import (
    METRICS_CONST_LABELS_ENV,
    _env_const_labels_cached,
    env_const_labels,
    get_prometheus_expfmt,
    parse_const_labels,
)

# Total runtime well under 1s — no need for parallel marker.
pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture(autouse=True)
def _reset_env_label_cache():
    _env_const_labels_cached.cache_clear()
    yield
    _env_const_labels_cached.cache_clear()


def _sample_line(text: str, metric_name: str) -> str:
    lines = [line for line in text.splitlines() if line.startswith(metric_name)]
    assert len(lines) == 1, f"expected one sample for {metric_name}, got: {text}"
    return lines[0]


def test_parse_const_labels_accepts_valid_pairs():
    assert parse_const_labels(" cluster=us-west-2, team = search ,, env=prod ") == {
        "cluster": "us-west-2",
        "team": "search",
        "env": "prod",
    }
    assert parse_const_labels("") == {}
    assert parse_const_labels(" , ") == {}
    # Only the first '=' splits; values may contain '='.
    assert parse_const_labels("k=a=b") == {"k": "a=b"}


@pytest.mark.parametrize(
    "bad",
    [
        "novalue",
        "=empty_name",
        "1bad=x",
        "bad-name=x",
        "__reserved=x",
        "dynamo_namespace=x",
        "dynamo_component=x",
        "dynamo_endpoint=x",
        "worker_id=1",
        "dup=1,dup=2",
    ],
)
def test_parse_const_labels_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_const_labels(bad)


def test_env_const_labels_is_read_once(monkeypatch):
    monkeypatch.setenv(METRICS_CONST_LABELS_ENV, "cluster=prod,team=search")
    assert env_const_labels() == {"cluster": "prod", "team": "search"}
    # Cached for the process lifetime; later env changes are not observed.
    monkeypatch.setenv(METRICS_CONST_LABELS_ENV, "cluster=changed")
    assert env_const_labels() == {"cluster": "prod", "team": "search"}


def test_env_const_labels_invalid_value_is_ignored_as_a_whole(monkeypatch):
    monkeypatch.setenv(METRICS_CONST_LABELS_ENV, "ok=1,broken")
    assert env_const_labels() == {}


def test_env_const_labels_unset_means_none(monkeypatch):
    monkeypatch.delenv(METRICS_CONST_LABELS_ENV, raising=False)
    assert env_const_labels() == {}


def test_expfmt_attaches_env_labels_without_overriding_existing(monkeypatch):
    monkeypatch.setenv(METRICS_CONST_LABELS_ENV, "cluster=from_env,team=search")
    registry = CollectorRegistry()
    Counter("demo_plain", "plain", registry=registry).inc()
    labeled = Counter(
        "demo_requests", "labeled", labelnames=["cluster"], registry=registry
    )
    labeled.labels(cluster="explicit").inc(2)

    out = get_prometheus_expfmt(registry)

    plain = _sample_line(out, "demo_plain_total{")
    assert 'cluster="from_env"' in plain and 'team="search"' in plain
    assert plain.endswith(" 1.0")

    labeled_line = _sample_line(out, "demo_requests_total{")
    assert 'cluster="explicit"' in labeled_line
    assert 'cluster="from_env"' not in labeled_line
    assert 'team="search"' in labeled_line
    assert labeled_line.endswith(" 2.0")


def test_explicit_custom_labels_win_over_env_labels(monkeypatch):
    monkeypatch.setenv(METRICS_CONST_LABELS_ENV, "team=env_team,region=eu")
    registry = CollectorRegistry()
    Counter("demo_plain", "plain", registry=registry).inc()

    out = get_prometheus_expfmt(registry, inject_custom_labels={"team": "custom"})

    line = _sample_line(out, "demo_plain_total{")
    assert 'team="custom"' in line
    assert 'team="env_team"' not in line
    assert 'region="eu"' in line


def test_expfmt_unchanged_when_env_unset(monkeypatch):
    monkeypatch.delenv(METRICS_CONST_LABELS_ENV, raising=False)
    registry = CollectorRegistry()
    Counter("demo_plain", "plain", registry=registry).inc()

    out = get_prometheus_expfmt(registry)

    assert "demo_plain_total 1.0" in out
