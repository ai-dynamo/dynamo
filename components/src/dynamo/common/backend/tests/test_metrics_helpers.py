# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for engine-facing helpers in
``dynamo.common.backend.metrics``.

Framework-only helpers (``make_component_metrics``,
``register_engine_registry``) are tested in ``test_internal_metrics.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest
from prometheus_client import CollectorRegistry, Gauge

from dynamo.common.backend.metrics import (
    gather_with_labels,
    register_global_registry,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class _StubMetrics:
    def __init__(self, auto_labels: dict[str, str] | None = None) -> None:
        self.auto_labels = dict(auto_labels) if auto_labels is not None else {}
        self.callbacks: list[Callable[[], str]] = []

    def register_prometheus_expfmt_callback(self, cb: Callable[[], str]) -> None:
        self.callbacks.append(cb)


def test_gather_with_labels_does_not_overwrite_existing_label():
    reg = CollectorRegistry()
    g = Gauge("kv_used_blocks", "...", labelnames=["model"], registry=reg)
    g.labels(model="explicit").set(7)

    text = gather_with_labels(reg, {"model": "auto-value"})
    assert 'model="explicit"' in text
    assert 'model="auto-value"' not in text


def test_register_global_registry_splits_on_multiprocesscollector_conflict(
    monkeypatch, tmp_path
):
    """K8s case: env var pre-set, MultiProcessCollector(REGISTRY) raises
    ValueError. Helper must register TWO callbacks — REGISTRY restricted
    to engine_prefix only, plus a fresh MultiProcessCollector registry
    covering engine_prefix + multiproc_only_prefixes."""
    multiproc_dir = tmp_path / "mp"
    multiproc_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(multiproc_dir))

    metrics = _StubMetrics()
    with patch(
        "prometheus_client.multiprocess.MultiProcessCollector",
        side_effect=[ValueError("metric already registered"), None],
    ):
        register_global_registry(
            metrics,
            engine_prefix="vllm:",
            multiproc_only_prefixes=["lmcache:"],
        )
    assert len(metrics.callbacks) == 2
