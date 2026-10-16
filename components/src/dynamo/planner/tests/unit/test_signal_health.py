# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from prometheus_client import CollectorRegistry

from dynamo.common.forward_pass_metrics import FPM_VERSION, ForwardPassMetrics, encode
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    FpmObservations,
    ScheduledTick,
    TickDiagnostics,
    TickInput,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.environment.metrics_provider.runtime_provider import (
    RuntimeFpmProvider,
)
from dynamo.planner.monitoring.planner_metrics import (
    SIGNAL_DEGRADATION_OBSERVATIONS,
    PluginFrameworkMetrics,
)
from dynamo.planner.monitoring.signal_health import PlannerSignalHealth

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


@pytest.fixture
def signal_health():
    registry = CollectorRegistry()
    metrics = PluginFrameworkMetrics(registry=registry)
    return PlannerSignalHealth(PlannerConfig(mode="agg"), metrics), registry


def test_missing_fpm_alarms_once_and_recovers(signal_health, caplog):
    health, registry = signal_health
    scheduled = ScheduledTick(at_s=0, need_worker_fpm=True)
    inputs = TickInput(now_s=0, worker_counts=WorkerCounts(ready_num_decode=1))
    labels = {"component": "decode", "signal": "fpm"}
    counter = "dynamo_planner_degraded_signals_total"
    with caplog.at_level(logging.WARNING):
        for _ in range(SIGNAL_DEGRADATION_OBSERVATIONS - 1):
            health.observe(scheduled, inputs, WorkerCapabilities(), TickDiagnostics())
        assert registry.get_sample_value(counter, labels) is None
        for _ in range(2):
            health.observe(scheduled, inputs, WorkerCapabilities(), TickDiagnostics())
    assert registry.get_sample_value(counter, labels) == 1
    assert caplog.text.count("Planner degraded signal") == 1
    inputs.fpm_observations = FpmObservations(decode={("w", 0): ForwardPassMetrics()})
    health.observe(scheduled, inputs, WorkerCapabilities(), TickDiagnostics())
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 0
    inputs.fpm_observations = None
    for _ in range(SIGNAL_DEGRADATION_OBSERVATIONS):
        health.observe(scheduled, inputs, WorkerCapabilities(), TickDiagnostics())
    assert registry.get_sample_value(counter, labels) == 2


@pytest.mark.parametrize(("sampled", "ready"), [(True, 0), (False, 1)])
def test_unsampled_ticks_and_empty_pools_do_not_alarm(signal_health, sampled, ready):
    health, registry = signal_health
    for _ in range(SIGNAL_DEGRADATION_OBSERVATIONS):
        health.observe(
            ScheduledTick(at_s=0, need_worker_fpm=sampled),
            TickInput(now_s=0, worker_counts=WorkerCounts(ready_num_decode=ready)),
            WorkerCapabilities(),
            TickDiagnostics(),
        )
    assert (
        registry.get_sample_value(
            "dynamo_planner_degraded_signals_total",
            {"component": "decode", "signal": "fpm"},
        )
        is None
    )


@pytest.fixture
def runtime_provider():
    provider = RuntimeFpmProvider(
        require_prefill=False,
        require_decode=True,
        backend="vllm",
        model_name="test",
        runtime=MagicMock(),
    )
    health = SimpleNamespace(
        snapshot={}, unjoined_worker_ids=[], stale_rank_count=0, missing_rank_count=0
    )
    provider._decode_fpm_sub = MagicMock()
    provider._decode_fpm_sub.get_recent_stats_with_health.return_value = health
    return provider, health


@pytest.mark.parametrize("reason", ["stale_fpm", "worker_join_failure", "missing_fpm"])
def test_provider_rejects_incomplete_pools_without_reusing_samples(
    runtime_provider, reason, caplog
):
    provider, health = runtime_provider
    health.snapshot = {("w", 0): encode(ForwardPassMetrics(worker_id="w"))}
    assert provider.collect_fpm().decode
    health.unjoined_worker_ids = ["orphan"] if reason == "worker_join_failure" else []
    health.stale_rank_count = int(reason == "stale_fpm")
    health.missing_rank_count = int(reason == "missing_fpm")
    observation = provider.collect_fpm()
    assert observation.decode is None
    assert observation.unavailable_reasons == {"decode": reason}
    if reason == "worker_join_failure":
        assert "FPM worker join failure" in caplog.text


@pytest.mark.parametrize(
    "invalid", [b"invalid", encode(ForwardPassMetrics(version=FPM_VERSION + 1))]
)
def test_provider_rejects_partial_decoding_and_recovers(runtime_provider, invalid):
    provider, health = runtime_provider
    health.snapshot = {("w", 0): encode(ForwardPassMetrics()), ("w", 1): invalid}
    observation = provider.collect_fpm()
    assert observation.decode is None
    assert observation.unavailable_reasons == {"decode": "invalid_fpm"}
    health.snapshot[("w", 1)] = encode(ForwardPassMetrics(dp_rank=1))
    assert set(provider.collect_fpm().decode) == {("w", 0), ("w", 1)}
    assert provider.collect_fpm().unavailable_reasons == {}


@pytest.mark.parametrize(
    ("path", "reason"),
    [("load", "insufficient_data"), ("throughput", "model_not_ready")],
)
def test_scaling_input_health_uses_decision_reasons(signal_health, path, reason):
    health, registry = signal_health
    scheduled = ScheduledTick(
        at_s=0, need_worker_fpm=True, run_load_scaling=True, run_throughput_scaling=True
    )
    inputs = TickInput(
        now_s=0,
        worker_counts=WorkerCounts(ready_num_decode=1),
        fpm_observations=FpmObservations(decode={("w", 0): ForwardPassMetrics()}),
    )
    labels = {"component": "decode", "signal": f"{path}_inputs"}
    diagnostics = TickDiagnostics(**{f"{path}_decision_reason": reason})
    health.observe(scheduled, inputs, WorkerCapabilities(), diagnostics)
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 1
    diagnostics = TickDiagnostics(**{f"{path}_decision_reason": "no_change"})
    health.observe(scheduled, inputs, WorkerCapabilities(), diagnostics)
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 0
