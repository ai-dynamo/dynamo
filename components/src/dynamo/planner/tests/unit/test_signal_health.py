# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from prometheus_client import CollectorRegistry

from dynamo.common.forward_pass_metrics import (
    FPM_VERSION,
    ForwardPassMetrics,
    ScheduledRequestMetrics,
    encode,
)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.state_machine import PlannerScalingState
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    ScheduledTick,
    TickDiagnostics,
    TickInput,
    TrafficObservation,
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


def _scaling_inputs():
    config = PlannerConfig.model_construct(
        mode="agg",
        optimization_target="sla",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
    )
    state = PlannerScalingState(
        config,
        WorkerCapabilities(
            decode=EngineCapabilities(max_num_batched_tokens=1024, max_kv_tokens=1024)
        ),
    )
    counts = WorkerCounts(ready_num_decode=1)
    state.observe_worker_counts(counts)
    # Active FPM reaches model readiness checks rather than the idle fast path.
    fpm = ForwardPassMetrics(
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=1, sum_decode_kv_tokens=64
        )
    )
    return state, TickInput(
        now_s=0,
        worker_counts=counts,
        fpm_observations=FpmObservations(decode={("d", 0): fpm}),
        traffic=TrafficObservation(duration_s=1, num_req=1, isl=64, osl=64),
    )


def _observe_scaling_health(state, inputs, health, path="load"):
    state.begin_tick()
    if path == "load":
        state.advance_load(inputs.fpm_observations)
    else:
        state.advance_throughput_from_prediction(
            inputs.traffic,
            predicted_num_req=1,
            predicted_isl=64,
            predicted_osl=64,
            predicted_kv_hit_rate=None,
        )
    health.observe(
        ScheduledTick(
            at_s=0,
            need_worker_fpm=True,
            run_load_scaling=path == "load",
            run_throughput_scaling=path == "throughput",
        ),
        inputs,
        state._capabilities,
        state.diagnostics(),
    )


def test_model_warmup_does_not_alarm(signal_health, caplog):
    health, registry = signal_health
    state, inputs = _scaling_inputs()
    for _ in range(SIGNAL_DEGRADATION_OBSERVATIONS + 1):
        _observe_scaling_health(state, inputs, health, "throughput")
    assert state.diagnostics().throughput_decision_reason == "model_not_ready"
    labels = {"component": "decode", "signal": "throughput_inputs"}
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 0
    assert (
        registry.get_sample_value("dynamo_planner_degraded_signals_total", labels)
        is None
    )
    assert "Planner degraded signal" not in caplog.text


def test_missing_metadata_alarms_and_recovers(signal_health):
    health, registry = signal_health
    state, inputs = _scaling_inputs()
    healthy_caps = state._capabilities
    state.update_capabilities(WorkerCapabilities())
    for _ in range(SIGNAL_DEGRADATION_OBSERVATIONS):
        _observe_scaling_health(state, inputs, health)
    assert state.diagnostics().load_decision_reason == "missing_engine_metadata"
    labels = {"component": "decode", "signal": "load_inputs"}
    counter = "dynamo_planner_degraded_signals_total"
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 1
    assert registry.get_sample_value(counter, labels) == 1
    state.update_capabilities(healthy_caps)
    _observe_scaling_health(state, inputs, health)
    assert state.diagnostics().load_decision_reason == "insufficient_data"
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 0
    assert registry.get_sample_value(counter, labels) == 1


def test_unevaluated_metadata_does_not_clear_an_alarm(signal_health, caplog):
    health, registry = signal_health
    state, inputs = _scaling_inputs()
    state.update_capabilities(WorkerCapabilities())
    for _ in range(SIGNAL_DEGRADATION_OBSERVATIONS):
        _observe_scaling_health(state, inputs, health)
    inputs.fpm_observations = FpmObservations()
    with caplog.at_level(logging.INFO):
        _observe_scaling_health(state, inputs, health)
    assert state.diagnostics().load_decision_reason == "no_fpm_data"
    assert state.diagnostics().load_metadata_available == {}
    labels = {"component": "decode", "signal": "load_inputs"}
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 1
    assert (
        registry.get_sample_value("dynamo_planner_degraded_signals_total", labels) == 1
    )
    assert "Planner signal recovered" not in caplog.text
