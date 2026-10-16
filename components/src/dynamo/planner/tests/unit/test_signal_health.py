# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import replace
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


def _scaling_inputs(mode, target="sla"):
    config = PlannerConfig.model_construct(
        mode=mode,
        optimization_target=target,
        enable_load_scaling=True,
        enable_throughput_scaling=True,
    )
    caps = EngineCapabilities(
        max_kv_tokens=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=16,
        context_length=1024,
    )
    state = PlannerScalingState(config, WorkerCapabilities(prefill=caps, decode=caps))
    counts = WorkerCounts(ready_num_prefill=1, ready_num_decode=1)
    state.observe_worker_counts(counts)
    # Active FPM must reach model readiness guards, not the idle fast path.
    fpm = ForwardPassMetrics(
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=1,
            sum_prefill_tokens=64,
            num_decode_requests=1,
            sum_decode_kv_tokens=64,
        )
    )
    inputs = TickInput(
        now_s=0,
        worker_counts=counts,
        fpm_observations=FpmObservations(
            prefill={("p", 0): fpm}, decode={("d", 0): fpm}
        ),
        traffic=TrafficObservation(duration_s=1, num_req=1, isl=64, osl=64),
    )
    registry = CollectorRegistry()
    health = PlannerSignalHealth(config, PluginFrameworkMetrics(registry=registry))
    return state, inputs, health, registry


def _observe_scaling_health(state, inputs, health, path):
    state.begin_tick()
    if path == "load":
        decision = state.advance_load(inputs.fpm_observations)
    else:
        decision = state.advance_throughput_from_prediction(
            inputs.traffic,
            predicted_num_req=inputs.traffic.num_req,
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
    return decision


@pytest.mark.parametrize("mode", ["prefill", "decode", "agg", "disagg"])
@pytest.mark.parametrize("path", ["load", "throughput"])
def test_model_warmup_does_not_alarm(mode, path, caplog):
    state, inputs, health, registry = _scaling_inputs(mode)
    for _ in range(SIGNAL_DEGRADATION_OBSERVATIONS + 1):
        assert _observe_scaling_health(state, inputs, health, path) is None
    reason = getattr(state.diagnostics(), f"{path}_decision_reason")
    assert reason == ("insufficient_data" if path == "load" else "model_not_ready")
    assert "Planner degraded signal" not in caplog.text
    for component in ("prefill", "decode"):
        labels = {"component": component, "signal": f"{path}_inputs"}
        active = mode == "disagg" or component == ("decode" if mode == "agg" else mode)
        unavailable = registry.get_sample_value(
            "dynamo_planner_signal_unavailable", labels
        )
        if active:
            assert unavailable == 0
        else:
            assert unavailable is None
        assert (
            registry.get_sample_value("dynamo_planner_degraded_signals_total", labels)
            is None
        )


@pytest.mark.parametrize(
    ("mode", "target", "path", "missing_field"),
    [
        ("prefill", "sla", "load", "max_num_batched_tokens"),
        ("agg", "sla", "load", "max_num_batched_tokens"),
        ("prefill", "throughput", "load", "context_length"),
        ("decode", "throughput", "load", "max_kv_tokens"),
        ("agg", "throughput", "load", "max_kv_tokens"),
        ("agg", "sla", "throughput", "max_num_batched_tokens"),
    ],
)
def test_missing_engine_metadata_alarms_and_recovers(
    mode, target, path, missing_field, monkeypatch, caplog
):
    state, inputs, health, registry = _scaling_inputs(mode, target)
    component = "prefill" if mode == "prefill" else "decode"
    healthy_caps = state._capabilities
    missing_caps = replace(getattr(healthy_caps, component), **{missing_field: None})
    state.update_capabilities(replace(healthy_caps, **{component: missing_caps}))
    if target == "sla" and path == "load":
        # Get past warmup to the real batching-metadata guard. Other model
        # methods remain real; missing estimates may still hold the decision.
        model = (
            state._prefill_regression if mode == "prefill" else state._agg_regression
        )
        monkeypatch.setattr(model, "has_sufficient_data", lambda: True)
    labels = {"component": component, "signal": f"{path}_inputs"}
    for _ in range(SIGNAL_DEGRADATION_OBSERVATIONS):
        assert _observe_scaling_health(state, inputs, health, path) is None
    diagnostics = state.diagnostics()
    reason = getattr(diagnostics, f"{path}_decision_reason_{component}") or getattr(
        diagnostics, f"{path}_decision_reason"
    )
    assert reason == "missing_engine_metadata"
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 1
    assert (
        registry.get_sample_value("dynamo_planner_degraded_signals_total", labels) == 1
    )
    assert caplog.text.count("Planner degraded signal") == 1
    state.update_capabilities(healthy_caps)
    _observe_scaling_health(state, inputs, health, path)
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 0
    assert (
        registry.get_sample_value("dynamo_planner_degraded_signals_total", labels) == 1
    )


def test_disagg_metadata_health_is_attributed_to_the_blocked_component():
    state, inputs, health, registry = _scaling_inputs("disagg", "throughput")
    state.update_capabilities(
        replace(state._capabilities, prefill=EngineCapabilities())
    )
    # Healthy decode can scale up even while prefill lacks context length.
    inputs.fpm_observations.decode = {
        ("d", 0): ForwardPassMetrics(
            scheduled_requests=ScheduledRequestMetrics(
                num_decode_requests=1, sum_decode_kv_tokens=2048
            )
        )
    }
    decision = _observe_scaling_health(state, inputs, health, "load")
    assert decision.num_prefill == 1
    assert decision.num_decode == 2
    for component, unavailable in (("prefill", 1), ("decode", 0)):
        assert (
            registry.get_sample_value(
                "dynamo_planner_signal_unavailable",
                {"component": component, "signal": "load_inputs"},
            )
            == unavailable
        )


@pytest.mark.parametrize("mode", ["prefill", "decode", "agg"])
@pytest.mark.parametrize("path", ["load", "throughput"])
@pytest.mark.parametrize(
    "invalid",
    ["capabilities", "max_num_batched_tokens", "max_num_seqs", "max_kv_tokens"],
)
def test_invalid_engine_limits_are_not_model_warmup(mode, path, invalid):
    state, inputs, health, registry = _scaling_inputs(mode)
    component = "prefill" if mode == "prefill" else "decode"
    caps = state._capabilities
    invalid_caps = (
        None
        if invalid == "capabilities"
        else replace(getattr(caps, component), **{invalid: 0})
    )
    state.update_capabilities(replace(caps, **{component: invalid_caps}))
    for _ in range(SIGNAL_DEGRADATION_OBSERVATIONS):
        assert _observe_scaling_health(state, inputs, health, path) is None
    assert (
        getattr(state.diagnostics(), f"{path}_decision_reason")
        == "missing_engine_metadata"
    )
    labels = {"component": component, "signal": f"{path}_inputs"}
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 1
    assert (
        registry.get_sample_value("dynamo_planner_degraded_signals_total", labels) == 1
    )


@pytest.mark.parametrize(
    ("path", "unsampled"),
    [
        ("load", "missing_fpm"),
        ("load", "scaling_in_progress"),
        ("load", "worker_count_mismatch"),
        ("load", "idle"),
        ("throughput", "no_traffic_data"),
        ("throughput", "zero_demand"),
    ],
)
def test_unevaluated_metadata_does_not_recover_an_alarm(path, unsampled, caplog):
    state, inputs, health, registry = _scaling_inputs("decode")
    healthy_caps = state._capabilities
    state.update_capabilities(replace(healthy_caps, decode=None))
    for _ in range(SIGNAL_DEGRADATION_OBSERVATIONS):
        _observe_scaling_health(state, inputs, health, path)
    labels = {"component": "decode", "signal": f"{path}_inputs"}
    original_fpm = inputs.fpm_observations
    original_traffic = inputs.traffic
    if unsampled == "missing_fpm":
        inputs.fpm_observations = FpmObservations()
    elif unsampled == "scaling_in_progress":
        state.observe_worker_counts(
            replace(inputs.worker_counts, decode_scaling_in_progress=True)
        )
    elif unsampled == "worker_count_mismatch":
        inputs.fpm_observations = FpmObservations(
            decode={("unexpected", 0): ForwardPassMetrics(), **original_fpm.decode}
        )
    elif unsampled == "idle":
        inputs.fpm_observations = FpmObservations(
            decode={("d", 0): ForwardPassMetrics()}
        )
    else:
        inputs.traffic = replace(
            inputs.traffic,
            **({"num_req": 0} if unsampled == "zero_demand" else {"duration_s": 0}),
        )
    with caplog.at_level(logging.INFO):
        _observe_scaling_health(state, inputs, health, path)
    assert getattr(state.diagnostics(), f"{path}_metadata_available") == {}
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 1
    assert (
        registry.get_sample_value("dynamo_planner_degraded_signals_total", labels) == 1
    )
    assert (
        f"Planner signal recovered: component=decode signal={path}_inputs"
        not in caplog.text
    )

    inputs.fpm_observations = original_fpm
    inputs.traffic = original_traffic
    state.observe_worker_counts(inputs.worker_counts)
    state.update_capabilities(healthy_caps)
    _observe_scaling_health(state, inputs, health, path)
    assert getattr(state.diagnostics(), f"{path}_metadata_available") == {
        "decode": True
    }
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 0


@pytest.mark.parametrize("mode", ["prefill", "agg"])
def test_partial_batching_metadata_does_not_recover_during_warmup(mode):
    state, inputs, health, registry = _scaling_inputs(mode)
    component = "prefill" if mode == "prefill" else "decode"
    healthy_caps = state._capabilities
    state.update_capabilities(replace(healthy_caps, **{component: None}))
    for _ in range(SIGNAL_DEGRADATION_OBSERVATIONS):
        _observe_scaling_health(state, inputs, health, "load")
    labels = {"component": component, "signal": "load_inputs"}
    partial_caps = replace(
        getattr(healthy_caps, component), max_num_batched_tokens=None
    )
    state.update_capabilities(replace(healthy_caps, **{component: partial_caps}))
    _observe_scaling_health(state, inputs, health, "load")
    assert state.diagnostics().load_decision_reason == "missing_engine_metadata"
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 1
    assert (
        registry.get_sample_value("dynamo_planner_degraded_signals_total", labels) == 1
    )
    state.update_capabilities(healthy_caps)
    _observe_scaling_health(state, inputs, health, "load")
    assert state.diagnostics().load_decision_reason == "insufficient_data"
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 0


@pytest.mark.parametrize("path", ["load", "throughput"])
def test_empty_pool_clears_metadata_alarm(path):
    state, inputs, health, registry = _scaling_inputs("decode")
    state.update_capabilities(replace(state._capabilities, decode=None))
    for _ in range(SIGNAL_DEGRADATION_OBSERVATIONS):
        _observe_scaling_health(state, inputs, health, path)
    labels = {"component": "decode", "signal": f"{path}_inputs"}
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 1
    inputs.worker_counts = WorkerCounts(ready_num_decode=0)
    inputs.fpm_observations = FpmObservations()
    state.begin_tick()
    health.observe(
        ScheduledTick(at_s=0, need_worker_fpm=True),
        inputs,
        state._capabilities,
        state.diagnostics(),
    )
    assert registry.get_sample_value("dynamo_planner_signal_unavailable", labels) == 0
    assert (
        registry.get_sample_value("dynamo_planner_degraded_signals_total", labels) == 1
    )
