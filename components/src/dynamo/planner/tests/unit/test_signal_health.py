# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import MagicMock

import pytest
from prometheus_client import CollectorRegistry

from dynamo.common.forward_pass_metrics import FPM_VERSION, ForwardPassMetrics, encode
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    ScheduledTick,
    TickInput,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.environment.metrics_provider.runtime_provider import (
    RuntimeFpmProvider,
)
from dynamo.planner.monitoring.planner_metrics import PluginFrameworkMetrics
from dynamo.planner.monitoring.signal_health import PlannerSignalHealth

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


def test_missing_fpm_alarms_once_after_three_observations_and_recovers(caplog):
    metrics = PluginFrameworkMetrics(registry=CollectorRegistry())
    health = PlannerSignalHealth(PlannerConfig(mode="agg"), metrics)
    scheduled = ScheduledTick(at_s=0, need_worker_fpm=True)
    inputs = TickInput(now_s=0, worker_counts=WorkerCounts(ready_num_decode=1))
    capabilities = WorkerCapabilities(decode=EngineCapabilities(max_kv_tokens=1024))
    counter = metrics.degraded_signals_total.labels(component="decode", signal="fpm")

    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            health.observe(scheduled, inputs, capabilities)
        assert counter._value.get() == 0
        assert "degraded signal" not in caplog.text
        for _ in range(3):
            health.observe(scheduled, inputs, capabilities)
    assert counter._value.get() == 1
    assert caplog.text.count("Planner degraded signal") == 1

    inputs.fpm_observations = FpmObservations(
        decode={("w", 0): ForwardPassMetrics(worker_id="w")}
    )
    health.observe(scheduled, inputs, capabilities)
    assert (
        metrics.signal_unavailable.labels(component="decode", signal="fpm")._value.get()
        == 0
    )
    inputs.fpm_observations = None
    for _ in range(3):
        health.observe(scheduled, inputs, capabilities)
    assert counter._value.get() == 2


@pytest.fixture
def runtime_provider():
    provider = RuntimeFpmProvider(
        require_prefill=False,
        require_decode=True,
        backend="vllm",
        model_name="test",
        runtime=MagicMock(),
    )
    subscriber = MagicMock()
    provider._decode_fpm_sub = subscriber
    return provider, subscriber


@pytest.mark.parametrize("reason", ["stale_fpm", "worker_join_failure", "missing_fpm"])
def test_provider_reports_filtered_fpm_and_does_not_reuse_previous_sample(
    runtime_provider, reason, caplog
):
    provider, subscriber = runtime_provider
    subscriber.get_recent_stats_with_health.return_value = (
        {("w", 0): encode(ForwardPassMetrics(worker_id="w"))},
        [],
        0,
        0,
    )
    assert provider.collect_fpm().decode
    subscriber.get_recent_stats_with_health.return_value = (
        {("w", 0): encode(ForwardPassMetrics(worker_id="w"))},
        ["orphan"] if reason == "worker_join_failure" else [],
        1 if reason == "stale_fpm" else 0,
        1 if reason == "missing_fpm" else 0,
    )
    observation = provider.collect_fpm()
    assert observation.decode is None
    assert observation.unavailable_reasons == {"decode": reason}
    if reason == "worker_join_failure":
        assert "FPM worker join failure" in caplog.text


@pytest.mark.parametrize(
    "invalid_payload",
    [
        b"invalid",
        encode(ForwardPassMetrics(version=FPM_VERSION + 1, worker_id="w", dp_rank=1)),
    ],
    ids=["malformed", "unsupported-version"],
)
def test_provider_rejects_partial_decoding_and_recovers(
    runtime_provider, invalid_payload
):
    provider, subscriber = runtime_provider
    first = encode(ForwardPassMetrics(worker_id="w", dp_rank=0))
    second = encode(ForwardPassMetrics(worker_id="w", dp_rank=1))
    subscriber.get_recent_stats_with_health.return_value = (
        {("w", 0): first, ("w", 1): invalid_payload},
        [],
        0,
        0,
    )
    observation = provider.collect_fpm()
    assert observation.decode is None
    assert observation.unavailable_reasons == {"decode": "invalid_fpm"}
    metrics = PluginFrameworkMetrics(registry=CollectorRegistry())
    health = PlannerSignalHealth(PlannerConfig(mode="agg"), metrics)
    tick = ScheduledTick(at_s=0, need_worker_fpm=True)
    inputs = TickInput(
        now_s=0,
        worker_counts=WorkerCounts(ready_num_decode=1),
        fpm_observations=observation,
    )
    caps = WorkerCapabilities(decode=EngineCapabilities(max_kv_tokens=1024))
    for _ in range(3):
        health.observe(tick, inputs, caps)
    assert (
        metrics.degraded_signals_total.labels(
            component="decode", signal="invalid_fpm"
        )._value.get()
        == 1
    )

    subscriber.get_recent_stats_with_health.return_value = (
        {("w", 0): first, ("w", 1): second},
        [],
        0,
        0,
    )
    recovered = provider.collect_fpm()
    assert set(recovered.decode) == {("w", 0), ("w", 1)}
    assert recovered.unavailable_reasons == {}
    inputs.fpm_observations = recovered
    health.observe(tick, inputs, caps)
    assert (
        metrics.signal_unavailable.labels(
            component="decode", signal="invalid_fpm"
        )._value.get()
        == 0
    )


def test_unsampled_ticks_and_empty_pools_do_not_alarm():
    metrics = PluginFrameworkMetrics(registry=CollectorRegistry())
    health = PlannerSignalHealth(PlannerConfig(mode="agg"), metrics)
    inputs = TickInput(now_s=0, worker_counts=WorkerCounts(ready_num_decode=0))
    for _ in range(5):
        health.observe(
            ScheduledTick(at_s=0, need_worker_fpm=True), inputs, WorkerCapabilities()
        )
    inputs.worker_counts.ready_num_decode = 1
    for _ in range(5):
        health.observe(
            ScheduledTick(at_s=0, need_worker_fpm=False), inputs, WorkerCapabilities()
        )
    assert (
        metrics.degraded_signals_total.labels(
            component="decode", signal="fpm"
        )._value.get()
        == 0
    )


def test_missing_engine_metadata_is_visible_even_with_fpm():
    metrics = PluginFrameworkMetrics(registry=CollectorRegistry())
    config = PlannerConfig(mode="agg", speculative_nextn=3)
    health = PlannerSignalHealth(config, metrics)
    inputs = TickInput(
        now_s=0,
        worker_counts=WorkerCounts(ready_num_decode=1),
        fpm_observations=FpmObservations(
            decode={("w", 0): ForwardPassMetrics(worker_id="w")}
        ),
    )
    scheduled = ScheduledTick(at_s=0, need_worker_fpm=True, need_traffic_metrics=True)
    for _ in range(3):
        health.observe(scheduled, inputs, WorkerCapabilities())
    for signal in ("engine_capacity", "accept_length"):
        assert (
            metrics.signal_unavailable.labels(
                component="decode", signal=signal
            )._value.get()
            == 1
        )

        assert (
            metrics.degraded_signals_total.labels(
                component="decode", signal=signal
            )._value.get()
            == 1
        )


@pytest.mark.parametrize(
    ("mode", "target", "unavailable"),
    [("agg", "sla", True), ("decode", "sla", False), ("agg", "load", False)],
)
def test_aggregate_sla_requires_batching_capacity_and_recovers(
    mode, target, unavailable
):
    metrics = PluginFrameworkMetrics(registry=CollectorRegistry())
    health = PlannerSignalHealth(
        PlannerConfig(
            mode=mode,
            optimization_target=target,
            enable_load_scaling=True,
            enable_throughput_scaling=False,
            decode_scale_up_kv_rate=90,
            decode_scale_down_kv_rate=10,
        ),
        metrics,
    )
    inputs = TickInput(
        now_s=0,
        worker_counts=WorkerCounts(ready_num_decode=1),
        fpm_observations=FpmObservations(
            decode={("w", 0): ForwardPassMetrics(worker_id="w")}
        ),
    )
    scheduled = ScheduledTick(at_s=0, need_worker_fpm=True)
    capabilities = WorkerCapabilities(decode=EngineCapabilities(max_kv_tokens=1024))
    gauge = metrics.signal_unavailable.labels(
        component="decode", signal="engine_capacity"
    )
    counter = metrics.degraded_signals_total.labels(
        component="decode", signal="engine_capacity"
    )
    for _ in range(3):
        health.observe(scheduled, inputs, capabilities)
    assert gauge._value.get() == int(unavailable)
    assert counter._value.get() == int(unavailable)
    capabilities.decode.max_num_batched_tokens = 256
    health.observe(scheduled, inputs, capabilities)
    assert gauge._value.get() == 0
