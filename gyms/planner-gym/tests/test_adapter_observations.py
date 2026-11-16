# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rival policies must distinguish rank measurements from missing telemetry."""

import asyncio
import importlib

import pytest

planner_types = pytest.importorskip("dynamo.planner.core.types")
metrics = importlib.import_module("dynamo.common.forward_pass_metrics")
adapters = importlib.import_module("autoscaling_arena.adapters")
aggregation = importlib.import_module("autoscaling_arena.adapters.aggregation")

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def _rank_metrics(worker_id, rank, *, kv_tokens=0, waiting=0):
    return metrics.ForwardPassMetrics(
        worker_id=worker_id,
        dp_rank=rank,
        scheduled_requests=metrics.ScheduledRequestMetrics(
            sum_decode_kv_tokens=kv_tokens,
        ),
        queued_requests=metrics.QueuedRequestMetrics(num_prefill_requests=waiting),
    )


def _fleet(dp_size, *, kv_tokens=1600):
    return {
        (str(worker), rank): _rank_metrics(str(worker), rank, kv_tokens=kv_tokens)
        for worker in range(4)
        for rank in range(dp_size)
    }


@pytest.mark.parametrize("dp_size", [1, 4])
@pytest.mark.parametrize("pool", ["prefill", "decode", "all"])
def test_saturated_rank_caches_mean_a_saturated_worker(dp_size, pool):
    role = "prefill" if pool == "prefill" else "decode"
    fpm = planner_types.FpmObservations(**{role: _fleet(dp_size)})
    caps = planner_types.WorkerCapabilities(
        **{role: planner_types.EngineCapabilities(max_kv_tokens=1600 * dp_size)}
    )

    assert aggregation.aggregate_kv_util(fpm, caps, pool=pool) == pytest.approx(1.0)


def test_worker_utilization_is_not_weighted_by_reported_rank_count():
    fpm = planner_types.FpmObservations(
        decode={
            ("a", 0): _rank_metrics("a", 0, kv_tokens=1600),
            ("a", 1): _rank_metrics("a", 1, kv_tokens=1600),
            ("b", 0): _rank_metrics("b", 0, kv_tokens=1600),
        }
    )
    caps = planner_types.WorkerCapabilities(
        decode=planner_types.EngineCapabilities(max_kv_tokens=6400)
    )
    # Worker a reports 0.5 of its total capacity; worker b reports 0.25.
    assert aggregation.aggregate_kv_util(fpm, caps) == pytest.approx(0.375)


def test_combined_pools_use_each_roles_capacity_and_worker_identity():
    fpm = planner_types.FpmObservations(
        prefill={("same-id", 0): _rank_metrics("same-id", 0, kv_tokens=800)},
        decode={("same-id", 0): _rank_metrics("same-id", 0, kv_tokens=2400)},
    )
    caps = planner_types.WorkerCapabilities(
        prefill=planner_types.EngineCapabilities(max_kv_tokens=1600),
        decode=planner_types.EngineCapabilities(max_kv_tokens=3200),
    )
    assert aggregation.aggregate_kv_util(fpm, caps, pool="all") == pytest.approx(0.625)


def _observations(kind, role):
    if kind == "absent":
        return None
    if kind == "unset":
        return planner_types.FpmObservations()
    if kind == "empty":
        return planner_types.FpmObservations(**{role: {}})
    if kind == "wrong_pool":
        other = "decode" if role == "prefill" else "prefill"
        return planner_types.FpmObservations(
            **{other: {("other", 0): _rank_metrics("other", 0)}}
        )
    return planner_types.FpmObservations(
        **{role: {("idle", 0): _rank_metrics("idle", 0)}}
    )


@pytest.mark.parametrize("kind", ["absent", "unset", "empty", "wrong_pool", "idle"])
def test_measurements_preserve_missing_vs_observed_zero(kind):
    fpm = _observations(kind, "prefill")
    queue = aggregation.aggregate_queue_depth(fpm, pool="prefill")
    caps = planner_types.WorkerCapabilities(
        prefill=planner_types.EngineCapabilities(max_kv_tokens=1600)
    )
    kv = aggregation.aggregate_kv_util(fpm, caps, pool="prefill")
    if kind == "idle":
        assert queue == 0
        assert kv == 0.0
    else:
        assert queue is None
        assert kv is None


def test_queue_totals_include_each_rank_in_the_selected_pool():
    fpm = planner_types.FpmObservations(
        prefill={
            ("a", 0): _rank_metrics("a", 0, waiting=2),
            ("a", 1): _rank_metrics("a", 1, waiting=3),
        },
        decode={("b", 0): _rank_metrics("b", 0, waiting=10)},
    )
    assert aggregation.aggregate_queue_depth(fpm, pool="prefill") == 5
    assert aggregation.aggregate_queue_depth(fpm, pool="all") == 15


def _tick(engine, fpm, *, ready=1):
    counts = planner_types.WorkerCounts(
        ready_num_prefill=ready,
        ready_num_decode=ready,
        expected_num_prefill=4,
        expected_num_decode=4,
        prefill_scaling_in_progress=ready < 4,
        decode_scaling_in_progress=ready < 4,
    )
    return asyncio.run(
        engine.tick(
            engine.initial_tick(15.0),
            planner_types.TickInput(
                now_s=15.0, worker_counts=counts, fpm_observations=fpm
            ),
        )
    )


@pytest.mark.parametrize(
    "factory", [adapters.KedaAutoscaler, adapters.ReactiveAutoscaler]
)
@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("kind", ["absent", "unset", "empty", "idle"])
def test_queue_policies_hold_missing_measurements_but_scale_down_measured_idle(
    factory, mode, kind
):
    role = "prefill" if mode == "disagg" else "decode"
    effects = _tick(factory(mode=mode), _observations(kind, role))
    target = (
        effects.scale_to.num_prefill
        if mode == "disagg"
        else effects.scale_to.num_decode
    )
    if kind == "idle":
        assert target < 4
    else:
        assert target == 4


@pytest.mark.parametrize(
    "factory", [adapters.KedaAutoscaler, adapters.ReactiveAutoscaler]
)
def test_disagg_queue_policies_ignore_other_pool_measurements(factory):
    effects = _tick(factory(mode="disagg"), _observations("wrong_pool", "prefill"))
    assert effects.scale_to.num_prefill == 4


@pytest.mark.parametrize(
    "factory", [adapters.KedaAutoscaler, adapters.ReactiveAutoscaler]
)
def test_dp4_cache_saturation_scales_up_instead_of_down(factory):
    caps = planner_types.WorkerCapabilities(
        decode=planner_types.EngineCapabilities(max_kv_tokens=6400)
    )
    effects = _tick(
        factory(mode="disagg", capabilities=caps),
        planner_types.FpmObservations(decode=_fleet(4)),
        ready=4,
    )
    assert effects.scale_to.num_decode == 5
