# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MPC: capacity model and control-loop contract."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
from feedback_fixtures import sample

dynamo_types = pytest.importorskip(
    "dynamo.planner.core.types",
    reason="MPC adapter tests require the optional Dynamo runtime",
)

from autoscaling_arena.adapters.mpc import (  # noqa: E402
    AISCapacityModel,
    CapacityModel,
    MPCAutoscaler,
    RooflineCapacityModel,
    _ArrivalForecaster,
)
from autoscaling_arena.match_config import (  # noqa: E402
    ReplicaCounts,
    SimAutoscalerConfig,
)
from autoscaling_arena.match_runner import _build_sim_factory  # noqa: E402
from autoscaling_arena.worker_limits import WorkerLimits  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

# Per-worker KV capacity of the published agg substrate: 29920 blocks x 16.
MAX_KV_TOKENS = 29920 * 16


@pytest.mark.parametrize("disagg", [False, True])
def test_ais_queries_use_deployed_pool_identity(monkeypatch, disagg):
    created = []
    queried = []

    def create_session(config):
        created.append(config)

        def prefill(**kwargs):
            queried.append(("prefill", config["worker_type"], config["tp"]))
            return 12.0

        def decode(**kwargs):
            queried.append(("decode", config["worker_type"], config["tp"]))
            return 15.0  # AIS totals the osl - 1 = 15 steps after the first token

        return SimpleNamespace(predict_prefill=prefill, predict_decode=decode)

    monkeypatch.setattr("dynamo._internal.ais.create_session", create_session)
    prefill_config = {
        "worker_type": "prefill" if disagg else "aggregated",
        "tp": 4 if disagg else 8,
    }
    decode_config = {"worker_type": "decode" if disagg else "aggregated", "tp": 8}
    model = AISCapacityModel(prefill_config, decode_config=decode_config)
    assert model.prefill_ms(256) == 12
    assert model.decode_step_ms(1, 256, 16) == 1
    assert len(created) == (2 if disagg else 1)
    assert queried == [
        ("prefill", prefill_config["worker_type"], prefill_config["tp"]),
        ("decode", decode_config["worker_type"], decode_config["tp"]),
    ]


def test_ais_step_time_and_capacity_count_the_steps_after_the_first_token(
    monkeypatch,
):
    """A 16-token output is 15 decode steps: 780 ms of generation is 52 ms a step."""

    def create_session(config):
        return SimpleNamespace(
            predict_prefill=lambda **kwargs: 100.0,
            predict_decode=lambda **kwargs: 780.0,
        )

    monkeypatch.setattr("dynamo._internal.ais.create_session", create_session)
    model = AISCapacityModel({"worker_type": "aggregated"}, max_num_seqs=1)
    assert model.decode_step_ms(1, 256, 16) == pytest.approx(52.0)
    # One request at a time: 100 ms prefill plus the 780 ms the 15 steps take.
    assert model.residence_s(1, 256, 16, 0.0) == pytest.approx(0.88)
    assert model.service_rate(1.0, 256, 16) == pytest.approx(1 / 0.88)


class _LinearCapacityModel(CapacityModel):
    """Deterministic engine: 0.03 ms per prefill token, 5 ms decode steps.

    At ISL 8000 / OSL 40 one saturated worker sustains ~4 req/s, so the
    5.4 rps scenarios below need two workers.
    """

    def prefill_ms(self, isl: float) -> float:
        return 0.03 * isl

    def decode_step_ms(self, concurrency: float, isl: float, osl: float) -> float:
        return 5.0 + 0.1 * concurrency


def test_service_rate_is_bounded_by_compute_and_kv():
    model = _LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS)
    # Short outputs: prefill-bound. 8000-token prefill = 240 ms -> < 4.17 req/s.
    short = model.service_rate(load_per_worker=5.0, isl=8000, osl=1)
    assert 3.0 < short < 4.17
    # Long outputs: KV-bound. cap = 0.9*478720/(4000+4000) = 53.9 concurrent.
    long_ = model.service_rate(load_per_worker=5.0, isl=4000, osl=4000)
    cap = model.kv_concurrency_cap(4000, 4000)
    assert cap == pytest.approx(0.9 * MAX_KV_TOKENS / 8000)
    assert long_ <= cap / model.residence_s(cap, 4000, 4000, 0.0) + 1e-9
    assert long_ < short
    # Capacity is a saturation property: it must not depend on the offered load.
    assert model.service_rate(0.1, 4000, 4000) == pytest.approx(long_)


def test_bias_moves_toward_measured_capacity_and_is_clamped():
    model = _LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS)
    base = model.service_rate(2.0, 4000, 200)
    model.observe_saturated(
        measured_rate_per_worker=base / 2,
        load_per_worker=2.0,
        isl=4000,
        osl=200,
        alpha=1.0,
    )
    assert model.bias == pytest.approx(0.5)
    assert model.service_rate(2.0, 4000, 200) == pytest.approx(base / 2)
    model.observe_saturated(
        measured_rate_per_worker=base * 100,
        load_per_worker=2.0,
        isl=4000,
        osl=200,
        alpha=1.0,
    )
    assert model.bias == pytest.approx(4.0)
    model.observe_saturated(
        measured_rate_per_worker=0.0, load_per_worker=2.0, isl=4000, osl=200
    )
    assert model.bias == pytest.approx(4.0)  # non-positive samples are ignored


def _tick(
    at_s: float,
    *,
    rps: float,
    isl: float,
    osl: float,
    queue: int,
    ready: int,
    expected: int,
):
    fpm = pytest.importorskip("dynamo.common.forward_pass_metrics")
    counts = dynamo_types.WorkerCounts(
        ready_num_decode=ready,
        expected_num_decode=expected,
        decode_scaling_in_progress=expected != ready,
    )
    traffic = dynamo_types.TrafficObservation(
        duration_s=5.0, num_req=rps * 5.0, isl=isl, osl=osl, kv_hit_rate=0.0
    )
    metrics = fpm.ForwardPassMetrics(
        worker_id="0",
        queued_requests=fpm.QueuedRequestMetrics(num_prefill_requests=queue),
    )
    observations = dynamo_types.FpmObservations(decode={("0", 0): metrics})
    tick = dynamo_types.ScheduledTick(
        at_s=at_s,
        need_worker_states=True,
        need_worker_fpm=True,
        need_traffic_metrics=True,
    )
    return tick, dynamo_types.TickInput(
        now_s=at_s, traffic=traffic, worker_counts=counts, fpm_observations=observations
    )


def _mpc(model: CapacityModel, **kwargs) -> MPCAutoscaler:
    defaults = {
        "mode": "agg",
        "horizon": 5,
        "slo_ttft_ms": 2000.0,
        "slo_itl_ms": 50.0,
        "poll_interval_s": 5.0,
        "scale_up_penalty": 0.0,
        "scale_down_penalty": 0.0,
        "reactive_queue_threshold": 6.0,
        "min_replicas": 1,
        "max_replicas": 8,
    }
    defaults.update(kwargs)
    return MPCAutoscaler(capacity_model=model, **defaults)


def test_aggregate_downsize_checks_itl_even_with_low_utilization_and_ttft():
    class DecodeSensitive(_LinearCapacityModel):
        def itl_ms(self, load_per_worker, isl, osl, kv_hit_rate, *, saturated=False):
            return 60.0 if load_per_worker > 0.75 else 30.0

    mpc = _mpc(DecodeSensitive(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=0)
    mpc._ready = mpc._expected = mpc._current_replicas = 2
    mpc._isl_now, mpc._osl_ema = 100.0, 2.0
    assert 1.0 < 0.75 * mpc._mu(1.0, 1)
    assert mpc._predict_ttft_ms(1.0, 0, 1) < mpc._slo_ttft * 0.5
    assert mpc._downsize_feasible([1.0], 0, 2)
    assert not mpc._downsize_feasible([1.0], 0, 1)


def test_aggregate_downsize_checks_the_wait_of_requests_already_queued():
    class FlatItl(_LinearCapacityModel):
        def itl_ms(self, load_per_worker, isl, osl, kv_hit_rate, *, saturated=False):
            return 30.0

    mpc = _mpc(FlatItl(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=0)
    mpc._ready = mpc._expected = mpc._current_replicas = 2
    mpc._isl_now, mpc._osl_ema = 500.0, 40.0
    mu = mpc._mu(1.0, 1)  # one worker serves about 50 requests/s
    assert 1.0 < 0.75 * mu
    # Sixty waiting requests clear within the first step, so the wait at its
    # end is only the prefill, but they themselves wait past half the limit.
    plan = mpc._queue_plan([1.0], 60, 1)
    assert plan[0][0] == 0 and plan[0][1] < mpc._slo_ttft * 0.5
    assert plan[0][4] > mpc._slo_ttft * 0.5
    assert not mpc._downsize_feasible([1.0], 60, 1)
    # An empty queue still allows the reduction.
    assert mpc._downsize_feasible([1.0], 0, 1)


@pytest.mark.parametrize("mode,p,d", [("agg", 0, 2), ("disagg", 2, 1)])
def test_mpc_uses_measured_goodput_per_resident_gpu_second(mode, p, d):
    mpc = _mpc(
        _LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS),
        mode=mode,
        cold_start_s=0,
        gpus_per_worker=8,
        prefill_gpus_per_worker=4,
    )
    slo = {"ttft_ms": 2000.0, "itl_ms": 50.0, "e2e_ms": None}
    mpc.on_telemetry(sample(p=p, d=d, slo=slo))
    mpc.on_telemetry(
        sample(
            5,
            p=p,
            d=d,
            slo=slo,
            terminal=100,
            completed=90,
            good=20,
            gpu_seconds=120,
            draining_d=1,
        )
    )
    mpc._observe_outcomes(5, (p, d))
    if mode == "agg":
        mpc._update_rho(100, 0)
    else:
        mpc._update_rho_disagg(100, (0, 0))
    assert mpc._rho == pytest.approx(20 / 120)
    assert mpc._observed_outcomes["good_rate"] == 0.2


def test_match_factory_passes_the_scorers_short_output_policy():
    entry = SimAutoscalerConfig(
        name="mpc",
        type="mpc",
        config={"forward_model": {"type": "roofline"}},
        start=ReplicaCounts(prefill=0, decode=2),
    )
    factory = _build_sim_factory(
        entry, topology="agg", short_output_itl="fail", feasible_targets=((0, 2),)
    )
    engine = factory(None, None)
    assert engine._feedback.short_output_itl == "fail"
    assert engine._latency_calibration.short_output_itl == "fail"
    assert engine._fleet_targets == ((0, 2),)


def test_min_stable_fleet_follows_the_capacity_model():
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS))
    mpc._isl_now, mpc._osl_ema = 8000.0, 40.0
    assert mpc._mu(5.4, 1) < 5.4  # one worker cannot sustain the load ...
    assert mpc._min_stable_fleet(5.4) >= 2  # ... so the stable fleet is larger
    assert mpc._min_stable_fleet(0.2) == 1


def test_mpc_does_not_collapse_to_one_replica_under_sustained_load():
    """Keep a sustainable fleet after the backlog clears.

    Constant 5.4 rps at ISL 8000 needs at least two workers under the linear
    engine (one saturated worker sustains ~4 req/s). After the fleet has drained the
    queue, MPC must keep at least the minimum stable fleet.
    """
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS))
    decisions = []
    # Backlog phase on one worker, fleet ramps.
    for i in range(6):
        tick, tick_input = _tick(
            5.0 * i,
            rps=5.4,
            isl=8000,
            osl=40,
            queue=20 + 5 * i,
            ready=1,
            expected=max(1, decisions[-1] if decisions else 1),
        )
        decisions.append(asyncio.run(mpc.tick(tick, tick_input)).scale_to.num_decode)
    assert max(decisions) >= 2
    # Drained phase: the whole fleet is ready and the queue is empty.
    fleet = decisions[-1]
    drained = []
    for i in range(6, 14):
        tick, tick_input = _tick(
            5.0 * i, rps=5.4, isl=8000, osl=40, queue=0, ready=fleet, expected=fleet
        )
        fleet = asyncio.run(mpc.tick(tick, tick_input)).scale_to.num_decode
        drained.append(fleet)
    assert min(drained) >= 2, drained


def test_mpc_scales_to_one_when_load_is_tiny():
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS))
    fleet = 4
    for i in range(12):
        tick, tick_input = _tick(
            5.0 * i, rps=0.2, isl=500, osl=40, queue=0, ready=fleet, expected=fleet
        )
        fleet = asyncio.run(mpc.tick(tick, tick_input)).scale_to.num_decode
    assert fleet == 1


@pytest.mark.parametrize("mode", ["disagg", "agg"])
def test_mpc_decisions_stay_within_bounds(mode: str):
    mpc = _mpc(
        RooflineCapacityModel(max_kv_tokens=MAX_KV_TOKENS),
        mode=mode,
        min_replicas=2,
        max_replicas=3,
    )
    initial = mpc.initial_tick(0.0)
    assert initial.need_traffic_metrics is True
    for i in range(3):
        tick, tick_input = _tick(
            5.0 * i, rps=3.0, isl=2000, osl=100, queue=10, ready=2, expected=2
        )
        decision = asyncio.run(mpc.tick(tick, tick_input)).scale_to
        assert 2 <= decision.num_decode <= 3
        if mode == "disagg":
            assert 2 <= decision.num_prefill <= 3
        else:
            assert decision.num_prefill is None
    assert mpc.supports_ais_bootstrap is False


@pytest.mark.skipif(
    os.environ.get("ARENA_AIS_TESTS") != "1",
    reason="set ARENA_AIS_TESTS=1 to validate against the AIS perf database",
)
@pytest.mark.parametrize(
    "isl,osl,rps,observed",
    [
        (8300, 39, 5.4, 4.0),  # prefill-bound
        (8190, 39, 5.0, 3.6),
        (4478, 1676, 1.7, 1.4),  # capacity-model regression input
        (4413, 1685, 1.4, 1.4),  # capacity-model regression input
    ],
)
def test_ais_capacity_matches_observed_saturated_rates(isl, osl, rps, observed):
    """Per-worker rates measured in saturated windows of aggregated replays."""
    model = AISCapacityModel(
        config={
            "model": "openai/gpt-oss-120b",
            "system": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "tp": 1,
            "moe_tp_size": 1,
            "moe_ep_size": 1,
            "attention_dp": 1,
            "worker_type": "aggregated",
            "estimation_mode": "auto",
            "fallback_policy": "deny",
        },
        max_kv_tokens=MAX_KV_TOKENS,
    )
    mu = model.service_rate(rps, isl, osl)
    assert mu == pytest.approx(observed, rel=0.35)


# --------------------------------------------------------------------------- #
# order book / actuation lag
# --------------------------------------------------------------------------- #


def test_order_book_delays_new_capacity_and_cancels_pending_first():
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
    mpc._t_s, mpc._ready, mpc._expected = 100.0, 1, 1
    assert mpc._plan_steps == 6 + 5  # ceil(30/5) + horizon
    # Ordering 3 more now: active stays 1 for the six lag steps, then 4.
    schedule = mpc._active_schedule(4)
    assert schedule[:6] == [1] * 6 and schedule[6:] == [4] * 5
    mpc._book_decision(100.0, 4)
    assert mpc.pending_orders == 3
    # Next tick the harness reports 3 starting workers: the book agrees.
    mpc._t_s = 105.0
    mpc._reconcile_orders(105.0, ready=1, expected=4)
    assert mpc.pending_orders == 3
    # A scale-down to 2 cancels two pending orders, keeps the ready worker.
    schedule = mpc._active_schedule(2)
    assert (
        schedule[:5] == [1] * 5 and schedule[5:] == [2] * 6
    )  # the kept order lands at t=130
    mpc._book_decision(105.0, 2)
    assert mpc.pending_orders == 1
    # When the harness reports them ready, the book empties.
    mpc._reconcile_orders(135.0, ready=2, expected=2)
    assert mpc.pending_orders == 0


def test_reconcile_books_unknown_starting_workers_and_retires_ready_ones():
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
    mpc._reconcile_orders(10.0, ready=1, expected=3)  # started by someone else
    assert mpc.pending_orders == 2 and mpc._orders[0][0] == pytest.approx(40.0)
    mpc._reconcile_orders(45.0, ready=2, expected=3)  # one came up
    assert mpc.pending_orders == 1


def test_late_worker_is_not_counted_as_serving_until_reported_ready():
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
    mpc._reconcile_orders(100.0, ready=1, expected=1)
    mpc._book_decision(100.0, 2)  # booked for t=130
    # Due at this tick: it serves from the first step.
    mpc._reconcile_orders(130.0, ready=1, expected=2)
    assert mpc._active_schedule(2)[:2] == [2, 2]
    # Five seconds late and still starting: only the ready worker serves now.
    mpc._reconcile_orders(135.0, ready=1, expected=2)
    assert mpc.pending_orders == 1
    assert mpc._active_schedule(2)[:3] == [1, 2, 2]
    # Still starting a tick later: the current step again has one worker.
    mpc._reconcile_orders(140.0, ready=1, expected=2)
    assert mpc._active_schedule(2)[:3] == [1, 2, 2]
    # Reported ready: the order retires and both workers serve.
    mpc._reconcile_orders(145.0, ready=2, expected=2)
    assert mpc.pending_orders == 0
    assert mpc._active_schedule(2)[0] == 2


def test_late_worker_rule_applies_to_each_disaggregated_pool():
    mpc = _mpc(
        _LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS),
        mode="disagg",
        cold_start_s=30.0,
    )
    book = mpc._book_p
    book.reconcile(100.0, 1, 1)
    book.book(100.0, 3)  # two prefill workers booked for t=130
    book.reconcile(130.0, 1, 3)
    assert book.schedule(3)[:2] == [3, 3]
    book.reconcile(135.0, 1, 3)
    assert book.pending == 2
    assert book.schedule(3)[:3] == [1, 3, 3]
    # One came up, one is still late.
    book.reconcile(140.0, 2, 3)
    assert book.pending == 1
    assert book.schedule(3)[:3] == [2, 3, 3]


def test_plan_serves_nothing_while_no_worker_is_ready():
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
    mpc._isl_now, mpc._osl_ema = 500.0, 40.0
    mpc._reconcile_orders(0.0, ready=0, expected=1)  # the only worker is starting
    assert mpc.pending_orders == 1
    schedule = mpc._active_schedule(1)
    assert schedule[:6] == [0] * 6 and schedule[6:] == [1] * 5
    plan = mpc._queue_plan([2.0] * mpc._plan_steps, 0, 1)
    # Nothing completes before the worker is due; arrivals accumulate.
    assert [step[2] for step in plan[:6]] == [0] * 6
    assert plan[5][0] == pytest.approx(2.0 * 30.0)
    assert plan[6][2] > 0
    assert not mpc._downsize_feasible([0.0] * mpc._plan_steps, 0, 1)
    # The model-based efficiency estimate sees no completions either.
    mpc._update_rho(2.0, 0)
    assert mpc._good_ema == 0


def test_disaggregated_plan_serves_nothing_while_a_pool_has_no_ready_worker():
    mpc = _mpc(
        _LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS),
        mode="disagg",
        cold_start_s=30.0,
    )
    mpc._isl_now, mpc._osl_ema = 500.0, 40.0
    mpc._book_p.reconcile(0.0, 1, 1)
    mpc._book_d.reconcile(0.0, 0, 1)  # the only decode worker is starting
    assert mpc._book_d.schedule(1)[:7] == [0] * 6 + [1]
    steps = mpc._plan_steps
    plan = mpc._plan_disagg([2.0] * steps, (0, 0), 1, 1)
    assert [step[2] for step in plan[:6]] == [0] * 6
    assert plan[6][2] > 0
    assert not mpc._feasible_disagg([0.0] * steps, (0, 0), 1, 1)
    mpc._update_rho_disagg(2.0, (0, 0))
    assert mpc._good_ema == 0


def test_queue_spike_floor_does_not_wind_up_while_orders_are_in_flight():
    """Pending workers must not cause a new replica order every tick.

    One worker at 5.4 rps / ISL 8000 builds a backlog. After the first
    scale-up is booked, the floor stays at the in-flight fleet as long as that
    fleet clears the backlog within the plan.
    """
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
    tick, tick_input = _tick(
        0.0, rps=5.4, isl=8000, osl=40, queue=20, ready=1, expected=1
    )
    first = asyncio.run(mpc.tick(tick, tick_input)).scale_to.num_decode
    assert first >= 2
    targets = []
    for i in range(1, 6):  # the cold start: workers starting, backlog persists
        tick, tick_input = _tick(
            5.0 * i,
            rps=5.4,
            isl=8000,
            osl=40,
            queue=20 + 4 * i,
            ready=1,
            expected=first,
        )
        targets.append(asyncio.run(mpc.tick(tick, tick_input)).scale_to.num_decode)
    assert max(targets) <= first + 1, targets


def test_queue_spike_floor_holds_in_flight_orders_that_clear_the_backlog():
    """A pending order that clears the backlog must not be cancelled next tick."""
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
    tick, tick_input = _tick(
        0.0, rps=5.4, isl=8000, osl=40, queue=20, ready=1, expected=1
    )
    first = asyncio.run(mpc.tick(tick, tick_input)).scale_to.num_decode
    assert first >= 2
    tick, tick_input = _tick(
        5.0, rps=5.4, isl=8000, osl=40, queue=12, ready=1, expected=first
    )
    second = asyncio.run(mpc.tick(tick, tick_input)).scale_to.num_decode
    assert second >= first, (first, second)
    assert mpc.pending_orders >= first - 1


# --------------------------------------------------------------------------- #
# scale-down rule
# --------------------------------------------------------------------------- #


def test_scale_down_waits_one_cold_start_and_needs_no_pending_orders():
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
    fleet = 4
    decisions = []
    for i in range(12):
        tick, tick_input = _tick(
            5.0 * i, rps=0.2, isl=500, osl=40, queue=0, ready=fleet, expected=fleet
        )
        fleet = asyncio.run(mpc.tick(tick, tick_input)).scale_to.num_decode
        decisions.append(fleet)
    # Held for the six lag ticks, then allowed to shrink.
    assert decisions[:5] == [4] * 5, decisions
    assert decisions[-1] == 1, decisions


def test_scale_down_is_blocked_while_orders_are_pending():
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
    mpc._downsize_ticks = 100  # would otherwise be eligible
    tick, tick_input = _tick(
        0.0, rps=0.2, isl=500, osl=40, queue=0, ready=2, expected=4
    )  # two starting
    decision = asyncio.run(mpc.tick(tick, tick_input)).scale_to.num_decode
    assert decision == 4
    assert mpc._downsize_ticks == 0


def test_upper_forecast_exceeds_mean_forecast():
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS))
    for _ in range(12):
        mpc._forecaster.update(1.5 * 5.0, 5.0)
    mean, hi = mpc._forecaster.forecast(mpc._plan_steps, 5.0)
    assert mean[0] == pytest.approx(1.5)
    assert all(h > m for m, h in zip(mean, hi))


# --------------------------------------------------------------------------- #
# goodput-per-GPU objective
# --------------------------------------------------------------------------- #


def test_good_fraction_is_half_at_each_slo_and_one_well_inside():
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS))
    assert mpc._good_fraction(2000.0, 50.0) == pytest.approx(0.25)
    assert mpc._good_fraction(200.0, 10.0) > 0.999
    assert mpc._good_fraction(6000.0, 10.0) < 1e-6


def test_itl_under_saturation_is_prefill_plus_decode_step():
    model = _LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS)
    c = model.saturated_concurrency(8000, 40)
    assert model.itl_ms(1.0, 8000, 40, saturated=True) == pytest.approx(
        0.03 * 8000 + 5.0 + 0.1 * c
    )
    assert model.itl_ms(0.1, 8000, 40) < 50.0  # light load: processor-sharing form


def test_rho_tracks_goodput_per_gpu_second():
    mpc = _mpc(
        _LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS),
        rho_min=0.01,
        rho_window_s=5.0,
    )
    # 2 rps served well on one worker: 10 good per 5 s over 5 GPU-seconds -> rho = 2.0
    tick, tick_input = _tick(
        0.0, rps=2.0, isl=500, osl=40, queue=0, ready=1, expected=1
    )
    asyncio.run(mpc.tick(tick, tick_input))
    assert mpc._rho == pytest.approx(2.0, rel=0.05)


def test_ratio_objective_prefers_one_saturated_replica_over_two_idle_ones():
    """With one worker inside both SLOs, a second GPU only lowers the ratio."""
    mpc = _mpc(
        _LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS),
        scale_up_penalty=0.0,
        scale_down_penalty=0.0,
    )
    mpc._t_s, mpc._ready, mpc._expected, mpc._current_replicas = 0.0, 1, 1, 1
    mpc._isl_now, mpc._osl_ema, mpc._rho = 500.0, 40.0, 1.0
    forecast = [2.0] * mpc._plan_steps
    assert mpc._evaluate_candidate(1, forecast, 0) < mpc._evaluate_candidate(
        2, forecast, 0
    )


# --------------------------------------------------------------------------- #
# arrival forecaster
# --------------------------------------------------------------------------- #


def _feed(f: _ArrivalForecaster, rates, dt=5.0):
    for r in rates:
        f.update(r * dt, dt)


def test_forecaster_level_is_flat_on_constant_traffic_with_noise():
    f = _ArrivalForecaster(bin_s=15.0, tau_s=60.0, trend_window_s=120.0, plan_s=55.0)
    _feed(f, [1.4, 1.6, 1.5, 1.7, 1.3, 1.5] * 6)  # 3 min of jittery 1.5 rps
    mean, hi = f.forecast(11, 5.0)
    assert mean[0] == pytest.approx(1.5, abs=0.1)
    assert mean[-1] == pytest.approx(mean[0])  # no significant trend on flat data
    assert hi[0] - mean[0] == pytest.approx(
        0.84 * (1.5 * (0.25 / (1.75 * 15.0) + 1 / 55.0)) ** 0.5, rel=0.1
    )


def test_forecaster_snaps_to_a_step_within_one_bin():
    f = _ArrivalForecaster(bin_s=15.0, tau_s=60.0)
    _feed(f, [1.0] * 12)  # one minute at 1 rps
    _feed(f, [6.0] * 3)  # one 15 s bin at 6 rps: 90 arrivals vs 15 expected
    assert f.level == pytest.approx(6.0)


def test_forecaster_uses_a_trend_only_when_significant():
    f = _ArrivalForecaster(bin_s=15.0, tau_s=60.0, trend_window_s=120.0)
    _feed(f, [1.0 + 0.1 * i for i in range(24)])  # clean ramp, 2 min
    mean, _ = f.forecast(11, 5.0)
    assert mean[-1] > mean[0]
    g = _ArrivalForecaster(bin_s=15.0, tau_s=60.0, trend_window_s=120.0)
    _feed(g, [2.0, 2.2, 1.8, 2.1, 1.9, 2.0] * 4)
    mean, _ = g.forecast(11, 5.0)
    assert mean[-1] == pytest.approx(mean[0])


# --------------------------------------------------------------------------- #
# disaggregated pools
# --------------------------------------------------------------------------- #


def _disagg_tick(
    at_s, *, rps, isl, osl, q_prefill, q_decode, ready_p, exp_p, ready_d, exp_d
):
    fpm = pytest.importorskip("dynamo.common.forward_pass_metrics")
    counts = dynamo_types.WorkerCounts(
        ready_num_prefill=ready_p,
        expected_num_prefill=exp_p,
        ready_num_decode=ready_d,
        expected_num_decode=exp_d,
    )
    traffic = dynamo_types.TrafficObservation(
        duration_s=5.0, num_req=rps * 5.0, isl=isl, osl=osl, kv_hit_rate=0.0
    )
    pm = fpm.ForwardPassMetrics(
        worker_id="p0",
        queued_requests=fpm.QueuedRequestMetrics(num_prefill_requests=q_prefill),
    )
    dm = fpm.ForwardPassMetrics(
        worker_id="d0",
        queued_requests=fpm.QueuedRequestMetrics(num_decode_requests=q_decode),
    )
    obs = dynamo_types.FpmObservations(prefill={("p0", 0): pm}, decode={("d0", 0): dm})
    tick = dynamo_types.ScheduledTick(
        at_s=at_s,
        need_worker_states=True,
        need_worker_fpm=True,
        need_traffic_metrics=True,
    )
    return tick, dynamo_types.TickInput(
        now_s=at_s, traffic=traffic, worker_counts=counts, fpm_observations=obs
    )


def test_disagg_pool_capacities_follow_their_binding_resources():
    mpc = _mpc(
        _LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS),
        mode="disagg",
        kv_transfer_gbps=100.0,
        kv_bytes_per_token=131072,
    )
    mpc._isl_now, mpc._osl_ema = 4500.0, 1680.0
    # prefill: 135 ms prefill + 47 ms transfer -> ~5.5 req/s per prefill worker
    assert mpc._kv_transfer_ms() == pytest.approx(
        4500 * 131072 * 8 / 100e9 * 1000, rel=1e-6
    )
    assert mpc._mu_prefill() == pytest.approx(
        1000 / (135 + mpc._kv_transfer_ms()), rel=1e-6
    )
    # decode: c_sat / ((osl - 1) * T_d(c_sat)) -- KV-bound, far below prefill capacity here
    c = mpc._capacity.saturated_concurrency(4500, 1680)
    assert mpc._mu_decode() == pytest.approx(
        c / (1679 * (5 + 0.1 * c) / 1000), rel=1e-6
    )
    assert mpc._mu_decode() < mpc._mu_prefill()


def test_disagg_sizes_pools_independently_on_a_prefill_backlog():
    """A prefill-only backlog with idle decode must grow the prefill pool, not both."""
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), mode="disagg")
    # Short outputs: decode is cheap (one worker sustains hundreds of req/s), prefill of 8000
    # tokens is the bottleneck at ~4 req/s per worker.
    tick, ti = _disagg_tick(
        0.0,
        rps=8.0,
        isl=8000,
        osl=8,
        q_prefill=30,
        q_decode=0,
        ready_p=1,
        exp_p=1,
        ready_d=1,
        exp_d=1,
    )
    d = asyncio.run(mpc.tick(tick, ti)).scale_to
    assert d.num_prefill >= 2
    assert d.num_decode == 1, (d.num_prefill, d.num_decode)


def test_disagg_plan_drains_each_stage_at_its_own_capacity():
    mpc = _mpc(
        _LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), mode="disagg", cold_start_s=0
    )
    mpc._isl_now, mpc._osl_ema = 8000.0, 8.0
    mpc._book_p.reconcile(0.0, 1, 1)
    mpc._book_d.reconcile(0.0, 1, 1)
    mu_p, mu_d = mpc._mu_prefill(), mpc._mu_decode()
    assert mu_p < 5 < 100 < mu_d
    idle = [0.0] * mpc._plan_steps
    # 200 requests waiting at decode: the decode worker clears them at once.
    at_decode = mpc._plan_disagg(idle, (0, 200), 1, 1)
    assert at_decode[0][0] == 0 and at_decode[0][2] == 200
    # The same backlog in front of prefill drains at the prefill rate.
    at_prefill = mpc._plan_disagg(idle, (200, 0), 1, 1)
    assert at_prefill[0][0] == pytest.approx(200 - 5 * mu_p)
    assert at_prefill[-1][0] > 50
    # Each wait is charged at its own stage's capacity.
    first_token_ms = mpc._prefill_ms() + mpc._kv_transfer_ms()
    assert at_prefill[0][1] == pytest.approx(200 / mu_p * 1000 + first_token_ms)
    assert at_decode[0][1] == pytest.approx(200 / mu_d * 1000 + first_token_ms)


def test_disagg_does_not_add_prefill_workers_for_a_decode_backlog():
    """Work waiting at decode has been prefilled: only decode capacity clears it."""
    mpc = _mpc(
        _LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), mode="disagg", cold_start_s=0
    )
    tick, ti = _disagg_tick(
        0.0,
        rps=1.0,
        isl=8000,
        osl=8,
        q_prefill=0,
        q_decode=200,
        ready_p=1,
        exp_p=1,
        ready_d=1,
        exp_d=1,
    )
    d = asyncio.run(mpc.tick(tick, ti)).scale_to
    # One prefill worker handles ~4 req/s, far above the 1 req/s arriving.
    assert (d.num_prefill, d.num_decode) == (1, 1)


def test_disagg_scale_down_needs_the_cold_start_hold_per_pool():
    mpc = _mpc(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), mode="disagg")
    fleet_p, fleet_d = 4, 4
    seen = []
    for i in range(12):
        tick, ti = _disagg_tick(
            5.0 * i,
            rps=0.2,
            isl=500,
            osl=40,
            q_prefill=0,
            q_decode=0,
            ready_p=fleet_p,
            exp_p=fleet_p,
            ready_d=fleet_d,
            exp_d=fleet_d,
        )
        d = asyncio.run(mpc.tick(tick, ti)).scale_to
        fleet_p, fleet_d = d.num_prefill, d.num_decode
        seen.append((fleet_p, fleet_d))
    assert seen[:5] == [(4, 4)] * 5, seen
    assert seen[-1] == (1, 1), seen


@pytest.mark.parametrize(
    "budget,p_cost,d_cost",
    [(12, 4, 8), (16, 8, 8), (80, 4, 8), (112, 4, 8), (128, 8, 8)],
)
def test_overload_uses_each_runs_budget_and_binding_pool(budget, p_cost, d_cost):
    targets = WorkerLimits(2, 16).targets(
        topology="disagg", gpu_budget=budget, prefill_gpus=p_cost, decode_gpus=d_cost
    )
    mpc = _mpc(
        _LinearCapacityModel(),
        mode="disagg",
        max_replicas=15,
        cold_start_s=0,
        feasible_targets=targets,
        prefill_gpus_per_worker=p_cost,
        gpus_per_worker=d_cost,
    )
    tick, inputs = _disagg_tick(
        5,
        rps=70,
        isl=8000,
        osl=8,
        q_prefill=1000,
        q_decode=0,
        ready_p=1,
        exp_p=1,
        ready_d=1,
        exp_d=1,
    )
    decision = asyncio.run(mpc.tick(tick, inputs)).scale_to
    pair = decision.num_prefill, decision.num_decode
    assert pair in targets
    assert pair == (min(15, (budget - d_cost) // p_cost), 1)
    assert pair[0] * p_cost + pair[1] * d_cost <= budget


def test_overload_reallocates_expensive_decode_workers_within_case_budget():
    targets = WorkerLimits(2, 16).targets(
        topology="disagg", gpu_budget=80, prefill_gpus=4, decode_gpus=8
    )
    mpc = _mpc(
        _LinearCapacityModel(),
        mode="disagg",
        max_replicas=15,
        cold_start_s=0,
        feasible_targets=targets,
        prefill_gpus_per_worker=4,
        gpus_per_worker=8,
    )
    tick, inputs = _disagg_tick(
        5,
        rps=70,
        isl=8000,
        osl=8,
        q_prefill=1000,
        q_decode=0,
        ready_p=4,
        exp_p=4,
        ready_d=8,
        exp_d=8,
    )
    decision = asyncio.run(mpc.tick(tick, inputs)).scale_to
    assert (decision.num_prefill, decision.num_decode) == (15, 1)


def test_startup_keeps_asymmetric_fleet_until_traffic_is_observed():
    mpc = _mpc(_LinearCapacityModel(), mode="disagg", max_replicas=15, cold_start_s=0)
    tick, inputs = _disagg_tick(
        0,
        rps=0,
        isl=0,
        osl=0,
        q_prefill=0,
        q_decode=0,
        ready_p=12,
        exp_p=12,
        ready_d=4,
        exp_d=4,
    )
    inputs.traffic.duration_s = 0
    decision = asyncio.run(mpc.tick(tick, inputs)).scale_to
    assert (decision.num_prefill, decision.num_decode) == (12, 4)
    assert mpc._forecaster._acc_time == 0


def test_attention_dp_changes_worker_capacity_without_inflating_rank_limits():
    serial = _LinearCapacityModel(max_kv_tokens=1024, max_num_seqs=32)
    parallel = _LinearCapacityModel(
        max_kv_tokens=8192, max_num_seqs=32, prefill_parallelism=8, decode_parallelism=8
    )
    a = _mpc(serial, mode="disagg")
    b = _mpc(parallel, mode="disagg")
    assert parallel.saturated_concurrency(512, 128) == serial.saturated_concurrency(
        512, 128
    )
    assert b._mu_prefill() == pytest.approx(8 * a._mu_prefill())
    assert b._mu_decode() == pytest.approx(8 * a._mu_decode())
    assert b._decode_itl_ms(8) == pytest.approx(a._decode_itl_ms(1))
    assert parallel.service_rate(8, 512, 128) == pytest.approx(
        8 * serial.service_rate(1, 512, 128)
    )


def test_efficiency_weight_can_follow_low_goodput_per_gpu():
    mpc = _mpc(_LinearCapacityModel(), mode="disagg", gpus_per_worker=8)
    mpc._book_p.reconcile(0, 1, 1)
    mpc._book_d.reconcile(0, 1, 1)
    mpc._update_rho_disagg(0.1, (0, 0))
    assert 0 < mpc._rho < 0.01
