# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CloudAI MPC v3: capacity model and control-loop contract."""

from __future__ import annotations

import asyncio
import os

import pytest

pytest.importorskip("numpy", reason="CloudAI adapters require numpy")
dynamo_types = pytest.importorskip(
    "dynamo.planner.core.types",
    reason="CloudAI adapter tests require the optional Dynamo runtime",
)

from autoscaling_arena.adapters.cloudai_mpc_v3 import (  # noqa: E402
    AISCapacityModel,
    CapacityModel,
    CloudAIMPCV3Autoscaler,
    RooflineCapacityModel,
)

# Per-worker KV capacity of the published agg substrate: 29920 blocks x 16.
MAX_KV_TOKENS = 29920 * 16


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


def _v3(model: CapacityModel, **kwargs) -> CloudAIMPCV3Autoscaler:
    defaults = dict(
        mode="agg",
        horizon=5,
        slo_ttft_ms=2000.0,
        slo_itl_ms=50.0,
        violation_weight=50.0,
        gpu_cost_per_s=100.0,
        poll_interval_s=5.0,
        scale_up_penalty=0.0,
        scale_down_penalty=0.0,
        headroom_decay_ticks=4,
        smoothing_alpha=0.4,
        reactive_queue_threshold=6.0,
        reactive_rps_jump_ratio=1.8,
        kv_pressure_threshold=0.75,
        min_replicas=1,
        max_replicas=8,
    )
    defaults.update(kwargs)
    return CloudAIMPCV3Autoscaler(capacity_model=model, **defaults)


def test_min_stable_fleet_follows_the_capacity_model():
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS))
    mpc._isl_now, mpc._osl_ema = 8000.0, 40.0
    assert mpc._mu(5.4, 1) < 5.4  # one worker cannot sustain the load ...
    assert mpc._min_stable_fleet(5.4) >= 2  # ... so the stable fleet is larger
    assert mpc._min_stable_fleet(0.2) == 1


def test_v3_does_not_collapse_to_one_replica_under_sustained_load():
    """Regression for v1's 8 -> 1 drop the tick the backlog clears.

    Constant 5.4 rps at ISL 8000 needs at least two workers under the linear
    engine (one saturated worker sustains ~4 req/s). After the fleet has drained the
    queue, v3 must keep at least the minimum stable fleet.
    """
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS))
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


def test_v3_scales_to_one_when_load_is_tiny():
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS))
    fleet = 4
    for i in range(12):
        tick, tick_input = _tick(
            5.0 * i, rps=0.2, isl=500, osl=40, queue=0, ready=fleet, expected=fleet
        )
        fleet = asyncio.run(mpc.tick(tick, tick_input)).scale_to.num_decode
    assert fleet == 1


@pytest.mark.parametrize("mode", ["disagg", "agg"])
def test_v3_decisions_stay_within_bounds(mode: str):
    mpc = _v3(
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
            assert (
                2 <= decision.num_prefill <= 3
            )  # pools are sized independently (point 6)
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
        (8300, 39, 5.4, 4.0),  # flat: prefill-bound
        (8190, 39, 5.0, 3.6),  # staircase
        (4478, 1676, 1.7, 1.4),  # steady-near-capacity: decode/KV-bound
        (4413, 1685, 1.4, 1.4),  # step-and-recovery
    ],
)
def test_ais_capacity_matches_observed_saturated_rates(isl, osl, rps, observed):
    """Per-worker rates measured in saturated windows of the published agg runs."""
    model = AISCapacityModel(
        config={
            "model": "openai/gpt-oss-120b",
            "system": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "tp": 1,
            "moe_tp_size": 1,
            "worker_type": "aggregated",
            "estimation_mode": "auto",
            "fallback_policy": "deny",
        },
        max_kv_tokens=MAX_KV_TOKENS,
    )
    mu = model.service_rate(rps, isl, osl)
    assert mu == pytest.approx(observed, rel=0.35)


# --------------------------------------------------------------------------- #
# Point 2: order book / actuation lag
# --------------------------------------------------------------------------- #


def test_order_book_delays_new_capacity_and_cancels_pending_first():
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
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
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
    mpc._reconcile_orders(10.0, ready=1, expected=3)  # started by someone else
    assert mpc.pending_orders == 2 and mpc._orders[0][0] == pytest.approx(40.0)
    mpc._reconcile_orders(45.0, ready=2, expected=3)  # one came up
    assert mpc.pending_orders == 1


def test_queue_spike_floor_does_not_wind_up_while_orders_are_in_flight():
    """v1 added a replica every tick of the 30 s cold start; v3 must not.

    One worker at 5.4 rps / ISL 8000 builds a backlog. After the first
    scale-up is booked, the floor stays at the in-flight fleet as long as that
    fleet clears the backlog within the plan.
    """
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
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
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
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
# Point 3: scale-down rule
# --------------------------------------------------------------------------- #


def test_scale_down_waits_one_cold_start_and_needs_no_pending_orders():
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
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
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), cold_start_s=30.0)
    mpc._downsize_ticks = 100  # would otherwise be eligible
    tick, tick_input = _tick(
        0.0, rps=0.2, isl=500, osl=40, queue=0, ready=2, expected=4
    )  # two starting
    decision = asyncio.run(mpc.tick(tick, tick_input)).scale_to.num_decode
    assert decision == 4
    assert mpc._downsize_ticks == 0


def test_upper_forecast_exceeds_mean_forecast():
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS))
    for _ in range(12):
        mpc._forecaster.update(1.5 * 5.0, 5.0)
    mean, hi = mpc._forecaster.forecast(mpc._plan_steps, 5.0)
    assert mean[0] == pytest.approx(1.5)
    assert all(h > m for m, h in zip(mean, hi))


# --------------------------------------------------------------------------- #
# Point 4: goodput-per-GPU objective
# --------------------------------------------------------------------------- #


def test_good_fraction_is_half_at_each_slo_and_one_well_inside():
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS))
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
    mpc = _v3(
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
    mpc = _v3(
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
# Point 5: arrival forecaster
# --------------------------------------------------------------------------- #

from autoscaling_arena.adapters.cloudai_mpc_v3 import _ArrivalForecaster  # noqa: E402


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
# Point 6: disaggregated pools
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
    mpc = _v3(
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
    # decode: c_sat / (osl * T_d(c_sat)) -- KV-bound, far below prefill capacity here
    c = mpc._capacity.saturated_concurrency(4500, 1680)
    assert mpc._mu_decode() == pytest.approx(
        c / (1680 * (5 + 0.1 * c) / 1000), rel=1e-6
    )
    assert mpc._mu_decode() < mpc._mu_prefill()


def test_disagg_sizes_pools_independently_on_a_prefill_backlog():
    """A prefill-only backlog with idle decode must grow the prefill pool, not both."""
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), mode="disagg")
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


def test_disagg_scale_down_needs_the_cold_start_hold_per_pool():
    mpc = _v3(_LinearCapacityModel(max_kv_tokens=MAX_KV_TOKENS), mode="disagg")
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
