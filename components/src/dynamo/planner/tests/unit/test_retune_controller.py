# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the controller imports the sweep.

"""Unit tests for the retune controller with a scripted ``run_sweep``."""

import json
import threading
from collections import deque

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

import dynamo.planner.retune.controller as controller_module
from dynamo.llm import KvRouterConfig
from dynamo.mocker.config import normalize_mocker_config
from dynamo.planner.config.planner_config import PlannerConfig, RetuneEngineSpec
from dynamo.planner.core.types import (
    EngineCapabilities,
    RetuneFeedback,
    RetuneInputs,
    RetuneParams,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.clock import VirtualClock, WallClock
from dynamo.planner.retune import RetuneController, RetuneTickResult
from dynamo.planner.retune.sweep import SweepResult
from dynamo.planner.simulation.load_predictor import (
    LOAD_PREDICTOR_PRESETS,
    predictor_fields,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

ORIGIN_MS = 1_000_000.0
INTERVAL = 60.0
GRACE = 5.0
STEP = 5.0
ROUTER = {"overlap_score_credit": [0.0, 1.0], "router_temperature": [0.0, 0.5]}
STARTUP = RetuneParams(router={"overlap_score_credit": 1.0, "router_temperature": 0.0})
BEST = RetuneParams(router={"overlap_score_credit": 0.0, "router_temperature": 0.5})
LSD = "load_scaling_down_sensitivity"


class FakeSweeps:
    """Scripted ``run_sweep``: records requests, returns queued results."""

    def __init__(self):
        self.requests = []
        self.cancels = []
        self.results = deque()
        self.default = None

    def __call__(self, request, *, cancel=None):
        self.requests.append(request)
        self.cancels.append(cancel)
        result = self.results.popleft() if self.results else self.default
        return result(request) if callable(result) else result


@pytest.fixture
def sweeps(monkeypatch):
    fake = FakeSweeps()
    monkeypatch.setattr(controller_module, "run_sweep", fake)
    return fake


def _scored(score_of, *, num_requests=10, **extra):
    def build(request):
        return SweepResult(
            status="ok",
            num_requests=num_requests,
            scores={
                candidate.key(): score_of(candidate, request)
                for candidate in request.candidates
            },
            wall_s=0.5,
            **extra,
        )

    return build


def _best_wins(best=BEST, best_score=2.0, other_score=1.0):
    return _scored(lambda c, _r: best_score if c == best else other_score)


def _config(mode="agg", retune=None, **overrides):
    retune_values = {
        "interval_seconds": INTERVAL,
        "grace_seconds": GRACE,
        "router": ROUTER,
        **(retune or {}),
    }
    return PlannerConfig(
        namespace="test-ns", mode=mode, retune=retune_values, **overrides
    )


def _s1(mode="agg", **retune):
    return _config(
        mode, retune, enable_load_scaling=False, enable_throughput_scaling=False
    )


def _s2(**retune):
    retune = {"router": {}, "planner": {LSD: [70, 80, 90]}, **retune}
    return _config("agg", retune, optimization_target="throughput")


def _engine(mode="agg", router_mode="kv_router", **router_config):
    config = {**json.loads(KvRouterConfig().to_json()), **router_config}
    if mode == "disagg":
        return RetuneEngineSpec(
            prefill_engine_args={},
            decode_engine_args={},
            router_mode=router_mode,
            router_config=config,
        )
    return RetuneEngineSpec(router_mode=router_mode, router_config=config)


def _controller(
    config,
    *,
    engine=None,
    clock=None,
    capabilities=None,
    ratio=1.0,
    start=0.0,
    history_paths=(),
    trace_paths=("trace.jsonl",),
):
    inputs = RetuneInputs(
        engine=engine or _engine(config.mode),
        trace_paths=trace_paths,
        trace_origin_ms=ORIGIN_MS,
        arrival_speedup_ratio=ratio,
        history_paths=history_paths,
    )
    controller = RetuneController(
        config,
        inputs,
        capabilities or WorkerCapabilities(),
        clock or VirtualClock(),
        STEP,
    )
    controller.start(start)
    return controller


def _counts(decode=2, prefill=None):
    return WorkerCounts(ready_num_prefill=prefill, ready_num_decode=decode)


def _apply_planner_changes(config):
    """Mimic the replay host: apply planner changes after the tick."""

    def apply(_now, result):
        for decision in result.decisions:
            if decision.outcome in ("applied", "rolled_back"):
                for key, value in decision.planner_changes.items():
                    setattr(config, key, value)

    return apply


async def _run(
    controller,
    until,
    *,
    start=0.0,
    counts=None,
    feedback=None,
    after_tick=None,
):
    ticks = []
    now = start
    while now <= until + 1e-9:
        tick_counts = counts(now) if callable(counts) else (counts or _counts())
        result = await controller.on_tick(
            now, tick_counts, (feedback or {}).get(now, [])
        )
        assert isinstance(result, RetuneTickResult)
        ticks.append((now, result))
        if after_tick is not None:
            after_tick(now, result)
        now += STEP
    return ticks


def _decisions(ticks):
    return [(now, decision) for now, result in ticks for decision in result.decisions]


def _events(ticks):
    return [(now, event) for now, result in ticks for event in result.audit_events]


def _startup_map(mode="agg", **overrides):
    config = KvRouterConfig.from_json(
        json.dumps({**json.loads(KvRouterConfig().to_json()), **overrides})
    )
    worker_types = ("prefill", "decode") if mode == "disagg" else ("aggregated",)
    return {
        worker_type: config.policy_parameters(worker_type)
        for worker_type in worker_types
    }


# ---------------------------------------------------------------------------
# Schedule and timing
# ---------------------------------------------------------------------------


async def test_windows_launch_at_l_k_and_decide_one_tick_later(sweeps):
    sweeps.default = _scored(lambda _c, _r: 1.0)
    controller = _controller(_s1())

    ticks = await _run(controller, 140)

    launches = [(now, e) for now, e in _events(ticks) if "launch" in e]
    assert launches == [(65.0, "retune:launch w=0"), (125.0, "retune:launch w=1")]
    decisions = _decisions(ticks)
    assert [(now, d.outcome, d.window_index) for now, d in decisions] == [
        (70.0, "kept", 0),
        (130.0, "kept", 1),
    ]
    first = decisions[0][1]
    assert first.decision_id == "retune-0-w0"
    assert (
        first.window_start_s,
        first.window_end_s,
        first.as_of_s,
        first.launched_at_s,
        first.decided_at_s,
    ) == (0.0, 60.0, 65.0, 65.0, 70.0)
    request = sweeps.requests[0]
    assert (request.window_start_ms, request.window_end_ms, request.as_of_ms) == (
        ORIGIN_MS,
        ORIGIN_MS + 60_000,
        ORIGIN_MS + 65_000,
    )
    assert request.history_start_ms == request.window_start_ms
    assert request.base_planner is None
    assert request.timeout_s == INTERVAL
    assert request.sla_ttft_ms == 500.0 and request.sla_itl_ms == 50.0
    assert ("retune:kept w=0") in [e for _, e in _events(ticks)]


async def test_window_mapping_uses_start_and_arrival_speedup(sweeps):
    sweeps.default = _scored(lambda _c, _r: 1.0)
    controller = _controller(_s1(), ratio=2.0, start=10.0)

    ticks = await _run(controller, 80, start=10.0)

    request = sweeps.requests[0]
    # W_0 = [10, 70) planner seconds; ms(s) = origin + s * 1000 * 2.
    assert (request.window_start_ms, request.window_end_ms, request.as_of_ms) == (
        ORIGIN_MS + 20_000,
        ORIGIN_MS + 140_000,
        ORIGIN_MS + 150_000,
    )
    assert request.window_s == INTERVAL
    assert [now for now, e in _events(ticks) if "launch" in e] == [75.0]
    assert _decisions(ticks)[0][1].decision_id == "retune-10-w0"


async def test_decision_latency_delays_the_decision(sweeps):
    sweeps.default = _scored(lambda _c, _r: 1.0)
    controller = _controller(_s1(decision_latency_seconds=20.0))

    ticks = await _run(controller, 100)

    assert [(now, d.outcome) for now, d in _decisions(ticks)] == [(85.0, "kept")]


async def test_busy_when_the_previous_sweep_is_still_running(sweeps):
    release = threading.Event()
    finish = _scored(lambda _c, _r: 1.0)

    def blocking(request):
        release.wait(30)
        return finish(request)

    sweeps.default = blocking
    controller = _controller(_s1(), clock=WallClock())
    try:
        launch = await controller.on_tick(65.0, _counts(), [])
        busy = await controller.on_tick(125.0, _counts(), [])
        release.set()
        controller._pending.future.result(timeout=30)
        harvest = await controller.on_tick(130.0, _counts(), [])
    finally:
        release.set()
        controller.close()

    assert launch.decisions == [] and launch.audit_events == ["retune:launch w=0"]
    assert [(d.outcome, d.window_index) for d in busy.decisions] == [("busy", 1)]
    assert busy.decisions[0].reason == "sweep in flight"
    assert [(d.outcome, d.window_index) for d in harvest.decisions] == [("kept", 0)]
    assert len(sweeps.requests) == 1


async def test_live_mode_waits_for_ready_at_even_when_done(sweeps):
    sweeps.default = _scored(lambda _c, _r: 1.0)
    controller = _controller(_s1(decision_latency_seconds=30.0), clock=WallClock())
    try:
        await controller.on_tick(65.0, _counts(), [])
        controller._pending.future.result(timeout=30)
        early = await controller.on_tick(90.0, _counts(), [])
        ready = await controller.on_tick(95.0, _counts(), [])
    finally:
        controller.close()

    assert early.decisions == []
    assert [d.outcome for d in ready.decisions] == ["kept"]


@pytest.mark.parametrize(
    ("status", "outcome", "reason"),
    [
        ("timeout", "timeout", "sweep exceeded its deadline"),
        ("failed", "failed", "FileNotFoundError: trace"),
        ("empty", "kept", "empty window"),
    ],
)
async def test_sweep_status_maps_to_outcome(sweeps, status, outcome, reason):
    sweeps.default = SweepResult(
        status=status, error=None if status == "empty" else reason
    )
    controller = _controller(_s1())

    ticks = await _run(controller, 70)

    ((_, decision),) = _decisions(ticks)
    assert (decision.outcome, decision.reason) == (outcome, reason)
    assert decision.chosen == decision.previous == STARTUP


async def test_a_raising_sweep_is_a_failed_decision(sweeps):
    def explode(_request):
        raise RuntimeError("boom")

    sweeps.default = explode
    controller = _controller(_s1())

    ticks = await _run(controller, 70)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "failed"
    assert "boom" in decision.reason


async def test_missing_worker_counts_fail_the_window(sweeps):
    controller = _controller(_s1())

    ticks = await _run(controller, 70, counts=lambda _now: None)

    ((now, decision),) = _decisions(ticks)
    assert (now, decision.outcome) == (65.0, "failed")
    assert "worker counts" in decision.reason
    assert sweeps.requests == []


async def test_replicas_come_from_the_window_start(sweeps):
    sweeps.default = _scored(lambda _c, _r: 1.0)
    controller = _controller(_s1("disagg"))

    await _run(
        controller,
        130,
        counts=lambda now: _counts(decode=2 if now < 60 else 3, prefill=1),
    )

    assert [request.replicas for request in sweeps.requests] == [
        {"prefill": 1, "decode": 2},
        {"prefill": 1, "decode": 3},
    ]


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------


async def test_baseline_failure_is_failed(sweeps):
    sweeps.default = _scored(lambda c, _r: None if c == STARTUP else 5.0)
    controller = _controller(_s1())

    ticks = await _run(controller, 70)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "failed"
    assert decision.reason.startswith("baseline replay failed")


async def test_ties_go_to_the_baseline(sweeps):
    sweeps.default = _scored(lambda _c, _r: 1.0)
    controller = _controller(_s1(min_improvement_percentage=0.0))

    ticks = await _run(controller, 70)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "kept"
    assert decision.top[0] == (STARTUP.key(), 1.0)
    assert decision.baseline_score == decision.best_score == 1.0
    assert len(decision.top) == 4


@pytest.mark.parametrize(
    ("best_score", "outcome"), [(103.0, "kept"), (103.01, "applied")]
)
async def test_min_improvement_percentage_boundary(sweeps, best_score, outcome):
    sweeps.default = _best_wins(best_score=best_score, other_score=100.0)
    controller = _controller(_s1())

    ticks = await _run(controller, 70)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == outcome
    assert decision.baseline_score == 100.0
    assert decision.best_score == best_score


async def test_dry_run_records_the_best_point_without_actuating(sweeps):
    sweeps.default = _best_wins()
    controller = _controller(_s1(dry_run=True))

    ticks = await _run(controller, 75)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "dry_run"
    assert decision.chosen == BEST and decision.previous == STARTUP
    assert decision.planner_changes == {}
    assert ticks[-1][1].router_parameters == _startup_map()
    assert controller.current_point() == STARTUP


async def test_applied_router_change_updates_the_desired_map(sweeps):
    sweeps.default = _best_wins()
    controller = _controller(_s1())

    ticks = await _run(controller, 75)

    assert ticks[0][1].router_parameters == _startup_map()
    ((now, decision),) = _decisions(ticks)
    assert (now, decision.outcome) == (70.0, "applied")
    assert decision.previous == STARTUP and decision.chosen == BEST
    assert decision.planner_changes == {}
    assert decision.predictor_warmup is None
    expected = _startup_map(overlap_score_credit=0.0, router_temperature=0.5)
    # Credit 0 zeroes the cache-tier weights in the Rust resolution.
    assert expected["aggregated"]["host_cache_hit_weight"] == 0.0
    assert ticks[-2][1].router_parameters == expected
    assert ticks[-1][1].router_parameters == expected
    assert controller.current_point() == BEST


async def test_disagg_desired_map_has_prefill_and_decode(sweeps):
    sweeps.default = _best_wins()
    controller = _controller(_s1("disagg"))

    ticks = await _run(controller, 70, counts=lambda _now: _counts(decode=2, prefill=1))

    assert ticks[0][1].router_parameters == _startup_map("disagg")
    assert ticks[-1][1].router_parameters == _startup_map(
        "disagg", overlap_score_credit=0.0, router_temperature=0.5
    )
    assert ticks[0][1].router_parameters["decode"]["overlap_score_credit"] == 0.0


async def test_candidate_set_is_canonical_and_window_independent(sweeps):
    sweeps.default = _best_wins()
    controller = _controller(_s1())

    await _run(controller, 130)

    first, second = sweeps.requests
    keys = sorted(
        RetuneParams(router={"overlap_score_credit": c, "router_temperature": t}).key()
        for c in (0.0, 1.0)
        for t in (0.0, 0.5)
    )
    assert [candidate.key() for candidate in first.candidates] == keys
    assert first.candidates == second.candidates
    assert (first.baseline, second.baseline) == (STARTUP, BEST)


async def test_startup_point_joins_the_grid(sweeps):
    sweeps.default = _scored(lambda _c, _r: 1.0)
    controller = _controller(_s1(router={"overlap_score_credit": [0.0, 0.5]}))

    await _run(controller, 70)

    (request,) = sweeps.requests
    assert [c.router for c in request.candidates] == [
        {"overlap_score_credit": 0.0},
        {"overlap_score_credit": 0.5},
        {"overlap_score_credit": 1.0},
    ]
    assert request.baseline.router == {"overlap_score_credit": 1.0}


async def test_router_parameters_are_none_without_a_router_domain(sweeps):
    sweeps.default = _scored(lambda _c, _r: 1.0)
    controller = _controller(_s2())

    ticks = await _run(controller, 70)

    assert all(result.router_parameters is None for _, result in ticks)
    (request,) = sweeps.requests
    assert request.base_planner["retune"] is None
    assert request.base_planner[LSD] == 80
    assert [c.planner for c in request.candidates] == [
        {LSD: 70},
        {LSD: 80},
        {LSD: 90},
    ]


# ---------------------------------------------------------------------------
# Planner changes and staleness
# ---------------------------------------------------------------------------


async def test_operator_change_during_the_sweep_is_stale(sweeps):
    config = _s2()
    sweeps.default = _scored(lambda c, _r: 2.0 if c.planner[LSD] == 70 else 1.0)
    controller = _controller(config)

    def operator_patch(now, _result):
        if now == 65.0:
            config.load_scaling_down_sensitivity = 90

    ticks = await _run(controller, 70, after_tick=operator_patch)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "stale"
    assert decision.planner_changes == {}


@pytest.mark.parametrize("host_applies", [True, False], ids=["replay", "live-lag"])
async def test_same_tick_planner_change_is_not_stale(sweeps, host_applies):
    config = _s2(decision_latency_seconds=INTERVAL)
    sweeps.default = _scored(lambda c, _r: 2.0 if c.planner[LSD] == 70 else 1.0)
    controller = _controller(config)

    ticks = await _run(
        controller,
        185,
        after_tick=_apply_planner_changes(config) if host_applies else None,
    )

    decisions = _decisions(ticks)
    assert [(now, d.outcome, d.window_index) for now, d in decisions] == [
        (125.0, "applied", 0),
        (185.0, "kept", 1),
    ]
    applied = decisions[0][1]
    assert applied.planner_changes == {LSD: 70}
    assert applied.chosen == RetuneParams(planner={LSD: 70})
    # Window 1 launched in the decision tick snapshots the new value.
    second = sweeps.requests[1]
    assert second.baseline == RetuneParams(planner={LSD: 70})
    assert second.base_planner[LSD] == 70
    assert config.load_scaling_down_sensitivity == (70 if host_applies else 80)


async def test_predictor_presweep_winner_is_applied_with_warmup(sweeps):
    config = _config(
        "agg",
        {"router": {}, "planner": "default"},
        optimization_target="sla",
        enable_load_scaling=False,
        enable_throughput_scaling=True,
        throughput_adjustment_interval_seconds=10,
    )
    fields = predictor_fields("kalman_default_raw")
    warmup = tuple(
        TrafficObservation(duration_s=10.0, num_req=float(i), isl=1.0, osl=1.0)
        for i in range(60)
    )
    presweep = {
        "winner": "kalman_default_raw",
        "fields": fields,
        "losses": {"kalman_default_raw": 0.1},
        "eval_steps": 6,
        "wall_s": 0.0,
    }
    sweeps.default = _scored(
        lambda c, r: 1.0 if c == r.baseline else 2.0,
        predictor_presweep=presweep,
        warmup_observations=warmup,
    )
    controller = _controller(config, history_paths=("history.jsonl",))

    ticks = await _run(controller, 70)

    (request,) = sweeps.requests
    assert request.predictor_presets == tuple(LOAD_PREDICTOR_PRESETS)
    assert request.trace_paths == ("trace.jsonl",)
    assert request.history_paths == ("history.jsonl",)
    history_s = config.retune.effective_history_seconds
    assert history_s > 0
    assert request.history_start_ms == ORIGIN_MS - history_s * 1000
    assert request.baseline.planner["load_predictor"] == "arima"
    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "applied"
    # The grid point takes the baseline's other predictor fields, then the
    # winner's.
    assert decision.chosen.planner == {**request.baseline.planner, **fields}
    assert decision.planner_changes == {"load_predictor": "kalman"}
    assert decision.predictor_warmup == warmup
    assert decision.predictor_presweep == presweep


def _sla_predictor_config(planner, **overrides):
    return _config(
        "agg",
        {"router": {}, "planner": planner, "interval_seconds": 600.0},
        optimization_target="sla",
        enable_load_scaling=False,
        enable_throughput_scaling=True,
        throughput_adjustment_interval_seconds=180,
        **overrides,
    )


def _window_only_warmup(count=4):
    # W_0 with no history before the trace: 600 s / 180 s -> 4 buckets.
    return tuple(
        TrafficObservation(duration_s=180.0, num_req=num_req, isl=900.0, osl=90.0)
        for num_req in (100.0, 130.0, 80.0, 150.0)[:count]
    )


@pytest.mark.parametrize(
    "planner, overrides, winner",
    [
        ({"load_predictor": ["arima", "kalman"]}, {}, {"load_predictor": "kalman"}),
        (
            {"kalman_min_points": [2, 20]},
            {"load_predictor": "kalman", "kalman_min_points": 2},
            {"kalman_min_points": 20},
        ),
    ],
    ids=["kalman-needs-5", "kalman-min-points-20"],
)
async def test_predictor_swap_with_too_short_a_warmup_fails(
    sweeps, planner, overrides, winner
):
    config = _sla_predictor_config(planner, **overrides)
    sweeps.default = _scored(
        lambda c, _r: 2.0 if all(c.planner[k] == v for k, v in winner.items()) else 1.0,
        warmup_observations=_window_only_warmup(),
    )
    controller = _controller(config)

    ticks = await _run(controller, 610)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "failed"
    assert "predictor warmup rejected" in decision.reason
    assert "warmup points; received 4" in decision.reason
    assert decision.planner_changes == {} and decision.predictor_warmup is None
    # Nothing is pending: the next sweep's baseline is still the current point.
    assert controller.current_point() == RetuneParams(
        planner={key: getattr(config, key) for key in planner}
    )


async def test_predictor_swap_with_enough_warmup_is_applied(sweeps):
    config = _sla_predictor_config({"load_predictor": ["arima", "constant"]})
    sweeps.default = _scored(
        lambda c, _r: 2.0 if c.planner["load_predictor"] == "constant" else 1.0,
        warmup_observations=_window_only_warmup(),
    )
    controller = _controller(config)

    ticks = await _run(controller, 610)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "applied"
    assert decision.planner_changes == {"load_predictor": "constant"}
    assert decision.predictor_warmup == _window_only_warmup()


async def test_a_baseline_duplicate_is_listed_once_and_kept(sweeps):
    # The pre-sweep winner equals the current predictor (arima_raw): the grid
    # point with the current sensitivity is the baseline after the pre-sweep.
    config = _config(
        "agg",
        {"router": {}, "planner": "default"},
        optimization_target="sla",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        throughput_adjustment_interval_seconds=10,
        load_adjustment_interval_seconds=5,
    )
    presweep = {
        "winner": "arima_raw",
        "fields": predictor_fields("arima_raw"),
        "losses": {"arima_raw": 0.1},
        "eval_steps": 6,
        "wall_s": 0.0,
    }

    # run_sweep scores equivalent candidates from one evaluation.
    sweeps.default = _scored(
        lambda c, _r: 2.0 if c.planner[LSD] == 80 else 1.0,
        predictor_presweep=presweep,
    )
    controller = _controller(config)

    ticks = await _run(controller, 70)

    ((_, decision),) = _decisions(ticks)
    (request,) = sweeps.requests
    # Sensitivity grid {70, 80, 90} plus the baseline (sensitivity 80 with
    # every predictor field); the 80 grid point duplicates the baseline.
    assert len(request.candidates) == 4
    assert decision.outcome == "kept"
    keys = [key for key, _ in decision.top]
    assert keys[0] == request.baseline.key()
    assert len(keys) == len(set(keys)) == 3


async def test_predictor_grid_change_without_warmup_fails(sweeps):
    config = _config(
        "agg",
        {"router": {}, "planner": {"load_predictor": ["arima", "constant"]}},
        optimization_target="sla",
        enable_load_scaling=False,
        enable_throughput_scaling=True,
        throughput_adjustment_interval_seconds=10,
    )
    sweeps.default = _scored(
        lambda c, _r: 2.0 if c.planner["load_predictor"] == "constant" else 1.0
    )
    controller = _controller(config)

    ticks = await _run(controller, 70)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "failed"
    assert "warmup" in decision.reason


# ---------------------------------------------------------------------------
# Feedback and rollback
# ---------------------------------------------------------------------------


async def test_router_feedback_rolls_back_and_failed_rollback_disables(sweeps):
    sweeps.default = _best_wins()
    controller = _controller(_s1())
    feedback = {
        75.0: [
            RetuneFeedback("retune-0-w0", "router", "422 invalid policy parameters"),
            RetuneFeedback("retune-9-w9", "router", "unrelated"),
        ],
        80.0: [RetuneFeedback("retune-0-w0-rb", "router", "422 again")],
    }

    ticks = await _run(controller, 200, feedback=feedback)

    decisions = _decisions(ticks)
    assert [(now, d.outcome, d.decision_id) for now, d in decisions] == [
        (70.0, "applied", "retune-0-w0"),
        (75.0, "rolled_back", "retune-0-w0-rb"),
        (80.0, "failed", "retune-0-w0-rb-failed"),
    ]
    rollback = decisions[1][1]
    assert rollback.chosen == STARTUP and rollback.previous == BEST
    assert "422 invalid policy parameters" in rollback.reason
    assert dict(ticks)[75.0].router_parameters == _startup_map()
    assert "retune disabled" in decisions[2][1].reason
    # Disabled: no further launches, the desired map stays at the rollback.
    assert [e for _, e in _events(ticks) if "launch" in e] == ["retune:launch w=0"]
    assert ticks[-1][1].router_parameters == _startup_map()


async def test_planner_feedback_drops_the_pending_change(sweeps):
    config = _s2()
    sweeps.default = _scored(lambda c, _r: 2.0 if c.planner[LSD] == 70 else 1.0)
    controller = _controller(config)
    feedback = {75.0: [RetuneFeedback("retune-0-w0", "planner", "conflict")]}

    ticks = await _run(controller, 75, feedback=feedback)

    decisions = _decisions(ticks)
    assert [d.outcome for _, d in decisions] == ["applied", "rolled_back"]
    assert controller.current_point() == RetuneParams(planner={LSD: 80})
    # The rejected change never landed, so nothing needs reverting.
    assert decisions[1][1].planner_changes == {}


async def test_planner_rollback_reverts_an_applied_change(sweeps):
    config = _s2()
    sweeps.default = _scored(lambda c, _r: 2.0 if c.planner[LSD] == 70 else 1.0)
    controller = _controller(config)
    feedback = {75.0: [RetuneFeedback("retune-0-w0", "router", "422")]}

    ticks = await _run(
        controller,
        75,
        feedback=feedback,
        after_tick=_apply_planner_changes(config),
    )

    rollback = _decisions(ticks)[1][1]
    assert rollback.outcome == "rolled_back"
    assert rollback.planner_changes == {LSD: 80}
    assert config.load_scaling_down_sensitivity == 80


# ---------------------------------------------------------------------------
# Stage-2 validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("engine", "match"),
    [
        (_engine(router_mode="round_robin"), "kv_router"),
        (_engine(router_policy_config={"name": "custom"}), "default router policy"),
        (_engine(overlap_score_weight=2.0), "overlap_score_weight"),
        (_engine(conditional_disagg_enabled=True), "conditional disaggregation"),
        (_engine("disagg"), "prefill_engine_args"),
    ],
    ids=["round-robin", "custom-policy", "overlap-weight", "cond-disagg", "e5"],
)
def test_router_domain_engine_requirements(engine, match):
    with pytest.raises(ValueError, match=match):
        _controller(_s1(), engine=engine)


@pytest.mark.parametrize(
    ("router", "match"),
    [
        ({"router_ttl_secs": [1.0]}, "not router policy parameters"),
        ({"host_cache_hit_weight": [0.5, 1.5]}, "not a valid router configuration"),
    ],
)
def test_router_keys_and_values_resolve_through_the_bindings(router, match):
    with pytest.raises(ValueError, match=match):
        _controller(_s1(router=router))


def test_zero_startup_credit_forbids_positive_credit():
    with pytest.raises(ValueError, match="startup credit is 0"):
        _controller(_s1(), engine=_engine(overlap_score_credit=0.0))


async def test_zero_startup_credit_collapses_the_default_domain(sweeps):
    sweeps.default = _scored(lambda _c, _r: 1.0)
    controller = _controller(
        _s1(router="default"), engine=_engine(overlap_score_credit=0.0)
    )

    await _run(controller, 65)

    (request,) = sweeps.requests
    assert len(request.candidates) == 36
    assert {c.router["overlap_score_credit"] for c in request.candidates} == {0.0}


def test_planner_candidates_must_fit_the_deployment():
    config = _config(
        "agg",
        {"router": {}, "planner": {"min_endpoint": [1, 4]}},
        optimization_target="throughput",
        max_gpu_budget=2,
    )
    capabilities = WorkerCapabilities(decode=EngineCapabilities(num_gpu=1))

    with pytest.raises(ValueError, match="retune.planner candidate"):
        _controller(config, capabilities=capabilities)


def test_scaling_retune_rejects_external_plugin_gateways():
    config = _s2()
    config.scheduling.gateway.enabled = True

    with pytest.raises(ValueError, match="gateway"):
        _controller(config)


def test_scoring_sla_is_frozen_at_construction(sweeps):
    config = _s1(sla_ttft_ms=200.0)
    controller = _controller(config)
    config.ttft_ms = 1.0
    config.itl_ms = 2.0

    assert controller._sla_ttft_ms == 200.0
    assert controller._sla_itl_ms == 50.0


async def test_on_tick_requires_start_and_close_cancels(sweeps):
    sweeps.default = _scored(lambda _c, _r: 1.0)
    inputs = RetuneInputs(engine=_engine(), trace_paths=("t.jsonl",))
    controller = RetuneController(
        _s1(), inputs, WorkerCapabilities(), VirtualClock(), STEP
    )
    with pytest.raises(RuntimeError, match="start"):
        await controller.on_tick(0.0, _counts(), [])

    controller.start(0.0)
    await _run(controller, 65)
    controller.close()

    assert sweeps.cancels[0].is_set()


@pytest.mark.timeout(240)
async def test_real_sweep_scores_every_candidate_and_decides(tmp_path):
    trace = tmp_path / "trace.jsonl"
    rows = []
    for index in range(80):
        received = int(ORIGIN_MS) + 750 * index
        rows.append(
            {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": received + 40,
                "request": {
                    "request_id": f"r{index}",
                    "request_received_ms": received,
                    "total_time_ms": 40.0,
                    "output_tokens": 16,
                    "replay": {
                        "trace_block_size": 64,
                        "input_length": 256,
                        "input_sequence_hashes": [
                            index % 4,
                            10 + index % 7,
                            100 + index,
                            1,
                        ],
                    },
                },
            }
        )
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    engine = RetuneEngineSpec(
        extra_engine_args=normalize_mocker_config(
            {
                "engine": {
                    "block_size": 64,
                    "num_gpu_blocks": 4096,
                    "speedup_ratio": 1000.0,
                }
            }
        ),
        router_config=json.loads(KvRouterConfig().to_json()),
    )
    controller = _controller(_s1(), engine=engine, trace_paths=(str(trace),))
    try:
        ticks = await _run(controller, 70)
    finally:
        controller.close()

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome in ("kept", "applied"), decision.reason
    assert decision.num_requests == 80
    assert decision.num_candidates == 4
    assert decision.baseline_score > 0
    assert len(decision.top) == 4
    assert decision.sweep_wall_s > 0
