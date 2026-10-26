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
from dynamo.planner.config.planner_config import PlannerConfig, RetuneEngineSpec
from dynamo.planner.core.types import (
    EngineCapabilities,
    RetuneInputs,
    RetuneParams,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.clock import VirtualClock, WallClock
from dynamo.planner.reconfiguration import PREDICTOR_FIELDS
from dynamo.planner.retune.controller import RetuneController, RetuneTickResult
from dynamo.planner.retune.sweep import SweepResult
from dynamo.planner.simulation.load_predictor import (
    LOAD_PREDICTOR_PRESETS,
    distinct_predictor_candidates,
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
# AISimulate recommend syntax; prefill_load_scale is pinned (an omitted knob
# searches its default domain), so it is a one-value dimension.
ROUTER = {
    "overlap_score_credit": {"choices": [0.0, 1.0]},
    "prefill_load_scale": 1.0,
    "temperature": {"choices": [0.0, 0.5]},
}


def _router_point(credit, temperature):
    return {
        "overlap_score_credit": credit,
        "prefill_load_scale": 1.0,
        "router_temperature": temperature,
    }


STARTUP = RetuneParams(router=_router_point(1.0, 0.0))
BEST = RetuneParams(router=_router_point(0.0, 0.5))
LSD = "load_scaling_down_sensitivity"
SENSITIVITY_GRID = {
    "load_sensitivity": {"preset": False},
    LSD: {"choices": [70, 80, 90]},
}


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


def _scored(score_of, *, num_requests=10, wall_s=0.5, **extra):
    def build(request):
        return SweepResult(
            status="ok",
            num_requests=num_requests,
            scores={
                candidate.key(): score_of(candidate, request)
                for candidate in request.candidates
            },
            wall_s=wall_s,
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


def _s1(mode="agg", *, advisory=False, **retune):
    return _config(
        mode,
        retune,
        enable_load_scaling=False,
        enable_throughput_scaling=False,
        advisory=advisory,
    )


def _s2(*, advisory=False, **retune):
    retune = {"router": None, "planner": SENSITIVITY_GRID, **retune}
    return _config("agg", retune, optimization_target="throughput", advisory=advisory)


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
            if decision.outcome == "applied":
                for key, value in decision.planner_changes.items():
                    setattr(config, key, value)

    return apply


async def _run(controller, until, *, start=0.0, counts=None, after_tick=None):
    ticks = []
    now = start
    while now <= until + 1e-9:
        tick_counts = counts(now) if callable(counts) else (counts or _counts())
        result = await controller.on_tick(now, tick_counts)
        assert isinstance(result, RetuneTickResult)
        ticks.append((now, result))
        if after_tick is not None:
            after_tick(now, result)
        now += STEP
    return ticks


def _blocking(finish):
    """A sweep that waits for ``release`` before it returns ``finish``."""

    release = threading.Event()

    def run(request):
        release.wait(30)
        return finish(request)

    return run, release


def _decisions(ticks):
    return [(now, decision) for now, result in ticks for decision in result.decisions]


def _events(ticks):
    return [(now, event) for now, result in ticks for event in result.audit_events]


def _presweep(winner, entry=None):
    return {
        "winner": winner,
        "fields": predictor_fields(winner if entry is None else entry),
        "losses": {winner: 0.1},
        "eval_steps": 6,
        "wall_s": 0.0,
    }


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
    config = _s1(sla_ttft_ms=200.0)
    controller = _controller(config, ratio=2.0, start=10.0)
    # The scoring SLA is frozen at construction.
    config.ttft_ms = 1.0
    config.itl_ms = 2.0

    ticks = await _run(controller, 140, start=10.0)

    launches = [(now, e) for now, e in _events(ticks) if "launch" in e]
    assert launches == [(75.0, "retune:launch w=0"), (135.0, "retune:launch w=1")]
    decisions = _decisions(ticks)
    assert [
        (now, d.decision_id, d.outcome, d.window_index) for now, d in decisions
    ] == [
        (80.0, "retune-10-w0", "kept", 0),
        (140.0, "retune-10-w1", "kept", 1),
    ]
    first = decisions[0][1]
    assert (
        first.window_start_s,
        first.window_end_s,
        first.as_of_s,
        first.launched_at_s,
        first.decided_at_s,
    ) == (10.0, 70.0, 75.0, 75.0, 80.0)
    request = sweeps.requests[0]
    # W_0 = [10, 70) planner seconds; ms(s) = origin + s * 1000 * 2.
    assert (request.window_start_ms, request.window_end_ms, request.as_of_ms) == (
        ORIGIN_MS + 20_000,
        ORIGIN_MS + 140_000,
        ORIGIN_MS + 150_000,
    )
    assert request.history_start_ms == request.window_start_ms
    assert request.base_planner is None
    assert request.window_s == request.timeout_s == INTERVAL
    # The retune ttft override and the planner's itl, as of construction.
    assert (request.sla_ttft_ms, request.sla_itl_ms) == (200.0, 50.0)
    assert ("retune:kept w=0") in [e for _, e in _events(ticks)]


async def test_busy_when_the_previous_sweep_is_still_running(sweeps):
    sweeps.default, release = _blocking(_scored(lambda _c, _r: 1.0))
    controller = _controller(_s1(), clock=WallClock())
    try:
        launch = await controller.on_tick(65.0, _counts())
        busy = await controller.on_tick(125.0, _counts())
        release.set()
        controller._pending.future.result(timeout=30)
        harvest = await controller.on_tick(130.0, _counts())
    finally:
        release.set()
        controller.close()

    assert launch.decisions == [] and launch.audit_events == ["retune:launch w=0"]
    assert [(d.outcome, d.window_index) for d in busy.decisions] == [("busy", 1)]
    assert busy.decisions[0].reason == "sweep in flight"
    assert [(d.outcome, d.window_index) for d in harvest.decisions] == [("kept", 0)]
    assert len(sweeps.requests) == 1


def _raise_boom(_request):
    raise RuntimeError("boom")


MEASURED = {"decision_latency_seconds": "measured"}


@pytest.mark.parametrize(
    ("retune", "result", "decided_at", "latency"),
    [
        ({}, _scored(lambda _c, _r: 1.0, wall_s=30.0), 70.0, 5.0),
        (
            {"decision_latency_seconds": 20.0},
            _scored(lambda _c, _r: 1.0, wall_s=30.0),
            85.0,
            20.0,
        ),
        (MEASURED, _scored(lambda _c, _r: 1.0, wall_s=0.5), 70.0, 5.0),
        (MEASURED, _scored(lambda _c, _r: 1.0, wall_s=22.0), 90.0, 22.0),
        (
            {**MEASURED, "timeout_seconds": 30.0, "decision_latency_scale": 2.0},
            _scored(lambda _c, _r: 1.0, wall_s=22.0),
            110.0,
            44.0,
        ),
        (
            {**MEASURED, "timeout_seconds": 20.0},
            _scored(lambda _c, _r: 1.0, wall_s=25.0),
            85.0,
            20.0,
        ),
        (MEASURED, _raise_boom, 70.0, 5.0),
        (
            {**MEASURED, "timeout_seconds": 20.0},
            _scored(lambda _c, _r: 1.0, wall_s=float("nan")),
            85.0,
            20.0,
        ),
    ],
    ids=[
        "constant-floor",
        "constant",
        "measured-floor",
        "measured",
        "measured-scaled",
        "measured-capped-at-timeout",
        "measured-raises",
        "measured-nan-uses-timeout",
    ],
)
async def test_decision_latency_modes(sweeps, retune, result, decided_at, latency):
    sweeps.default = result
    controller = _controller(_s1(**retune))

    ticks = await _run(controller, 115)

    # Launched at L_0 = 65; the old parameters stay until the decision tick.
    ((now, decision),) = _decisions(ticks)
    assert (now, decision.launched_at_s, decision.decided_at_s) == (
        decided_at,
        65.0,
        decided_at,
    )
    assert decision.decision_latency_s == latency
    assert decision.to_dict()["decision_latency_s"] == latency


async def test_measured_timeout_decides_in_the_next_launch_tick(sweeps):
    # D == interval with a tick that accumulates rounding error (0.2 s):
    # the decision must land in the tick that launches the next window,
    # which must not be skipped as busy.
    step = 0.2
    sweeps.default = SweepResult(
        status="timeout", error="sweep exceeded its deadline", wall_s=INTERVAL
    )
    config = _config(
        "agg",
        {**MEASURED, "grace_seconds": INTERVAL},
        enable_load_scaling=False,
        enable_throughput_scaling=False,
        scheduling={"scale_interval_seconds": step},
    )
    controller = RetuneController(
        config,
        RetuneInputs(
            engine=_engine(),
            trace_paths=("trace.jsonl",),
            trace_origin_ms=ORIGIN_MS,
            arrival_speedup_ratio=1.0,
        ),
        WorkerCapabilities(),
        VirtualClock(),
        step,
    )
    controller.start(0.0)
    outcomes = []
    now = 0.0
    # Accumulate like the engine adapter does (last tick + interval).
    for _ in range(round(6 * INTERVAL / step)):
        result = await controller.on_tick(now, _counts())
        outcomes.extend((d.outcome, d.window_index) for d in result.decisions)
        now += step

    # Windows launch at 120, 180, ... and each decides at the next launch.
    assert outcomes == [("timeout", k) for k in range(3)]
    assert len(sweeps.requests) == 4


async def test_measured_latency_live_decides_after_the_sweep_finishes(sweeps):
    sweeps.default, release = _blocking(_scored(lambda _c, _r: 1.0, wall_s=12.0))
    controller = _controller(_s1(**MEASURED), clock=WallClock())
    try:
        launch = await controller.on_tick(65.0, _counts())
        running = await controller.on_tick(70.0, _counts())
        release.set()
        controller._pending.future.result(timeout=30)
        early = await controller.on_tick(75.0, _counts())
        harvest = await controller.on_tick(80.0, _counts())
    finally:
        release.set()
        controller.close()

    assert launch.audit_events == ["retune:launch w=0"]
    # Not done at 70; done at 75 but 65 + 12 s has not passed yet.
    assert running.decisions == [] and early.decisions == []
    ((decision),) = harvest.decisions
    assert (decision.outcome, decision.decided_at_s) == ("kept", 80.0)
    assert (decision.sweep_wall_s, decision.decision_latency_s) == (12.0, 12.0)


@pytest.mark.parametrize(
    ("result", "outcome", "reason"),
    [
        (
            SweepResult(status="timeout", error="sweep exceeded its deadline"),
            "timeout",
            "sweep exceeded its deadline",
        ),
        (
            SweepResult(status="failed", error="FileNotFoundError: trace"),
            "failed",
            "FileNotFoundError: trace",
        ),
        (SweepResult(status="empty"), "kept", "empty window"),
        (_raise_boom, "failed", "RuntimeError: boom"),
        (
            _scored(lambda c, _r: None if c == STARTUP else 5.0),
            "failed",
            "baseline replay failed",
        ),
    ],
    ids=["timeout", "failed", "empty", "raises", "baseline-missing"],
)
async def test_sweep_status_maps_to_outcome(sweeps, result, outcome, reason):
    sweeps.default = result
    controller = _controller(_s1())

    ticks = await _run(controller, 70)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == outcome
    assert reason in decision.reason
    assert decision.chosen == decision.previous == STARTUP


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


async def test_advisory_records_the_best_point_without_actuating(sweeps):
    sweeps.default = _best_wins()
    controller = _controller(_s1(advisory=True))

    ticks = await _run(controller, 75)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "advisory"
    assert decision.chosen == BEST and decision.previous == STARTUP
    assert decision.planner_changes == {}
    assert decision.predictor_warmup is None
    assert ("retune:advisory w=0") in [e for _, e in _events(ticks)]
    assert ticks[-1][1].router_parameters == _startup_map()
    assert controller.current_point() == STARTUP


async def test_advisory_recommends_from_the_deployed_point_every_window(sweeps):
    config = _s2(advisory=True)
    sweeps.default = _scored(lambda c, _r: 2.0 if c.planner[LSD] == 70 else 1.0)
    controller = _controller(config)

    # Even a host that applied every applied decision changes nothing.
    ticks = await _run(controller, 185, after_tick=_apply_planner_changes(config))

    decisions = _decisions(ticks)
    assert [(now, d.outcome, d.window_index) for now, d in decisions] == [
        (70.0, "advisory", 0),
        (130.0, "advisory", 1),
    ]
    for _, decision in decisions:
        # The recommended changes, from the deployed value each time.
        assert decision.previous == RetuneParams(planner={LSD: 80})
        assert decision.chosen == RetuneParams(planner={LSD: 70})
        assert decision.planner_changes == {LSD: 70}
    # Neither the baseline nor the inner planner moved to the recommendation.
    assert [request.baseline for request in sweeps.requests] == [
        RetuneParams(planner={LSD: 80})
    ] * 3
    assert [request.base_planner[LSD] for request in sweeps.requests] == [80] * 3
    assert all(request.base_planner["advisory"] is True for request in sweeps.requests)
    assert config.load_scaling_down_sensitivity == 80
    assert controller.current_point() == RetuneParams(planner={LSD: 80})


async def test_advisory_predictor_recommendation_carries_no_warmup(sweeps):
    config = _config(
        "agg",
        {"router": None, "planner": "default"},
        optimization_target="sla",
        enable_load_scaling=False,
        enable_throughput_scaling=True,
        throughput_adjustment_interval_seconds=10,
        advisory=True,
    )
    presweep = _presweep("kalman_default_raw")
    warmup = tuple(
        TrafficObservation(duration_s=10.0, num_req=float(i), isl=1.0, osl=1.0)
        for i in range(60)
    )
    sweeps.default = _scored(
        lambda c, r: 1.0 if c == r.baseline else 2.0,
        predictor_presweep=presweep,
        warmup_observations=warmup,
    )
    controller = _controller(config)

    ticks = await _run(controller, 70)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "advisory"
    assert decision.planner_changes == {"load_predictor": "kalman"}
    assert decision.predictor_warmup is None
    assert decision.to_dict()["predictor_warmup"] is None
    assert config.load_predictor == "arima"


@pytest.mark.parametrize("mode", ["agg", "disagg"])
async def test_applied_router_change_updates_the_desired_map(sweeps, mode):
    sweeps.default = _best_wins()
    controller = _controller(_s1(mode))

    ticks = await _run(controller, 75, counts=_counts(prefill=1))

    assert ticks[0][1].router_parameters == _startup_map(mode)
    ((now, decision),) = _decisions(ticks)
    assert (now, decision.outcome) == (70.0, "applied")
    assert decision.previous == STARTUP and decision.chosen == BEST
    assert decision.planner_changes == {}
    assert decision.predictor_warmup is None
    expected = _startup_map(mode, overlap_score_credit=0.0, router_temperature=0.5)
    # Credit 0 zeroes the cache-tier weights in the Rust resolution.
    assert all(values["host_cache_hit_weight"] == 0.0 for values in expected.values())
    assert ticks[-2][1].router_parameters == expected
    assert ticks[-1][1].router_parameters == expected
    assert controller.current_point() == BEST


async def test_candidate_set_is_canonical_and_window_independent(sweeps):
    sweeps.default = _best_wins()
    grid = {
        "overlap_score_credit": {"choices": [0.0, 0.5]},
        "prefill_load_scale": 1.0,
        "temperature": {"choices": [0.0, 0.5]},
    }
    controller = _controller(_s1(router=grid))

    await _run(controller, 130)

    first, second = sweeps.requests
    # The startup point (credit 1.0) joins the grid as a fifth candidate.
    points = [(c, t) for c in (0.0, 0.5) for t in (0.0, 0.5)] + [(1.0, 0.0)]
    keys = sorted(RetuneParams(router=_router_point(c, t)).key() for c, t in points)
    assert [candidate.key() for candidate in first.candidates] == keys
    assert first.candidates == second.candidates
    assert (first.baseline, second.baseline) == (STARTUP, BEST)


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


@pytest.mark.parametrize(
    ("history", "history_s"),
    # Unset, the history covers the throughput interval (10 s) x the largest
    # prophet_window_size (50) of the current and candidate predictors.
    [({}, 500.0), ({"history_seconds": 3600.0}, 3600.0)],
    ids=["auto-history", "explicit-history"],
)
async def test_predictor_presweep_winner_is_applied_with_warmup(
    sweeps, history, history_s
):
    config = _config(
        "agg",
        {"router": None, "planner": "default", **history},
        optimization_target="sla",
        enable_load_scaling=False,
        enable_throughput_scaling=True,
        throughput_adjustment_interval_seconds=10,
    )
    presweep = _presweep("kalman_default_raw")
    fields = presweep["fields"]
    warmup = tuple(
        TrafficObservation(duration_s=10.0, num_req=float(i), isl=1.0, osl=1.0)
        for i in range(60)
    )
    sweeps.default = _scored(
        lambda c, r: 1.0 if c == r.baseline else 2.0,
        predictor_presweep=presweep,
        warmup_observations=warmup,
    )
    controller = _controller(config, history_paths=("history.jsonl",))

    ticks = await _run(controller, 70)

    (request,) = sweeps.requests
    # (label, complete public mapping) for each of the 11 presets.
    assert request.predictor_candidates == tuple(
        distinct_predictor_candidates(list(LOAD_PREDICTOR_PRESETS))
    )
    assert [label for label, _ in request.predictor_candidates] == list(
        LOAD_PREDICTOR_PRESETS
    )
    assert request.history_paths == ("history.jsonl",)
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
        {"router": None, "planner": planner, "interval_seconds": 600.0},
        optimization_target="sla",
        enable_load_scaling=False,
        enable_throughput_scaling=True,
        throughput_adjustment_interval_seconds=180,
        **overrides,
    )


def _window_only_warmup():
    # W_0 with no history before the trace: 600 s / 180 s -> 4 buckets.
    return tuple(
        TrafficObservation(duration_s=180.0, num_req=num_req, isl=900.0, osl=90.0)
        for num_req in (100.0, 130.0, 80.0, 150.0)
    )


def _too_short(min_points):
    return (
        f"ValueError: KalmanPredictor needs {min_points} effective warmup "
        "points; received 4"
    )


KALMAN_MIN_POINTS_60 = {
    "load_predictor": "kalman",
    "load_predictor_log1p": False,
    "prophet_window_size": 50,
    "kalman_q_level": 1.0,
    "kalman_q_trend": 0.1,
    "kalman_r": 10.0,
    "kalman_min_points": 60,
}


@pytest.mark.parametrize(
    ("candidates", "overrides", "winner", "labels", "history_s", "warmup_error"),
    [
        (
            ["arima_raw", "kalman_default_raw"],
            {},
            ("kalman_default_raw", "kalman_default_raw"),
            ["arima_raw", "kalman_default_raw"],
            180.0 * 50,
            _too_short(5),
        ),
        (
            # A mapping candidate; its kalman_min_points also sets the history
            # every sweep reads.
            [KALMAN_MIN_POINTS_60],
            {"load_predictor": "kalman", "kalman_min_points": 2},
            ("custom_0", KALMAN_MIN_POINTS_60),
            ["custom_0"],
            180.0 * 60,
            _too_short(60),
        ),
        (
            ["arima_raw", "constant_last"],
            {},
            ("constant_last", "constant_last"),
            ["arima_raw", "constant_last"],
            180.0 * 50,
            None,
        ),
    ],
    ids=["kalman-needs-5", "kalman-min-points-60", "no-warmup"],
)
# Advisory recommends only what an active planner would apply.
@pytest.mark.parametrize("advisory", [False, True], ids=["active", "advisory"])
async def test_predictor_swap_with_too_short_a_warmup_fails(
    sweeps, candidates, overrides, winner, labels, history_s, warmup_error, advisory
):
    config = _sla_predictor_config(
        {"load_predictor": {"preset": candidates}}, advisory=advisory, **overrides
    )
    # The pre-sweep winner is fixed into every non-baseline candidate; the
    # sweep prepared it on the warmup it returns.
    sweeps.default = _scored(
        lambda c, r: 1.0 if c == r.baseline else 2.0,
        predictor_presweep=_presweep(*winner),
        warmup_observations=None if warmup_error is None else _window_only_warmup(),
        predictor_warmup_error=warmup_error,
    )
    controller = _controller(config)

    ticks = await _run(controller, 610)

    (request,) = sweeps.requests
    assert [label for label, _ in request.predictor_candidates] == labels
    assert request.history_start_ms == ORIGIN_MS - history_s * 1000
    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "failed"
    assert decision.reason == (
        "predictor change without warmup observations"
        if warmup_error is None
        else f"predictor warmup rejected: {warmup_error}"
    )
    assert decision.planner_changes == {} and decision.predictor_warmup is None
    # Nothing is pending: the next sweep's baseline is still the current point.
    assert controller.current_point() == RetuneParams(
        planner={key: getattr(config, key) for key in PREDICTOR_FIELDS}
    )


async def test_a_baseline_duplicate_is_listed_once_and_kept(sweeps):
    # The pre-sweep winner equals the current predictor (arima_raw): the grid
    # point with the current sensitivity is the baseline after the pre-sweep.
    config = _config(
        "agg",
        {"router": None, "planner": "default"},
        optimization_target="sla",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        throughput_adjustment_interval_seconds=10,
        load_adjustment_interval_seconds=5,
    )
    # run_sweep scores equivalent candidates from one evaluation.
    sweeps.default = _scored(
        lambda c, _r: 2.0 if c.planner[LSD] == 80 else 1.0,
        predictor_presweep=_presweep("arima_raw"),
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


# ---------------------------------------------------------------------------
# Controller construction checks
# ---------------------------------------------------------------------------


# Engines are built inside the test: _engine() calls the native bindings, which
# test collection (e.g. the pytest marker report hook) may stub out.
@pytest.mark.parametrize(
    ("retune", "engine_kwargs", "match"),
    [
        ({}, {"router_mode": "round_robin"}, "kv_router"),
        ({}, {"router_policy_config": {"name": "custom"}}, "default router policy"),
        ({}, {"conditional_disagg_enabled": True}, "conditional disaggregation"),
        (
            {"router": {**ROUTER, "prefill_load_model": {"type": "ais"}}},
            {},
            "prefill_load_model.type is fixed at startup",
        ),
        ({}, {"overlap_score_credit": 0.0}, "startup credit is 0"),
    ],
    ids=[
        "round-robin",
        "custom-policy",
        "cond-disagg",
        "load-model-pin",
        "zero-credit-forbids",
    ],
)
def test_router_domain_engine_requirements(retune, engine_kwargs, match):
    engine = _engine(**engine_kwargs)
    with pytest.raises(ValueError, match=match):
        _controller(_s1(**retune), engine=engine)


@pytest.mark.parametrize("written", [None, "ais", {"choices": ["ais"]}])
def test_startup_ais_load_model_matches_a_written_ais(written):
    # The startup KvRouterConfig serializes the AIS load model as "aic".
    engine = _engine(router_prefill_load_model="ais")
    assert engine.router_config["router_prefill_load_model"] == "ais"
    assert (
        json.loads(
            KvRouterConfig.from_json(json.dumps(engine.router_config)).to_json()
        )["router_prefill_load_model"]
        == "aic"
    )
    router = dict(ROUTER)
    if written is not None:
        router["prefill_load_model"] = {"type": written}
    controller = _controller(_s1(router=router), engine=engine)
    assert len(controller._grid) == 4

    with pytest.raises(ValueError, match="keeps the running 'ais'"):
        _controller(
            _s1(router={**ROUTER, "prefill_load_model": {"type": "none"}}),
            engine=engine,
        )


async def test_zero_startup_credit_collapses_the_default_domain(sweeps):
    sweeps.default = _scored(lambda _c, _r: 1.0)
    controller = _controller(
        _s1(router="default"), engine=_engine(overlap_score_credit=0.0)
    )

    await _run(controller, 65)

    (request,) = sweeps.requests
    assert len(request.candidates) == 36
    assert {c.router["overlap_score_credit"] for c in request.candidates} == {0.0}


def test_planner_grid_points_that_do_not_fit_are_dropped(caplog):
    # Scale-up must stay above scale-down: (0.6, 0.7) is invalid, but every
    # value still appears in a valid point.
    config = _config(
        "agg",
        {
            "router": None,
            "planner": {
                **SENSITIVITY_GRID,
                "decode_scale_up_kv_rate": {"choices": [0.6, 0.8]},
                "decode_scale_down_kv_rate": {"choices": [0.4, 0.7]},
            },
        },
        optimization_target="load",
        decode_scale_up_kv_rate=0.8,
        decode_scale_down_kv_rate=0.4,
    )

    with caplog.at_level("WARNING", logger=controller_module.__name__):
        controller = _controller(config)

    rates = {
        (p.planner["decode_scale_up_kv_rate"], p.planner["decode_scale_down_kv_rate"])
        for p in controller._grid
    }
    assert rates == {(0.6, 0.4), (0.8, 0.4), (0.8, 0.7)}
    assert len(controller._grid) == 9
    (record,) = [r for r in caplog.records if "drops" in r.getMessage()]
    assert "retune drops 3 of 12 planner grid points" in record.getMessage()


def test_planner_values_that_never_fit_the_deployment_raise():
    config = _config(
        "agg",
        {"router": None, "planner": {"min_workers": {"choices": [1, 4]}}},
        optimization_target="throughput",
        max_gpu_budget=2,
    )
    capabilities = WorkerCapabilities(decode=EngineCapabilities(num_gpu=1))

    with pytest.raises(
        ValueError, match="no retune.planner candidate with min_endpoint=4 fits"
    ):
        _controller(config, capabilities=capabilities)


async def test_planner_changes_that_no_longer_fit_fail_the_decision(sweeps):
    # A grid point fit at startup, but the budget dropped before the decision.
    config = _config(
        "agg",
        {"router": None, "planner": {"min_workers": {"choices": [1, 2]}}},
        optimization_target="throughput",
        max_gpu_budget=4,
    )
    sweeps.default = _scored(
        lambda c, _r: 2.0 if c.planner["min_endpoint"] == 2 else 1.0
    )
    controller = _controller(
        config, capabilities=WorkerCapabilities(decode=EngineCapabilities(num_gpu=1))
    )

    def lower_budget(now, _result):
        if now == 65.0:
            config.max_gpu_budget = 1

    ticks = await _run(controller, 70, after_tick=lower_budget)

    ((_, decision),) = _decisions(ticks)
    assert decision.outcome == "failed"
    assert decision.reason.startswith("planner candidate ")
    assert decision.reason.endswith(
        "cannot run on this deployment: minimum endpoint footprint requires "
        "2 GPUs, exceeding max_gpu_budget=1"
    )
    assert decision.planner_changes == {}
    assert controller.current_point().planner["min_endpoint"] == 1


def test_scaling_retune_rejects_external_plugin_gateways():
    config = _s2()
    config.scheduling.gateway.enabled = True

    with pytest.raises(ValueError, match="gateway"):
        _controller(config)


async def test_close_cancels_the_running_sweep(sweeps):
    sweeps.default = _scored(lambda _c, _r: 1.0)
    controller = _controller(_s1())
    await _run(controller, 65)

    controller.close()

    assert sweeps.cancels[0].is_set()
