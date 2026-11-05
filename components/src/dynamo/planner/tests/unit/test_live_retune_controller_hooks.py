# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the controller imports the sweep.

"""Live retune hooks of ``RetuneController`` with a scripted sweep function.

Covers what a live host adds on top of the replay loop: epoch-phase
anchoring, the launch gate, harvest gating and abandoned sweeps,
``drop_pending``/``rewind``, the DGD spec replicas check, ``restore_desired``
and the structured failure reason of a decision. The sweep thread shared
across generations is tested with the engine adapter.
"""

from __future__ import annotations

import json
import threading
from collections import deque

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.llm import KvRouterConfig
from dynamo.planner.config.planner_config import PlannerConfig, RetuneEngineSpec
from dynamo.planner.core.types import (
    RetuneInputs,
    RetuneParams,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.clock import VirtualClock, WallClock
from dynamo.planner.retune.controller import RetuneController
from dynamo.planner.retune.live.contracts import NON_DECISION_FAILURES, GateResult
from dynamo.planner.retune.sweep import SweepResult

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

INTERVAL = 60.0
GRACE = 5.0
STEP = 5.0
# An epoch-like planner time on a window boundary (phase 0): W_K = [T0-60, T0).
T0 = 1_800_000_000.0
K = int(T0 // INTERVAL) - 1
L_K = T0 + GRACE
ROUTER = {
    "overlap_score_credit": {"choices": [0.0, 1.0]},
    "prefill_load_scale": 1.0,
    "temperature": {"choices": [0.0, 0.5]},
}
ROUTER_DOMAIN = {
    "overlap_score_credit": (0.0, 1.0),
    "prefill_load_scale": (1.0,),
    "router_temperature": (0.0, 0.5),
}
SPEC = {"aggregated": 2}


def _point(credit, temperature):
    return RetuneParams(
        router={
            "overlap_score_credit": credit,
            "prefill_load_scale": 1.0,
            "router_temperature": temperature,
        }
    )


STARTUP = _point(1.0, 0.0)
BEST = _point(0.0, 0.5)


class FakeSweeps:
    """Scripted sweep function: records each call, returns queued results."""

    def __init__(self):
        self.requests = []
        self.cancels = []
        self.aborts = []
        self.results = deque()
        self.default = None

    def __call__(self, request, *, cancel=None, abort=None):
        self.requests.append(request)
        self.cancels.append(cancel)
        self.aborts.append(abort)
        result = self.results.popleft() if self.results else self.default
        return result(request) if callable(result) else result


class Gate:
    """Scripted ``launch_gate``: a queue of results, then ``default``."""

    def __init__(self, *results, default=None):
        self.results = deque(results)
        self.default = default if default is not None else GateResult()
        self.calls = []

    def __call__(self, k, window):
        self.calls.append((k, window))
        result = self.results.popleft() if self.results else self.default
        if isinstance(result, Exception):
            raise result
        return result


class Harvest:
    """Scripted ``harvest_allowed``."""

    def __init__(self, *answers, default=True):
        self.answers = deque(answers)
        self.default = default
        self.calls = []

    def __call__(self, generation):
        self.calls.append(generation)
        return self.answers.popleft() if self.answers else self.default


def _scored(score_of, *, wall_s=0.5, **extra):
    def build(request):
        return SweepResult(
            status="ok",
            num_requests=10,
            scores={c.key(): score_of(c) for c in request.candidates},
            wall_s=wall_s,
            **extra,
        )

    return build


def _flat(score=1.0, **extra):
    return _scored(lambda _c: score, **extra)


def _config(**retune):
    return PlannerConfig(
        namespace="test-ns",
        mode="agg",
        enable_load_scaling=False,
        enable_throughput_scaling=False,
        retune={
            "interval_seconds": INTERVAL,
            "grace_seconds": GRACE,
            "router": ROUTER,
            **retune,
        },
    )


def _inputs(origin_ms=0.0):
    return RetuneInputs(
        engine=RetuneEngineSpec(
            router_mode="kv_router",
            router_config=json.loads(KvRouterConfig().to_json()),
        ),
        trace_paths=("/trace/segments/dgd",),
        trace_origin_ms=origin_ms,
    )


def _live(
    sweeps,
    *,
    gate=None,
    harvest=None,
    clock=None,
    generation=3,
    start=T0,
    phase=0.0,
    config=None,
    executor=None,
):
    controller = RetuneController(
        config or _config(),
        _inputs(),
        WorkerCapabilities(),
        clock or VirtualClock(),
        STEP,
        sweep_fn=sweeps,
        launch_gate=(
            gate if gate is not None else Gate(default=GateResult(spec_replicas=SPEC))
        ),
        harvest_allowed=harvest,
        generation=generation,
        executor=executor,
    )
    controller.start(start, phase)
    return controller


def _counts(decode=2):
    return WorkerCounts(ready_num_decode=decode)


async def _tick(controller, now, decode=2):
    return await controller.on_tick(now, _counts(decode))


def _window_ms(k):
    return k * INTERVAL * 1000.0, (k + 1) * INTERVAL * 1000.0


# ---------------------------------------------------------------------------
# Phase anchoring
# ---------------------------------------------------------------------------


async def test_phase_anchoring_gives_restart_stable_windows_and_ids():
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    # Two processes started at different times share the window schedule.
    first = _live(sweeps, start=T0 - 7.0)
    second = _live(sweeps, start=T0 + 3.0)
    try:
        for controller in (first, second):
            await _tick(controller, L_K)
            result = await _tick(controller, L_K + STEP)
            (decision,) = result.decisions
            start_ms, end_ms = _window_ms(K)
            assert decision.window_index == K
            assert decision.decision_id == f"retune-w{int(end_ms)}"
            assert (decision.window_start_s, decision.window_end_s) == (
                T0 - INTERVAL,
                T0,
            )
        first_request, second_request = sweeps.requests
        assert (first_request.window_start_ms, first_request.window_end_ms) == (
            start_ms,
            end_ms,
        )
        assert first_request.as_of_ms == (T0 + GRACE) * 1000.0
        assert (second_request.window_start_ms, second_request.window_end_ms) == (
            start_ms,
            end_ms,
        )
    finally:
        first.close()
        second.close()


# ---------------------------------------------------------------------------
# Launch gate
# ---------------------------------------------------------------------------


async def test_gated_launch_passes_abort_and_live_request_fields():
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    abort = threading.Event()
    gate = Gate(GateResult(spec_replicas=SPEC, abort=abort))
    controller = _live(sweeps, gate=gate, generation=4)
    try:
        result = await _tick(controller, L_K)
    finally:
        controller.close()

    assert result.audit_events == [f"retune:launch w={K}"]
    assert gate.calls == [(K, (T0 - INTERVAL, T0, T0 + GRACE))]
    (request,) = sweeps.requests
    assert sweeps.aborts == [abort]
    assert sweeps.cancels == [controller._cancel]
    assert request.router_domain == ROUTER_DOMAIN
    # D_k = L_k + timeout (timeout defaults to the interval).
    assert request.deadline_s == L_K + INTERVAL
    assert request.spec_replicas == SPEC
    assert request.replicas == SPEC
    # Started at the window end: no recorded count covers the window start.
    assert request.replicas_verified is False
    assert request.generation == 4
    assert request.trace_paths == ("/trace/segments/dgd",)


@pytest.mark.parametrize("reason", ["not_leader", "not_ready"])
async def test_retry_reasons_hold_the_window_until_it_may_launch(reason):
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    gate = Gate(
        GateResult(reason=reason),
        GateResult(reason=reason),
        default=GateResult(spec_replicas=SPEC),
    )
    controller = _live(sweeps, gate=gate)
    try:
        held = [await _tick(controller, L_K + i * STEP) for i in range(2)]
        launched = await _tick(controller, L_K + 2 * STEP)
        decided = await _tick(controller, L_K + 3 * STEP)
    finally:
        controller.close()

    for result in held:
        assert result.decisions == [] and result.audit_events == []
    assert [k for k, _ in gate.calls] == [K, K, K]
    assert launched.audit_events == [f"retune:launch w={K}"]
    assert [d.window_index for d in decided.decisions] == [K]
    assert decided.decisions[0].launched_at_s == L_K + 2 * STEP
    assert len(sweeps.requests) == 1


async def test_already_decided_consumes_the_window_without_a_decision():
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    gate = Gate(
        GateResult(reason="already_decided"), default=GateResult(spec_replicas=SPEC)
    )
    controller = _live(sweeps, gate=gate)
    try:
        consumed = await _tick(controller, L_K)
        quiet = await _tick(controller, L_K + STEP)
        next_window = await _tick(controller, L_K + INTERVAL)
    finally:
        controller.close()

    assert consumed.decisions == [] and consumed.audit_events == []
    assert quiet.decisions == [] and quiet.audit_events == []
    # Asked once for K (consumed), then for K + 1 at its launch time.
    assert [k for k, _ in gate.calls] == [K, K + 1]
    assert next_window.audit_events == [f"retune:launch w={K + 1}"]
    assert [r.window_index for r in sweeps.requests] == [K + 1]


async def test_gate_runs_before_the_busy_branch():
    release = threading.Event()
    sweeps = FakeSweeps()

    def blocking(request):
        release.wait(30)
        return _flat()(request)

    sweeps.default = blocking
    gate = Gate(
        GateResult(spec_replicas=SPEC),
        GateResult(reason="not_leader"),
        default=GateResult(spec_replicas=SPEC),
    )
    controller = _live(sweeps, gate=gate, clock=WallClock())
    try:
        await _tick(controller, L_K)
        held = await _tick(controller, L_K + INTERVAL)
        busy = await _tick(controller, L_K + INTERVAL + STEP)
    finally:
        release.set()
        controller.close()

    # A window this process may not work on is not recorded busy.
    assert held.decisions == []
    assert [(d.outcome, d.window_index) for d in busy.decisions] == [("busy", K + 1)]
    assert busy.decisions[0].failure_reason is None
    assert len(sweeps.requests) == 1


# ---------------------------------------------------------------------------
# Harvest gating and abandoned sweeps
# ---------------------------------------------------------------------------


async def test_refused_harvest_drops_the_result_and_relaunches_the_window():
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    harvest = Harvest(False)
    controller = _live(sweeps, harvest=harvest, generation=5)
    try:
        await _tick(controller, L_K)
        refused = await _tick(controller, L_K + STEP)
        decided = await _tick(controller, L_K + 2 * STEP)
    finally:
        controller.close()

    assert harvest.calls == [5, 5]
    # No decision; the window is due again and relaunches in the same tick.
    assert refused.decisions == []
    assert refused.audit_events == [f"retune:dropped w={K}", f"retune:launch w={K}"]
    assert [r.window_index for r in sweeps.requests] == [K, K]
    assert [(d.outcome, d.window_index) for d in decided.decisions] == [("kept", K)]


@pytest.mark.parametrize("reason", sorted(NON_DECISION_FAILURES))
async def test_non_decision_failures_record_nothing_and_rewind(reason):
    sweeps = FakeSweeps()
    sweeps.results.append(
        SweepResult(status="failed", error="stopped", failure_reason=reason)
    )
    sweeps.default = _flat()
    controller = _live(sweeps)
    try:
        await _tick(controller, L_K)
        abandoned = await _tick(controller, L_K + STEP)
        decided = await _tick(controller, L_K + 2 * STEP)
    finally:
        controller.close()

    assert abandoned.decisions == []
    assert abandoned.audit_events == [f"retune:dropped w={K}", f"retune:launch w={K}"]
    assert [(d.outcome, d.window_index) for d in decided.decisions] == [("kept", K)]


async def test_abandoned_window_is_not_relaunched_once_its_successor_is_due():
    sweeps = FakeSweeps()
    sweeps.results.append(SweepResult(status="failed", failure_reason="shutdown"))
    sweeps.default = _flat()
    controller = _live(sweeps, clock=WallClock())
    try:
        await _tick(controller, L_K)
        controller._pending.future.result(timeout=30)
        # Harvested only after the next window is due: K + 1 launches.
        result = await _tick(controller, L_K + INTERVAL)
    finally:
        controller.close()

    assert result.audit_events == [f"retune:dropped w={K}", f"retune:launch w={K + 1}"]


# ---------------------------------------------------------------------------
# drop_pending and rewind
# ---------------------------------------------------------------------------


async def test_drop_pending_and_rewind_relaunch_instead_of_busy():
    release = threading.Event()
    sweeps = FakeSweeps()

    def blocking(request):
        release.wait(30)
        return _flat()(request)

    sweeps.results.append(blocking)
    sweeps.default = _flat()
    abort = threading.Event()
    gate = Gate(
        GateResult(spec_replicas=SPEC, abort=abort),
        default=GateResult(spec_replicas=SPEC),
    )
    controller = _live(sweeps, gate=gate, clock=WallClock())
    try:
        await _tick(controller, L_K)
        dropped = controller._pending.future
        # A pending sweep keeps its window: rewind alone does nothing.
        controller.rewind(K - 1)
        assert controller._last_k == K

        controller.drop_pending()
        assert abort.is_set()
        assert controller._pending is None
        controller.rewind(K - 1)
        relaunch = await _tick(controller, L_K + STEP)
        release.set()
        dropped.result(timeout=30)
        # The relaunch queued behind the dropped sweep; it is decided once
        # its own sweep is done.
        controller._pending.future.result(timeout=30)
        decided = await _tick(controller, L_K + 3 * STEP)
    finally:
        release.set()
        controller.close()

    assert relaunch.decisions == []
    assert relaunch.audit_events == [f"retune:launch w={K}"]
    assert [r.window_index for r in sweeps.requests] == [K, K]
    assert [(d.outcome, d.window_index) for d in decided.decisions] == [("kept", K)]


# ---------------------------------------------------------------------------
# DGD spec replicas
# ---------------------------------------------------------------------------


async def _run_window(controller, counts_at, *, start, until):
    """Tick from ``start`` to ``until``; return window K's decisions.

    Earlier windows are due from the first tick too; only K is asserted.
    """

    decisions = []
    now = start
    while now <= until + 1e-9:
        result = await controller.on_tick(now, _counts(counts_at(now)))
        decisions.extend((now, d) for d in result.decisions if d.window_index == K)
        now += STEP
    return decisions


def _requests_for_k(sweeps):
    return [request for request in sweeps.requests if request.window_index == K]


async def test_a_ready_count_change_in_the_window_fails_it():
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    start = T0 - INTERVAL - 2 * STEP

    def counts(now):
        # One worker restarts for a tick in the middle of the window.
        return 1 if now == T0 - INTERVAL / 2 else 2

    controller = _live(sweeps, start=start)
    try:
        ((now, decision),) = await _run_window(
            controller, counts, start=start, until=L_K
        )
    finally:
        controller.close()

    assert now == L_K
    assert (decision.outcome, decision.failure_reason) == ("failed", "replicas_changed")
    assert decision.details == {
        "spec_replicas": SPEC,
        "observed_replicas": {"aggregated": 1},
    }
    assert decision.decision_id == f"retune-w{int(T0 * 1000)}"
    assert decision.launched_at_s is None
    assert _requests_for_k(sweeps) == []


@pytest.mark.parametrize(
    "counts, failed",
    [
        # 3 ready before the window, 2 from the tick at its start: clean.
        (lambda now: 3 if now < T0 - INTERVAL else 2, False),
        # The tick at the window start still reports 3: that count was in
        # effect when the window began, so it fails.
        (lambda now: 3 if now <= T0 - INTERVAL else 2, True),
    ],
    ids=["changed-before-start", "changed-after-start"],
)
async def test_the_count_in_effect_at_window_start_must_match_too(counts, failed):
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    start = T0 - INTERVAL - 2 * STEP
    controller = _live(sweeps, start=start)
    try:
        decisions = await _run_window(controller, counts, start=start, until=L_K)
    finally:
        controller.close()

    if failed:
        assert [(d.outcome, d.failure_reason) for _, d in decisions] == [
            ("failed", "replicas_changed")
        ]
        assert _requests_for_k(sweeps) == []
    else:
        assert decisions == []
        (request,) = _requests_for_k(sweeps)
        assert request.replicas_verified is True


async def test_window_started_before_the_controller_launches_unverified():
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    start = T0 - INTERVAL / 2
    controller = _live(sweeps, start=start)
    try:
        decisions = await _run_window(
            controller, lambda _now: 2, start=start, until=L_K
        )
    finally:
        controller.close()

    assert decisions == []
    (request,) = _requests_for_k(sweeps)
    assert request.replicas_verified is False
    assert request.replicas == SPEC


def _scaling_config():
    # Scenario 2: load scaling (easy mode) with the default sensitivities.
    return PlannerConfig(
        namespace="test-ns",
        mode="agg",
        enable_throughput_scaling=False,
        retune={
            "interval_seconds": INTERVAL,
            "grace_seconds": GRACE,
            "router": ROUTER,
            "planner": "default",
        },
    )


@pytest.mark.parametrize(
    ("start", "replicas", "verified"),
    [
        # Ready counts recorded from before the window: its start's count.
        (T0 - INTERVAL - 2 * STEP, {"aggregated": 3}, True),
        # The window began before the controller: the spec stands in.
        (T0 - INTERVAL / 2, SPEC, False),
    ],
    ids=["recorded", "unrecorded"],
)
async def test_a_scaling_planner_sweeps_from_the_ready_counts_at_window_start(
    start, replicas, verified
):
    # The planner scales within the window: 3 workers at its start, 1 in
    # the middle, 2 (the spec) at its end. That is not replicas_changed.
    sweeps = FakeSweeps()
    sweeps.default = _flat()

    def counts(now):
        if now < T0 - INTERVAL / 2:
            return 3
        return 1 if now < T0 - STEP else 2

    controller = _live(sweeps, start=start, config=_scaling_config())
    try:
        decisions = await _run_window(controller, counts, start=start, until=L_K)
    finally:
        controller.close()

    assert decisions == []
    (request,) = _requests_for_k(sweeps)
    assert request.replicas == replicas
    assert request.replicas_verified is verified
    # The executor starts each inner planner from SweepRequest.replicas.
    assert request.spec_replicas is None
    assert request.base_planner is not None
    assert {c.planner["load_scaling_down_sensitivity"] for c in request.candidates} == {
        70,
        80,
        90,
    }
    # The planner axis the DGDR sweep shards: the grid's planner points.
    assert sorted(
        planner["load_scaling_down_sensitivity"] for planner in request.grid_planners
    ) == [70, 80, 90]


# ---------------------------------------------------------------------------
# restore_desired
# ---------------------------------------------------------------------------


async def test_restore_desired_sets_the_deployed_point_but_not_startup():
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    controller = _live(sweeps)
    try:
        controller.restore_desired(
            {**BEST.router, "decode_active_request_weight": 4.0}  # extra: ignored
        )
        assert controller.current_point() == BEST
        result = await _tick(controller, L_K)
    finally:
        controller.close()

    expected = KvRouterConfig.from_json(
        json.dumps({**json.loads(KvRouterConfig().to_json()), **BEST.router})
    ).policy_parameters("aggregated")
    assert result.router_parameters == {"aggregated": dict(expected)}
    (request,) = sweeps.requests
    # The restored point is the baseline; startup stays a candidate.
    assert request.baseline == BEST
    assert STARTUP in request.candidates
