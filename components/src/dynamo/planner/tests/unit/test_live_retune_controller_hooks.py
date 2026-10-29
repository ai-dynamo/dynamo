# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the controller imports the sweep.

"""Live retune hooks of ``RetuneController`` with a scripted sweep function.

Covers what a live host adds on top of the replay loop: epoch-phase
anchoring, the launch gate, harvest gating and abandoned sweeps,
``drop_pending``/``rewind``, the predecessor generation's sweep, the DGD
spec replicas check, ``restore_desired`` and the structured failure reason
of a decision.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from concurrent.futures import Future

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

import dynamo.planner.retune.controller as controller_module
from dynamo.llm import KvRouterConfig
from dynamo.planner.config.planner_config import PlannerConfig, RetuneEngineSpec
from dynamo.planner.core.types import (
    RetuneDecision,
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


class ManualClock(WallClock):
    """A wall clock (no inline await) whose monotonic time tests set."""

    def __init__(self):
        self.mono = 0.0

    def monotonic(self) -> float:
        return self.mono


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
    predecessor=None,
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
        predecessor=predecessor,
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
# Construction and properties
# ---------------------------------------------------------------------------


def test_launch_gate_requires_a_sweep_fn():
    with pytest.raises(ValueError, match="launch_gate requires a sweep_fn"):
        RetuneController(
            _config(),
            _inputs(),
            WorkerCapabilities(),
            VirtualClock(),
            STEP,
            launch_gate=Gate(),
        )


def test_generation_router_keys_and_domain_are_exposed():
    controller = _live(FakeSweeps(), generation=7)
    try:
        assert controller.generation == 7
        assert controller.router_keys == (
            "overlap_score_credit",
            "prefill_load_scale",
            "router_temperature",
        )
        assert controller.router_domain == {
            key: list(values) for key, values in ROUTER_DOMAIN.items()
        }
        # A copy: callers cannot change the controller's domain.
        controller.router_domain["router_temperature"].append(9.0)
        assert controller.router_domain["router_temperature"] == [0.0, 0.5]
    finally:
        controller.close()


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


async def test_replay_start_keeps_its_decision_ids(monkeypatch):
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    monkeypatch.setattr(controller_module, "run_sweep", sweeps)
    controller = RetuneController(
        _config(), _inputs(), WorkerCapabilities(), VirtualClock(), STEP
    )
    controller.start(10.0)
    try:
        await _tick(controller, 75.0)
        (decision,) = (await _tick(controller, 80.0)).decisions
    finally:
        controller.close()

    assert decision.decision_id == "retune-10-w0"
    (request,) = sweeps.requests
    # Replay requests keep the live-only fields at their defaults.
    assert request.router_domain is None
    assert request.deadline_s is None
    assert request.spec_replicas is None
    assert request.replicas_verified is True
    assert request.generation == 0
    assert sweeps.aborts == [None]


def test_phase_must_be_finite():
    controller = _live(FakeSweeps())
    try:
        with pytest.raises(ValueError, match="phase_s must be finite"):
            controller.start(T0, float("nan"))
    finally:
        controller.close()


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


@pytest.mark.parametrize(
    "bad",
    [RuntimeError("supervisor bug"), GateResult(reason="bogus")],  # type: ignore[arg-type]
    ids=["raises", "unknown-reason"],
)
async def test_a_failing_gate_holds_the_window(bad):
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    gate = Gate(bad, default=GateResult(spec_replicas=SPEC))
    controller = _live(sweeps, gate=gate)
    try:
        held = await _tick(controller, L_K)
        launched = await _tick(controller, L_K + STEP)
    finally:
        controller.close()

    assert held.decisions == [] and held.audit_events == []
    assert launched.audit_events == [f"retune:launch w={K}"]


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


async def test_harvest_allowed_that_raises_counts_as_refused():
    sweeps = FakeSweeps()
    sweeps.default = _flat()

    def harvest(_generation):
        raise RuntimeError("lease read failed")

    controller = _live(sweeps, harvest=harvest)
    try:
        await _tick(controller, L_K)
        refused = await _tick(controller, L_K + STEP)
    finally:
        controller.close()

    assert refused.decisions == []
    assert f"retune:dropped w={K}" in refused.audit_events


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


@pytest.mark.parametrize(
    "result, outcome",
    [
        (
            SweepResult(
                status="failed",
                error="probe scores differ",
                failure_reason="crn_mismatch",
                details={"probe": {"expected": 1.0, "got": 1.5}},
            ),
            "failed",
        ),
        (
            SweepResult(
                status="timeout",
                error="window deadline",
                failure_reason="timeout",
                details={"requests": ["a", "b"]},
            ),
            "timeout",
        ),
        (
            _flat(failure_reason=None, details={"runtime_image": "ais:0.13"}),
            "kept",
        ),
    ],
    ids=["failed", "timeout", "kept"],
)
async def test_failure_reason_and_details_reach_the_decision(result, outcome):
    sweeps = FakeSweeps()
    sweeps.default = result
    controller = _live(sweeps)
    try:
        await _tick(controller, L_K)
        (decision,) = (await _tick(controller, L_K + STEP)).decisions
    finally:
        controller.close()

    expected = result if isinstance(result, SweepResult) else result(sweeps.requests[0])
    assert decision.outcome == outcome
    assert decision.failure_reason == expected.failure_reason
    assert decision.details == expected.details
    record = decision.to_dict()
    assert record["failure_reason"] == expected.failure_reason
    assert record["details"] == expected.details


async def test_a_raising_sweep_fn_is_a_failed_decision_without_reason():
    def raises(_request, *, cancel, abort):
        raise RuntimeError("kube unreachable")

    controller = _live(raises)
    try:
        await _tick(controller, L_K)
        (decision,) = (await _tick(controller, L_K + STEP)).decisions
    finally:
        controller.close()

    assert decision.outcome == "failed"
    assert decision.reason == "RuntimeError: kube unreachable"
    assert decision.failure_reason is None


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
        controller._pending.future.result(timeout=30)
        # The relaunch queued behind the dropped sweep, so it launched (and
        # becomes ready) a little after L_K + STEP.
        decided = await _tick(controller, L_K + 3 * STEP)
    finally:
        release.set()
        controller.close()

    assert relaunch.decisions == []
    assert relaunch.audit_events == [f"retune:launch w={K}"]
    assert [r.window_index for r in sweeps.requests] == [K, K]
    assert [(d.outcome, d.window_index) for d in decided.decisions] == [("kept", K)]


async def test_drop_pending_cancels_a_sweep_that_has_not_started():
    release = threading.Event()
    sweeps = FakeSweeps()

    def blocking(request):
        release.wait(30)
        return _flat()(request)

    sweeps.default = blocking
    controller = _live(sweeps, clock=WallClock())
    try:
        # Occupy the sweep thread so the next submission stays queued.
        blocker = controller._executor.submit(release.wait, 30)
        await _tick(controller, L_K)
        queued = controller._pending.future
        controller.drop_pending()
        assert queued.cancelled()
        # Nothing holds the thread on its behalf: no queue delay to track.
        assert controller._dropped is None
        release.set()
        blocker.result(timeout=30)
    finally:
        release.set()
        controller.close()

    assert sweeps.requests == []


def test_drop_pending_and_rewind_without_a_sweep():
    controller = _live(FakeSweeps())
    try:
        controller.drop_pending()
        controller._last_k = K
        controller.rewind(K + 3)
        assert controller._last_k == K
        controller.rewind(K - 2)
        assert controller._last_k == K - 2
    finally:
        controller.close()


async def test_a_sweep_queued_behind_a_dropped_one_launches_when_it_starts():
    release = threading.Event()
    sweeps = FakeSweeps()

    def blocking(request):
        release.wait(30)
        return _flat()(request)

    sweeps.results.append(blocking)
    sweeps.default = _flat()
    clock = ManualClock()
    controller = _live(sweeps, clock=clock)
    try:
        clock.mono = 100.0
        await _tick(controller, L_K)
        dropped = controller._pending.future
        controller.drop_pending()
        controller.rewind(K - 1)
        await _tick(controller, L_K + STEP)  # submitted at monotonic 100
        clock.mono = 107.0  # the dropped sweep returns 7 s later
        release.set()
        dropped.result(timeout=30)
        controller._pending.future.result(timeout=30)
        (decision,) = (await _tick(controller, L_K + 4 * STEP)).decisions
    finally:
        release.set()
        controller.close()

    assert decision.launched_at_s == L_K + STEP + 7.0


async def test_repeated_drops_keep_tracking_the_sweep_holding_the_thread():
    release = threading.Event()
    sweeps = FakeSweeps()

    def blocking(request):
        release.wait(30)
        return _flat()(request)

    sweeps.results.append(blocking)
    sweeps.default = _flat()
    clock = ManualClock()
    controller = _live(sweeps, clock=clock)
    try:
        clock.mono = 100.0
        await _tick(controller, L_K)
        running = controller._pending.future
        controller.drop_pending()
        controller.rewind(K - 1)
        await _tick(controller, L_K + STEP)  # queued behind the running one
        queued = controller._pending.future
        controller.drop_pending()
        assert queued.cancelled()
        controller.rewind(K - 1)
        await _tick(controller, L_K + 2 * STEP)  # still behind the first one
        clock.mono = 104.0
        release.set()
        running.result(timeout=30)
        controller._pending.future.result(timeout=30)
        (decision,) = (await _tick(controller, L_K + 5 * STEP)).decisions
    finally:
        release.set()
        controller.close()

    # The cancelled sweep never ran; the last one started when the first ended.
    assert len(sweeps.requests) == 2
    assert decision.launched_at_s == L_K + 2 * STEP + 4.0


# ---------------------------------------------------------------------------
# Predecessor generation and running_future
# ---------------------------------------------------------------------------


def _blocking_sweeps(release, started=None):
    """``FakeSweeps`` whose first sweep runs until ``release`` is set."""

    sweeps = FakeSweeps()

    def blocking(request):
        if started is not None:
            started.set()
        release.wait(30)
        return _flat()(request)

    sweeps.results.append(blocking)
    sweeps.default = _flat()
    return sweeps


async def test_a_first_launch_queues_behind_the_predecessor_generation():
    # The previous generation's sweep, still running on its own thread.
    predecessor: Future = Future()
    assert predecessor.set_running_or_notify_cancel()
    ran = threading.Event()
    sweeps = FakeSweeps()

    def record(request):
        ran.set()
        return _flat()(request)

    sweeps.default = record
    clock = ManualClock()
    controller = _live(sweeps, clock=clock, predecessor=predecessor)
    try:
        assert controller.running_future() is predecessor
        clock.mono = 100.0
        launch = await _tick(controller, L_K)
        assert launch.audit_events == [f"retune:launch w={K}"]
        queued = controller._pending.future
        # Waits for the predecessor; it holds no other thread, so it runs.
        assert controller.running_future() is queued
        assert not ran.wait(0.2)
        assert sweeps.requests == []

        clock.mono = 106.0  # the predecessor returns 6 s later
        predecessor.set_result(SweepResult(status="failed"))
        queued.result(timeout=30)
        assert controller.running_future() is None
        (decision,) = (await _tick(controller, L_K + 4 * STEP)).decisions
    finally:
        if not predecessor.done():
            predecessor.cancel()
        controller.close()

    assert [r.window_index for r in sweeps.requests] == [K]
    assert decision.launched_at_s == L_K + 6.0


async def test_a_done_predecessor_does_not_delay_the_first_launch():
    predecessor: Future = Future()
    predecessor.set_result(SweepResult(status="failed"))
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    controller = _live(sweeps, clock=WallClock(), predecessor=predecessor)
    try:
        assert controller.running_future() is None
        await _tick(controller, L_K)
        assert controller._pending.queued is None
        controller._pending.future.result(timeout=30)
        (decision,) = (await _tick(controller, L_K + 2 * STEP)).decisions
    finally:
        controller.close()

    assert decision.launched_at_s == L_K


async def test_a_dropped_launch_hands_the_predecessor_wait_on():
    # A launch dropped while it waits for the predecessor still holds this
    # controller's thread: the relaunch, and a successor, queue behind it.
    predecessor: Future = Future()
    assert predecessor.set_running_or_notify_cancel()
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    controller = _live(sweeps, clock=WallClock(), predecessor=predecessor)
    try:
        await _tick(controller, L_K)
        waiting = controller._pending.future
        controller.drop_pending()
        assert controller._dropped is waiting
        assert controller.running_future() is waiting
        controller.rewind(K - 1)
        await _tick(controller, L_K + STEP)
        relaunch = controller._pending.future
        assert relaunch.running() is False
        # The relaunch has not started: the waiting launch holds the thread.
        assert controller.running_future() is waiting

        predecessor.set_result(SweepResult(status="failed"))
        waiting.result(timeout=30)
        relaunch.result(timeout=30)
        assert controller.running_future() is None
    finally:
        if not predecessor.done():
            predecessor.cancel()
        controller.close()

    assert [r.window_index for r in sweeps.requests] == [K, K]


async def test_running_future_tracks_the_sweep_holding_the_thread():
    release = threading.Event()
    started = threading.Event()
    sweeps = _blocking_sweeps(release, started)
    controller = _live(sweeps, clock=WallClock())
    try:
        assert controller.running_future() is None
        await _tick(controller, L_K)
        running = controller._pending.future
        assert started.wait(30)
        assert controller.running_future() is running

        controller.drop_pending()
        assert controller.running_future() is running
        controller.rewind(K - 1)
        await _tick(controller, L_K + STEP)  # queued behind the dropped one
        queued = controller._pending.future
        assert controller.running_future() is running

        release.set()
        running.result(timeout=30)
        queued.result(timeout=30)
        assert controller.running_future() is None
    finally:
        release.set()
        controller.close()


async def test_running_future_after_close_skips_a_cancelled_queued_sweep():
    # The host reads running_future after close: a queued sweep that close
    # cancelled never runs, so only the running one is waited on.
    release = threading.Event()
    started = threading.Event()
    sweeps = _blocking_sweeps(release, started)
    controller = _live(sweeps, clock=WallClock())
    try:
        await _tick(controller, L_K)
        running = controller._pending.future
        assert started.wait(30)
        controller.drop_pending()
        controller.rewind(K - 1)
        await _tick(controller, L_K + STEP)
        queued = controller._pending.future
        controller.close()
        assert queued.cancelled()
        assert controller.running_future() is running
    finally:
        release.set()
        controller.close()
    running.result(timeout=30)
    assert controller.running_future() is None


def test_predecessor_must_be_a_future():
    with pytest.raises(ValueError, match="predecessor must be"):
        _live(FakeSweeps(), predecessor=object())


async def test_generations_chained_by_predecessor_never_sweep_at_once():
    # Reviewer scenario: generation 0 is slow to see its abort while
    # generation 1 relaunches the same window. Replacing generation 1 before
    # it launched hands generation 0's sweep on to generation 2 as well.
    lock = threading.Lock()
    running: list[int] = []
    overlaps: list[list[int]] = []
    order: list[tuple[int, int]] = []
    release = threading.Event()
    started = threading.Event()

    def sweep(request, *, cancel, abort):
        with lock:
            running.append(request.generation)
            order.append((request.generation, request.window_index))
            if len(running) > 1:
                overlaps.append(list(running))
        try:
            if request.generation == 0:
                started.set()
                release.wait(30)
                return SweepResult(status="failed", failure_reason="lost_leadership")
            return _flat()(request)
        finally:
            with lock:
                running.remove(request.generation)

    old = _live(sweep, clock=WallClock(), generation=0)
    middle = new = None
    try:
        await _tick(old, L_K)
        assert started.wait(30)
        old.drop_pending()
        old.close()
        middle = _live(
            sweep, clock=WallClock(), generation=1, predecessor=old.running_future()
        )
        middle.close()
        new = _live(
            sweep, clock=WallClock(), generation=2, predecessor=middle.running_future()
        )
        launched = await _tick(new, L_K + STEP)
        assert launched.audit_events == [f"retune:launch w={K}"]
        assert new._pending.queued is not None
        relaunch = new._pending.future
        assert not relaunch.done()
        release.set()
        relaunch.result(timeout=30)
    finally:
        release.set()
        for controller in (old, middle, new):
            if controller is not None:
                controller.close()

    assert overlaps == []
    assert order == [(0, K), (2, K)]


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


async def test_replicas_verified_when_the_counts_cover_the_window():
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    start = T0 - INTERVAL - 2 * STEP
    controller = _live(sweeps, start=start)
    try:
        decisions = await _run_window(
            controller, lambda _now: 2, start=start, until=L_K
        )
    finally:
        controller.close()

    assert decisions == []
    (request,) = _requests_for_k(sweeps)
    assert request.replicas_verified is True
    assert request.replicas == request.spec_replicas == SPEC


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


async def test_unverified_window_still_fails_on_an_observed_change():
    sweeps = FakeSweeps()
    sweeps.default = _flat()
    start = T0 - INTERVAL / 2
    controller = _live(sweeps, start=start)
    try:
        decisions = await _run_window(
            controller, lambda _now: 3, start=start, until=L_K
        )
    finally:
        controller.close()

    assert [(d.outcome, d.failure_reason) for _, d in decisions] == [
        ("failed", "replicas_changed")
    ]
    assert _requests_for_k(sweeps) == []


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


@pytest.mark.parametrize(
    "flat, match",
    [
        (
            {"overlap_score_credit": 0.0, "prefill_load_scale": 1.0},
            "router_temperature",
        ),
        ({**BEST.router, "router_temperature": -1.0}, "non-negative"),
        ({**BEST.router, "router_temperature": float("inf")}, "finite"),
    ],
    ids=["missing-key", "negative", "infinite"],
)
def test_restore_desired_rejects_bad_values_and_keeps_the_desired_point(flat, match):
    controller = _live(FakeSweeps())
    try:
        with pytest.raises(ValueError, match=match):
            controller.restore_desired(flat)
        assert controller.current_point() == STARTUP
    finally:
        controller.close()


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------


def test_decision_record_carries_failure_reason_and_a_private_details_copy():
    details = {"shards": [{"id": "g0", "state": "Completed"}]}
    decision = RetuneDecision(
        decision_id="retune-w1800000000000",
        outcome="failed",
        window_index=K,
        window_start_s=T0 - INTERVAL,
        window_end_s=T0,
        as_of_s=L_K,
        launched_at_s=L_K,
        decided_at_s=L_K + STEP,
        num_requests=0,
        num_candidates=2,
        previous=STARTUP,
        chosen=STARTUP,
        baseline_score=None,
        best_score=None,
        reason="probe scores differ",
        failure_reason="crn_mismatch",
        details=details,
    )
    details["shards"] = []

    record = decision.to_dict()
    assert record["failure_reason"] == "crn_mismatch"
    assert record["details"] == {"shards": [{"id": "g0", "state": "Completed"}]}
    json.dumps(record)
    with pytest.raises(ValueError, match="failure_reason"):
        RetuneDecision(**{**decision.__dict__, "failure_reason": ""})


def test_replay_decision_record_has_empty_failure_fields():
    decision = RetuneDecision(
        decision_id="retune-0-w0",
        outcome="kept",
        window_index=0,
        window_start_s=0.0,
        window_end_s=60.0,
        as_of_s=65.0,
        launched_at_s=65.0,
        decided_at_s=70.0,
        num_requests=1,
        num_candidates=1,
        previous=STARTUP,
        chosen=STARTUP,
        baseline_score=1.0,
        best_score=1.0,
    )

    record = decision.to_dict()
    assert (record["failure_reason"], record["details"]) == (None, {})
