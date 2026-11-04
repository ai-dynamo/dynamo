# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live retune wiring of the engine adapter, the planner base and ``__main__``.

The retune controller is the recording stub of ``retune_stubs`` (one test
builds the real one), and the live supervisor is a fake ``LiveRetuneHandle``,
so these tests cover only the host contract: the disabled state,
``replace_retune``, the base's supervisor lifecycle, its runtime
configuration API and the SIGTERM path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from dynamo.planner import __main__ as planner_main
from dynamo.planner.config.planner_config import PlannerConfig, RetuneConfig
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.types import (
    PlannerEffects,
    RetuneDecision,
    RetuneInputs,
    RetuneParams,
    ScheduledTick,
    TickInput,
    WorkerCounts,
)
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.plugins.clock import WallClock
from dynamo.planner.plugins.orchestrator.engine_adapter import (
    OrchestratorEngineAdapter,
    RetuneHooks,
)
from dynamo.planner.reconfiguration import ReconfigureRequest, candidate_config
from dynamo.planner.retune.live.contracts import GateResult
from dynamo.planner.tests.retune_stubs import (
    ROUTER_MAP,
    StubTickResult,
    caps,
    install_stub_controller,
    retune_inputs,
    retune_only_config,
)
from dynamo.planner.tests.unit.test_live_retune_state import live_config

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

PHASE = 120.0


def _sweep(request, *, cancel, abort):
    raise AssertionError("the stub controller never sweeps")


def _hooks(**overrides) -> RetuneHooks:
    values = {
        "sweep_fn": _sweep,
        "launch_gate": lambda _k, _window: GateResult(),
        "harvest_allowed": lambda _generation: True,
        "phase_s": PHASE,
    }
    values.update(overrides)
    return RetuneHooks(**values)


@pytest.fixture
def stub_controller(monkeypatch):
    return install_stub_controller(monkeypatch)


def _adapter(*, inputs=None, hooks=None, clock=None, **overrides):
    return OrchestratorEngineAdapter(
        retune_only_config(**overrides),
        caps(),
        clock=clock or WallClock(),
        retune_inputs=inputs,
        retune_hooks=hooks if hooks is not None else _hooks(),
    )


def _decision() -> RetuneDecision:
    point = RetuneParams(router={"router_temperature": 0.5})
    return RetuneDecision(
        decision_id="retune-w60000",
        outcome="kept",
        window_index=0,
        window_start_s=0.0,
        window_end_s=60.0,
        as_of_s=120.0,
        launched_at_s=120.0,
        decided_at_s=125.0,
        num_requests=3,
        num_candidates=2,
        previous=point,
        chosen=point,
        baseline_score=1.0,
        best_score=1.0,
    )


# ---------------------------------------------------------------------------
# Engine adapter: construction and the disabled state
# ---------------------------------------------------------------------------


def test_live_controller_gets_the_hooks_and_the_phase(stub_controller):
    hooks = _hooks()
    engine = _adapter(inputs=retune_inputs(), hooks=hooks)

    (controller,) = stub_controller.instances
    assert engine.retune_controller is controller
    executor = controller.hooks.pop("executor")
    assert isinstance(executor, ThreadPoolExecutor)
    assert controller.hooks == {
        "sweep_fn": hooks.sweep_fn,
        "launch_gate": hooks.launch_gate,
        "harvest_allowed": hooks.harvest_allowed,
        "generation": 0,
    }
    engine.initial_tick(1_000.0)
    assert (controller.started, controller.phases) == ([1_000.0], [PHASE])


def test_live_without_inputs_or_a_valid_controller_starts_disabled(
    stub_controller, caplog
):
    with caplog.at_level(logging.WARNING):
        engine = _adapter(inputs=None)
    assert engine.retune_controller is None and stub_controller.instances == []
    assert "retune stays disabled" in caplog.text
    engine.initial_tick(0.0)  # no controller to start

    stub_controller.construct_error = ValueError("retune.router requires kv_router")
    with caplog.at_level(logging.ERROR):
        engine = _adapter(inputs=retune_inputs())
    assert engine.retune_controller is None
    assert "requires kv_router" in caplog.text


async def test_disabled_retune_only_planner_never_reconciles_replicas(
    stub_controller,
):
    # retune_only follows the config, not the controller: a disabled live
    # retune keeps the deployment's replicas (here below min_endpoint) and
    # stays quiet.
    engine = _adapter(inputs=None, optimization_target="sla", min_endpoint=2)
    counts = WorkerCounts(ready_num_decode=1, expected_num_decode=1)

    scheduled = engine.initial_tick(0.0)
    for _ in range(10):
        assert not scheduled.run_load_scaling
        assert not scheduled.need_worker_fpm
        effects = await engine.tick(
            scheduled, TickInput(now_s=scheduled.at_s, worker_counts=counts)
        )
        assert effects.scale_to is None
        assert effects.router_parameters is None
        assert effects.retune_decisions == []
        assert effects.diagnostics.audit_events == []
        assert effects.next_tick is not None
        scheduled = effects.next_tick


# ---------------------------------------------------------------------------
# Engine adapter: replace_retune
# ---------------------------------------------------------------------------


async def test_replace_retune_swaps_the_controller(stub_controller):
    engine = _adapter(inputs=retune_inputs())
    engine.initial_tick(0.0)
    (old,) = stub_controller.instances
    inputs = RetuneInputs(engine=retune_inputs().engine, trace_paths=("/trace/b",))

    engine.replace_retune(inputs, generation=4, now_s=500.0)

    old_, new = stub_controller.instances
    assert old_ is old
    # Abort the old sweep first, then close the old controller.
    assert old.calls == ["start", "drop_pending", "close"]
    assert new.args[1] is inputs
    assert new.generation == 4
    # Every generation submits to the adapter's one sweep thread.
    assert new.hooks["executor"] is old.hooks["executor"]
    assert new.hooks["launch_gate"] is old.hooks["launch_gate"]
    assert new.calls == ["start"]
    assert (new.started, new.phases) == ([500.0], [PHASE])
    assert engine.retune_controller is new

    # The next tick runs the new controller only.
    new.results.append(
        StubTickResult(decisions=[_decision()], router_parameters=ROUTER_MAP)
    )
    scheduled = ScheduledTick(at_s=505.0, at_monotonic_s=505.0)
    effects = await engine.tick(
        scheduled,
        TickInput(now_s=505.0, worker_counts=WorkerCounts(ready_num_decode=1)),
    )
    assert old.ticks == [] and len(new.ticks) == 1
    assert effects.retune_decisions == [_decision()]
    assert effects.router_parameters == ROUTER_MAP

    await engine.shutdown()
    assert (old.closed, new.closed) == (1, 1)


def test_replace_retune_with_no_inputs_disables_retune(stub_controller):
    engine = _adapter(inputs=retune_inputs())
    (old,) = stub_controller.instances

    engine.replace_retune(None, generation=2, now_s=10.0)

    assert engine.retune_controller is None
    assert old.calls == ["drop_pending", "close"]
    assert len(stub_controller.instances) == 1


@pytest.mark.timeout(10)
async def test_rebuilt_real_controllers_never_sweep_at_once():
    """Two generations of one window never run their sweeps concurrently.

    Generation 0's sweep of window k is dropped but, like a DGDR sweep that
    polls, only returns a while later; generation 1 relaunches window k in
    the meantime. Its sweep must start only after generation 0's returned,
    and it is decided once it is done.
    """

    pytest.importorskip(
        "aisimulate.sweeper",
        reason="AI Simulate is an optional Dynamo simulation dependency",
    )
    from dynamo.planner.retune.sweep import SweepResult

    lock = threading.Lock()
    running: list[int] = []
    overlaps: list[list[int]] = []
    order: list[tuple[int, int]] = []
    gen0_started = threading.Event()
    gen1_started = threading.Event()
    release = threading.Event()

    def sweep(request, *, cancel, abort):
        with lock:
            running.append(request.generation)
            order.append((request.generation, request.window_index))
            if len(running) > 1:
                overlaps.append(list(running))
        try:
            if request.generation == 0:
                gen0_started.set()
                # Sees the abort late, as a DGDR sweep at its next poll.
                release.wait(30)
                return SweepResult(status="failed", failure_reason="lost_leadership")
            gen1_started.set()
            return SweepResult(
                status="ok",
                num_requests=1,
                scores={candidate.key(): 1.0 for candidate in request.candidates},
            )
        finally:
            with lock:
                running.remove(request.generation)

    engine = _adapter(
        inputs=retune_inputs(),
        hooks=_hooks(
            sweep_fn=sweep,
            launch_gate=lambda _k, _window: GateResult(
                spec_replicas={"aggregated": 2}, abort=threading.Event()
            ),
            phase_s=0.0,
        ),
    )
    counts = WorkerCounts(ready_num_decode=2, expected_num_decode=2)
    # On a window boundary: W_k = [now - 120, now - 60) launches at now and
    # stays the only due window until now + 60.
    now = 1_800_000_000.0
    try:
        scheduled = engine.initial_tick(now)
        await engine.tick(scheduled, TickInput(now_s=now, worker_counts=counts))
        assert gen0_started.wait(30)

        engine.replace_retune(retune_inputs(), generation=1, now_s=now)
        await engine.tick(scheduled, TickInput(now_s=now + 1.0, worker_counts=counts))
        controller = engine.retune_controller
        assert controller is not None and controller.generation == 1
        relaunch = controller._pending.future
        # Queued behind generation 0's sweep, which still runs.
        assert not gen1_started.wait(0.2)

        release.set()
        relaunch.result(timeout=30)
        effects = await engine.tick(
            scheduled, TickInput(now_s=now + 30.0, worker_counts=counts)
        )
    finally:
        release.set()
        await engine.shutdown()

    assert overlaps == []
    ((_, k0), (_, k1)) = order
    assert [g for g, _ in order] == [0, 1] and k0 == k1
    (decision,) = effects.retune_decisions
    assert decision.window_index == k0


# ---------------------------------------------------------------------------
# Planner base wiring
# ---------------------------------------------------------------------------


class _Environment:
    def __init__(self) -> None:
        self.state = DeploymentState()
        self.state.decode.num_gpus = 1
        self.calls: list[str] = []

    def deployment_state(self) -> DeploymentState:
        return self.state

    async def initialize(self) -> None:
        self.calls.append("environment.initialize")

    async def refresh(self) -> None:
        pass

    async def shutdown(self) -> None:
        self.calls.append("environment.shutdown")


class FakeLiveRetune:
    """A ``LiveRetuneHandle`` that records the base's calls."""

    def __init__(self, calls, *, inputs=None, hooks=None):
        self.calls = calls
        self.inputs = inputs
        self._hooks = hooks or _hooks()
        self.effects: list[PlannerEffects] = []
        self.host = None
        self.on_effects_error: Exception | None = None
        # The host's engine when stop ran (it must still be up).
        self.engine_at_stop = None

    def retune_inputs(self):
        self.calls.append("live.retune_inputs")
        return self.inputs

    def hooks(self):
        return self._hooks

    async def start(self, host):
        self.calls.append("live.start")
        self.host = host

    async def stop(self):
        self.calls.append("live.stop")
        if self.host is not None:
            self.engine_at_stop = self.host._engine

    def on_effects(self, effects):
        self.effects.append(effects)
        if self.on_effects_error is not None:
            raise self.on_effects_error


def _live_config(executor: str = "dgdr") -> PlannerConfig:
    config = PlannerConfig(
        namespace="test-ns",
        mode="agg",
        enable_load_scaling=False,
        enable_throughput_scaling=False,
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        control_api_port=0,
        max_gpu_budget=8,
    )
    retune = {
        "interval_seconds": 21600,
        "grace_seconds": 120,
        "timeout_seconds": 20400,
        "router": {"overlap_score_credit": {"choices": [0.0, 1.0]}},
        "executor": executor,
    }
    if executor == "dgdr":
        retune["dgdr"] = {
            "claim_name": "request-traces",
            "model": "Qwen/Qwen3-0.6B",
            "sku": "h200_sxm",
            "engine": {"mode": "aggregated", "backend": "vllm"},
        }
    # Assigned after construction: the base wiring is under test here, not
    # the live retune config validation of PlannerConfig.
    config.retune = RetuneConfig.model_validate(retune)
    return config


def _base(config: PlannerConfig, environment: _Environment) -> NativePlannerBase:
    with patch("dynamo.planner.core.base.PlannerPrometheusMetrics"):
        return NativePlannerBase(None, config, environment)


def _planner(executor="dgdr", **live_kwargs):
    environment = _Environment()
    planner = _base(_live_config(executor), environment)
    live = FakeLiveRetune(environment.calls, **live_kwargs)

    async def build():
        environment.calls.append("live.build")
        return live

    planner._build_live_retune = build  # type: ignore[method-assign]
    return planner, environment, live


def test_live_planner_rejects_the_local_executor():
    with pytest.raises(
        ValueError, match="replay-only with executor 'local'; use executor: dgdr"
    ):
        _base(_live_config("local"), _Environment())


async def test_async_init_builds_the_engine_with_live_retune_and_starts_it(
    stub_controller,
):
    planner, environment, live = _planner(inputs=retune_inputs())

    async def bootstrap_regression():
        environment.calls.append("bootstrap_regression")
        planner._ensure_engine()

    planner._bootstrap_regression = bootstrap_regression  # type: ignore[method-assign]
    await planner._async_init()
    try:
        assert environment.calls[:5] == [
            "environment.initialize",
            "live.build",
            "bootstrap_regression",
            "live.retune_inputs",
            "live.start",
        ]
        assert live.host is planner
        engine = planner._engine
        assert isinstance(engine, OrchestratorEngineAdapter)
        (controller,) = stub_controller.instances
        assert engine.retune_controller is controller
        assert controller.hooks["launch_gate"] is live.hooks().launch_gate
    finally:
        await planner._shutdown_runtime()

    # The supervisor stops while the engine is up (it waits for the engine's
    # sweep thread), then the engine closes the controller.
    assert live.engine_at_stop is engine
    assert controller.closed == 1
    assert environment.calls[-2:] == ["live.stop", "environment.shutdown"]
    assert planner._live_retune is None


async def test_failed_init_still_stops_live_retune(stub_controller):
    planner, environment, live = _planner(inputs=retune_inputs())

    async def failing_start(_host):
        environment.calls.append("live.start")
        raise RuntimeError("lease API unavailable")

    live.start = failing_start  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="lease API unavailable"):
        await planner._async_init()

    assert environment.calls[-2:] == ["live.stop", "environment.shutdown"]


class _FakeEngine:
    def __init__(self, effects):
        self.effects = effects

    async def observe(self, tick, now_s):
        return TickInput(now_s=now_s)

    async def tick(self, tick, tick_input):
        return self.effects


async def test_each_tick_hands_its_effects_to_live_retune(caplog):
    planner, _, live = _planner()
    planner._live_retune = live
    effects = PlannerEffects(
        next_tick=ScheduledTick(at_s=10.0),
        router_parameters=ROUTER_MAP,
        retune_decisions=[_decision()],
    )
    applied = []

    async def apply_effects(applied_effects):
        applied.append(applied_effects)

    planner._apply_effects = apply_effects  # type: ignore[method-assign]
    engine = _FakeEngine(effects)

    next_tick = await planner._run_one_tick(engine, ScheduledTick(at_s=5.0))
    assert next_tick.at_s == 10.0
    assert live.effects == [effects]
    assert applied == [effects]

    # A failing outbox never fails the tick or blocks the effects.
    live.on_effects_error = RuntimeError("outbox full")
    with caplog.at_level(logging.ERROR):
        await planner._run_one_tick(engine, ScheduledTick(at_s=5.0))
    assert "Live retune failed to take the tick's effects" in caplog.text
    assert len(applied) == 2


@pytest.mark.timeout(10)
async def test_replace_live_retune_holds_the_config_lock(stub_controller):
    planner, _, _ = _planner(inputs=None)
    await planner._async_init()
    try:
        await planner._config_lock.acquire()
        task = asyncio.create_task(
            planner.replace_live_retune(retune_inputs(), generation=2)
        )
        for _ in range(5):  # let the task run until it waits for the lock
            await asyncio.sleep(0)
        assert not task.done() and stub_controller.instances == []
        planner._config_lock.release()
        await asyncio.wait_for(task, timeout=5)

        (controller,) = stub_controller.instances
        assert planner._engine.retune_controller is controller  # type: ignore[union-attr]
        assert controller.generation == 2
        assert controller.phases == [PHASE]
    finally:
        await planner._shutdown_runtime()


# ---------------------------------------------------------------------------
# Runtime reconfiguration
# ---------------------------------------------------------------------------


def _sla_planner(**retune) -> NativePlannerBase:
    """A planner whose validated dgdr config scores with SLA 500/50 ms by
    default, with its live retune supervisor installed."""

    config = live_config(
        ttft_ms=500.0,
        itl_ms=50.0,
        retune=retune,
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        control_api_port=0,
    )
    planner = _base(config, _Environment())
    planner._live_retune = FakeLiveRetune([])
    return planner


async def _patch(planner: NativePlannerBase, changes: dict) -> dict:
    """PATCH ``changes`` through the runtime API; an accepted update is
    awaited and answered with its operation."""

    reconfiguration = planner.reconfiguration
    request_id = f"update-{planner._config_generation}"
    body = ReconfigureRequest(
        instance_id=reconfiguration.instance_id,
        expected_revision=planner._config_generation,
        request_id=request_id,
        config=changes,
    ).model_dump()
    response = await reconfiguration.handle_request(
        {"method": "PATCH", "path": "config", "body": body}
    )
    if response["status"] != 202:
        return response
    await reconfiguration._task
    return {"status": 202, "body": reconfiguration.get_operation(request_id)}


@pytest.mark.parametrize(
    ("retune", "changes", "inherited"),
    [
        ({}, {"ttft_ms": 250.0}, "ttft_ms"),
        ({}, {"itl_ms": 25.0}, "itl_ms"),
        ({}, {"ttft_ms": 250.0, "itl_ms": 25.0}, "ttft_ms and itl_ms"),
        ({}, {"ttft_ms": 250.0, "max_gpu_budget": 16}, "ttft_ms"),
        ({"sla_ttft_ms": 400.0}, {"itl_ms": 25.0}, "itl_ms"),
    ],
)
async def test_live_retune_rejects_a_runtime_change_of_its_inherited_sla(
    retune, changes, inherited
):
    planner = _sla_planner(**retune)
    # The update passes every other runtime check: only live retune stops it.
    candidate = candidate_config(planner.config, changes)
    assert {field: getattr(candidate, field) for field in changes} == changes
    before = await planner.reconfiguration.get_config()

    response = await _patch(planner, changes)

    assert response["status"] == 422
    overrides = " and ".join(f"retune.sla_{f}" for f in inherited.split(" and "))
    assert response["body"]["error"] == (
        f"cannot change {inherited} at runtime while retune.executor is 'dgdr': "
        f"live retune inherits its scoring SLA from {inherited} ({overrides} unset); "
        f"restart the planner with the new {inherited}, or restart it with "
        f"{overrides} set to keep {inherited} changeable at runtime"
    )
    assert await planner.reconfiguration.get_config() == before
    assert (planner.config.ttft_ms, planner.config.itl_ms) == (500.0, 50.0)
    await planner._shutdown_runtime()


@pytest.mark.parametrize(
    ("retune", "changes"),
    [
        (
            {"sla_ttft_ms": 400.0, "sla_itl_ms": 40.0},
            {"ttft_ms": 250.0, "itl_ms": 25.0},
        ),
        ({"sla_ttft_ms": 400.0}, {"ttft_ms": 250.0}),
        ({}, {"ttft_ms": 500.0, "max_gpu_budget": 16}),
    ],
)
async def test_live_retune_accepts_runtime_updates_that_keep_its_scoring_sla(
    retune, changes
):
    planner = _sla_planner(**retune)
    config = planner.config
    assert config.retune is not None
    scoring_sla = config.retune.scoring_sla(config.ttft_ms, config.itl_ms)

    response = await _patch(planner, changes)

    assert response["body"]["state"] == "applied"
    assert {field: getattr(config, field) for field in changes} == changes
    assert config.retune.scoring_sla(config.ttft_ms, config.itl_ms) == scoring_sla
    await planner._shutdown_runtime()


async def test_planner_without_live_retune_changes_its_sla_at_runtime():
    config = PlannerConfig(
        namespace="test-ns",
        mode="agg",
        metric_reporting_prometheus_port=0,
        live_dashboard_port=0,
        control_api_port=0,
    )
    planner = _base(config, _Environment())

    response = await _patch(planner, {"ttft_ms": 250.0, "itl_ms": 25.0})

    assert response["body"]["state"] == "applied"
    assert (planner.config.ttft_ms, planner.config.itl_ms) == (250.0, 25.0)
    await planner._shutdown_runtime()


# ---------------------------------------------------------------------------
# SIGTERM
# ---------------------------------------------------------------------------


class _Runtime:
    def __init__(self) -> None:
        self.stopped = asyncio.Event()
        self.shutdowns = 0

    def shutdown(self) -> None:
        self.shutdowns += 1
        self.stopped.set()


def _found() -> asyncio.Event:
    found = asyncio.Event()
    found.set()
    return found


def _sigterm_installed() -> bool:
    loop = asyncio.get_running_loop()
    return signal.SIGTERM in loop._signal_handlers  # type: ignore[attr-defined]


async def test_sigterm_cancels_the_planner_run_and_stops_serving():
    runtime = _Runtime()
    events: list[str] = []
    running = asyncio.Event()

    async def serve():
        await runtime.stopped.wait()
        events.append("serve stopped")

    async def run():
        running.set()
        try:
            await asyncio.sleep(3600)
        finally:
            # run()'s finally: the planner shutdown that releases the Lease.
            await asyncio.sleep(0)
            events.append("run finally")

    main = asyncio.create_task(
        planner_main._run_until_sigterm(runtime, serve(), run(), _found())
    )
    await asyncio.wait_for(running.wait(), timeout=5)
    assert _sigterm_installed()
    os.kill(os.getpid(), signal.SIGTERM)
    os.kill(os.getpid(), signal.SIGTERM)  # a second one changes nothing
    await asyncio.wait_for(main, timeout=5)

    assert events == ["run finally", "serve stopped"]
    assert runtime.shutdowns == 1
    assert not _sigterm_installed()


async def test_sigterm_is_handled_only_once_a_retune_lease_is_found():
    # The clean-up check of a removed live retune runs in the background;
    # a planner whose check finds no Lease keeps the default SIGTERM.
    runtime = _Runtime()
    found = asyncio.Event()
    running = asyncio.Event()

    async def serve():
        await runtime.stopped.wait()

    async def run():
        running.set()
        await asyncio.sleep(3600)

    main = asyncio.create_task(
        planner_main._run_until_sigterm(runtime, serve(), run(), found)
    )
    await asyncio.wait_for(running.wait(), timeout=5)
    assert not _sigterm_installed()
    found.set()
    for _ in range(3):
        await asyncio.sleep(0)
    assert _sigterm_installed()
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(main, timeout=5)
    assert runtime.shutdowns == 1 and not _sigterm_installed()


async def test_run_failure_and_outer_cancellation_propagate():
    runtime = _Runtime()

    async def serve():
        await runtime.stopped.wait()

    async def failing():
        raise RuntimeError("tick failed")

    with pytest.raises(RuntimeError, match="tick failed"):
        await planner_main._run_until_sigterm(runtime, serve(), failing(), _found())
    assert runtime.shutdowns == 0 and not _sigterm_installed()

    started = asyncio.Event()

    async def run():
        started.set()
        await asyncio.sleep(3600)

    # Never found: the waiting installer is cancelled with the call.
    main = asyncio.create_task(
        planner_main._run_until_sigterm(runtime, serve(), run(), asyncio.Event())
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    main.cancel()
    with pytest.raises(asyncio.CancelledError):
        await main
    assert runtime.shutdowns == 0
    runtime.shutdown()  # let the serve tasks finish


async def test_start_planner_runs_the_sigterm_path_only_when_a_lease_may_be_held(
    monkeypatch,
):
    # With live retune, or the clean-up check of a removed one (every
    # Kubernetes planner without retune config), the planner may hold the
    # retune leadership Lease.
    seen: list[asyncio.Event] = []

    class _Planner:
        def __init__(self, may_hold: bool) -> None:
            self.may_hold_retune_lease = may_hold
            self.retune_lease_found = asyncio.Event()

        async def _async_init(self):
            pass

        async def run(self):
            return None

    class _Endpoint:
        async def serve_endpoint(self, *_args, **_kwargs):
            return None

    class _StartRuntime:
        def endpoint(self, _name):
            return _Endpoint()

    async def fake_run_until_sigterm(runtime, serve, run, lease_found):
        seen.append(lease_found)
        await asyncio.gather(serve, run)

    planners = iter([_Planner(False), _Planner(True)])
    monkeypatch.setattr(planner_main, "construct_planner", lambda **_: next(planners))
    monkeypatch.setattr(planner_main, "_run_until_sigterm", fake_run_until_sigterm)

    await planner_main.start_planner(_StartRuntime(), PlannerConfig(mode="agg"))
    assert seen == []
    await planner_main.start_planner(_StartRuntime(), PlannerConfig(mode="agg"))
    assert len(seen) == 1 and not seen[0].is_set()


async def test_live_retune_finds_its_lease_at_start(stub_controller):
    planner, _, _ = _planner(inputs=None)
    assert not planner.may_hold_retune_lease
    await planner._async_init()
    try:
        assert planner.may_hold_retune_lease
        assert planner.retune_lease_found.is_set()
    finally:
        await planner._shutdown_runtime()
