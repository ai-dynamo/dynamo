# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the supervisor imports the sweep.

"""Unit tests for ``LiveRetuneSupervisor`` and its bootstrap, plus a seeded
interleaving harness that checks its invariants: at most one leader writes,
a decision is persisted before any router changes, no window is wasted, and
the routers settle on the persisted state.

The supervisor drives the real engine adapter and ``RetuneController``; their
sweeps run on the adapter's sweep thread, in ``StubSweeps``, until the test
finishes them. Kubernetes is ``FakeRetuneKube`` (shared Leases), the routers
are ``FakeFrontend`` (PATCH merges and bumps the revision, reset restores
the initial) behind ``FakeActuator``, and clocks are advanced by hand.
Kubernetes calls run inline.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import itertools
import json
import logging
import math
import random
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any, Optional

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.planner.config.planner_config import PlannerConfig, RetuneEngineSpec
from dynamo.planner.core.types import (
    EngineCapabilities,
    RetuneDecision,
    RetuneInputs,
    RetuneParams,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.clock import WallClock
from dynamo.planner.plugins.orchestrator.engine_adapter import (
    OrchestratorEngineAdapter,
    RetuneHooks,
)
from dynamo.planner.retune import controller as controller_module
from dynamo.planner.retune.controller import RetuneController
from dynamo.planner.retune.live import bootstrap
from dynamo.planner.retune.live.actuator import RouterActuator
from dynamo.planner.retune.live.contracts import (
    LABEL_DGD_UID,
    ActuationReport,
    ActuationTarget,
    DgdRef,
    FrontendStartup,
    KubeConflict,
    KubeError,
    NullMetrics,
    PolicyCallResult,
    PremiseReport,
    RouterInfo,
)
from dynamo.planner.retune.live.state import (
    STATE_ANNOTATION,
    RetuneState,
    StateStore,
    append_history,
    leadership_lease_name,
    state_lease_name,
)
from dynamo.planner.retune.live.supervisor import (
    LiveRetuneSupervisor,
    SpecEvaluation,
    SupervisorSettings,
    current_window_end_ms,
    due_window_end_ms,
    plan_initialization,
    window_index,
)
from dynamo.planner.retune.sweep import SweepResult
from dynamo.planner.tests.live_retune_fakes import FakeRetuneKube
from dynamo.planner.tests.unit.test_live_retune_state import live_config

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

INTERVAL = 600.0
GRACE = 30.0
TIMEOUT = 540.0
LEASE = 60.0
START_WALL = 1_800_000_100.0
DGD_NAME = "graph"
STATE_NAME = state_lease_name(DGD_NAME)
LEADER_NAME = leadership_lease_name(DGD_NAME)
CREDIT = "overlap_score_credit"
SCALE = "prefill_load_scale"
TEMP = "router_temperature"
KEYS = (CREDIT, SCALE, TEMP)
# The searched router domain of live_config(), by AISimulate knob name.
ROUTER = {
    "overlap_score_credit": {"choices": [0.5, 1.0]},
    "prefill_load_scale": 1.0,
    "temperature": {"choices": [0.0, 0.2]},
}
# The same domain by live knob name.
DOMAIN = {CREDIT: [0.5, 1.0], SCALE: [1.0], TEMP: [0.0, 0.2]}
# Two frontend startups: the KvRouterConfig JSON the frontend parses to.
ROUTER_CONFIG = {"a": {CREDIT: 1.0}, "b": {CREDIT: 0.5}}
SPEC_COUNTS = WorkerCounts(ready_num_decode=2)
# The due window at START_WALL.
DUE0 = math.floor((START_WALL - GRACE) / INTERVAL) - 1
# The window in progress at START_WALL (the one after DUE0). First enable
# records it as decided: neither it (its trace may have begun mid-way) nor
# the due window is swept.
K0 = math.floor(START_WALL / INTERVAL)
assert K0 == DUE0 + 1


def resolver(variant: str) -> bootstrap.KvRouterResolver:
    return bootstrap.KvRouterResolver(ROUTER_CONFIG[variant], ("aggregated",))


def by_wt(variant: str, flat: dict[str, float]) -> dict[str, dict[str, float]]:
    return resolver(variant).by_wt(flat)


def startup_flat(variant: str) -> dict[str, float]:
    return resolver(variant).startup_flat(KEYS)


# policy_parameters("aggregated") of each startup: its routers' 8 parameters.
_DEFAULT_POLICY = {
    "decode_active_request_weight": 0.0,
    "disk_cache_hit_weight": 0.25,
    "host_cache_hit_weight": 0.75,
    "overlap_score_credit": 1.0,
    "overlap_score_credit_decay": 0.0,
    "prefill_load_scale": 1.0,
    "router_temperature": 0.0,
    "shared_cache_multiplier": 0.0,
}
VARIANTS = {
    variant: {"aggregated": {**_DEFAULT_POLICY, **config}}
    for variant, config in ROUTER_CONFIG.items()
}


def window(k: int) -> tuple[float, float, float]:
    start = k * INTERVAL
    return start, start + INTERVAL, start + INTERVAL + GRACE


def end_ms(k: int) -> int:
    return int(round(window(k)[1] * 1000))


# ------------------------------------------------------------------ fakes


class Clock:
    def __init__(self) -> None:
        self.wall = START_WALL
        self.mono = 10_000.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds


class AdapterClock(WallClock):
    """The engine adapter's clock: monotonic time is the test's."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def now(self) -> float:
        return self._clock.time()

    def monotonic(self) -> float:
        return self._clock.monotonic()


async def inline(call):
    return call()


async def no_sleep(_seconds: float) -> None:
    await asyncio.sleep(0)


class RecordingMetrics(NullMetrics):
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.state: dict[str, Any] = {}

    def decision(self, outcome: str) -> None:
        self.events.append(("decision", outcome))

    def failure(self, reason: str) -> None:
        self.events.append(("failure", reason))

    def skipped(self, reason: str) -> None:
        self.events.append(("skipped", reason))

    def supervisor_state(self, **kwargs: Any) -> None:
        self.state = kwargs

    def scores(self, **kwargs: Any) -> None:
        self.events.append(("scores", kwargs))

    def dropped_members(self, count: int) -> None:
        self.events.append(("dropped_members", count))

    def of(self, kind: str) -> list[Any]:
        return [value for event, value in self.events if event == kind]


@dataclasses.dataclass
class Router:
    router_id: str
    worker_type: str
    revision: int
    config: dict[str, float]
    initial: dict[str, float]
    manual: bool = False


class FakeFrontend:
    """Routers with the runtime.rs semantics: PATCH merges and bumps the
    revision, reset restores the initial and bumps the revision."""

    def __init__(self, maps: dict[str, dict[str, float]]) -> None:
        self._ids = itertools.count()
        self.down = False
        # (writer, advisory, kind, target mode, router_id)
        self.writes: list[tuple[str, bool, str, str, str]] = []
        self.manual_patches = 0
        self.routers: dict[str, Router] = {}
        self.restart(maps)

    def restart(self, maps: dict[str, dict[str, float]]) -> None:
        self.routers = {}
        for worker_type, params in maps.items():
            router_id = f"r{next(self._ids)}"
            self.routers[router_id] = Router(
                router_id, worker_type, 0, dict(params), dict(params)
            )

    def patch(self, router: Router, params, writer, advisory, mode) -> None:
        router.config.update(params)
        router.revision += 1
        router.manual = False
        self.writes.append((writer, advisory, "patch", mode, router.router_id))

    def reset(self, router: Router, writer, advisory, mode) -> None:
        router.config = dict(router.initial)
        router.revision += 1
        router.manual = False
        self.writes.append((writer, advisory, "reset", mode, router.router_id))

    def manual_patch(self, key: str, value: float) -> None:
        self.manual_patches += 1
        for router in self.routers.values():
            router.config[key] = value
            router.revision += 1
            router.manual = True


class FakeActuator:
    """The router actuator's rules over a ``FakeFrontend``: latch, apply,
    rollback, observe and parity. Like ``RouterActuator`` it PATCHes a router
    only once the persisted ``initials`` hold its initial."""

    def __init__(
        self, frontend: FakeFrontend, owner: str, advisory: bool, maps
    ) -> None:
        self.frontend = frontend
        self.owner = owner
        self.advisory = advisory
        self.startup_maps = {wt: dict(p) for wt, p in maps.items()}
        self.latched: dict[str, dict[str, float]] = {}
        self.targets: list[ActuationTarget] = []
        self.last_pods = ()

    def set_startup_maps(self, maps) -> None:
        self.startup_maps = {wt: dict(p) for wt, p in maps.items()}

    async def reconcile(self, target, *, initials, expected_frontends):
        self.targets.append(target)
        frontend = self.frontend
        if frontend.down:
            return ActuationReport(
                converged=False,
                seen_router_ids=frozenset(initials) | frozenset(self.latched),
                errors={"list_pods": 1},
            )
        routers = list(frontend.routers.values())

        def initial(router: Router) -> Optional[dict[str, float]]:
            return self.latched.get(router.router_id) or initials.get(router.router_id)

        for router in routers:
            if router.revision == 0 and initial(router) is None:
                self.latched[router.router_id] = dict(router.config)
        for router in routers:
            known = initial(router)
            if target.mode == "rollback":
                if known is None and not target.learn_unknown:
                    continue  # advisory: never written, never waited for
                if known is None or router.config != known:
                    frontend.reset(router, self.owner, self.advisory, target.mode)
                    self.latched[router.router_id] = dict(router.config)
            elif target.mode == "apply":
                if known is None:
                    frontend.reset(router, self.owner, self.advisory, target.mode)
                    self.latched[router.router_id] = dict(router.config)
                    known = router.config
                if initials.get(router.router_id) != known:
                    continue  # held until the state records the initial
                if known != self.startup_maps.get(router.worker_type):
                    continue
                desired = target.desired_by_wt[router.worker_type]
                diff = {k: v for k, v in desired.items() if router.config.get(k) != v}
                if diff:
                    frontend.patch(router, diff, self.owner, self.advisory, target.mode)

        def expected(router: Router):
            if target.mode == "apply":
                return target.desired_by_wt[router.worker_type]
            return initial(router)

        converged = bool(routers) and all(
            expected(router) is not None and router.config == expected(router)
            for router in routers
            if not (
                target.mode == "rollback"
                and not target.learn_unknown
                and initial(router) is None
            )
        )
        known = [router for router in routers if initial(router) is not None]
        mismatched = [
            router
            for router in known
            if initial(router) != self.startup_maps.get(router.worker_type)
        ]
        if mismatched:
            parity: Optional[bool] = False
        elif known and len(known) == len(routers):
            parity = True
        else:
            parity = None
        seen = frozenset(router.router_id for router in routers)
        self.latched = {rid: cfg for rid, cfg in self.latched.items() if rid in seen}
        new_initials = {
            rid: dict(cfg)
            for rid, cfg in self.latched.items()
            if initials.get(rid) != cfg
        }
        return ActuationReport(
            converged=converged,
            new_initials=new_initials,
            seen_router_ids=seen,
            parity_ok=parity,
        )


class StubSweeps:
    """The ``sweep_runner`` of every process: a sweep runs on its adapter's
    sweep thread until the test finishes it, or until its abort or cancel
    event is set. Results are made on the test's thread, so a seeded run
    is deterministic."""

    @dataclasses.dataclass
    class Running:
        cancel: threading.Event
        abort: Optional[threading.Event]
        done: threading.Event = dataclasses.field(default_factory=threading.Event)
        result: Optional[SweepResult] = None

        def stopping(self) -> bool:
            return self.cancel.is_set() or (
                self.abort is not None and self.abort.is_set()
            )

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._running: dict[int, StubSweeps.Running] = {}
        self._closed = False
        self.requests: list[Any] = []

    def __call__(self, request, *, cancel, abort, context):
        running = StubSweeps.Running(cancel, abort)
        with self._cond:
            self._running[id(request)] = running
            self.requests.append(request)
            self._cond.notify_all()
        try:
            while not running.done.wait(0.002):
                if self._closed or cancel.is_set():
                    return SweepResult(status="failed", failure_reason="shutdown")
                if abort is not None and abort.is_set():
                    return SweepResult(
                        status="failed", failure_reason="lost_leadership"
                    )
            assert running.result is not None
            return running.result
        finally:
            with self._cond:
                del self._running[id(request)]
                self._cond.notify_all()

    def finish(self, controller: RetuneController, result_for) -> bool:
        """Finish the controller's pending sweep with ``result_for(request)``
        and wait until it returned; ``False`` when none is pending."""

        pending = controller._pending
        if pending is None or pending.future.done():
            return False
        with self._cond:
            assert self._cond.wait_for(
                lambda: id(pending.request) in self._running or pending.future.done(),
                timeout=10,
            )
            running = self._running.get(id(pending.request))
            if running is None:
                return False
            running.result = result_for(pending.request)
            running.done.set()
        pending.future.result(timeout=10)
        return True

    def quiesce(self, controllers) -> None:
        """Wait until every stopped sweep returned and every pending sweep
        runs, so no sweep starts after the test's clock moved on."""

        def quiet() -> bool:
            if any(running.stopping() for running in self._running.values()):
                return False
            for controller in controllers:
                pending = controller._pending
                if (
                    pending is not None
                    and not pending.future.done()
                    and id(pending.request) not in self._running
                ):
                    return False
            return True

        with self._cond:
            assert self._cond.wait_for(quiet, timeout=10)

    def close(self) -> None:
        with self._cond:
            self._closed = True


def failed(request) -> SweepResult:
    return SweepResult(status="failed", error="injected", failure_reason="crn_mismatch")


def scored(best=None):
    """A sweep result whose ``best`` candidate (else none) beats the rest."""

    def result_for(request) -> SweepResult:
        return SweepResult(
            status="ok",
            num_requests=10,
            scores={
                candidate.key(): (2.0 if candidate.router == best else 1.0)
                for candidate in request.candidates
            },
            wall_s=1.0,
        )

    return result_for


class TracedController(RetuneController):
    """The real controller, remembering the windows the supervisor dropped."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dropped: list[int] = []

    def drop_pending(self) -> None:
        if self._pending is not None:
            self.dropped.append(self._pending.request.window_index)
        super().drop_pending()


@pytest.fixture(autouse=True)
def traced_controllers(monkeypatch):
    # The engine adapter looks the controller class up when it builds one.
    monkeypatch.setattr(controller_module, "RetuneController", TracedController)
    yield
    while WORLDS:
        WORLDS.pop().close()


def controller_config(*, advisory: bool = False) -> PlannerConfig:
    """The real controller's config, with this file's windows.

    Built with the replay executor: the live config's timeout must leave
    600 s before the next window, which 600 s windows cannot, and the
    controller never reads the executor.
    """

    return PlannerConfig(
        namespace="test-ns",
        mode="agg",
        advisory=advisory,
        enable_load_scaling=False,
        enable_throughput_scaling=False,
        retune={
            "interval_seconds": INTERVAL,
            "grace_seconds": GRACE,
            "timeout_seconds": TIMEOUT,
            "router": ROUTER,
        },
    )


def retune_inputs(variant: str) -> RetuneInputs:
    return RetuneInputs(
        engine=RetuneEngineSpec(
            router_mode="kv_router", router_config=dict(ROUTER_CONFIG[variant])
        ),
        trace_paths=("/trace/segments/graph",),
    )


def variant_of(inputs: RetuneInputs) -> str:
    (variant,) = [
        v for v, c in ROUTER_CONFIG.items() if c == inputs.engine.router_config
    ]
    return variant


def make_decision(
    k: int,
    outcome: str,
    previous: Optional[RetuneParams] = None,
    *,
    chosen: Optional[dict[str, float]] = None,
    failure_reason: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
) -> RetuneDecision:
    previous = previous or RetuneParams(router=startup_flat("a"))
    start_s, end_s, as_of_s = window(k)
    return RetuneDecision(
        decision_id=f"retune-w{end_ms(k)}",
        outcome=outcome,
        window_index=k,
        window_start_s=start_s,
        window_end_s=end_s,
        as_of_s=as_of_s,
        launched_at_s=None if outcome == "busy" else as_of_s,
        decided_at_s=as_of_s + 60.0,
        num_requests=10,
        num_candidates=4,
        previous=previous,
        chosen=previous if chosen is None else RetuneParams(router=chosen),
        baseline_score=1.0,
        best_score=1.5,
        sweep_wall_s=50.0,
        failure_reason=failure_reason,
        details=details or {},
    )


class FakeHost:
    """``NativePlannerBase`` as the supervisor sees it: the config lock and
    the real engine adapter, whose controllers share one sweep thread."""

    def __init__(
        self,
        world: World,
        hooks: RetuneHooks,
        inputs: Optional[RetuneInputs],
        *,
        advisory: bool,
    ) -> None:
        self.clock = world.clock
        self._config_lock = asyncio.Lock()
        self._engine = OrchestratorEngineAdapter(
            controller_config(advisory=advisory),
            WorkerCapabilities(
                decode=EngineCapabilities(
                    num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384
                )
            ),
            clock=AdapterClock(world.clock),
            retune_inputs=inputs,
            retune_hooks=hooks,
        )
        if self.controller is not None:
            # What the adapter's initial_tick does.
            self.controller.start(world.clock.time(), hooks.phase_s)
        # The frontend startup the current controller was built for.
        self.variant = "a" if inputs is None else variant_of(inputs)
        self.replacements: list[tuple[Any, int]] = []
        self.fail_replace = False

    @property
    def controller(self) -> Optional[TracedController]:
        return self._engine.retune_controller

    async def replace_live_retune(self, inputs, *, generation):
        async with self._config_lock:
            self.replacements.append((inputs, generation))
            now = self.clock.time()
            if self.fail_replace:
                self._engine.replace_retune(None, generation=generation, now_s=now)
                raise ValueError("controller construction failed")
            self._engine.replace_retune(inputs, generation=generation, now_s=now)
            if inputs is not None:
                self.variant = variant_of(inputs)

    def close(self) -> None:
        if self.controller is not None:
            self.controller.close()
        executor = self._engine._retune_executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


class Spec:
    """The DGD spec as the premise evaluator sees it (tests mutate it)."""

    def __init__(self) -> None:
        self.variant = "a"
        self.spec_ok = True
        self.p4_ok = True
        self.replicas: Optional[dict[str, int]] = {"aggregated": 2}
        self.uid = "uid-1"
        self.dynamo_namespace = "dyn"
        self.frontend_replicas = 1

    def startup(self) -> FrontendStartup:
        return FrontendStartup(
            router_mode="kv_router",
            router_config=dict(ROUTER_CONFIG[self.variant]),
            startup_maps=copy.deepcopy(VARIANTS[self.variant]),
            frontend_component="Frontend",
            replicas=self.frontend_replicas,
            replica_sync=False,
            fingerprint_source=self.variant,
        )

    def evaluation(self, _dgd: Optional[dict] = None) -> SpecEvaluation:
        reasons = [] if self.p4_ok else ["P4: router mode round-robin"]
        if not self.spec_ok:
            reasons.append("P8: two frontends without replica sync")
        return SpecEvaluation(
            report=PremiseReport(
                p4_ok=self.p4_ok,
                spec_ok=self.p4_ok and self.spec_ok,
                reasons=tuple(reasons),
            ),
            startup=self.startup(),
            inputs=retune_inputs(self.variant) if self.p4_ok else None,
            spec_replicas=None if self.replicas is None else dict(self.replicas),
            dgd=DgdRef(DGD_NAME, "test-ns", self.uid, self.dynamo_namespace),
        )


# Worlds of the running test, closed after it (sweep threads end).
WORLDS: list[World] = []


class World:
    """Shared kube, frontend, spec, clock and sweeps of one test."""

    def __init__(self, kube: Optional[FakeRetuneKube] = None) -> None:
        self.clock = Clock()
        self.kube = kube or FakeRetuneKube(now=self.clock.time)
        self.spec = Spec()
        self.frontend = FakeFrontend(VARIANTS["a"])
        self.sweeps = StubSweeps()
        self.hosts: list[FakeHost] = []
        self.kube.put_dgd(
            {"metadata": {"name": DGD_NAME, "uid": "uid-1"}, "spec": {"components": []}}
        )
        WORLDS.append(self)

    def shared_state(self) -> Optional[RetuneState]:
        lease = self.kube.leases.get(STATE_NAME)
        if lease is None:
            return None
        return RetuneState.from_annotations(lease["metadata"].get("annotations"))

    def holder(self) -> Optional[str]:
        lease = self.kube.leases.get(LEADER_NAME)
        return None if lease is None else lease["spec"].get("holderIdentity")

    def close(self) -> None:
        self.sweeps.close()
        for host in self.hosts:
            host.close()


@dataclasses.dataclass
class Process:
    name: str
    sup: LiveRetuneSupervisor
    host: FakeHost
    metrics: RecordingMetrics
    advisory: bool
    kube: Any
    alive: bool = True

    @property
    def controller(self) -> Optional[TracedController]:
        return self.host.controller


def settings_for(
    *, advisory: bool = False, promote: bool = False, **changes: Any
) -> SupervisorSettings:
    values: dict[str, Any] = dict(
        interval_s=INTERVAL,
        grace_s=GRACE,
        timeout_s=TIMEOUT,
        advisory=advisory,
        promote_last_advisory=promote,
        router_keys=KEYS,
        lease_duration_s=LEASE,
        actuator_interval_s=15.0,
        k8s_timeout_s=5.0,
        write_failure_limit=3,
    )
    values.update(changes)
    return SupervisorSettings(**values)


def make_process(
    world: World,
    name: str = "p1",
    *,
    advisory: bool = False,
    promote: bool = False,
    kube: Any = None,
    canary=None,
    reconciler=None,
    sweep_runner=None,
    context_factory=lambda supervisor: None,
    settings: Optional[SupervisorSettings] = None,
    blocking=None,
    actuator_factory=None,
) -> Process:
    kube = kube or world.kube
    metrics = RecordingMetrics()
    evaluation = world.spec.evaluation()
    settings = settings or settings_for(advisory=advisory, promote=promote)
    sup = LiveRetuneSupervisor(
        settings=settings,
        kube=kube,
        evaluation=evaluation,
        identity=f"{name}/{len(world.kube.calls)}",
        premise_evaluator=world.spec.evaluation,
        actuator_factory=actuator_factory
        or (lambda dgd, maps: FakeActuator(world.frontend, name, advisory, maps)),
        fingerprint_fn=lambda startup: f"fp-{None if startup is None else startup.fingerprint_source}",
        resolver_factory=lambda startup: bootstrap.KvRouterResolver(
            startup.router_config, ("aggregated",)
        ),
        reconciler=reconciler,
        metrics=metrics,
        canary=canary,
        sweep_runner=sweep_runner or world.sweeps,
        context_factory=context_factory,
        clock=world.clock.time,
        monotonic=world.clock.monotonic,
        sleep=no_sleep,
        blocking=blocking or inline,
    )
    host = FakeHost(world, sup.hooks(), evaluation.inputs, advisory=advisory)
    world.hosts.append(host)
    return Process(name, sup, host, metrics, advisory, kube)


async def started(world: World, name: str = "p1", **kwargs) -> Process:
    process = make_process(world, name, **kwargs)
    await process.sup.start(process.host, background=False)
    return process


async def leading(world: World, name: str = "p1", **kwargs) -> Process:
    """A started process that leads, is initialized and is ready."""

    process = await started(world, name, **kwargs)
    await process.sup.lease_round()
    assert process.sup.is_leader() and process.sup.initialized
    await process.sup.control_round()
    return process


async def settle(processes: list[Process], world: World, rounds: int = 6) -> None:
    for _ in range(rounds):
        world.clock.advance(20)
        for process in processes:
            if process.alive:
                await process.sup.lease_round()
                await process.sup.outbox_round()
                await process.sup.control_round()


def enqueue(process: Process, *decisions: RetuneDecision) -> None:
    """Hand the supervisor a tick's decisions with the controller's routers."""

    controller = process.controller
    desired = (
        controller.current_point().router
        if controller is not None
        else startup_flat("a")
    )
    process.sup.on_effects(
        SimpleNamespace(
            retune_decisions=list(decisions),
            router_parameters=by_wt(process.host.variant, desired),
        )
    )


def frontend_configs(world: World) -> list[dict[str, float]]:
    return [router.config for router in world.frontend.routers.values()]


# ------------------------------------------------------- pure functions


def claimed(**changes) -> RetuneState:
    base = RetuneState(dgd_uid="uid-1", leader_term="1/p/2")
    return dataclasses.replace(base, **changes)


def deployed(**changes) -> RetuneState:
    values = dict(
        fingerprint="fp",
        mode="active",
        deploy_seq=4,
        flat={CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2},
        by_wt=by_wt("a", {CREDIT: 0.5, TEMP: 0.2}),
        last_window_end_ms=777,
    )
    values.update(changes)
    return claimed(**values)


def plan(state: RetuneState, **overrides):
    kwargs = dict(
        spec_ok=True,
        advisory=False,
        fingerprint="fp",
        startup_flat=startup_flat("a"),
        startup_by_wt=VARIANTS["a"],
        router_keys=KEYS,
        promote=False,
        now_ms=10_000_000,
        interval_ms=int(INTERVAL * 1000),
        first_window_end_ms=9_999,
    )
    kwargs.update(overrides)
    return plan_initialization(state, **kwargs)


def test_init_table_first_enable_persists_the_due_window() -> None:
    result = plan(claimed())
    assert result.row == "active_reset" and result.write
    assert result.state.mode == "active" and not result.state.rollback_pending
    assert result.state.by_wt == VARIANTS["a"] and result.state.flat == startup_flat(
        "a"
    )
    assert result.state.last_window_end_ms == 9_999
    assert result.state.deploy_seq == 1
    assert result.restore_flat == startup_flat("a")


def test_init_table_spec_not_ok_is_the_first_row() -> None:
    # Deployed X: roll back (startup, pending), controller untouched.
    result = plan(deployed(), spec_ok=False)
    assert result.row == "spec_not_ok" and result.restore_flat is None
    assert result.state.rollback_pending and result.state.by_wt == VARIANTS["a"]
    assert result.state.deploy_seq == 5
    # Nothing deployed and the record current: nothing to write.
    clean = deployed(by_wt=VARIANTS["a"], flat=startup_flat("a"))
    assert not plan(clean, spec_ok=False).write
    # Even an advisory planner takes this row (with its own mode).
    advisory = plan(deployed(), spec_ok=False, advisory=True)
    assert advisory.row == "spec_not_ok" and advisory.state.mode == "advisory"
    assert advisory.state.rollback_pending
    # Unknown startup and an active record: roll back to the latched initials.
    unknown = plan(
        deployed(by_wt=VARIANTS["a"]),
        spec_ok=False,
        startup_flat=None,
        startup_by_wt=None,
    )
    assert unknown.state.rollback_pending and unknown.state.by_wt == {}


def test_init_table_advisory_rows() -> None:
    rollback = plan(deployed(), advisory=True)
    assert rollback.row == "advisory_rollback" and rollback.restore_flat is None
    assert rollback.state.mode == "advisory" and rollback.state.rollback_pending
    assert rollback.state.by_wt == VARIANTS["a"]

    observe = plan(deployed(by_wt=VARIANTS["a"], flat=startup_flat("a")), advisory=True)
    assert observe.row == "advisory_observe" and observe.write
    assert observe.state.mode == "advisory" and not observe.state.rollback_pending
    # Already advisory and current: no write.
    again = plan(observe.state, advisory=True)
    assert again.row == "advisory_observe" and not again.write
    # An advisory record of an older startup is not a deployment.
    older = plan(observe.state, advisory=True, startup_by_wt=VARIANTS["b"])
    assert older.row == "advisory_observe" and not older.state.rollback_pending


def test_init_table_active_rows() -> None:
    restore = plan(deployed())
    assert restore.row == "active_restore" and not restore.write
    assert restore.restore_flat == {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}

    for state, pending in (
        (deployed(fingerprint="old"), True),
        (deployed(mode="advisory"), False),
        (deployed(rollback_pending=True), True),
        (deployed(flat={CREDIT: 0.5}), True),
        (deployed(fingerprint="old", by_wt=VARIANTS["a"]), False),
    ):
        result = plan(state)
        assert result.row == "active_reset", state
        assert result.state.rollback_pending is pending, state
        assert result.state.by_wt == VARIANTS["a"] and result.state.mode == "active"
        assert result.restore_flat == startup_flat("a")
        assert result.state.last_window_end_ms == 777


def test_init_table_promotes_a_recent_matching_advisory() -> None:
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    entry = {
        "decision_id": "retune-w9000000",
        "outcome": "advisory",
        "window_end_ms": 9_900_000,
        "fingerprint": "fp",
        "flat": chosen,
        "by_wt": by_wt("a", chosen),
    }
    state = append_history(
        deployed(mode="advisory", by_wt=VARIANTS["a"], flat=startup_flat("a")), entry
    )
    result = plan(state, promote=True)
    assert result.row == "promote" and result.restore_flat == chosen
    assert result.state.mode == "active" and result.state.by_wt == by_wt("a", chosen)
    assert not result.state.rollback_pending and result.state.deploy_seq == 5
    assert result.state.history[-1]["outcome"] == "promoted"
    # Not promoted: disabled, stale fingerprint, too old, or not the latest.
    assert plan(state).row == "active_reset"
    assert plan(state, promote=True, fingerprint="new").row == "active_reset"
    assert plan(state, promote=True, now_ms=9_900_000 + 600_001).row == "active_reset"
    later = append_history(state, {"decision_id": "x", "outcome": "kept"})
    assert plan(later, promote=True).row == "active_reset"


def test_window_arithmetic_matches_the_controller() -> None:
    now = START_WALL
    end = due_window_end_ms(now, phase_s=0.0, interval_s=INTERVAL, grace_s=GRACE)
    assert end == end_ms(DUE0)
    assert window_index(end, phase_s=0.0, interval_s=INTERVAL) == DUE0
    # Exactly at L_k the window is due.
    at_launch = window(DUE0 + 1)[2]
    assert due_window_end_ms(
        at_launch, phase_s=0.0, interval_s=INTERVAL, grace_s=GRACE
    ) == end_ms(DUE0 + 1)
    phased = due_window_end_ms(1000.0, phase_s=100.0, interval_s=300.0, grace_s=0.0)
    assert phased == 1_000_000
    assert window_index(phased, phase_s=100.0, interval_s=300.0) == 2


def test_current_window_is_after_the_due_and_the_not_yet_due_windows() -> None:
    def current(now_s: float, phase_s: float = 0.0) -> int:
        return current_window_end_ms(now_s, phase_s=phase_s, interval_s=INTERVAL)

    # Mid-window: the window in progress, one after the due window.
    assert current(START_WALL) == end_ms(K0) == end_ms(DUE0 + 1)
    # Within the grace after a window ended: that window is not due yet and
    # the one in progress is two after the due window; both are skipped.
    start = window(K0)[0]
    within_grace = start + GRACE / 2
    due = due_window_end_ms(
        within_grace, phase_s=0.0, interval_s=INTERVAL, grace_s=GRACE
    )
    assert due == end_ms(K0 - 2) and current(within_grace) == end_ms(K0)
    # A window boundary belongs to the window starting there.
    assert current(start) == end_ms(K0)
    # Phased: windows [100 + k T, 100 + (k + 1) T); 1000 is in [700, 1300).
    assert current(1000.0, phase_s=100.0) == 1_300_000


# ----------------------------------------------------------- gate/harvest


async def test_gate_reasons_and_the_skip_metric() -> None:
    world = World()
    process = await started(world)
    sup = process.sup
    assert sup.launch_gate(K0 + 1, window(K0 + 1)).reason == "not_leader"
    assert sup.launch_gate(K0 + 1, window(K0 + 1)).reason == "not_leader"
    await sup.lease_round()
    # Initialized but no health yet (no actuator round).
    assert sup.initialized
    assert sup.launch_gate(K0 + 1, window(K0 + 1)).reason == "not_ready"
    await sup.control_round()
    assert sup.ready()
    # First enable recorded the window in progress at start as decided: it
    # and the due window before it are skipped.
    assert world.shared_state().last_window_end_ms == end_ms(K0)
    assert sup.launch_gate(DUE0, window(DUE0)).reason == "already_decided"
    assert sup.launch_gate(K0, window(K0)).reason == "already_decided"
    gate = sup.launch_gate(K0 + 1, window(K0 + 1))
    assert gate.reason is None and gate.spec_replicas == {"aggregated": 2}
    assert isinstance(gate.abort, threading.Event) and not gate.abort.is_set()
    # Each (k, reason) counts once.
    assert process.metrics.of("skipped") == [
        "not_leader",
        "not_ready",
        "already_decided",
        "already_decided",
    ]
    # Health and spec gate only launches.
    world.spec.replicas = None
    await sup.control_round()
    assert sup.launch_gate(K0 + 2, window(K0 + 2)).reason == "not_ready"


async def test_harvest_allowed_needs_leader_initialized_and_generation() -> None:
    world = World()
    process = await leading(world)
    sup = process.sup
    assert sup.harvest_allowed(0)
    assert not sup.harvest_allowed(1)
    world.spec.spec_ok = False
    await sup.control_round()  # rebuild: generation 1
    assert sup.generation == 1
    assert not sup.harvest_allowed(0) and sup.harvest_allowed(1)
    world.clock.advance(LEASE)  # leadership lapses without renewal
    assert not sup.harvest_allowed(1)


# ---------------------------------------------------------------- outbox


async def test_applied_decision_is_persisted_then_deployed() -> None:
    world = World()
    process = await leading(world)
    sup, controller = process.sup, process.controller
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    controller.restore_desired(chosen)
    enqueue(process, make_decision(K0 + 1, "applied", chosen=chosen))
    assert sup.outbox_size == 1
    assert process.metrics.of("decision") == []  # counted only once persisted
    await sup.outbox_round()
    state = world.shared_state()
    assert sup.outbox_size == 0
    assert state.flat == chosen and state.by_wt == by_wt("a", chosen)
    assert state.mode == "active" and state.deploy_seq == 2
    assert state.last_window_end_ms == end_ms(K0 + 1)
    entry = state.history[-1]
    assert entry["decision_id"] == f"retune-w{end_ms(K0 + 1)}"
    assert entry["outcome"] == "applied" and entry["by_wt"] == by_wt("a", chosen)
    assert entry["fingerprint"] == "fp-a" and entry["baseline"] == 1.0
    assert process.metrics.of("decision") == ["applied"]
    assert sup.target() == ActuationTarget(
        mode="apply", desired_by_wt=by_wt("a", chosen)
    )
    await sup.control_round()
    assert frontend_configs(world) == [by_wt("a", chosen)["aggregated"]]
    # The next window launches; the decided one does not.
    assert sup.launch_gate(K0 + 1, window(K0 + 1)).reason == "already_decided"


async def test_advisory_decision_records_the_recommendation_only() -> None:
    world = World()
    process = await leading(world, advisory=True)
    sup = process.sup
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    enqueue(
        process,
        make_decision(
            K0 + 1,
            "advisory",
            chosen=chosen,
            details={"runtime_image": "img:1", "missing_candidates": 2},
        ),
    )
    await sup.outbox_round()
    state = world.shared_state()
    assert state.mode == "advisory" and state.by_wt == VARIANTS["a"]
    entry = state.history[-1]
    assert entry["outcome"] == "advisory" and entry["flat"] == chosen
    assert entry["by_wt"] == by_wt("a", chosen)
    assert entry["image"] == {"runtime_image": "img:1"}
    scores = process.metrics.of("scores")[-1]
    assert scores["missing_candidates"] == 2 and scores["best"] == 1.5
    assert sup.target() == ActuationTarget(mode="observe")
    world.frontend.manual_patch(TEMP, 0.7)
    await settle([process], world)
    assert world.frontend.writes == []


async def test_failed_decision_counts_its_reason() -> None:
    world = World()
    process = await leading(world)
    enqueue(
        process,
        make_decision(
            K0 + 1,
            "failed",
            failure_reason="crn_mismatch",
            details={"window": {"dropped_members": 3, "cut": True}},
        ),
    )
    await process.sup.outbox_round()
    assert process.metrics.of("failure") == ["crn_mismatch"]
    assert process.metrics.of("dropped_members") == [3]
    assert world.shared_state().history[-1]["failure_reason"] == "crn_mismatch"


async def test_outbox_timeout_that_landed_and_conflict() -> None:
    world = World()
    process = await leading(world)
    sup = process.sup
    # A write that timed out but landed: the re-read finds it.
    world.kube.fail(
        "replace_lease",
        KubeError("timeout"),
        landed=True,
        match=lambda target: target[0] == STATE_NAME,
    )
    enqueue(process, make_decision(K0 + 1, "kept"))
    await sup.outbox_round()
    ids = [item["decision_id"] for item in world.shared_state().history]
    assert ids.count(f"retune-w{end_ms(K0 + 1)}") == 1
    assert sup.outbox_size == 0 and process.metrics.of("decision") == ["kept"]
    # A stale cached resourceVersion: conflict, re-read, retry.
    world.kube.bump(world.kube.leases[STATE_NAME])
    enqueue(process, make_decision(K0 + 2, "kept"))
    await sup.outbox_round()
    ids = [item["decision_id"] for item in world.shared_state().history]
    assert ids[-2:] == [f"retune-w{end_ms(K0 + 1)}", f"retune-w{end_ms(K0 + 2)}"]
    assert sup.outbox_size == 0 and process.metrics.of("decision") == ["kept", "kept"]


async def test_consecutive_write_failures_reinitialize_and_rewind() -> None:
    world = World()
    process = await leading(world)
    sup, controller = process.sup, process.controller
    term_seq = sup.term_seq
    # Window K0 + 2 launches (K0 + 1 is skipped: K0 + 2 is due already).
    world.clock.wall = window(K0 + 2)[2]
    await controller.on_tick(world.clock.time(), SPEC_COUNTS)
    assert controller._pending.request.window_index == K0 + 2
    world.kube.fail(
        "replace_lease",
        KubeError("timeout"),
        times=3,
        match=lambda target: target[0] == STATE_NAME,
    )
    enqueue(process, make_decision(K0 + 1, "kept"))
    for _ in range(3):
        await sup.outbox_round()
    assert not sup.initialized  # re-init requested
    assert process.metrics.of("failure") == ["state_write"] * 3
    await sup.lease_round()
    assert sup.initialized and sup.term_seq == term_seq + 1
    assert sup.outbox_size == 0
    # The sweep was dropped and the controller rewound to the last decided
    # window, so both windows can still run.
    assert controller._pending is None and controller.dropped == [K0 + 2]
    assert controller._last_k == K0
    assert world.shared_state().leader_term == sup.term


async def test_every_failed_state_write_counts_as_state_write() -> None:
    world = World()
    process = await started(world)
    sup = process.sup

    def state_lease(target) -> bool:
        return target[0] == STATE_NAME

    def failures() -> list[str]:
        return process.metrics.of("failure")

    # First enable: the claim creates the state, the plan write fails.
    world.kube.fail("replace_lease", KubeError("quota"), match=state_lease)
    await sup.lease_round()
    assert not sup.initialized and failures() == ["state_write"]
    await sup.lease_round()
    assert sup.initialized
    # The claim write of a re-initialization.
    world.kube.fail("replace_lease", KubeError("forbidden"), match=state_lease)
    sup._request_reinit("test")
    await sup.lease_round()
    assert not sup.initialized and failures() == ["state_write"] * 2
    await sup.lease_round()
    assert sup.initialized
    # The initials of a restarted frontend (retried next round).
    world.frontend.restart(VARIANTS["a"])
    world.kube.fail("replace_lease", KubeError("too large"), match=state_lease)
    await sup.control_round()
    assert failures() == ["state_write"] * 3
    await sup.control_round()
    assert set(world.shared_state().initials) == set(world.frontend.routers)
    # A lost term is not a failed write.
    store = StateStore(world.kube, sup.dgd)
    snapshot = store.read()
    store.write(dataclasses.replace(snapshot.state, leader_term="9/x/1"), snapshot)
    enqueue(process, make_decision(K0 + 1, "kept"))
    await sup.outbox_round()
    assert not sup.initialized and failures() == ["state_write"] * 3


async def test_decisions_are_dropped_and_rewound_without_leadership() -> None:
    world = World()
    process = await started(world)
    controller = process.controller
    controller._last_k = K0 + 1
    enqueue(process, make_decision(K0 + 1, "kept"))
    assert process.sup.outbox_size == 0
    assert controller._last_k == -1  # no state: every window is due again


async def test_a_foreign_term_forces_reinitialization() -> None:
    world = World()
    process = await leading(world)
    sup = process.sup
    store = StateStore(world.kube, sup.dgd)
    snapshot = store.read()
    store.write(dataclasses.replace(snapshot.state, leader_term="9/x/1"), snapshot)
    enqueue(process, make_decision(K0 + 1, "kept"))
    await sup.outbox_round()
    assert not sup.initialized and sup.outbox_size == 1
    await sup.lease_round()
    assert sup.initialized and world.shared_state().leader_term == sup.term
    assert sup.outbox_size == 0


async def test_a_late_write_of_an_earlier_term_fails() -> None:
    world = World()
    process = await leading(world)
    sup = process.sup
    store = StateStore(world.kube, sup.dgd)
    stale = store.read()
    # Same-process re-initialization (term_seq + 1) moves the term on.
    sup._request_reinit("test")
    await sup.lease_round()
    assert world.shared_state().leader_term == sup.term != stale.state.leader_term
    with pytest.raises(KubeConflict):
        store.write(append_history(stale.state, {"decision_id": "late"}), stale)


def steal_lease(world: World, identity: str) -> None:
    """The leadership Lease changes hands without expiring (an admin edit,
    or a contender whose clock runs fast)."""

    lease = world.kube.leases[LEADER_NAME]
    lease["spec"]["holderIdentity"] = identity
    lease["spec"]["leaseTransitions"] = lease["spec"].get("leaseTransitions", 0) + 1
    world.kube.bump(lease)


async def test_a_stolen_lease_rejects_the_old_leaders_writes() -> None:
    """Two processes believe they lead; the state term precondition keeps
    the old one from writing once the new one has claimed."""

    world = World()
    old = await leading(world)
    new = await started(world, "p2")
    steal_lease(world, new.sup.identity)
    await new.sup.lease_round()
    assert new.sup.initialized and old.sup.is_leader()  # both believe
    enqueue(old, make_decision(K0 + 1, "kept"))
    await old.sup.outbox_round()
    state = world.shared_state()
    assert state.leader_term == new.sup.term
    assert all(
        item["decision_id"] != f"retune-w{end_ms(K0 + 1)}" for item in state.history
    )
    assert not old.sup.initialized
    await old.sup.lease_round()  # sees the new holder: local loss
    assert not old.sup.is_leader() and old.sup.term is None


async def test_a_stolen_lease_aborts_the_old_leaders_claim() -> None:
    world = World()
    old = await leading(world)
    term = world.shared_state().leader_term
    steal_lease(world, "intruder/0")
    assert old.sup.is_leader()  # not seen yet
    world.spec.spec_ok = False
    await old.sup.control_round()  # a rebuild re-initializes: no claim
    assert world.shared_state().leader_term == term
    assert not old.sup.is_leader() and not old.sup.initialized


# ----------------------------------------------------- initialization


async def test_restart_restores_the_persisted_point() -> None:
    world = World()
    first = await leading(world)
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    first.controller.restore_desired(chosen)
    enqueue(first, make_decision(K0 + 1, "applied", chosen=chosen))
    await first.sup.outbox_round()
    await first.sup.stop()
    assert world.holder() is None  # released
    second = await leading(world, "p2")
    assert second.controller.current_point().router == chosen
    # The persisted window is decided: the gate skips it, the next launches.
    assert second.sup.launch_gate(K0 + 1, window(K0 + 1)).reason == "already_decided"
    assert second.sup.launch_gate(K0 + 2, window(K0 + 2)).reason is None
    state = world.shared_state()
    assert state.leader_term == second.sup.term and state.deploy_seq == 2
    assert second.sup.target().desired_by_wt == by_wt("a", chosen)


async def test_fingerprint_change_resets_with_a_two_phase_rollback() -> None:
    world = World()
    first = await leading(world)
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    first.controller.restore_desired(chosen)
    enqueue(first, make_decision(K0 + 1, "applied", chosen=chosen))
    await first.sup.outbox_round()
    await first.sup.control_round()
    assert frontend_configs(world) == [by_wt("a", chosen)["aggregated"]]
    # The frontend args change (new startup): restart under it.
    await first.sup.stop()
    world.spec.variant = "b"
    world.frontend.restart(VARIANTS["b"])
    world.frontend.manual_patch(TEMP, 0.9)
    second = await leading(world, "p2")
    state = world.shared_state()
    assert state.rollback_pending and state.by_wt == VARIANTS["b"]
    assert state.fingerprint == "fp-b" and state.deploy_seq == 3
    assert second.controller.current_point().router == startup_flat("b")
    assert second.sup.target().mode == "rollback"
    # Reset once, then converged: completes after a lease duration.
    resets = len(world.frontend.writes)
    await second.sup.control_round()
    assert len(world.frontend.writes) == resets
    assert world.shared_state().rollback_pending
    await settle([second], world, rounds=2)
    assert world.shared_state().rollback_pending  # 40s: not yet
    await settle([second], world, rounds=1)
    state = world.shared_state()
    assert not state.rollback_pending and state.deploy_seq == 3
    assert state.history[-1]["outcome"] == "rolled_back"
    assert second.sup.target().desired_by_wt == VARIANTS["b"]
    await settle([second], world, rounds=3)
    assert len(world.frontend.writes) == resets  # no further writes


async def test_applied_during_a_pending_rollback_wins() -> None:
    world = World()
    first = await leading(world)
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    first.controller.restore_desired(chosen)
    enqueue(first, make_decision(K0 + 1, "applied", chosen=chosen))
    await first.sup.outbox_round()
    await first.sup.stop()
    world.spec.variant = "b"
    world.frontend.restart(VARIANTS["b"])
    second = await leading(world, "p2")
    assert world.shared_state().rollback_pending
    seq = world.shared_state().deploy_seq
    # An applied decision lands while the rollback is pending.
    applied = {CREDIT: 1.0, SCALE: 1.0, TEMP: 0.2}
    second.controller.restore_desired(applied)
    enqueue(second, make_decision(K0 + 2, "applied", chosen=applied))
    await second.sup.outbox_round()
    state = world.shared_state()
    assert not state.rollback_pending and state.deploy_seq == seq + 1
    assert second.sup.target().mode == "apply"
    # A rollback-complete write racing it is discarded, never retried.
    stale_seq = seq
    from dynamo.planner.retune.live.state import rollback_complete

    assert rollback_complete(state, expected_deploy_seq=stale_seq) is None


@pytest.mark.parametrize("racing", ["deploy", "rollback_start"])
async def test_a_stale_rollback_complete_is_discarded(racing: str) -> None:
    """Right before the rollback-complete write of deploy_seq S, a write of
    the same term moves deploy_seq on (an applied, or a new rollback): the
    completion of S is discarded, never retried, never applied to S+1."""

    from dynamo.planner.retune.live.state import deploy, rollback_start

    world = World()
    first = await leading(world)
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    first.controller.restore_desired(chosen)
    enqueue(first, make_decision(K0 + 1, "applied", chosen=chosen))
    await first.sup.outbox_round()
    await first.sup.stop()
    world.spec.variant = "b"
    world.frontend.restart(VARIANTS["b"])
    second = await leading(world, "p2")
    sup = second.sup
    seq = world.shared_state().deploy_seq
    assert world.shared_state().rollback_pending
    store = sup._store
    original = store.write
    raced = []

    def racing_write(state, snapshot):
        if not raced:
            raced.append(1)
            current = store.read()
            rule = deploy if racing == "deploy" else rollback_start
            original(
                rule(
                    current.state,
                    fingerprint="fp-b",
                    mode="active",
                    flat=startup_flat("b"),
                    by_wt=VARIANTS["b"],
                ),
                current,
            )
        return original(state, snapshot)

    store.write = racing_write
    for _ in range(6):
        if raced:
            break
        await settle([second], world, rounds=1)
    assert raced
    state = world.shared_state()
    assert state.deploy_seq == seq + 1
    assert state.rollback_pending is (racing == "rollback_start")
    assert all(item.get("outcome") != "rolled_back" for item in state.history)
    # The new rollback completes on its own deploy_seq.
    if racing == "rollback_start":
        await settle([second], world, rounds=4)
        state = world.shared_state()
        assert not state.rollback_pending
        assert state.history[-1]["decision_id"] == f"rolled_back-{seq + 1}"


async def test_advisory_switch_rolls_back_only_what_was_deployed() -> None:
    world = World()
    first = await leading(world)
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    first.controller.restore_desired(chosen)
    enqueue(first, make_decision(K0 + 1, "applied", chosen=chosen))
    await first.sup.outbox_round()
    await first.sup.control_round()
    await first.sup.stop()
    second = await leading(world, "p2", advisory=True)
    state = world.shared_state()
    assert state.mode == "advisory" and state.rollback_pending
    # Advisory never restores a deployed point into the controller.
    assert second.controller.current_point().router == startup_flat("a")
    await settle([second], world, rounds=5)
    state = world.shared_state()
    assert not state.rollback_pending
    assert frontend_configs(world) == [VARIANTS["a"]["aggregated"]]
    planner_writes = [w for w in world.frontend.writes if w[0] == "p2"]
    assert planner_writes and all(w[2] == "reset" for w in planner_writes)
    # Now advisory only observes: a manual change is left alone.
    world.frontend.manual_patch(TEMP, 0.9)
    before = len(world.frontend.writes)
    await settle([second], world, rounds=3)
    assert len(world.frontend.writes) == before


async def test_advisory_rollback_leaves_unlatched_routers_alone() -> None:
    """A router changed at revision > 0 before anyone latched it is reset to
    learn its initial only in active mode."""

    world = World()
    first = await leading(world)
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    first.controller.restore_desired(chosen)
    enqueue(first, make_decision(K0 + 1, "applied", chosen=chosen))
    await first.sup.outbox_round()
    await first.sup.control_round()
    target = first.sup.target()
    assert target.mode == "apply" and target.learn_unknown
    await first.sup.stop()
    # The frontend restarts and an operator patches its router before any
    # planner latched it.
    world.frontend.restart(VARIANTS["a"])
    world.frontend.manual_patch(TEMP, 0.7)
    (manual,) = world.frontend.routers.values()
    second = await leading(world, "p2", advisory=True)
    target = second.sup.target()
    assert target.mode == "rollback" and not target.learn_unknown
    await settle([second], world, rounds=5)
    # Never written, and the rollback still completes without it.
    assert not [w for w in world.frontend.writes if w[0] == "p2"]
    assert manual.config[TEMP] == 0.7 and manual.revision == 1
    assert not world.shared_state().rollback_pending
    await second.sup.stop()


def test_rollback_targets_learn_unknown_routers_only_when_active() -> None:
    world = World()
    for advisory, decommission, learn in (
        (False, False, True),
        (True, False, False),
        (True, True, False),
    ):
        process = make_process(
            world,
            settings=settings_for(advisory=advisory, decommission=decommission),
        )
        sup = process.sup
        sup._initialized = True
        sup._snapshot = SimpleNamespace(state=claimed(rollback_pending=True))
        target = sup.target()
        assert target.mode == "rollback" and target.include_not_ready
        assert target.learn_unknown is learn
    # A failed spec premise rolls back an active planner's own deployment.
    sup = make_process(world).sup
    sup._initialized = True
    sup._snapshot = SimpleNamespace(state=claimed())
    sup._spec_ok = False
    target = sup.target()
    assert target.mode == "rollback" and target.learn_unknown


async def test_promote_applies_the_last_advisory_recommendation() -> None:
    world = World()
    first = await leading(world, advisory=True)
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    enqueue(first, make_decision(K0 + 1, "advisory", chosen=chosen))
    await first.sup.outbox_round()
    await first.sup.stop()
    world.clock.advance(100)
    second = await leading(world, "p2", promote=True)
    state = world.shared_state()
    assert state.mode == "active" and state.by_wt == by_wt("a", chosen)
    assert state.history[-1]["outcome"] == "promoted"
    assert second.controller.current_point().router == chosen
    assert frontend_configs(world) == [by_wt("a", chosen)["aggregated"]]


async def test_a_state_of_another_dgd_uid_is_recreated() -> None:
    world = World()
    store = StateStore(world.kube, DgdRef(DGD_NAME, "test-ns", "old-uid", "dyn"))
    snapshot = store.read()
    store.write(
        RetuneState(
            dgd_uid="old-uid",
            leader_term="0/x/1",
            fingerprint="fp-a",
            mode="active",
            last_window_end_ms=123,
        ),
        snapshot,
    )
    world.kube.create_request(
        {"metadata": {"name": "old-req", "labels": {LABEL_DGD_UID: "old-uid"}}}
    )
    reconciler = SimpleNamespace(
        deleted=[],
        protect=lambda *a: None,
        release=lambda *a: None,
        run_once=lambda **kwargs: None,
    )
    reconciler.enqueue_delete = lambda names: reconciler.deleted.extend(names)
    process = await leading(world, reconciler=reconciler)
    state = world.shared_state()
    assert state.dgd_uid == "uid-1" and state.last_window_end_ms == end_ms(K0)
    owner = world.kube.leases[STATE_NAME]["metadata"]["ownerReferences"][0]
    assert owner["uid"] == "uid-1"
    assert reconciler.deleted == ["old-req"]
    assert process.sup.initialized


# ----------------------------------------------------------- revalidation


async def test_spec_flip_rebuilds_rolls_back_and_recovers() -> None:
    world = World()
    process = await leading(world)
    sup, host = process.sup, process.host
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    process.controller.restore_desired(chosen)
    enqueue(process, make_decision(K0 + 1, "applied", chosen=chosen))
    await sup.outbox_round()
    await sup.control_round()
    old_controller = process.controller
    # P8 breaks (a second frontend without replica sync).
    world.spec.spec_ok = False
    await sup.control_round()
    assert host.replacements[-1][1] == 1 and sup.generation == 1
    assert process.controller is not old_controller
    assert old_controller._cancel.is_set()  # closed
    state = world.shared_state()
    assert state.rollback_pending and state.by_wt == VARIANTS["a"]
    assert sup.target().mode == "rollback" and not sup.ready()
    assert frontend_configs(world) == [VARIANTS["a"]["aggregated"]]
    assert sup.launch_gate(K0 + 2, window(K0 + 2)).reason == "not_ready"
    # Fixed: rebuilt again, back to ready.
    world.spec.spec_ok = True
    await settle([process], world, rounds=5)
    assert sup.generation == 2 and sup.ready()
    assert not world.shared_state().rollback_pending
    assert sup.launch_gate(K0 + 2, window(K0 + 2)).reason is None


async def test_p4_failure_disables_until_it_recovers() -> None:
    world = World()
    world.spec.p4_ok = False
    process = await leading(world)
    sup, host = process.sup, process.host
    assert sup.retune_inputs() is None and process.controller is None
    assert not sup.ready()
    world.spec.p4_ok = True
    await sup.control_round()
    assert variant_of(host.replacements[-1][0]) == "a"
    assert process.controller is not None and sup.generation == 1
    await sup.control_round()
    assert sup.ready()


async def test_controller_construction_failure_keeps_not_ready() -> None:
    world = World()
    process = await leading(world)
    process.host.fail_replace = True
    world.spec.variant = "b"
    world.frontend.restart(VARIANTS["b"])
    await process.sup.control_round()  # no exception escapes
    assert process.controller is None and not process.sup.ready()
    assert process.sup.initialized  # still leads, rolls back, persists


async def test_p14_parity_failure_is_a_spec_failure() -> None:
    world = World()
    process = await leading(world)
    sup = process.sup
    # A frontend restarts with startup values the planner did not parse.
    world.frontend.restart(VARIANTS["b"])
    await sup.control_round()
    assert sup._p14_ok is False
    await sup.control_round()  # revalidation sees spec_ok flip
    assert not sup.spec_ok and sup.generation == 1
    assert sup.target().mode == "rollback"
    world.frontend.restart(VARIANTS["a"])
    await settle([process], world, rounds=3)
    assert sup.spec_ok and sup.ready()


async def test_namespace_change_swaps_the_actuator_only() -> None:
    world = World()
    process = await leading(world)
    old = process.sup.actuator
    world.spec.dynamo_namespace = "dyn-v2"
    await process.sup.control_round()
    assert process.sup.actuator is not old
    assert process.sup.dgd.dynamo_namespace == "dyn-v2"
    assert process.sup.generation == 0 and process.host.replacements == []


async def test_health_premises_only_gate_launches() -> None:
    world = World()
    verdict = {"ok": False}
    process = await leading(world, canary=lambda sup: verdict["ok"])
    sup = process.sup
    rv = world.kube.leases[STATE_NAME]["metadata"]["resourceVersion"]
    assert not sup.ready()
    world.frontend.down = True
    await sup.control_round()
    world.frontend.down = False
    verdict["ok"] = True
    assert world.kube.leases[STATE_NAME]["metadata"]["resourceVersion"] == rv
    await sup.control_round()
    assert sup.ready()


async def test_an_actuator_round_overtaken_by_a_local_loss_is_discarded() -> None:
    world = World()
    process = await leading(world)
    sup = process.sup
    report_before = sup.last_report
    rv = world.kube.leases[STATE_NAME]["metadata"]["resourceVersion"]

    class OvertakenActuator(FakeActuator):
        async def reconcile(self, target, *, initials, expected_frontends):
            sup._epoch += 1  # a local loss ran while this round awaited I/O
            return ActuationReport(
                converged=False,
                new_initials={"stale-router": {CREDIT: 9.0}},
                seen_router_ids=frozenset({"stale-router"}),
                parity_ok=False,
            )

    sup._actuator = OvertakenActuator(world.frontend, "p1", False, VARIANTS["a"])
    await sup.control_round()
    assert sup.last_report is report_before and sup._p14_ok is not False
    assert world.kube.leases[STATE_NAME]["metadata"]["resourceVersion"] == rv


async def test_initials_are_persisted_and_pruned() -> None:
    world = World()
    process = await leading(world)
    first_ids = set(world.frontend.routers)
    assert set(world.shared_state().initials) == first_ids
    world.frontend.restart(VARIANTS["a"])
    await process.sup.control_round()
    assert set(world.shared_state().initials) == set(world.frontend.routers)
    assert not first_ids & set(world.shared_state().initials)


# ------------------------------------------------------------ reconciler


class FakeReconciler:
    """Records each round's ``should_continue``; a round can hang (a
    hard-mounted NFS call) until ``unblock`` is set."""

    def __init__(self, *, hang: bool = False) -> None:
        self.rounds: list[Any] = []
        self.answers_after: list[bool] = []
        self.entered = threading.Event()
        self.unblock = threading.Event()
        if not hang:
            self.unblock.set()
        self.events: list[tuple[str, Any]] = []

    def protect(self, window_end_ms, names) -> None:
        pass

    def release(self, window_end_ms) -> None:
        pass

    def enqueue_delete(self, names) -> None:
        self.events.append(("enqueue_delete", sorted(names)))

    def set_dgd(self, dgd: DgdRef) -> None:
        self.events.append(("set_dgd", dgd.uid))

    def run_once(
        self,
        *,
        last_window_end_ms,
        due_window_end_ms,
        deadline_for,
        frontend_pods,
        should_continue=None,
    ):
        self.rounds.append(should_continue)
        self.entered.set()
        assert self.unblock.wait(10), "never unblocked"
        self.answers_after.append(should_continue())


def reconciler_on_a_thread(reconciler: FakeReconciler):
    """Run only the reconciler's round on a worker thread (as live)."""

    async def blocking(call):
        if getattr(call, "func", None) == reconciler.run_once:
            return await asyncio.to_thread(call)
        return call()

    return blocking


async def reconciled(sup: LiveRetuneSupervisor) -> None:
    task = sup.reconcile_task
    if task is not None:
        await task


async def test_a_hung_reconciler_never_blocks_revalidation_or_actuation() -> None:
    world = World()
    reconciler = FakeReconciler(hang=True)
    process = await leading(
        world,
        reconciler=reconciler,
        blocking=reconciler_on_a_thread(reconciler),
        settings=settings_for(reconcile_budget_s=0.05),
    )
    sup = process.sup
    try:
        assert await asyncio.to_thread(reconciler.entered.wait, 5)
        # The round runs in its own task; the control loop goes on and never
        # starts a second round while the first one's call still runs.
        chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
        process.controller.restore_desired(chosen)
        enqueue(process, make_decision(K0 + 1, "applied", chosen=chosen))
        await sup.outbox_round()
        await asyncio.wait_for(sup.control_round(), 5)
        assert frontend_configs(world) == [by_wt("a", chosen)["aggregated"]]
        assert len(reconciler.rounds) == 1 and sup.reconcile_running()
        # Past its budget the round's task ends and the reconciler is told
        # to stop before its next delete; its call still runs.
        await asyncio.wait_for(reconciled(sup), 5)
        assert sup.reconcile_running() and reconciler.rounds[0]() is False
        # A spec break is still revalidated and rolled back meanwhile.
        world.spec.spec_ok = False
        await asyncio.wait_for(sup.control_round(), 5)
        await asyncio.wait_for(sup.control_round(), 5)
        assert not sup.spec_ok and sup.target().mode == "rollback"
        assert frontend_configs(world) == [VARIANTS["a"]["aggregated"]]
        assert len(reconciler.rounds) == 1
    finally:
        reconciler.unblock.set()
    work = sup._reconcile_work
    assert work is not None
    await asyncio.wait_for(work, 5)
    assert reconciler.answers_after == [False]
    # Once the call returned, the next control round starts a new round.
    assert not sup.reconcile_running()
    await sup.control_round()
    await reconciled(sup)
    await asyncio.wait_for(sup._reconcile_work, 5)
    assert len(reconciler.rounds) == 2 and reconciler.answers_after[-1] is True
    await sup.stop()


async def test_reconciler_runs_only_for_the_leader_of_its_epoch() -> None:
    world = World()
    reconciler = FakeReconciler()
    process = await leading(world, reconciler=reconciler)
    sup = process.sup
    await reconciled(sup)
    assert len(reconciler.rounds) == 1 and reconciler.rounds[0]() is True
    # Another process took the term; the actuator round finds out while
    # persisting new initials: no reconciler round starts after it.
    store = StateStore(world.kube, sup.dgd)
    snapshot = store.read()
    store.write(dataclasses.replace(snapshot.state, leader_term="9/x/1"), snapshot)
    world.frontend.restart(VARIANTS["a"])
    await sup.control_round()
    await reconciled(sup)
    assert not sup.initialized
    assert len(reconciler.rounds) == 1 and reconciler.rounds[0]() is False
    # Re-initialized (a new epoch): the old round's should_continue stays
    # False, the new round's is True until leadership lapses.
    await sup.lease_round()
    await sup.control_round()
    await reconciled(sup)
    assert len(reconciler.rounds) == 2
    assert reconciler.rounds[0]() is False and reconciler.rounds[1]() is True
    world.clock.advance(LEASE)
    assert not sup.is_leader() and reconciler.rounds[1]() is False
    await sup.control_round()
    assert len(reconciler.rounds) == 2


async def test_recreated_dgd_moves_the_reconciler_first() -> None:
    world = World()
    reconciler = FakeReconciler()
    process = await leading(world, reconciler=reconciler)
    world.kube.create_request(
        {"metadata": {"name": "old-req", "labels": {LABEL_DGD_UID: "uid-1"}}}
    )
    world.spec.uid = "uid-2"  # deleted and recreated under the same name
    await process.sup.control_round()
    assert process.sup.dgd.uid == "uid-2" and process.sup.initialized
    assert world.shared_state().dgd_uid == "uid-2"
    # The reconciler follows the new uid before the old uid's Requests are
    # queued for deletion, so the new DGD's Requests stay its own.
    assert reconciler.events == [("set_dgd", "uid-2"), ("enqueue_delete", ["old-req"])]
    # A namespace-only change keeps the reconciler's identity.
    world.spec.dynamo_namespace = "dyn-2"
    await process.sup.control_round()
    assert reconciler.events[-1] == ("enqueue_delete", ["old-req"])


# ----------------------------------------------------------- sweeps, stop


async def test_sweep_fn_binds_the_context_and_never_raises() -> None:
    world = World()
    seen = {}

    def runner(request, *, cancel, abort, context):
        seen.update(request=request, cancel=cancel, abort=abort, context=context)
        return "result"

    process = await started(
        world, sweep_runner=runner, context_factory=lambda sup: ("ctx", sup.dgd)
    )
    cancel, abort = threading.Event(), threading.Event()
    request = SimpleNamespace(window_index=5)
    assert process.sup.sweep_fn(request, cancel=cancel, abort=abort) == "result"
    assert seen["context"] == ("ctx", process.sup.dgd) and seen["abort"] is abort

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    process.sup._sweep_runner = broken
    result = process.sup.sweep_fn(request, cancel=cancel, abort=abort)
    assert result.status == "failed" and result.failure_reason == "sweep_error"
    assert "boom" in result.error
    no_context = await started(world, "p2", sweep_runner=runner, context_factory=None)
    result = no_context.sup.sweep_fn(request, cancel=cancel, abort=abort)
    assert result.status == "failed"


async def test_stop_aborts_the_sweep_and_waits_for_the_sweep_thread() -> None:
    world = World()
    process = await leading(world)
    sup, controller = process.sup, process.controller
    world.clock.wall = window(K0 + 1)[2]
    await controller.on_tick(world.clock.time(), SPEC_COUNTS)
    pending = controller._pending
    world.sweeps.quiesce([controller])  # running on the sweep thread
    abort = pending.abort

    await sup.stop()

    # drop_pending set the sweep's abort; stop waited for the sweep thread.
    assert abort.is_set() and pending.future.done()
    assert pending.future.result().failure_reason == "lost_leadership"
    assert world.holder() is None and not sup.is_leader()
    await sup.stop()  # idempotent


async def test_stop_gives_up_on_a_sweep_that_never_returns(caplog) -> None:
    world = World()
    release = threading.Event()

    def stuck(request, *, cancel, abort, context):
        release.wait(10)  # ignores its abort
        return SweepResult(status="failed")

    process = await leading(
        world, sweep_runner=stuck, settings=settings_for(stop_timeout_s=0.05)
    )
    sup, controller = process.sup, process.controller
    world.clock.wall = window(K0 + 1)[2]
    await controller.on_tick(world.clock.time(), SPEC_COUNTS)
    try:
        with caplog.at_level(logging.WARNING):
            await sup.stop()
        assert "a sweep is still running at shutdown" in caplog.text
        assert world.holder() is None
    finally:
        release.set()


async def test_background_loops_run_and_stop() -> None:
    world = World()
    process = make_process(world)
    sup = process.sup
    sup._sleep = lambda s: asyncio.sleep(0.001)
    await sup.start(process.host)
    for _ in range(500):
        await asyncio.sleep(0.001)
        if sup.initialized and sup.ready():
            break
    assert sup.initialized and sup.ready()
    enqueue(process, make_decision(K0 + 1, "kept"))
    for _ in range(500):
        await asyncio.sleep(0.001)
        if sup.outbox_size == 0:
            break
    assert world.shared_state().history[-1]["outcome"] == "kept"
    await sup.stop()
    assert world.holder() is None
    assert process.metrics.state["leader"] is False


async def test_hooks_are_the_adapter_hooks() -> None:
    world = World()
    process = await started(world)
    hooks = process.sup.hooks()
    assert isinstance(hooks, RetuneHooks)
    assert hooks.launch_gate == process.sup.launch_gate
    assert hooks.harvest_allowed == process.sup.harvest_allowed
    assert hooks.sweep_fn == process.sup.sweep_fn and hooks.phase_s == 0.0


def decommission_supervisor(world: World) -> LiveRetuneSupervisor:
    """The clean-up supervisor of a planner whose retune was removed."""

    frontend = SimpleNamespace(
        parse_frontend=lambda dgd, planner_mode: world.spec.startup()
    )
    return bootstrap.build_decommission_supervisor(
        live_config(),
        kube=world.kube,
        dgd=world.kube.get_dgd(DGD_NAME),
        dynamo_namespace="dyn",
        metrics=NullMetrics(),
        actuator_factory=lambda dgd, maps: FakeActuator(
            world.frontend, "gc", True, maps
        ),
        frontend=frontend,
        clock=world.clock.time,
        monotonic=world.clock.monotonic,
        sleep=no_sleep,
        blocking=inline,
    )


async def test_decommission_rolls_back_then_deletes_everything() -> None:
    world = World()
    first = await leading(world)
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    first.controller.restore_desired(chosen)
    enqueue(first, make_decision(K0 + 1, "applied", chosen=chosen))
    await first.sup.outbox_round()
    await first.sup.control_round()
    await first.sup.stop()
    world.kube.create_request(
        {"metadata": {"name": "rt-1", "labels": {LABEL_DGD_UID: "uid-1"}}}
    )
    sup = decommission_supervisor(world)
    await sup.start(SimpleNamespace(), background=False)
    for _ in range(6):
        if sup.decommissioned:
            break
        await sup.lease_round()
        await sup.control_round()
        world.clock.advance(30)
    assert sup.decommissioned
    assert frontend_configs(world) == [VARIANTS["a"]["aggregated"]]
    assert world.kube.requests == {} and world.kube.leases == {}
    assert sup.launch_gate(K0 + 1, window(K0 + 1)).reason == "not_leader"
    await sup.stop()


async def test_decommission_never_deletes_after_losing_its_term() -> None:
    world = World()
    first = await leading(world, advisory=True)  # nothing deployed
    await first.sup.stop()
    world.kube.create_request(
        {"metadata": {"name": "rt-1", "labels": {LABEL_DGD_UID: "uid-1"}}}
    )
    sup = decommission_supervisor(world)
    await sup.start(SimpleNamespace(), background=False)
    await sup.lease_round()
    assert sup.initialized and not world.shared_state().rollback_pending
    # Another replica claims the state; this round's initials write (a
    # restarted frontend) finds out before the clean-up would delete.
    store = StateStore(world.kube, sup.dgd)
    snapshot = store.read()
    store.write(dataclasses.replace(snapshot.state, leader_term="9/x/1"), snapshot)
    world.frontend.restart(VARIANTS["a"])
    await sup.control_round()
    assert not sup.decommissioned and not sup.initialized
    assert set(world.kube.requests) == {"rt-1"}
    assert set(world.kube.leases) == {STATE_NAME, LEADER_NAME}
    await sup.stop()


# -------------------------------------------------------------- bootstrap


def dgd_object(mode: str = "agg", **worker) -> dict:
    components = [{"name": "Frontend", "type": "frontend", "replicas": 1}]
    if mode == "disagg":
        components += [
            {"name": "prefill", "type": "prefill", "replicas": 2},
            {"name": "decode", "type": "decode", **worker},
        ]
    else:
        components.append({"name": "worker", "type": "worker", **worker})
    return {
        "metadata": {
            "name": DGD_NAME,
            "namespace": "test-ns",
            "uid": "uid-1",
            "generation": 3,
        },
        "spec": {"components": components},
    }


def test_worker_spec_replicas() -> None:
    assert bootstrap.worker_spec_replicas(dgd_object(replicas=3), "agg") == {
        "aggregated": 3
    }
    # Omitted replicas are the workload default of 1.
    assert bootstrap.worker_spec_replicas(dgd_object(), "agg") == {"aggregated": 1}
    assert bootstrap.worker_spec_replicas(dgd_object(replicas=0), "agg") is None
    assert bootstrap.worker_spec_replicas(
        dgd_object("disagg", replicas=4), "disagg"
    ) == {"prefill": 2, "decode": 4}
    assert bootstrap.worker_spec_replicas({"spec": {"components": []}}, "agg") is None


def test_build_retune_inputs_agg_and_disagg() -> None:
    config = live_config()
    startup = Spec().startup()
    inputs = bootstrap.build_retune_inputs(
        config, startup, dgd_name="graph", model_name="m"
    )
    assert inputs.trace_paths == ("/trace/segments/graph",)
    assert (inputs.trace_origin_ms, inputs.arrival_speedup_ratio) == (0.0, 1.0)
    assert inputs.engine.router_mode == "kv_router" and inputs.engine.model_name == "m"
    assert not inputs.engine.is_disagg and inputs.engine.extra_engine_args is None
    disagg = bootstrap.build_retune_inputs(
        config.model_copy(update={"mode": "disagg"}),
        startup,
        dgd_name="graph",
        model_name=None,
    )
    assert disagg.engine.is_disagg and disagg.engine.prefill_engine_args == {}


def test_kv_router_resolver_matches_the_router_config() -> None:
    llm = pytest.importorskip("dynamo.llm")
    assert VARIANTS == {variant: by_wt(variant, {}) for variant in ROUTER_CONFIG}
    resolver = bootstrap.KvRouterResolver({}, ("prefill", "decode"))
    flat = resolver.startup_flat(KEYS)
    default = llm.KvRouterConfig.from_json("{}")
    serialized = json.loads(default.to_json())
    resolved = dict(default.policy_parameters("aggregated"))
    assert flat == {
        key: float(resolved[key] if serialized.get(key) is None else serialized[key])
        for key in KEYS
    }
    resolved = resolver.by_wt({CREDIT: 0.5})
    config = llm.KvRouterConfig.from_json(json.dumps({CREDIT: 0.5}))
    assert resolved == {
        wt: {k: float(v) for k, v in dict(config.policy_parameters(wt)).items()}
        for wt in ("prefill", "decode")
    }


class FakeFrontendModule:
    def __init__(self, startup: Optional[FrontendStartup], p4_ok=True, spec_ok=True):
        self.startup = startup
        self.p4_ok = p4_ok
        self.spec_ok = spec_ok

    def parse_frontend(self, dgd, *, planner_mode):
        return self.startup if self.startup is not None else "P4: shell-wrapped"

    def evaluate_premises(self, dgd, *, planner_config):
        parsed = self.startup is not None
        return PremiseReport(
            p4_ok=self.p4_ok and parsed,
            spec_ok=self.spec_ok and self.p4_ok and parsed,
            reasons=() if parsed else ("P4: shell-wrapped",),
            startup=self.startup,
        )

    def served_model_name(self, dgd):
        return "served"


def test_evaluate_spec() -> None:
    config = live_config()
    frontend = FakeFrontendModule(Spec().startup())
    evaluation = bootstrap.evaluate_spec(
        dgd_object(replicas=2),
        config=config,
        k8s_namespace="ns",
        dynamo_namespace="dyn",
        frontend=frontend,
    )
    assert evaluation.dgd == DgdRef(DGD_NAME, "test-ns", "uid-1", "dyn")
    assert evaluation.spec_replicas == {"aggregated": 2} and evaluation.generation == 3
    assert evaluation.inputs.engine.model_name == "served"
    assert evaluation.report.p4_ok and evaluation.startup is not None

    unparsed = FakeFrontendModule(None)
    evaluation = bootstrap.evaluate_spec(
        dgd_object(),
        config=config,
        k8s_namespace="ns",
        dynamo_namespace="dyn",
        frontend=unparsed,
    )
    assert evaluation.inputs is None and evaluation.startup is None
    assert evaluation.report.reasons == ("P4: shell-wrapped",)
    with pytest.raises(ValueError):
        bootstrap.evaluate_spec(
            {"metadata": {"name": "x"}},
            config=config,
            k8s_namespace="ns",
            dynamo_namespace="dyn",
            frontend=frontend,
        )


async def test_build_supervisor_wires_the_collaborators() -> None:
    world = World()
    world.kube.put_dgd(dgd_object(replicas=2))
    frontend = FakeFrontendModule(Spec().startup())
    reconciler = SimpleNamespace(
        protect=lambda *a: None,
        release=lambda *a: None,
        enqueue_delete=lambda *a: None,
        run_once=lambda **kwargs: None,
    )
    sup = await bootstrap.build_supervisor(
        live_config(),
        kube=world.kube,
        dgd_name=DGD_NAME,
        namespace_fn=lambda: "dyn",
        metrics=NullMetrics(),
        identity="pod/1",
        frontend=frontend,
        actuator_factory=lambda dgd, maps: FakeActuator(
            world.frontend, "x", False, maps
        ),
        reconciler=reconciler,
        canary=lambda s: True,
    )
    assert sup.identity == "pod/1" and sup.registry is reconciler
    assert sup.retune_inputs() is not None
    assert sup.evaluation.spec_replicas == {"aggregated": 2}
    assert sup._settings.router_keys == KEYS
    assert sup._settings.interval_s == 21600 and sup._settings.lease_duration_s == 60
    assert len(sup.fingerprint) == 16


async def test_build_live_retune_requires_the_dgdr_executor() -> None:
    host = SimpleNamespace(
        config=live_config(
            retune={"executor": "local", "dgdr": None, "timeout_seconds": None}
        )
    )
    with pytest.raises(ValueError, match="replay-only"):
        await bootstrap.build_live_retune(host)


async def test_build_live_retune_uses_the_connector(monkeypatch) -> None:
    world = World()
    world.kube.put_dgd(dgd_object(replicas=2))
    from dynamo.planner.retune.live import kube as kube_module

    monkeypatch.setattr(
        kube_module.KubernetesRetuneKube,
        "from_kube_api",
        classmethod(lambda cls, api: world.kube),
    )
    captured = {}

    async def fake_build(config, **kwargs):
        captured.update(kwargs, config=config)
        return "supervisor"

    monkeypatch.setattr(bootstrap, "build_supervisor", fake_build)
    host = SimpleNamespace(
        config=live_config(),
        environment=SimpleNamespace(
            controller=SimpleNamespace(kube_api=object(), parent_dgd_name=DGD_NAME),
            runtime_namespace=lambda: "dyn-hash",
        ),
    )
    assert await bootstrap.build_live_retune(host) == "supervisor"
    assert captured["kube"] is world.kube and captured["dgd_name"] == DGD_NAME
    assert captured["namespace_fn"]() == "dyn-hash"


async def test_build_legacy_cleanup(monkeypatch) -> None:
    world = World()
    world.kube.put_dgd(dgd_object(replicas=2))
    monkeypatch.setattr(
        bootstrap.frontend_config,
        "parse_frontend",
        lambda dgd, planner_mode: "P4: no frontend",
    )
    monkeypatch.setattr(bootstrap, "RetuneMetrics", NullMetrics)
    host = SimpleNamespace(
        config=live_config().model_copy(update={"retune": None}),
        environment=SimpleNamespace(
            controller=SimpleNamespace(parent_dgd_name=DGD_NAME),
            runtime_namespace=lambda: "",
        ),
    )
    sup = await bootstrap.build_legacy_cleanup(host, world.kube)
    assert sup._settings.decommission and sup.retune_inputs() is None
    # The planner's configured namespace when the runtime has none.
    assert sup.dgd == DgdRef(DGD_NAME, "test-ns", "uid-1", host.config.namespace)
    world.kube.fail("get_dgd", KubeError("forbidden"))
    with pytest.raises(KubeError):
        await bootstrap.build_legacy_cleanup(host, world.kube)


class PlannerEnvironment:
    """The ``PlannerEnvironment`` surface ``NativePlannerBase`` startup uses."""

    def __init__(self, controller: Any) -> None:
        from dynamo.planner.environment.state import DeploymentState

        self.controller = controller
        self.state = DeploymentState()
        self.state.decode.num_gpus = 1
        self.calls: list[str] = []

    def deployment_state(self):
        return self.state

    async def initialize(self) -> None:
        self.calls.append("environment.initialize")

    async def refresh(self) -> None:
        pass

    async def shutdown(self) -> None:
        self.calls.append("environment.shutdown")


class FakeCleanup:
    def __init__(self, calls: list[str]):
        self.calls = calls
        self.host = None

    async def start(self, host) -> None:
        self.calls.append("cleanup.start")
        self.host = host

    async def stop(self) -> None:
        self.calls.append("cleanup.stop")


def forbidden() -> KubeError:
    error = KubeError("read leases/graph-planner-retune: HTTP 403: forbidden")
    error.status = 403  # type: ignore[attr-defined]
    return error


def planner_without_retune(monkeypatch, controller: Any, *, leases=(), build=None):
    """A ``NativePlannerBase`` without retune config; the engine bootstrap
    is a no-op. ``leases`` is what the Lease check finds (or raises) and
    ``build(host, kube)`` builds the clean-up."""

    from unittest.mock import patch

    from dynamo.planner.core import base as base_module

    config = live_config(
        metric_reporting_prometheus_port=0, live_dashboard_port=0, control_api_port=0
    ).model_copy(update={"retune": None})
    environment = PlannerEnvironment(controller)
    kube = object()

    def find_retune_leases(found_kube, dgd_name, *, timeout_s):
        assert found_kube is kube and dgd_name == DGD_NAME
        environment.calls.append("leases.check")
        if isinstance(leases, Exception):
            raise leases
        return tuple(leases)

    async def build_legacy_cleanup(host, found_kube, *, timeout_s):
        assert found_kube is kube
        environment.calls.append("cleanup.build")
        return await build(host)

    monkeypatch.setattr(base_module, "find_retune_leases", find_retune_leases)
    monkeypatch.setattr(
        base_module.KubernetesRetuneKube,
        "from_kube_api",
        classmethod(lambda cls, api: kube),
    )
    monkeypatch.setattr(bootstrap, "build_legacy_cleanup", build_legacy_cleanup)
    with patch("dynamo.planner.core.base.PlannerPrometheusMetrics"):
        planner = base_module.NativePlannerBase(None, config, environment)

    async def bootstrap_engine():
        environment.calls.append("engine")

    async def nothing():
        pass

    planner._bootstrap_regression = bootstrap_engine  # type: ignore[method-assign]
    planner._bootstrap_engine_plugins_if_needed = nothing  # type: ignore[method-assign]
    return planner, environment


def kubernetes_connector() -> Any:
    from unittest.mock import MagicMock

    from dynamo.planner.connectors.kubernetes import KubernetesConnector

    connector = MagicMock(spec=KubernetesConnector)
    connector.kube_api = object()
    connector.parent_dgd_name = DGD_NAME
    return connector


async def test_a_planner_without_retune_cleans_up_in_the_background(
    monkeypatch,
) -> None:
    cleanups: list[FakeCleanup] = []

    async def build(host):
        cleanups.append(FakeCleanup(environment.calls))
        return cleanups[-1]

    planner, environment = planner_without_retune(
        monkeypatch, kubernetes_connector(), leases=(STATE_NAME,), build=build
    )
    await planner._async_init()
    try:
        # Startup never waits for the check.
        assert environment.calls == ["environment.initialize", "engine"]
        assert planner.may_hold_retune_lease
        await planner._legacy_retune_task
        assert environment.calls[2:] == [
            "leases.check",
            "cleanup.build",
            "cleanup.start",
        ]
        assert cleanups[0].host is planner and planner._live_retune is None
    finally:
        await planner._shutdown_runtime()
    assert environment.calls[-2:] == ["cleanup.stop", "environment.shutdown"]
    assert planner._legacy_retune_cleanup is None


@pytest.mark.parametrize("outcome", ["nothing_left", "forbidden", "not_found", "bug"])
async def test_the_legacy_check_never_fails_the_planner(
    monkeypatch, caplog, outcome
) -> None:
    from dynamo.planner.core import base as base_module
    from dynamo.planner.retune.live.contracts import KubeNotFound

    leases: Any = (LEADER_NAME,)
    if outcome == "nothing_left":
        leases = ()
    elif outcome == "forbidden":
        leases = forbidden()

    async def build(host):
        if outcome == "not_found":
            raise KubeNotFound("read dynamographdeployments/graph: HTTP 404")
        raise RuntimeError("a bug")

    if outcome in ("nothing_left", "forbidden"):
        # The live retune stack is never loaded.
        monkeypatch.setattr(base_module, "_live_retune_bootstrap", None)
    planner, environment = planner_without_retune(
        monkeypatch, kubernetes_connector(), leases=leases, build=build
    )
    with caplog.at_level(logging.DEBUG, logger="dynamo.planner.core.base"):
        await planner._async_init()
        try:
            await asyncio.wait([planner._legacy_retune_task])
            assert planner._legacy_retune_cleanup is None
            records = [r for r in caplog.records if "legacy clean-up" in r.message]
            levels = {r.levelname for r in records}
            if outcome == "nothing_left":
                assert records == []
            elif outcome == "forbidden":
                assert levels == {"DEBUG"}  # the default chart: no Lease RBAC
            elif outcome == "not_found":
                assert levels == {"INFO"} and "skipped" in caplog.text
            else:
                assert levels == {"ERROR"} and "a bug" in caplog.text
        finally:
            await planner._shutdown_runtime()
    assert "cleanup.stop" not in environment.calls


async def test_the_legacy_check_is_kubernetes_only(monkeypatch) -> None:
    for controller in (None, SimpleNamespace(kube_api=object())):
        planner, environment = planner_without_retune(monkeypatch, controller)
        await planner._async_init()
        try:
            assert planner._legacy_retune_task is None
            assert not planner.may_hold_retune_lease
        finally:
            await planner._shutdown_runtime()
        assert "leases.check" not in environment.calls


async def test_dgdr_canary_and_context() -> None:
    world = World()
    world.kube.put_dgd(dgd_object(replicas=2))
    config = live_config()
    process = await leading(world)
    canary = bootstrap.dgdr_canary(config)
    assert canary(process.sup) is True
    created = world.kube.calls_of("create_request")[-1]
    assert created.kwargs["dry_run"] is True and world.kube.requests == {}
    world.kube.reject_create = (
        "strict decoding error: unknown field spec.workload.trace"
    )
    assert canary(process.sup) is False
    world.spec.replicas = None
    await process.sup.control_round()
    assert canary(process.sup) is False

    context = bootstrap.dgdr_context_factory(config, cut_window="cutter")(process.sup)
    assert context.dgd == process.sup.dgd and context.kube is world.kube
    assert context.cut_window == "cutter" and context.backend == config.backend
    assert context.is_leader() is True


# ------------------------------------------- initials before router writes


class FrontendPolicyClient:
    """``RouterPolicyClient`` of the real ``RouterActuator`` over the
    world's ``FakeFrontend`` (every pod answers with all of its routers)."""

    def __init__(self, world: World, writer: str) -> None:
        self.frontend = world.frontend
        self.dynamo_namespace = world.spec.dynamo_namespace
        self.writer = writer

    def _info(self, router: Router) -> RouterInfo:
        return RouterInfo(
            router_id=router.router_id,
            namespace=self.dynamo_namespace,
            worker_type=router.worker_type,
            revision=router.revision,
            configuration=dict(router.config),
            model="model",
        )

    async def get(self, pod) -> PolicyCallResult:
        routers = tuple(self._info(r) for r in self.frontend.routers.values())
        return PolicyCallResult(status=200, routers=routers)

    async def patch(self, pod, router_id, expected_revision, parameters):
        router = self.frontend.routers[router_id]
        self.frontend.patch(router, dict(parameters), self.writer, False, "apply")
        return PolicyCallResult(status=200, router=self._info(router))

    async def reset(self, pod, router_id, expected_revision):
        router = self.frontend.routers[router_id]
        self.frontend.reset(router, self.writer, False, "rollback")
        return PolicyCallResult(status=200, router=self._info(router))


def real_actuators(world: World, writer: str):
    world.kube.add_pod("frontend-0", dgd=DGD_NAME)
    return lambda dgd, maps: RouterActuator(
        world.kube, FrontendPolicyClient(world, writer), dgd, maps, NullMetrics()
    )


async def test_a_stop_before_the_initials_write_leaves_no_router_retuned() -> None:
    """A router seen at revision 0 is latched and would be PATCHed in the same
    round. Stopped (SIGTERM) before the state records its initial, the next
    planner rolls back with an advisory target, which never resets a router
    whose initial it does not know: so no router may be PATCHed before its
    initial is persisted."""

    world = World()
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    first = await started(world, actuator_factory=real_actuators(world, "p1"))
    store = StateStore(world.kube, first.sup.dgd)
    await first.sup.lease_round()
    snapshot = store.read()
    # An active deployment is persisted, and the routers restart before the
    # actuator has latched them (revision 0, initials unknown).
    store.write(
        dataclasses.replace(
            snapshot.state, flat=chosen, by_wt=by_wt("a", chosen), deploy_seq=2
        ),
        snapshot,
    )
    first.sup._snapshot = store.read()
    world.kube.fail(
        "replace_lease",
        KubeError("timeout"),
        match=lambda target: target[0] == STATE_NAME,
    )
    await first.sup.control_round()
    assert world.frontend.writes == []  # held: the initial is not persisted
    assert world.shared_state().initials == {}
    await first.sup.stop()

    second = await leading(
        world, "p2", advisory=True, actuator_factory=real_actuators(world, "p2")
    )
    await settle([second], world, rounds=4)
    assert frontend_configs(world) == [VARIANTS["a"]["aggregated"]]
    assert not world.shared_state().rollback_pending
    assert world.frontend.writes == []


async def test_initials_are_persisted_before_the_routers_are_patched() -> None:
    world = World()
    process = await leading(world, actuator_factory=real_actuators(world, "p1"))
    chosen = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}
    process.controller.restore_desired(chosen)
    enqueue(process, make_decision(K0 + 1, "applied", chosen=chosen))
    await process.sup.outbox_round()
    world.frontend.restart(VARIANTS["a"])  # new routers at revision 0
    (router_id,) = world.frontend.routers
    patched_with: list[dict] = []
    original = world.frontend.patch

    def patch(router, params, *args):
        patched_with.append(dict(world.shared_state().initials))
        original(router, params, *args)

    world.frontend.patch = patch  # type: ignore[method-assign]
    # One round: latch, persist the initial, then PATCH.
    await process.sup.control_round()
    assert patched_with and all(router_id in seen for seen in patched_with)
    assert frontend_configs(world) == [by_wt("a", chosen)["aggregated"]]
    assert process.sup.last_report.converged


async def test_a_state_of_an_unknown_format_version_is_never_overwritten() -> None:
    world = World()
    newer = json.dumps({"version": 99, "dgd_uid": "uid-1", "anything": "new"})
    world.kube.create_lease(
        {
            "metadata": {"name": STATE_NAME, "annotations": {STATE_ANNOTATION: newer}},
            "spec": {},
        }
    )
    process = await started(world)
    for _ in range(2):
        await process.sup.lease_round()
        assert process.sup.is_leader() and not process.sup.initialized
    lease = world.kube.leases[STATE_NAME]
    assert lease["metadata"]["annotations"][STATE_ANNOTATION] == newer
    assert process.metrics.of("failure") == ["state_version", "state_version"]


# ------------------------------------------- with the real retune controller


async def test_real_controller_launches_harvests_and_persists() -> None:
    """The supervisor's hooks drive ``RetuneController`` end to end: gate,
    launch with the spec replicas and the abort event, sweep_fn, harvest
    gating, on_effects, outbox and the persisted deployment."""

    world = World()
    process = await leading(world)
    sup, controller = process.sup, process.controller
    assert sup.ready() and controller.router_keys == KEYS
    best = {CREDIT: 0.5, SCALE: 1.0, TEMP: 0.2}

    async def tick(now_s):
        world.clock.wall = now_s
        result = await controller.on_tick(now_s, SPEC_COUNTS)
        sup.on_effects(
            SimpleNamespace(
                retune_decisions=result.decisions,
                router_parameters=result.router_parameters,
            )
        )
        await sup.outbox_round()
        return result

    # First enable recorded the window in progress at start as decided:
    # neither it (its trace began mid-way) nor the due window launches.
    await tick(START_WALL)
    await tick(window(K0)[2])
    assert controller._pending is None and world.sweeps.requests == []
    # The next window launches at its L_k with the spec replicas.
    await tick(window(K0 + 1)[2])
    pending = controller._pending
    assert pending is not None and pending.abort is not None
    assert world.sweeps.finish(controller, scored(best))
    (request,) = world.sweeps.requests
    assert request.window_index == K0 + 1 and request.generation == 0
    assert request.spec_replicas == {"aggregated": 2} and request.replicas_verified
    # Harvested on the first tick at least a scale interval after the launch:
    # applied, persisted, then deployed.
    result = await tick(window(K0 + 1)[2] + 5.0)
    (decided,) = result.decisions
    assert decided.outcome == "applied"
    assert decided.decision_id == f"retune-w{end_ms(K0 + 1)}"
    state = world.shared_state()
    assert state.flat == best and state.deploy_seq == 2
    assert state.last_window_end_ms == end_ms(K0 + 1)
    assert state.by_wt == by_wt("a", best)
    await sup.control_round()
    assert frontend_configs(world) == [by_wt("a", best)["aggregated"]]
    # A re-initialization restores the persisted point into the controller.
    controller.restore_desired(startup_flat("a"))
    sup._request_reinit("test")
    await sup.lease_round()
    assert controller.current_point().router == best


# =================================================== interleaving harness
#
# Two planner processes share a FakeRetuneKube, a frontend and a clock. A
# seeded random schedule interleaves lease renewals and failures, takeovers,
# state writes that succeed, conflict, time out, time out but land, or land
# late, ticks of the real controllers that launch and harvest sweeps,
# frontend restarts and manual patches, spec changes (startup, spec_ok, P4),
# advisory switches, SIGTERM and SIGKILL. After every step the invariants
# are asserted; at the end the system settles with no infinite resets or
# PATCHes.


class CheckedKube(FakeRetuneKube):
    """Asserts at every successful state write that a term change (a claim)
    is made by the leadership Lease holder, and that a write never changes
    the term otherwise (an earlier term's write cannot land after a claim)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state_writes = 0
        self.late = False
        # Identity of the process whose call is executing (set by ProcessKube).
        self.writer: Optional[str] = None

    def _holder(self) -> Optional[str]:
        lease = self.leases.get(LEADER_NAME)
        return None if lease is None else lease["spec"].get("holderIdentity")

    def _check(self, before: Optional[dict], after: dict) -> None:
        def term(obj: Optional[dict]) -> Optional[str]:
            if obj is None:
                return None
            raw = (obj["metadata"].get("annotations") or {}).get(STATE_ANNOTATION)
            return None if raw is None else json.loads(raw).get("leader_term")

        new = term(after)
        identity = None if not new else new.split("/", 1)[1].rsplit("/", 1)[0]
        if self.writer is not None and new:
            # At most one leader writes: a write lands only under its
            # writer's own term (an earlier term's writer must fail).
            assert identity == self.writer, (identity, self.writer)
        if new != term(before):
            assert not self.late, "a late write changed the leader_term"
            assert identity == self._holder(), (identity, self._holder())
        self.state_writes += 1

    def replace_lease(self, name, body, *, timeout_s=10.0):
        before = copy.deepcopy(self.leases.get(name))
        result = super().replace_lease(name, body, timeout_s=timeout_s)
        if name == STATE_NAME:
            self._check(before, result)
        return result

    def create_lease(self, body, *, timeout_s=10.0):
        result = super().create_lease(body, timeout_s=timeout_s)
        if result["metadata"]["name"] == STATE_NAME:
            self._check(None, result)
        return result


class ProcessKube:
    """One process's view of the shared kube, with its own faults."""

    def __init__(self, shared: CheckedKube) -> None:
        self._shared = shared
        self.namespace = shared.namespace
        self.identity: Optional[str] = None
        self.lease_get_failures = 0
        self.state_faults: deque[str] = deque()
        self.deferred: list[tuple[str, dict]] = []

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self._shared, attribute)

    def _as_writer(self, call):
        self._shared.writer = self.identity
        try:
            return call()
        finally:
            self._shared.writer = None

    def create_lease(self, body, *, timeout_s=10.0):
        return self._as_writer(
            lambda: self._shared.create_lease(body, timeout_s=timeout_s)
        )

    def get_lease(self, name, *, timeout_s=10.0):
        if name == LEADER_NAME and self.lease_get_failures > 0:
            self.lease_get_failures -= 1
            raise KubeError("injected renewal failure")
        return self._shared.get_lease(name, timeout_s=timeout_s)

    def replace_lease(self, name, body, *, timeout_s=10.0):
        if name == STATE_NAME and self.state_faults:
            fault = self.state_faults.popleft()
            current = self._shared.leases.get(name)
            same_term = current is not None and (
                json.loads(current["metadata"]["annotations"][STATE_ANNOTATION]).get(
                    "leader_term"
                )
                == json.loads(body["metadata"]["annotations"][STATE_ANNOTATION]).get(
                    "leader_term"
                )
            )
            if fault == "timeout":
                raise KubeError("injected timeout")
            if fault == "conflict":
                raise KubeConflict("injected conflict")
            if fault == "landed":
                self._as_writer(
                    lambda: self._shared.replace_lease(name, body, timeout_s=timeout_s)
                )
                raise KubeError("injected timeout (landed)")
            if fault == "defer" and same_term:
                self.deferred.append((name, copy.deepcopy(body)))
                raise KubeError("injected timeout (still in flight)")
        return self._as_writer(
            lambda: self._shared.replace_lease(name, body, timeout_s=timeout_s)
        )

    def land_deferred(self) -> None:
        while self.deferred:
            name, body = self.deferred.pop(0)
            self._shared.late = True
            try:
                self._as_writer(lambda: self._shared.replace_lease(name, body))
            except KubeConflict:
                pass
            finally:
                self._shared.late = False


class Harness:
    def __init__(self, seed: int, *, advisory_only: bool = False) -> None:
        self.rng = random.Random(seed)
        self.advisory_only = advisory_only
        self.world = World(CheckedKube())
        self.world.kube.now = self.world.clock.time
        self.processes: dict[str, Process] = {}
        self.spawned = 0
        self.steps = 0
        self.transients: dict[str, int] = {}
        # Monotonic time until which two processes may both believe they
        # lead (the lease was taken over without expiring).
        self.stolen_until = -math.inf
        self.launched = 0
        self.decided: list[str] = []

    # -- processes --

    async def spawn(self, slot: str) -> None:
        self.spawned += 1
        kube = ProcessKube(self.world.kube)
        advisory = True if self.advisory_only else self.rng.random() < 0.3
        promote = self.rng.random() < 0.5
        process = make_process(
            self.world,
            f"{slot}-{self.spawned}",
            advisory=advisory,
            promote=promote,
            kube=kube,
        )
        kube.identity = process.sup.identity
        await process.sup.start(process.host, background=False)
        self.processes[slot] = process

    def live(self) -> list[Process]:
        return [p for p in self.processes.values() if p.alive]

    # -- the real controllers --

    async def tick(self, process: Process) -> None:
        controller = process.controller
        if controller is None:
            return
        had = controller._pending
        result = await controller.on_tick(self.world.clock.time(), SPEC_COUNTS)
        if controller._pending is not None and controller._pending is not had:
            self.launched += 1
        for event in result.audit_events:
            if event.startswith("retune:dropped w="):
                controller.dropped.append(int(event.split("=", 1)[1]))
        self.decided.extend(decision.outcome for decision in result.decisions)
        process.sup.on_effects(
            SimpleNamespace(
                retune_decisions=result.decisions,
                router_parameters=result.router_parameters,
            )
        )

    def finish_sweep(self, process: Process) -> None:
        """The process's sweep returns: failed, flat (kept) or with a best
        candidate (applied, or advisory)."""

        controller = process.controller
        if controller is None:
            return
        roll = self.rng.random()
        if roll < 0.15:
            self.world.sweeps.finish(controller, failed)
            return
        best = None
        if roll >= 0.35:
            best = {key: self.rng.choice(values) for key, values in DOMAIN.items()}
        self.world.sweeps.finish(controller, scored(best))

    def quiesce(self) -> None:
        controllers = [p.controller for p in self.live() if p.controller is not None]
        self.world.sweeps.quiesce(controllers)

    # -- events --

    def _restore_transients(self) -> None:
        """Faults of the frontend and the spec last a few steps, then heal."""

        world, now = self.world, self.steps
        for name, attribute, healthy in (
            ("frontend_down", "down", False),
            ("spec_broken", "spec_ok", True),
            ("p4_broken", "p4_ok", True),
        ):
            until = self.transients.get(name)
            if until is not None and now >= until:
                target = world.frontend if attribute == "down" else world.spec
                setattr(target, attribute, healthy)
                del self.transients[name]

    def _transient(self, name: str, target: Any, attribute: str, broken: Any) -> None:
        if name not in self.transients:
            setattr(target, attribute, broken)
            self.transients[name] = self.steps + self.rng.randint(2, 25)

    async def step(self) -> str:
        rng = self.rng
        self.steps += 1
        self._restore_transients()
        slot = rng.choice(["p1", "p2"])
        process = self.processes.get(slot)
        if process is None or not process.alive:
            if rng.random() < 0.5:
                await self.spawn(slot)
                return f"spawn {slot}"
            slot = "p2" if slot == "p1" else "p1"
            process = self.processes.get(slot)
            if process is None or not process.alive:
                return "idle"
        sup = process.sup
        roll = rng.random()
        if roll < 0.14:
            await sup.lease_round()
            return f"lease {slot}"
        if roll < 0.28:
            await sup.control_round()
            return f"control {slot}"
        if roll < 0.38:
            await sup.outbox_round()
            return f"outbox {slot}"
        if roll < 0.56:
            await self.tick(process)
            return f"tick {slot}"
        if roll < 0.64:
            self.finish_sweep(process)
            return f"sweep_done {slot}"
        if roll < 0.78:
            await self.advance(rng.choice([5, 20, 20, 40, 60, 120, 240]))
            return "advance"
        if roll < 0.82:
            kind = rng.choice(["timeout", "conflict", "landed", "defer"])
            process.kube.state_faults.extend([kind] * rng.choice([1, 1, 2, 3]))
            return f"fault {kind} {slot}"
        if roll < 0.84:
            process.kube.land_deferred()
            return f"land {slot}"
        if roll < 0.855:
            process.kube.lease_get_failures += rng.choice([1, 2, 4])
            return f"renew_fail {slot}"
        if roll < 0.87:
            self.world.frontend.restart(VARIANTS[self.world.spec.variant])
            return "frontend_restart"
        if roll < 0.89:
            self.world.frontend.manual_patch(TEMP, rng.choice([0.3, 0.7]))
            return "manual_patch"
        if roll < 0.905:
            self._transient("frontend_down", self.world.frontend, "down", True)
            return "frontend_down"
        if roll < 0.915:
            spec = self.world.spec
            spec.variant = "b" if spec.variant == "a" else "a"
            self.world.frontend.restart(VARIANTS[spec.variant])
            return "startup_change"
        if roll < 0.93:
            self._transient("spec_broken", self.world.spec, "spec_ok", False)
            return "spec_break"
        if roll < 0.937:
            self._transient("p4_broken", self.world.spec, "p4_ok", False)
            return "p4_break"
        if roll < 0.945:
            # SIGTERM: the supervisor stops while the engine is up, then the
            # engine closes the controller.
            await sup.stop()
            process.host.close()
            process.alive = False
            return f"sigterm {slot}"
        if roll < 0.951:
            process.alive = False
            return f"sigkill {slot}"
        if roll < 0.957:
            self.steal()
            return "steal"
        return "noop"

    def steal(self) -> None:
        """The leadership Lease changes hands without expiring (an admin edit,
        or a contender with a skewed clock): to the other live process or
        to an intruder that never renews."""

        lease = self.world.kube.leases.get(LEADER_NAME)
        if lease is None:
            return
        holder = lease["spec"].get("holderIdentity")
        others = [p.sup.identity for p in self.live() if p.sup.identity != holder]
        lease["spec"]["holderIdentity"] = (
            self.rng.choice(others)
            if others and self.rng.random() < 0.7
            else "intruder/0"
        )
        lease["spec"]["leaseTransitions"] = lease["spec"].get("leaseTransitions", 0) + 1
        self.world.kube.bump(lease)
        self.stolen_until = self.world.clock.monotonic() + LEASE

    async def advance(self, seconds: float) -> None:
        """Time passes in chunks of at most 20s while the background loops
        mostly keep running (each may miss a chunk)."""

        rng = self.rng
        remaining = seconds
        while remaining > 0:
            chunk = min(20.0, remaining)
            remaining -= chunk
            self.quiesce()
            self.world.clock.advance(chunk)
            self._restore_transients()
            for process in self.live():
                sup = process.sup
                if rng.random() < 0.85:
                    await sup.lease_round()
                if rng.random() < 0.6:
                    await sup.outbox_round()
                if rng.random() < 0.6:
                    await sup.control_round()
                if rng.random() < 0.3:
                    self.finish_sweep(process)
                if rng.random() < 0.6:
                    await self.tick(process)
                self.quiesce()
                self.check("advance")

    # -- invariants --

    def check(self, event: str) -> None:
        world = self.world
        live = self.live()
        leaders = [p.name for p in live if p.sup.is_leader()]
        if world.clock.monotonic() >= self.stolen_until:
            assert len(leaders) <= 1, (event, leaders)
        shared = world.shared_state()
        if shared is not None and shared.rollback_pending:
            # A pending rollback always records the startup (or nothing).
            assert shared.by_wt in (VARIANTS["a"], VARIANTS["b"], {}), (
                event,
                shared.by_wt,
            )
        for process in live:
            sup, controller = process.sup, process.controller
            # No window is wasted: a dropped sweep or decision leaves its
            # window launchable.
            if controller is not None:
                decided = (
                    -1
                    if shared is None or shared.last_window_end_ms is None
                    else window_index(
                        shared.last_window_end_ms, phase_s=0.0, interval_s=INTERVAL
                    )
                )
                for k in controller.dropped:
                    assert (
                        controller._last_k < k
                        or controller._pending is not None
                        or k <= decided
                    ), (event, process.name, k, controller._last_k)
                controller.dropped.clear()
            # An advisory planner never PATCHes, and resets only to finish a
            # rollback of what was deployed.
            for writer, advisory, kind, mode, _ in world.frontend.writes:
                if advisory:
                    assert kind == "reset" and mode == "rollback", (event, writer, kind)
            if not (sup.is_leader() and sup.initialized and sup.outbox_size == 0):
                continue
            snapshot = sup._snapshot
            if snapshot is None or shared is None:
                continue
            lease = world.kube.leases[STATE_NAME]
            if snapshot.resource_version != lease["metadata"]["resourceVersion"]:
                continue  # a late write landed; the next write re-reads
            assert shared.leader_term == sup.term, event
            # State, controller point and target agree.
            if (
                shared.mode == "active"
                and not shared.rollback_pending
                and sup.spec_ok
                and controller is not None
            ):
                assert controller.current_point().router == shared.flat, (
                    event,
                    controller.current_point().router,
                    shared.flat,
                )
            target = sup.target()
            if shared.rollback_pending:
                assert target.mode == "rollback", event
            elif process.advisory:
                assert target.mode == "observe", event
            elif not sup.spec_ok:
                assert target.mode == "rollback", event
            else:
                assert target.desired_by_wt == shared.by_wt, event

    async def settle(self) -> None:
        world = self.world
        world.frontend.down = False
        world.spec.spec_ok = True
        world.spec.p4_ok = True
        self.transients.clear()
        for process in self.live():
            process.kube.state_faults.clear()
            process.kube.lease_get_failures = 0
            process.kube.deferred.clear()
        if not self.live():
            await self.spawn("p1")
        live = self.live()
        for _ in range(12):
            self.quiesce()
            world.clock.advance(20)
            for process in live:
                await process.sup.lease_round()
                await process.sup.outbox_round()
                await process.sup.control_round()
                self.check("settle")
        writes = len(world.frontend.writes)
        for _ in range(4):
            world.clock.advance(20)
            for process in live:
                await process.sup.lease_round()
                await process.sup.outbox_round()
                await process.sup.control_round()
                self.check("settle")
        # No infinite reset or PATCH.
        assert len(world.frontend.writes) == writes, world.frontend.writes[writes:]
        leader = next((p for p in live if p.sup.is_leader()), None)
        assert leader is not None and leader.sup.initialized
        state = world.shared_state()
        # Two-phase rollbacks complete once converged.
        assert not state.rollback_pending
        target = leader.sup.target()
        configs = frontend_configs(world)
        if target.mode == "apply":
            assert configs == [state.by_wt["aggregated"]]
        elif target.mode == "rollback":
            assert configs == [VARIANTS[world.spec.variant]["aggregated"]]

    async def run(self, steps: int) -> None:
        await self.spawn("p1")
        await self.spawn("p2")
        for _ in range(steps):
            event = await self.step()
            self.quiesce()
            self.check(event)
        await self.settle()


@pytest.mark.timeout(120)
@pytest.mark.parametrize("seed", range(24))
async def test_interleavings_keep_the_invariants(seed: int) -> None:
    harness = Harness(seed)
    await harness.run(steps=300)
    assert harness.world.kube.state_writes > 0


@pytest.mark.timeout(120)
@pytest.mark.parametrize("seed", range(6))
async def test_advisory_only_never_writes_to_routers(seed: int) -> None:
    harness = Harness(1000 + seed, advisory_only=True)
    await harness.run(steps=250)
    assert harness.world.frontend.writes == []


async def test_the_interleavings_launch_and_decide_windows() -> None:
    # The schedule reaches the real controller's launch and harvest paths.
    harness = Harness(0)
    await harness.run(steps=300)
    assert harness.launched > 0
    assert {"applied", "kept", "failed"} <= set(harness.decided)
