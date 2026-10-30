# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``LiveRetuneSupervisor``: leadership, persisted state and actuation (design §4.7).

The retune controller decides exactly as in replay; the supervisor decides
when it may launch and harvest, persists every decision before anything is
applied, and owns the routers' parameters.

Hooks the controller calls on the planner's event loop:

- ``launch_gate(k, window)`` before both the launch and the busy branch:
  ``not_leader``/``not_ready`` keep the window due, ``already_decided`` skips
  a window the state already records, otherwise it hands out the DGD spec
  replicas and a fresh abort event.
- ``harvest_allowed(generation)``: leader, initialized and the sweep's
  controller generation is current.
- ``sweep_fn(request, *, cancel, abort)`` on the sweep thread: binds a
  ``DgdrContext`` and runs ``dgdr.run_dgdr_sweep``; never raises.

The host calls ``on_effects`` after every engine tick; it enqueues the tick's
decisions in the outbox bound to (term_seq, generation, fingerprint).

Background tasks (all catch every exception; every Kubernetes call runs in a
thread with a timeout):

- lease: renew or acquire ``<dgd>-planner-retune`` every third of the lease
  duration. Becoming leader, re-acquiring after losing track, or a re-init
  request enters initialization (§4.7.3) with ``term_seq + 1``; losing
  leadership runs the local loss (§4.7.5).
- control: re-validate the spec premises (§4.7.7, rebuilding the controller
  through ``host.replace_live_retune`` when the startup, ``spec_ok`` or P4
  change), then, as an initialized leader, the P11 canary and one actuator
  round (§4.8), and it starts one reconciler round (§4.4.5).
- reconciler: one round at a time in its own task, never awaited by the
  control loop, within ``reconcile_budget_s``. A round still running (a hung
  filesystem or slow API calls) skips the next ones; the reconciler stops
  before its next delete once this process is no longer the initialized
  leader of the round's epoch or the budget has passed.
- outbox: write each decision with the term and resourceVersion
  preconditions (§4.7.4); N consecutive failures re-initialize.

Local leadership loss, a failed outbox, and a controller rebuild share one
path: clear ``initialized``, stop the outbox writer and the in-flight
actuator/reconciler round, clear the outbox, abort the sweep and drop it
(``drop_pending``), and rewind the controller to the last persisted window,
so a due window can be relaunched or adopted before its deadline.

The actuation target follows the state: ``rollback`` while
``rollback_pending`` (an advisory planner resets only routers whose initial
it latched, §4.8); ``observe`` (never write) in advisory; ``rollback`` when a
spec premise fails; otherwise ``apply`` the state's ``by_wt``.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import inspect
import logging
import math
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol, Union

from dynamo.planner.retune.live.contracts import (
    LABEL_DGD_UID,
    ActuationReport,
    ActuationTarget,
    DgdRef,
    FrontendStartup,
    GateReason,
    GateResult,
    KubeConflict,
    KubeError,
    KubeNotFound,
    NullMetrics,
    PremiseReport,
    RetuneKube,
    RetuneMetricsSink,
    SweepRegistry,
)
from dynamo.planner.retune.live.state import (
    LeaseLeadership,
    RetuneMode,
    RetuneState,
    StateSnapshot,
    StateStore,
    append_history,
    claim,
    deploy,
    deployed_off,
    has_decision,
    maps_equal,
    merge_initials,
    record_decision,
    reset,
    rollback_complete,
    rollback_start,
)

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.core.types import PlannerEffects, RetuneDecision, RetuneInputs
    from dynamo.planner.retune.sweep import SweepRequest, SweepResult

logger = logging.getLogger(__name__)

_EPSILON = 1e-9
# Claim-write attempts on resourceVersion conflicts before giving up the
# initialization (the next lease round retries with term_seq + 1).
_CLAIM_ATTEMPTS = 3
# Conflict retries of one state write.
_WRITE_ATTEMPTS = 3
# Remembered (k, reason) pairs of skipped launches (the metric counts each
# pair once).
_SKIP_MEMORY = 64

BlockingRunner = Callable[[Callable[[], Any]], Awaitable[Any]]


async def _to_thread(call: Callable[[], Any]) -> Any:
    return await asyncio.to_thread(call)


# ---------------------------------------------------------------- types


@dataclass(frozen=True)
class SpecEvaluation:
    """One evaluation of the DGD spec (frontend parse and spec premises).

    ``inputs`` is the live ``RetuneInputs``, ``None`` iff P4 fails.
    ``spec_replicas`` maps worker type to the DGD spec replicas (``None``
    when the workers cannot be resolved). ``dgd`` carries the DGD's current
    uid and the routers' dynamo namespace.
    """

    report: PremiseReport
    startup: Optional[FrontendStartup]
    inputs: Optional[RetuneInputs]
    spec_replicas: Optional[dict[str, int]]
    dgd: DgdRef
    generation: Optional[int] = None


# DGD object -> evaluation; may raise (the round is then skipped).
PremiseEvaluator = Callable[[dict], SpecEvaluation]


class RouterResolver(Protocol):
    """Resolves router points against the parsed startup ``KvRouterConfig``."""

    def startup_flat(self, keys: Sequence[str]) -> dict[str, float]:
        """The startup value of each searched key (the controller's start)."""
        ...

    def by_wt(self, flat: Mapping[str, float]) -> dict[str, dict[str, float]]:
        """``policy_parameters(wt)`` of ``startup ⊕ flat`` per worker type."""
        ...


ResolverFactory = Callable[[Optional[FrontendStartup]], Optional[RouterResolver]]


class ActuatorLike(Protocol):
    async def reconcile(
        self,
        target: ActuationTarget,
        *,
        initials: Mapping[str, Mapping[str, float]],
        expected_frontends: int,
    ) -> ActuationReport:
        ...

    def set_startup_maps(self, startup_maps: Mapping[str, Mapping[str, float]]) -> None:
        ...


# (DgdRef, startup maps per worker type) -> actuator
ActuatorFactory = Callable[[DgdRef, dict[str, dict[str, float]]], ActuatorLike]
SweepRunner = Callable[..., "SweepResult"]


class LiveRetuneHost(Protocol):
    """What the supervisor needs from ``NativePlannerBase``.

    ``replace_live_retune`` takes the planner's config lock and calls the
    engine adapter's ``replace_retune``. The supervisor reaches the current
    controller through ``host._engine.retune_controller`` and serializes its
    calls on it with ``host._config_lock``.
    """

    async def replace_live_retune(
        self,
        inputs: Optional[RetuneInputs],
        *,
        generation: int,
        last_decided_k: Optional[int],
    ) -> None:
        ...


@dataclass(frozen=True)
class SupervisorSettings:
    """Timing and policy of one supervisor (from ``PlannerConfig``)."""

    interval_s: float
    grace_s: float
    timeout_s: float
    phase_s: float = 0.0
    advisory: bool = False
    promote_last_advisory: bool = False
    worker_types: tuple[str, ...] = ("aggregated",)
    # Searched router keys (live names); the controller's router_keys.
    router_keys: tuple[str, ...] = ()
    lease_duration_s: float = 60.0
    actuator_interval_s: float = 15.0
    k8s_timeout_s: float = 5.0
    write_failure_limit: int = 3
    canary_interval_s: float = 600.0
    stop_timeout_s: float = 10.0
    # How long one reconciler round may run before the reconciler is told to
    # stop (a blocked filesystem call cannot be interrupted; later rounds
    # wait for it instead of piling up threads).
    reconcile_budget_s: float = 120.0
    # Legacy clean-up (§4.11): roll back what an earlier retune deployed,
    # then delete the Requests and both Leases.
    decommission: bool = False

    def __post_init__(self) -> None:
        for name in (
            "interval_s",
            "lease_duration_s",
            "actuator_interval_s",
            "reconcile_budget_s",
        ):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive and finite")
        if self.write_failure_limit < 1:
            raise ValueError("write_failure_limit must be >= 1")

    @property
    def renew_interval_s(self) -> float:
        return self.lease_duration_s / 3.0

    @property
    def io_timeout_s(self) -> float:
        """Backstop on one blocking call; the client applies its own."""

        return 2.0 * self.k8s_timeout_s + 1.0

    @classmethod
    def from_config(
        cls, config: PlannerConfig, *, router_keys: Sequence[str] = ()
    ) -> SupervisorSettings:
        retune = config.retune
        if retune is None or retune.dgdr is None:
            raise ValueError("live retune requires retune.executor 'dgdr'")
        dgdr = retune.dgdr
        return cls(
            interval_s=float(retune.interval_seconds),
            grace_s=float(retune.grace_seconds),
            timeout_s=float(retune.effective_timeout_seconds),
            phase_s=float(dgdr.phase_seconds),
            advisory=bool(config.advisory),
            promote_last_advisory=bool(dgdr.promote_last_advisory),
            worker_types=worker_types_for(config.mode),
            router_keys=tuple(sorted(router_keys)),
            lease_duration_s=float(dgdr.lease_duration_seconds),
            actuator_interval_s=float(dgdr.actuator_interval_seconds),
        )


def worker_types_for(planner_mode: str) -> tuple[str, ...]:
    return ("prefill", "decode") if planner_mode == "disagg" else ("aggregated",)


def due_window_end_ms(
    now_s: float, *, phase_s: float, interval_s: float, grace_s: float
) -> int:
    """End of the latest window whose launch time is at or before ``now_s``.

    Same arithmetic as ``RetuneController._due_window`` with ``s0 = phase``.
    """

    k = math.floor((now_s - phase_s - grace_s) / interval_s + _EPSILON) - 1
    return int(round((phase_s + (k + 1) * interval_s) * 1000.0))


def current_window_end_ms(now_s: float, *, phase_s: float, interval_s: float) -> int:
    """End of the window in progress at ``now_s`` (it contains ``now_s``).

    It ends after the due window (``due_window_end_ms``) and after every
    window that ended but is not due yet (within the grace).
    """

    k = math.floor((now_s - phase_s) / interval_s + _EPSILON)
    return int(round((phase_s + (k + 1) * interval_s) * 1000.0))


def window_index(window_end_ms: int, *, phase_s: float, interval_s: float) -> int:
    """The controller's ``k`` of the window ending at ``window_end_ms``."""

    return int(round((window_end_ms / 1000.0 - phase_s) / interval_s)) - 1


# ----------------------------------------------------- initialization (§4.7.3)

InitRow = Literal[
    "spec_not_ok",
    "advisory_rollback",
    "advisory_observe",
    "active_restore",
    "active_reset",
    "promote",
]


@dataclass(frozen=True)
class InitPlan:
    """What initialization writes and restores.

    ``state`` is the snapshot to persist (``write`` says whether it differs
    from the claimed one); ``restore_flat`` is passed to the controller's
    ``restore_desired`` (``None``: leave the controller alone).
    """

    row: InitRow
    state: RetuneState
    write: bool
    restore_flat: Optional[dict[str, float]]


def plan_initialization(
    state: RetuneState,
    *,
    spec_ok: bool,
    advisory: bool,
    fingerprint: str,
    startup_flat: Optional[Mapping[str, float]],
    startup_by_wt: Optional[Mapping[str, Mapping[str, float]]],
    router_keys: Sequence[str],
    promote: bool,
    now_ms: int,
    interval_ms: int,
    first_window_end_ms: int,
) -> InitPlan:
    """The initialization table of design §4.7.3, first matching row wins.

    ``state`` is the snapshot after the claim write. A state without a
    deployment record is "no state". The first initialization of a DGD also
    persists ``last_window_end_ms = first_window_end_ms`` (the end of the
    window in progress now): neither the due window nor the window whose
    trace began mid-way (tracing enabled with the planner) is ever swept,
    even after a restart.
    """

    mode: RetuneMode = "advisory" if advisory else "active"
    off = deployed_off(state, startup_by_wt)
    flat0 = {key: float(value) for key, value in (startup_flat or {}).items()}
    by0 = {wt: dict(params) for wt, params in (startup_by_wt or {}).items()}
    stale = (
        not state.has_deployment
        or state.mode != mode
        or state.fingerprint != fingerprint
    )
    restore: Optional[dict[str, float]] = None

    if not spec_ok:
        row: InitRow = "spec_not_ok"
        if off:
            new = rollback_start(
                state, fingerprint=fingerprint, mode=mode, flat=flat0, by_wt=by0
            )
        elif stale:
            new = reset(
                state,
                fingerprint=fingerprint,
                mode=mode,
                flat=flat0,
                by_wt=by0,
                rollback_pending=False,
            )
        else:
            new = state
    elif advisory:
        if off:
            row = "advisory_rollback"
            new = rollback_start(
                state, fingerprint=fingerprint, mode=mode, flat=flat0, by_wt=by0
            )
        else:
            row = "advisory_observe"
            new = (
                reset(
                    state,
                    fingerprint=fingerprint,
                    mode=mode,
                    flat=flat0,
                    by_wt=by0,
                    rollback_pending=False,
                )
                if stale
                else state
            )
    elif (
        not stale
        and not state.rollback_pending
        and set(state.flat) == set(router_keys)
        and state.by_wt
    ):
        row = "active_restore"
        new = state
        restore = dict(state.flat)
    else:
        entry = (
            _promotable(state, fingerprint, router_keys, now_ms, interval_ms)
            if promote
            else None
        )
        if entry is not None:
            row = "promote"
            flat = {key: float(value) for key, value in entry["flat"].items()}
            by_wt = {wt: dict(params) for wt, params in entry["by_wt"].items()}
            new = deploy(
                state, fingerprint=fingerprint, mode="active", flat=flat, by_wt=by_wt
            )
            new = append_history(
                new,
                {
                    "decision_id": f"promote-{entry.get('decision_id')}",
                    "outcome": "promoted",
                    "window_end_ms": entry.get("window_end_ms"),
                    "fingerprint": fingerprint,
                    "flat": flat,
                    "by_wt": by_wt,
                    "deploy_seq": new.deploy_seq,
                    "at_ms": now_ms,
                },
            )
            restore = flat
        else:
            row = "active_reset"
            new = reset(
                state,
                fingerprint=fingerprint,
                mode="active",
                flat=flat0,
                by_wt=by0,
                rollback_pending=off,
            )
            restore = flat0
    if new.last_window_end_ms is None:
        new = dataclasses.replace(new, last_window_end_ms=int(first_window_end_ms))
    return InitPlan(row=row, state=new, write=new != state, restore_flat=restore)


def _promotable(
    state: RetuneState,
    fingerprint: str,
    router_keys: Sequence[str],
    now_ms: int,
    interval_ms: int,
) -> Optional[Mapping[str, Any]]:
    """The last advisory recommendation, if §4.7.6 lets it be promoted."""

    if not state.history:
        return None
    entry = state.history[-1]
    if entry.get("outcome") != "advisory" or entry.get("fingerprint") != fingerprint:
        return None
    window_end_ms = entry.get("window_end_ms")
    if (
        not isinstance(window_end_ms, (int, float))
        or now_ms - window_end_ms > interval_ms
    ):
        return None
    flat = entry.get("flat")
    by_wt = entry.get("by_wt")
    if not isinstance(flat, Mapping) or set(flat) != set(router_keys):
        return None
    if not isinstance(by_wt, Mapping) or not by_wt:
        return None
    return entry


# ------------------------------------------------------------------ outbox


@dataclass(frozen=True)
class OutboxEntry:
    """One decision waiting to be persisted, bound to the term that made it."""

    decision: RetuneDecision
    router_parameters: Optional[dict[str, dict[str, float]]]
    term_seq: int
    generation: int
    fingerprint: str
    epoch: int


WriteOutcome = Literal["ok", "already", "discarded", "lost_term", "error"]
_ALREADY = object()
Mutation = Callable[[RetuneState], Union[RetuneState, None, object]]


class _NullRegistry:
    def protect(self, window_end_ms: int, request_names: Any) -> None:
        pass

    def release(self, window_end_ms: int) -> None:
        pass

    def enqueue_delete(self, request_names: Any) -> None:
        pass


# -------------------------------------------------------------- supervisor


class LiveRetuneSupervisor:
    """Leadership, state, outbox, actuation and clean-up of live retune.

    Collaborators are injected so tests never need a cluster: ``kube`` (a
    ``RetuneKube``), ``premise_evaluator`` (DGD -> ``SpecEvaluation``),
    ``actuator_factory`` (a ``RouterActuator``), ``reconciler`` (a
    ``Reconciler``, also the sweep registry), ``canary`` (P11),
    ``sweep_runner``/``context_factory`` (``dgdr.run_dgdr_sweep`` and its
    ``DgdrContext``), the clocks, ``sleep`` and ``blocking`` (how a blocking
    call runs off the event loop).
    """

    def __init__(
        self,
        *,
        settings: SupervisorSettings,
        kube: RetuneKube,
        evaluation: SpecEvaluation,
        identity: str,
        premise_evaluator: PremiseEvaluator,
        actuator_factory: ActuatorFactory,
        fingerprint_fn: Callable[[Optional[FrontendStartup]], str],
        resolver_factory: Optional[ResolverFactory] = None,
        reconciler: Optional[Any] = None,
        metrics: Optional[RetuneMetricsSink] = None,
        canary: Optional[Callable[[LiveRetuneSupervisor], bool]] = None,
        sweep_runner: Optional[SweepRunner] = None,
        context_factory: Optional[Callable[[LiveRetuneSupervisor], Any]] = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        blocking: Optional[BlockingRunner] = None,
        leadership: Optional[LeaseLeadership] = None,
        store: Optional[StateStore] = None,
    ) -> None:
        self._settings = settings
        self._kube = kube
        self._identity = identity
        self._premise_evaluator = premise_evaluator
        self._actuator_factory = actuator_factory
        self._fingerprint_fn = fingerprint_fn
        self._resolver_factory = resolver_factory
        self._reconciler = reconciler
        self._metrics: RetuneMetricsSink = (
            metrics if metrics is not None else NullMetrics()
        )
        self._canary = canary
        self._sweep_runner = sweep_runner
        self._context_factory = context_factory
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._blocking: BlockingRunner = (
            blocking if blocking is not None else _to_thread
        )

        self._dgd = evaluation.dgd
        self._leadership = leadership or LeaseLeadership(
            kube,
            evaluation.dgd,
            identity=identity,
            lease_duration_s=settings.lease_duration_s,
            margin_s=3.0 * settings.k8s_timeout_s,
            timeout_s=settings.k8s_timeout_s,
            monotonic=monotonic,
            clock=clock,
        )
        self._store = store or StateStore(
            kube, evaluation.dgd, timeout_s=settings.k8s_timeout_s, clock=clock
        )

        # Spec evaluation and what is derived from it.
        self._evaluation = evaluation
        self._p14_ok: Optional[bool] = None
        self._spec_ok = bool(evaluation.report.spec_ok)
        self._fingerprint = fingerprint_fn(evaluation.startup)
        self._resolver = self._make_resolver(evaluation.startup)
        self._actuator = actuator_factory(
            evaluation.dgd, self._actuator_maps(evaluation.startup)
        )
        self._log_report(evaluation.report)

        # Leadership, term and the cached state snapshot.
        self._term_seq = 0
        self._term: Optional[str] = None
        self._snapshot: Optional[StateSnapshot] = None
        self._initialized = False
        self._reinit_requested = False
        # Bumped by every local loss: results of work started before it are
        # discarded.
        self._epoch = 0

        # Controller.
        self._host: Optional[Any] = None
        self._generation = 0
        self._controller_ok = evaluation.inputs is not None

        # Outbox.
        self._outbox: deque[OutboxEntry] = deque()
        self._write_failures = 0

        # Health (P11, P12) and actuation.
        self._canary_ok: Optional[bool] = None
        self._canary_at: Optional[float] = None
        self._last_report: Optional[ActuationReport] = None
        # (deploy_seq, monotonic since) of a converged rollback.
        self._rollback_since: Optional[tuple[int, float]] = None

        # Sweeps.
        self._aborts: list[threading.Event] = []
        self._aborts_lock = threading.Lock()
        self._sweeps_in_flight = 0
        self._skips_seen: deque[tuple[int, str]] = deque(maxlen=_SKIP_MEMORY)

        # Tasks and locks.
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._tasks: list[asyncio.Task[None]] = []
        self._outbox_task: Optional[asyncio.Task[None]] = None
        self._round_task: Optional[asyncio.Task[None]] = None
        # The reconciler round's task (it waits up to the budget) and its
        # work (the reconciler call itself, done only once it has returned).
        self._reconcile_task: Optional[asyncio.Task[None]] = None
        self._reconcile_work: Optional[asyncio.Future[Any]] = None
        self._lease_wake = asyncio.Event()
        self._outbox_wake = asyncio.Event()
        self._background = False
        self._stopping = False
        self._stopped = False
        self.decommissioned = False

    # ------------------------------------------------------------ queries

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def term(self) -> Optional[str]:
        return self._term

    @property
    def term_seq(self) -> int:
        return self._term_seq

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def state(self) -> Optional[RetuneState]:
        return None if self._snapshot is None else self._snapshot.state

    @property
    def spec_ok(self) -> bool:
        return self._spec_ok

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def evaluation(self) -> SpecEvaluation:
        return self._evaluation

    @property
    def dgd(self) -> DgdRef:
        return self._dgd

    @property
    def kube(self) -> RetuneKube:
        return self._kube

    @property
    def registry(self) -> SweepRegistry:
        reconciler = self._reconciler
        if reconciler is not None and hasattr(reconciler, "protect"):
            return reconciler
        return _NullRegistry()

    @property
    def actuator(self) -> ActuatorLike:
        return self._actuator

    @property
    def outbox_size(self) -> int:
        return len(self._outbox)

    @property
    def last_report(self) -> Optional[ActuationReport]:
        return self._last_report

    @property
    def reconcile_task(self) -> Optional[asyncio.Task[None]]:
        """The latest reconciler round's task (done once the round returned
        or its budget passed); tests await it after ``control_round``."""

        return self._reconcile_task

    def is_leader(self) -> bool:
        """Thread-safe; ``False`` once ``leader_until`` has passed."""

        return self._leadership.is_leader()

    def ready(self) -> bool:
        """Whether a due window may launch (spec, controller and health)."""

        return (
            self._spec_ok
            and self._controller_ok
            and self._evaluation.spec_replicas is not None
            and self._canary_ok is True
            and self._p12_ok()
            and not self._settings.decommission
        )

    def retune_inputs(self) -> Optional[RetuneInputs]:
        """Inputs for the engine's first controller; ``None`` if P4 fails."""

        return self._evaluation.inputs

    def hooks(self) -> Any:
        """``RetuneHooks`` for ``OrchestratorEngineAdapter``."""

        from dynamo.planner.plugins.orchestrator.engine_adapter import RetuneHooks

        return RetuneHooks(
            sweep_fn=self.sweep_fn,
            launch_gate=self.launch_gate,
            harvest_allowed=self.harvest_allowed,
            phase_s=self._settings.phase_s,
        )

    def target(self) -> Optional[ActuationTarget]:
        """What the actuator converges to now (``None``: nothing)."""

        state = self.state
        if not self._initialized or state is None:
            return None
        if state.rollback_pending:
            # §4.8: only an active planner resets a router it never latched
            # (revision > 0) to learn its initial; an advisory planner leaves
            # it alone.
            return ActuationTarget(
                mode="rollback",
                include_not_ready=True,
                learn_unknown=not (
                    self._settings.advisory or self._settings.decommission
                ),
            )
        if self._settings.advisory or self._settings.decommission:
            return ActuationTarget(mode="observe")
        if not self._spec_ok:
            return ActuationTarget(mode="rollback", include_not_ready=True)
        if not state.by_wt:
            return ActuationTarget(mode="observe")
        return ActuationTarget(
            mode="apply",
            desired_by_wt={wt: dict(params) for wt, params in state.by_wt.items()},
        )

    # ----------------------------------------------------- controller hooks

    def launch_gate(self, k: int, window: tuple[float, float, float]) -> GateResult:
        """Whether window ``k`` may launch now; see the module docstring."""

        reason: Optional[GateReason]
        _, end_s, _ = window
        state = self.state
        if not self.is_leader():
            reason = "not_leader"
        elif not self._initialized or state is None or not self.ready():
            reason = "not_ready"
        elif (
            state.last_window_end_ms is not None
            and int(round(end_s * 1000.0)) <= state.last_window_end_ms
        ):
            reason = "already_decided"
        else:
            reason = None
        if reason is not None:
            if (k, reason) not in self._skips_seen:
                self._skips_seen.append((k, reason))
                self._metrics.skipped(reason)
                logger.info("retune gate: window %d %s", k, reason)
            return GateResult(reason=reason)
        abort = threading.Event()
        with self._aborts_lock:
            self._aborts = [event for event in self._aborts if not event.is_set()][-8:]
            self._aborts.append(abort)
        return GateResult(
            reason=None,
            spec_replicas=dict(self._evaluation.spec_replicas or {}),
            abort=abort,
        )

    def harvest_allowed(self, generation: int) -> bool:
        """Read-only: leader, initialized, and the sweep's generation is current."""

        return (
            self.is_leader()
            and self._initialized
            and int(generation) == self._generation
        )

    def sweep_fn(
        self,
        request: SweepRequest,
        *,
        cancel: threading.Event,
        abort: Optional[threading.Event] = None,
    ) -> SweepResult:
        """Run one DGDR sweep on the controller's sweep thread; never raises."""

        with self._aborts_lock:
            self._sweeps_in_flight += 1
        try:
            runner = self._sweep_runner
            if runner is None:
                from dynamo.planner.retune.live.dgdr import run_dgdr_sweep

                runner = run_dgdr_sweep
            if self._context_factory is None:
                raise RuntimeError("no DGDR sweep context factory is configured")
            context = self._context_factory(self)
            return runner(request, cancel=cancel, abort=abort, context=context)
        except Exception as error:  # the sweep contract: never raise
            logger.exception("retune sweep of window %s failed", request.window_index)
            from dynamo.planner.retune.sweep import SweepResult

            return SweepResult(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                failure_reason="sweep_error",
            )
        finally:
            with self._aborts_lock:
                self._sweeps_in_flight -= 1

    def on_effects(self, effects: PlannerEffects) -> None:
        """Enqueue the tick's decisions; never raises into the tick."""

        decisions = list(getattr(effects, "retune_decisions", None) or ())
        if not decisions:
            return
        if not (self.is_leader() and self._initialized):
            logger.warning(
                "retune: dropping %d decision(s) %s: not an initialized leader",
                len(decisions),
                [decision.decision_id for decision in decisions],
            )
            self._rewind_now()
            return
        router_parameters = getattr(effects, "router_parameters", None)
        for decision in decisions:
            self._outbox.append(
                OutboxEntry(
                    decision=decision,
                    router_parameters=(
                        None
                        if router_parameters is None
                        else {wt: dict(p) for wt, p in router_parameters.items()}
                    ),
                    term_seq=self._term_seq,
                    generation=self._generation,
                    fingerprint=self._fingerprint,
                    epoch=self._epoch,
                )
            )
        self._outbox_wake.set()

    # ------------------------------------------------------------ lifecycle

    async def start(self, host: Any, *, background: bool = True) -> None:
        """Bind the planner host and start the background tasks.

        With ``background=False`` (tests) nothing runs on its own: drive
        ``lease_round``, ``control_round`` and ``outbox_round`` directly.
        ``control_round`` still starts the reconciler round as a task (it is
        never waited for); await ``reconcile_task`` to finish it.
        """

        self._host = host
        controller = self._controller()
        if controller is not None:
            self._generation = int(getattr(controller, "generation", self._generation))
        self._controller_ok = controller is not None
        self._background = background
        self._publish()
        if not background:
            return
        self._tasks = [
            asyncio.create_task(self._lease_loop(), name="planner-retune-lease"),
            asyncio.create_task(self._control_loop(), name="planner-retune-control"),
        ]

    async def stop(self) -> None:
        """Abort sweeps, wait for them briefly and release the Lease (§4.7.8)."""

        if self._stopped:
            return
        self._stopping = True
        tasks = list(self._tasks)
        self._tasks = []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks)
        try:
            async with self._init_lock:
                await self._local_loss_locked("shutdown")
        except Exception:
            logger.exception("retune supervisor: shutdown clean-up failed")
        self._abort_all()
        deadline = self._monotonic() + self._settings.stop_timeout_s
        while self._sweeps_in_flight > 0 and self._monotonic() < deadline:
            await self._sleep(0.1)
        if self._sweeps_in_flight > 0:
            logger.warning(
                "retune supervisor: %d sweep(s) still running at shutdown",
                self._sweeps_in_flight,
            )
        try:
            await self._io(self._leadership.release)
        except Exception:
            logger.exception("retune supervisor: releasing the lease failed")
        await self._close_actuator(self._actuator)
        self._stopped = True
        self._publish()

    # ------------------------------------------------------------- rounds

    async def lease_round(self) -> None:
        """One renewal/acquisition; initialize or run the local loss."""

        if self._stopping:
            return
        try:
            step = await self._io(self._leadership.step)
        except Exception as error:
            logger.warning("retune lease round failed: %s", error)
            step = "error"
        try:
            if self.is_leader():
                if (
                    step == "acquired"
                    or self._reinit_requested
                    or not self._initialized
                ):
                    async with self._init_lock:
                        if self.is_leader() and not self._stopping:
                            await self._initialize_locked()
            elif self._initialized or self._term is not None:
                async with self._init_lock:
                    if self._term is not None:
                        await self._local_loss_locked(f"not leader ({step})")
                        self._term = None
        except Exception:
            logger.exception("retune lease round: initialization failed")
        self._publish()

    async def control_round(self) -> None:
        """Re-validate the spec; as an initialized leader, actuate and start
        a reconciler round (never waited for)."""

        if self._stopping:
            return
        try:
            await self.revalidate()
        except Exception:
            logger.exception("retune revalidation failed")
        if self._initialized and self.is_leader():
            task = asyncio.create_task(self._leader_round())
            self._round_task = task
            await asyncio.wait([task])
            if self._round_task is task:
                self._round_task = None
            if not task.cancelled() and task.exception() is not None:
                logger.error("retune actuation round failed", exc_info=task.exception())
        self._publish()

    async def outbox_round(self) -> None:
        """Write the queued decisions in order; stop at the first failure."""

        while self._outbox and not self._stopping:
            entry = self._outbox[0]
            if not self._entry_current(entry):
                self._outbox.popleft()
                logger.info(
                    "retune outbox: dropped %s of an earlier term/generation",
                    entry.decision.decision_id,
                )
                continue
            outcome = await self._write_decision(entry)
            if outcome in ("ok", "already", "discarded"):
                if self._outbox and self._outbox[0] is entry:
                    self._outbox.popleft()
                self._write_failures = 0
                continue
            if outcome == "lost_term":
                self._request_reinit("the state's leader_term is not ours")
                return
            # "error": counted as failures_total{reason="state_write"} by
            # _write_state.
            self._write_failures += 1
            logger.warning(
                "retune outbox: writing %s failed (%d/%d)",
                entry.decision.decision_id,
                self._write_failures,
                self._settings.write_failure_limit,
            )
            if self._write_failures >= self._settings.write_failure_limit:
                self._request_reinit("consecutive state write failures")
            return

    async def revalidate(self) -> None:
        """§4.7.7: re-read the DGD and rebuild on a spec-class change."""

        try:
            dgd_obj = await self._io(self._kube.get_dgd, self._dgd.name)
        except KubeNotFound:
            logger.warning("retune: DGD %s not found", self._dgd.name)
            return
        except (KubeError, asyncio.TimeoutError) as error:
            logger.warning("retune: reading DGD %s failed: %s", self._dgd.name, error)
            return
        evaluation = self._premise_evaluator(dgd_obj)
        await self.apply_evaluation(evaluation)

    async def apply_evaluation(self, evaluation: SpecEvaluation) -> None:
        """Adopt an evaluation; rebuild when the startup, ``spec_ok``, P4 or
        the DGD identity changed."""

        old = self._evaluation
        startup_changed = _startup_key(old.startup) != _startup_key(evaluation.startup)
        p14 = None if startup_changed else self._p14_ok
        spec_ok = bool(evaluation.report.spec_ok) and p14 is not False
        reasons = []
        if startup_changed:
            reasons.append("startup changed")
        if spec_ok != self._spec_ok:
            reasons.append(f"spec_ok {self._spec_ok} -> {spec_ok}")
        if bool(old.report.p4_ok) != bool(evaluation.report.p4_ok):
            reasons.append(f"P4 {old.report.p4_ok} -> {evaluation.report.p4_ok}")
        if (old.dgd.name, old.dgd.namespace, old.dgd.uid) != (
            evaluation.dgd.name,
            evaluation.dgd.namespace,
            evaluation.dgd.uid,
        ):
            reasons.append(f"DGD uid {old.dgd.uid} -> {evaluation.dgd.uid}")
        if reasons:
            await self.rebuild(evaluation, "; ".join(reasons))
            return
        # Replicas, warnings or the routers' dynamo namespace (a worker
        # rollout) change without a rebuild.
        self._evaluation = evaluation
        if old.dgd != evaluation.dgd:
            logger.info(
                "retune: routers' dynamo namespace %s -> %s",
                old.dgd.dynamo_namespace,
                evaluation.dgd.dynamo_namespace,
            )
            await self._swap_actuator(evaluation)

    async def rebuild(self, evaluation: SpecEvaluation, reason: str) -> None:
        """Local loss, generation + 1, ``replace_live_retune``, re-initialize."""

        async with self._init_lock:
            logger.info("retune: rebuilding the controller: %s", reason)
            await self._local_loss_locked(f"rebuild: {reason}")
            await self._install_evaluation(evaluation)
            self._generation += 1
            host = self._host
            if host is not None:
                try:
                    await host.replace_live_retune(
                        evaluation.inputs,
                        generation=self._generation,
                        last_decided_k=self._last_decided_k(),
                    )
                    self._controller_ok = evaluation.inputs is not None
                except Exception:
                    logger.exception(
                        "retune: installing controller generation %d failed; not ready",
                        self._generation,
                    )
                    self._controller_ok = False
            else:
                self._controller_ok = evaluation.inputs is not None
            if self.is_leader() and not self._stopping:
                await self._initialize_locked()
        self._publish()

    # ------------------------------------------------- init and local loss

    async def _initialize_locked(self) -> bool:
        """§4.7.3 under ``_init_lock``; ``True`` once initialized."""

        await self._local_loss_locked("initialize")
        # Confirm on the API server right before the claim: a holder change
        # this process has not seen yet (it believes it leads until
        # leader_until) aborts the initialization instead of claiming.
        try:
            await self._io(self._leadership.step)
        except Exception as error:
            logger.warning("retune init: confirming leadership failed: %s", error)
        if not self.is_leader():
            return False
        self._term_seq += 1
        self._reinit_requested = False
        term = f"{self._leadership.transitions}/{self._identity}/{self._term_seq}"
        self._term = term
        epoch = self._epoch
        settings = self._settings
        async with self._write_lock:
            snapshot = await self._read_own_snapshot()
            if snapshot is None:
                return False
            # Claim: only leader_term, conditional on the resourceVersion.
            for attempt in range(_CLAIM_ATTEMPTS):
                if not self.is_leader() or epoch != self._epoch:
                    return False
                claimed = claim(snapshot.state, dgd_uid=self._dgd.uid, term=term)
                try:
                    snapshot = await self._io(self._store.write, claimed, snapshot)
                    break
                except KubeConflict:
                    snapshot = await self._read_own_snapshot()
                    if snapshot is None:
                        return False
                except (KubeError, asyncio.TimeoutError) as error:
                    logger.warning("retune init: claim write failed: %s", error)
                    self._metrics.failure("state_write")
                    return False
            else:
                logger.warning("retune init: claim write kept conflicting")
                self._metrics.failure("state_write")
                return False
            assert snapshot.state is not None
            now_s = self._clock()
            plan = plan_initialization(
                snapshot.state,
                spec_ok=self._spec_ok,
                advisory=settings.advisory or settings.decommission,
                fingerprint=self._fingerprint,
                startup_flat=self._startup_flat(),
                startup_by_wt=self._startup_by_wt(),
                router_keys=self._router_keys(),
                promote=settings.promote_last_advisory,
                now_ms=int(round(now_s * 1000.0)),
                interval_ms=int(round(settings.interval_s * 1000.0)),
                # §4.7.2 first enable: skip the due window and the window in
                # progress, whose trace may have begun mid-way.
                first_window_end_ms=current_window_end_ms(
                    now_s, phase_s=settings.phase_s, interval_s=settings.interval_s
                ),
            )
            if plan.write:
                if not self.is_leader() or epoch != self._epoch:
                    return False
                try:
                    snapshot = await self._io(self._store.write, plan.state, snapshot)
                except (KubeError, asyncio.TimeoutError) as error:
                    logger.warning("retune init: %s write failed: %s", plan.row, error)
                    self._metrics.failure("state_write")
                    return False
            self._snapshot = snapshot
        logger.info(
            "retune init: term %s row %s (mode=%s rollback_pending=%s deploy_seq=%s "
            "last_window_end_ms=%s spec_ok=%s)",
            term,
            plan.row,
            plan.state.mode,
            plan.state.rollback_pending,
            plan.state.deploy_seq,
            plan.state.last_window_end_ms,
            self._spec_ok,
        )
        controller = self._controller()
        async with self._host_lock():
            if plan.restore_flat is not None and controller is not None:
                try:
                    controller.restore_desired(plan.restore_flat)
                except Exception:
                    logger.exception("retune init: restore_desired failed")
                    return False
            if controller is not None:
                self._rewind(controller)
        if epoch != self._epoch or not self.is_leader():
            return False
        self._initialized = True
        self._rollback_since = None
        if self._background:
            self._outbox_task = asyncio.create_task(
                self._outbox_loop(), name="planner-retune-outbox"
            )
        self._outbox_wake.set()
        return True

    async def _read_own_snapshot(self) -> Optional[StateSnapshot]:
        """Read the state; drop a snapshot of another DGD uid (§4.7.3 step 2)."""

        try:
            snapshot = await self._io(self._store.read)
            if snapshot.state is not None and snapshot.state.dgd_uid != self._dgd.uid:
                logger.warning(
                    "retune init: state belongs to DGD uid %s, not %s; recreating it",
                    snapshot.state.dgd_uid,
                    self._dgd.uid,
                )
                await self._io(self._store.delete)
                await self._delete_foreign_requests(snapshot.state.dgd_uid)
                snapshot = StateSnapshot(state=None, resource_version=None, lease=None)
            elif snapshot.corrupt:
                # Replaced by the claim (conditional on its resourceVersion).
                snapshot = dataclasses.replace(snapshot, state=None)
            return snapshot
        except (KubeError, asyncio.TimeoutError) as error:
            logger.warning("retune init: reading the state failed: %s", error)
            return None

    async def _delete_foreign_requests(self, dgd_uid: str) -> None:
        try:
            requests = await self._io(
                self._kube.list_requests, f"{LABEL_DGD_UID}={dgd_uid}"
            )
        except (KubeError, asyncio.TimeoutError) as error:
            logger.warning(
                "retune: listing Requests of DGD uid %s failed: %s", dgd_uid, error
            )
            return
        names = [(request.get("metadata") or {}).get("name") for request in requests]
        self.registry.enqueue_delete([name for name in names if name])

    async def _local_loss_locked(self, reason: str) -> None:
        """§4.7.5 under ``_init_lock``."""

        was = self._initialized or bool(self._outbox) or self._term is not None
        self._initialized = False
        self._epoch += 1
        self._rollback_since = None
        # Health is re-established by the next term's first actuator round.
        self._last_report = None
        await self._cancel(self._outbox_task)
        self._outbox_task = None
        await self._cancel(self._round_task)
        self._round_task = None
        # The reconciler stops before its next delete (its should_continue
        # checks the epoch); _reconcile_work stays pending until it returns.
        await self._cancel(self._reconcile_task)
        self._reconcile_task = None
        dropped = len(self._outbox)
        self._outbox.clear()
        self._write_failures = 0
        self._abort_all()
        controller = self._controller()
        if controller is not None:
            async with self._host_lock():
                try:
                    controller.drop_pending()
                except Exception:
                    logger.exception("retune: drop_pending failed")
                self._rewind(controller)
        if was:
            logger.info(
                "retune: local leadership loss (%s); dropped %d queued decision(s)",
                reason,
                dropped,
            )

    def _request_reinit(self, reason: str) -> None:
        logger.warning("retune: re-initializing: %s", reason)
        self._initialized = False
        self._reinit_requested = True
        self._lease_wake.set()

    # ------------------------------------------------------ state writes

    async def _write_state(self, mutate: Mutation, what: str) -> WriteOutcome:
        """Conditional write (term and resourceVersion) of ``mutate(state)``.

        ``mutate`` returns the new state, ``None`` to discard, or
        ``_ALREADY`` when the server state already has it. A timeout is
        resolved by re-reading: a write that landed is a success. Every
        failed write (``error``) counts as ``failures_total{state_write}``.
        """

        outcome = await self._conditional_write(mutate, what)
        if outcome == "error":
            self._metrics.failure("state_write")
        return outcome

    async def _conditional_write(self, mutate: Mutation, what: str) -> WriteOutcome:
        epoch = self._epoch
        async with self._write_lock:
            snapshot = self._snapshot
            for attempt in range(_WRITE_ATTEMPTS):
                if epoch != self._epoch or not self.is_leader():
                    return "discarded"
                if snapshot is None or snapshot.state is None:
                    snapshot = await self._reread()
                    if snapshot is None or snapshot.state is None:
                        return "error"
                state = snapshot.state
                if state.leader_term != self._term:
                    return "lost_term"
                new = mutate(state)
                if new is _ALREADY:
                    self._snapshot = snapshot
                    return "already"
                if new is None:
                    self._snapshot = snapshot
                    return "discarded"
                assert isinstance(new, RetuneState)
                try:
                    written = await self._io(self._store.write, new, snapshot)
                except KubeConflict:
                    snapshot = await self._reread()
                    continue
                except (KubeError, asyncio.TimeoutError) as error:
                    logger.warning("retune state write (%s) failed: %s", what, error)
                    snapshot = await self._reread()
                    if (
                        snapshot is not None
                        and snapshot.state is not None
                        and snapshot.state.leader_term == self._term
                        and mutate(snapshot.state) is _ALREADY
                    ):
                        self._snapshot = snapshot
                        return "ok"
                    return "error"
                if epoch != self._epoch:
                    return "discarded"
                self._snapshot = written
                return "ok"
            return "error"

    async def _reread(self) -> Optional[StateSnapshot]:
        try:
            snapshot = await self._io(self._store.read)
        except (KubeError, asyncio.TimeoutError) as error:
            logger.warning("retune: re-reading the state failed: %s", error)
            return None
        if snapshot.state is not None and snapshot.state.dgd_uid != self._dgd.uid:
            return StateSnapshot(state=None, resource_version=None, lease=None)
        self._snapshot = snapshot
        return snapshot

    async def _write_decision(self, entry: OutboxEntry) -> WriteOutcome:
        decision = entry.decision
        history = self._history_entry(entry)
        window_end_ms = history["window_end_ms"]
        applied = decision.outcome == "applied"
        by_wt = entry.router_parameters

        def mutate(state: RetuneState) -> Union[RetuneState, None, object]:
            if not self._entry_current(entry):
                return None
            if has_decision(state, decision.decision_id):
                return _ALREADY
            new = record_decision(state, history, window_end_ms)
            if applied:
                if not by_wt:
                    logger.error(
                        "retune: applied decision %s has no router parameters",
                        decision.decision_id,
                    )
                    return new
                new = deploy(
                    new,
                    fingerprint=entry.fingerprint,
                    mode="active",
                    flat=dict(decision.chosen.router),
                    by_wt=by_wt,
                )
            return new

        outcome = await self._write_state(mutate, f"decision {decision.decision_id}")
        if outcome in ("ok", "already"):
            self._count_decision(decision)
            logger.info(
                "retune: persisted decision %s (%s)%s",
                decision.decision_id,
                decision.outcome,
                " - deploying" if applied else "",
            )
        return outcome

    def _count_decision(self, decision: RetuneDecision) -> None:
        self._metrics.decision(decision.outcome)
        failure_reason = getattr(decision, "failure_reason", None)
        if decision.outcome in ("failed", "timeout"):
            self._metrics.failure(failure_reason or decision.outcome)
        details = getattr(decision, "details", None) or {}
        # The DGDR executor reports the window cut under details["window"].
        window = details.get("window")
        dropped = (
            window.get("dropped_members")
            if isinstance(window, Mapping)
            else details.get("dropped_members")
        )
        if isinstance(dropped, int) and dropped > 0:
            self._metrics.dropped_members(dropped)
        if decision.launched_at_s is not None:
            missing = details.get("missing_candidates", 0)
            self._metrics.scores(
                baseline=_finite(decision.baseline_score),
                best=_finite(decision.best_score),
                sweep_seconds=_finite(decision.sweep_wall_s),
                missing_candidates=int(missing) if isinstance(missing, int) else 0,
            )

    def _history_entry(self, entry: OutboxEntry) -> dict[str, Any]:
        decision = entry.decision
        flat = {key: float(value) for key, value in decision.chosen.router.items()}
        item: dict[str, Any] = {
            "decision_id": decision.decision_id,
            "window_end_ms": int(round(decision.window_end_s * 1000.0)),
            "outcome": decision.outcome,
            "failure_reason": getattr(decision, "failure_reason", None),
            "reason": decision.reason,
            "fingerprint": entry.fingerprint,
            "baseline": _finite(decision.baseline_score),
            "best": _finite(decision.best_score),
            "flat": flat,
            "decided_ms": int(round(decision.decided_at_s * 1000.0)),
        }
        if decision.outcome == "applied" and entry.router_parameters is not None:
            item["by_wt"] = entry.router_parameters
        elif decision.outcome == "advisory":
            item["by_wt"] = self._resolve_by_wt(flat)
        details = getattr(decision, "details", None) or {}
        image = {
            key: details[key]
            for key in ("runtime_image", "runtime_version", "runtime_version_override")
            if isinstance(details.get(key), str)
        }
        if image:
            item["image"] = image
        return item

    def _entry_current(self, entry: OutboxEntry) -> bool:
        return (
            self._initialized
            and self.is_leader()
            and entry.epoch == self._epoch
            and entry.term_seq == self._term_seq
            and entry.generation == self._generation
            and entry.fingerprint == self._fingerprint
        )

    # ----------------------------------------------------------- actuation

    async def _leader_round(self) -> None:
        epoch = self._epoch
        if self._settings.decommission:
            await self._actuate()
            await self._finish_decommission()
            return
        await self._check_canary()
        await self._actuate()
        # Off the control loop's critical path: revalidation and actuation
        # never wait for the reconciler.
        self._start_reconcile(epoch)

    async def _actuate(self) -> None:
        target = self.target()
        state = self.state
        if target is None or state is None:
            return
        epoch = self._epoch
        startup = self._evaluation.startup
        report = await self._actuator.reconcile(
            target,
            initials=state.initials,
            expected_frontends=max(1, startup.replicas if startup is not None else 1),
        )
        if epoch != self._epoch:
            return
        self._last_report = report
        if startup is not None and report.parity_ok is not None:
            if report.parity_ok != self._p14_ok:
                logger.info(
                    "retune: P14 parity %s -> %s", self._p14_ok, report.parity_ok
                )
            self._p14_ok = report.parity_ok
        await self._persist_initials(report)
        await self._track_rollback(target, report)

    async def _persist_initials(self, report: ActuationReport) -> None:
        state = self.state
        if state is None:
            return
        merged = merge_initials(state, report.new_initials, keep=report.seen_router_ids)
        if maps_equal(merged.initials, state.initials):
            return
        latched = dict(report.new_initials)
        keep = frozenset(report.seen_router_ids)

        def mutate(current: RetuneState) -> Union[RetuneState, None, object]:
            new = merge_initials(current, latched, keep=keep)
            return _ALREADY if maps_equal(new.initials, current.initials) else new

        outcome = await self._write_state(mutate, "initials")
        if outcome == "lost_term":
            self._request_reinit("the state's leader_term is not ours")

    async def _track_rollback(
        self, target: ActuationTarget, report: ActuationReport
    ) -> None:
        """Write "rollback complete" after a lease duration of convergence."""

        state = self.state
        if state is None or target.mode != "rollback" or not state.rollback_pending:
            self._rollback_since = None
            return
        if not report.converged:
            self._rollback_since = None
            return
        now = self._monotonic()
        since = self._rollback_since
        if since is None or since[0] != state.deploy_seq:
            self._rollback_since = (state.deploy_seq, now)
            return
        if now - since[1] < self._settings.lease_duration_s:
            return
        expected = since[0]
        entry = {
            "decision_id": f"rolled_back-{expected}",
            "outcome": "rolled_back",
            "deploy_seq": expected,
            "fingerprint": self._fingerprint,
            "at_ms": int(round(self._clock() * 1000.0)),
        }

        def mutate(current: RetuneState) -> Union[RetuneState, None, object]:
            return rollback_complete(current, expected_deploy_seq=expected, entry=entry)

        outcome = await self._write_state(mutate, "rollback complete")
        self._rollback_since = None
        if outcome == "ok":
            logger.info("retune: rollback complete (deploy_seq %d)", expected)
        elif outcome == "lost_term":
            self._request_reinit("the state's leader_term is not ours")

    async def _check_canary(self) -> None:
        """P11: a dry-run Strict Request with ``workload.trace``."""

        canary = self._canary
        if canary is None:
            self._canary_ok = True
            return
        now = self._monotonic()
        if (
            self._canary_ok
            and self._canary_at is not None
            and now - self._canary_at < self._settings.canary_interval_s
        ):
            return
        try:
            ok = bool(await self._io(canary, self))
        except Exception as error:
            logger.warning("retune canary (P11) failed: %s", error)
            ok = False
        if ok != self._canary_ok:
            logger.info("retune canary (P11): %s", "ok" if ok else "failing")
        self._canary_ok = ok
        self._canary_at = now

    def _reconcile_allowed(self, epoch: int) -> bool:
        """§4.4.5: only the initialized leader of ``epoch`` cleans up.

        Called from the reconciler's thread too (``is_leader`` is
        thread-safe, the rest are single reads).
        """

        return (
            not self._stopping
            and self._initialized
            and epoch == self._epoch
            and self.is_leader()
        )

    def reconcile_running(self) -> bool:
        """Whether a reconciler round (its task or its call) is still running."""

        task, work = self._reconcile_task, self._reconcile_work
        return (task is not None and not task.done()) or (
            work is not None and not work.done()
        )

    def _start_reconcile(self, epoch: int) -> None:
        """Start one reconciler round in its own task, unless one still runs."""

        if self._reconciler is None or not self._reconcile_allowed(epoch):
            return
        if self.reconcile_running():
            logger.debug("retune reconciler: the previous round is still running")
            return
        self._reconcile_task = asyncio.create_task(
            self._reconcile(epoch), name="planner-retune-reconcile"
        )

    async def _reconcile(self, epoch: int) -> None:
        """One reconciler round, waited for at most ``reconcile_budget_s``.

        The reconciler calls ``should_continue`` before every delete or
        unlink and stops once it is ``False``: this process is no longer the
        initialized leader of ``epoch``, or the budget has passed. A blocking
        call cannot be cancelled, so ``_reconcile_work`` stays pending until
        it returns and no other round starts meanwhile.
        """

        reconciler = self._reconciler
        if reconciler is None or not self._reconcile_allowed(epoch):
            return
        state = self.state
        settings = self._settings
        expired = threading.Event()

        def should_continue() -> bool:
            return not expired.is_set() and self._reconcile_allowed(epoch)

        call = functools.partial(
            reconciler.run_once,
            last_window_end_ms=None if state is None else state.last_window_end_ms,
            due_window_end_ms=due_window_end_ms(
                self._clock(),
                phase_s=settings.phase_s,
                interval_s=settings.interval_s,
                grace_s=settings.grace_s,
            ),
            deadline_for=self._deadline_for,
            frontend_pods=getattr(self._actuator, "last_pods", None),
            should_continue=should_continue,
        )
        cancellable = inspect.iscoroutinefunction(reconciler.run_once)
        work: asyncio.Future[Any] = asyncio.ensure_future(
            call() if cancellable else self._blocking(call)
        )
        work.add_done_callback(_log_reconcile_failure)
        self._reconcile_work = work
        try:
            await asyncio.wait_for(
                asyncio.shield(work), timeout=settings.reconcile_budget_s
            )
        except asyncio.TimeoutError:
            logger.warning(
                "retune reconciler: the round exceeded its %.0fs budget; it "
                "stops before its next delete, later rounds wait for it",
                settings.reconcile_budget_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            pass  # logged by _log_reconcile_failure
        finally:
            if not work.done():
                expired.set()
                if cancellable:
                    work.cancel()

    def _deadline_for(self, window_end_ms: int) -> float:
        """``D_k = L_k + timeout`` in epoch seconds."""

        settings = self._settings
        return window_end_ms / 1000.0 + settings.grace_s + settings.timeout_s

    async def _finish_decommission(self) -> None:
        """§4.11: once nothing is left to roll back, delete Requests and Leases."""

        state = self.state
        if self.decommissioned or state is None or state.rollback_pending:
            return
        dgd = self._dgd
        try:
            requests = await self._io(
                self._kube.list_requests, f"{LABEL_DGD_UID}={dgd.uid}"
            )
            for request in requests:
                name = (request.get("metadata") or {}).get("name")
                if name:
                    await self._io(self._kube.delete_request, name)
            await self._io(self._store.delete)
        except (KubeError, asyncio.TimeoutError) as error:
            logger.warning("retune clean-up: deleting Requests/state failed: %s", error)
            return
        try:
            await self._io(self._kube.delete_lease, self._leadership.name)
        except (KubeError, asyncio.TimeoutError) as error:
            logger.warning("retune clean-up: deleting the lease failed: %s", error)
        self._leadership.drop()
        self._initialized = False
        self.decommissioned = True
        self._stopping = True
        logger.info("retune clean-up of DGD %s finished", dgd.name)

    # ------------------------------------------------------------- helpers

    async def _install_evaluation(self, evaluation: SpecEvaluation) -> None:
        old = self._evaluation
        startup_changed = _startup_key(old.startup) != _startup_key(evaluation.startup)
        if startup_changed:
            self._p14_ok = None
        self._evaluation = evaluation
        self._spec_ok = bool(evaluation.report.spec_ok) and self._p14_ok is not False
        self._fingerprint = self._fingerprint_fn(evaluation.startup)
        if startup_changed:
            self._resolver = self._make_resolver(evaluation.startup)
        if old.dgd != evaluation.dgd:
            if old.dgd.uid != evaluation.dgd.uid:
                # Before the re-initialization deletes the old uid's
                # Requests: the reconciler must treat the new uid's Requests
                # as its own and the old ones as another DGD's.
                self._set_reconciler_dgd(evaluation.dgd)
                self._store = StateStore(
                    self._kube,
                    evaluation.dgd,
                    timeout_s=self._settings.k8s_timeout_s,
                    clock=self._clock,
                )
                self._leadership.set_owner(evaluation.dgd)
            await self._swap_actuator(evaluation)
        elif startup_changed:
            self._actuator.set_startup_maps(self._actuator_maps(evaluation.startup))
            self._last_report = None
        self._log_report(evaluation.report)

    def _set_reconciler_dgd(self, dgd: DgdRef) -> None:
        setter = getattr(self._reconciler, "set_dgd", None)
        if setter is None:
            return
        try:
            setter(dgd)
        except Exception:
            logger.exception(
                "retune: moving the reconciler to DGD uid %s failed", dgd.uid
            )

    async def _swap_actuator(self, evaluation: SpecEvaluation) -> None:
        """A new actuator for a new DGD identity or dynamo namespace."""

        self._dgd = evaluation.dgd
        previous = self._actuator
        self._actuator = self._actuator_factory(
            evaluation.dgd, self._actuator_maps(evaluation.startup)
        )
        self._last_report = None
        await self._close_actuator(previous)

    def _actuator_maps(
        self, startup: Optional[FrontendStartup]
    ) -> dict[str, dict[str, float]]:
        if startup is not None and startup.startup_maps:
            return {wt: dict(params) for wt, params in startup.startup_maps.items()}
        # Unknown startup: rollback needs only the latched initials.
        return {wt: {} for wt in self._settings.worker_types}

    def _make_resolver(
        self, startup: Optional[FrontendStartup]
    ) -> Optional[RouterResolver]:
        if startup is None or self._resolver_factory is None:
            return None
        try:
            return self._resolver_factory(startup)
        except Exception:
            logger.exception("retune: building the router resolver failed")
            return None

    def _router_keys(self) -> tuple[str, ...]:
        controller = self._controller()
        keys = (
            getattr(controller, "router_keys", None) if controller is not None else None
        )
        return tuple(keys) if keys else self._settings.router_keys

    def _startup_by_wt(self) -> Optional[dict[str, dict[str, float]]]:
        startup = self._evaluation.startup
        if startup is None:
            return None
        return {wt: dict(params) for wt, params in startup.startup_maps.items()}

    def startup_flat(self) -> Optional[dict[str, float]]:
        """The controller's startup router point (live names), if known."""

        return self._startup_flat()

    def _startup_flat(self) -> Optional[dict[str, float]]:
        resolver = self._resolver
        keys = self._router_keys()
        if resolver is None or not keys:
            return None
        try:
            return resolver.startup_flat(keys)
        except Exception:
            logger.exception("retune: resolving the startup router point failed")
            return None

    def _resolve_by_wt(
        self, flat: Mapping[str, float]
    ) -> Optional[dict[str, dict[str, float]]]:
        resolver = self._resolver
        if resolver is None:
            return None
        try:
            return resolver.by_wt(flat)
        except Exception:
            logger.exception("retune: resolving router point %s failed", dict(flat))
            return None

    def _p12_ok(self) -> bool:
        """P12 from the last actuator round (routers present, admin API on)."""

        report = self._last_report
        if report is None or report.admin_api_disabled:
            return False
        present = report.routers_present
        return bool(report.converged if present is None else present)

    def _last_decided_k(self) -> Optional[int]:
        state = self.state
        if state is None or state.last_window_end_ms is None:
            return None
        return window_index(
            state.last_window_end_ms,
            phase_s=self._settings.phase_s,
            interval_s=self._settings.interval_s,
        )

    def _rewind(self, controller: Any) -> None:
        last = self._last_decided_k()
        try:
            controller.rewind(-1 if last is None else last)
        except Exception:
            logger.exception("retune: rewind failed")

    def _rewind_now(self) -> None:
        controller = self._controller()
        if controller is not None:
            self._rewind(controller)

    def _abort_all(self) -> None:
        with self._aborts_lock:
            aborts, self._aborts = self._aborts, []
        for event in aborts:
            event.set()

    def _controller(self) -> Any:
        host = self._host
        if host is None:
            return None
        engine = getattr(host, "_engine", None)
        return getattr(engine, "retune_controller", None)

    @contextlib.asynccontextmanager
    async def _host_lock(self) -> AsyncIterator[None]:
        lock = getattr(self._host, "_config_lock", None)
        if lock is None:
            yield
            return
        async with lock:
            yield

    async def _io(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        call = functools.partial(function, *args, **kwargs)
        return await asyncio.wait_for(
            self._blocking(call), timeout=self._settings.io_timeout_s
        )

    @staticmethod
    async def _cancel(task: Optional[asyncio.Task[Any]]) -> None:
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.wait([task])

    @staticmethod
    async def _close_actuator(actuator: Any) -> None:
        client = getattr(actuator, "_client", None)
        close = getattr(client, "close", None)
        if close is None:
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug(
                "retune: closing the router policy client failed", exc_info=True
            )

    def _log_report(self, report: PremiseReport) -> None:
        for reason in report.reasons:
            logger.warning("retune premise failed: %s", reason)
        for warning in report.warnings:
            logger.warning("retune premise warning: %s", warning)

    def _publish(self) -> None:
        state = self.state
        try:
            self._metrics.supervisor_state(
                leader=self.is_leader(),
                ready=self.ready(),
                initialized=self._initialized,
                rollback_pending=bool(state is not None and state.rollback_pending),
                generation=self._generation,
            )
        except Exception:
            logger.debug("retune: publishing supervisor metrics failed", exc_info=True)

    # ---------------------------------------------------------------- loops

    async def _lease_loop(self) -> None:
        while not self._stopping:
            try:
                await self.lease_round()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("retune lease loop iteration failed")
            await self._wait(self._lease_wake, self._settings.renew_interval_s)

    async def _control_loop(self) -> None:
        while not self._stopping:
            try:
                await self.control_round()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("retune control loop iteration failed")
            await self._wait(None, self._settings.actuator_interval_s)

    async def _outbox_loop(self) -> None:
        while not self._stopping:
            await self._outbox_wake.wait()
            self._outbox_wake.clear()
            try:
                await self.outbox_round()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("retune outbox round failed")
            if self._reinit_requested:
                return
            if self._outbox:
                # A failed write: retry after a short pause.
                await self._sleep(min(5.0, self._settings.k8s_timeout_s))
                self._outbox_wake.set()

    async def _wait(self, event: Optional[asyncio.Event], timeout_s: float) -> None:
        """Sleep ``timeout_s`` (the injectable sleep) or until ``event`` is set."""

        sleeper = asyncio.ensure_future(self._sleep(timeout_s))
        waiters: set[asyncio.Future[Any]] = {sleeper}
        if event is not None:
            waiters.add(asyncio.ensure_future(event.wait()))
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()
            await asyncio.wait(waiters)
        if event is not None:
            event.clear()


def _log_reconcile_failure(work: asyncio.Future[Any]) -> None:
    """Done callback of a reconciler call (it may outlive its round's task)."""

    if work.cancelled():
        return
    error = work.exception()
    if error is not None:
        logger.error("retune reconciler round failed", exc_info=error)


def _startup_key(startup: Optional[FrontendStartup]) -> Optional[str]:
    return None if startup is None else startup.fingerprint_source


def _finite(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
