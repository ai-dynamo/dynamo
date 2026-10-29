# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-owned retune decision loop, identical in replay and live hosts.

``RetuneController.on_tick`` runs once per planner tick inside the engine.
Every ``interval_seconds`` it launches a sweep of the window that just ended,
on a single sweep thread, and on a later tick turns the sweep result into a
``RetuneDecision``. The only mode branch is how a launched sweep is awaited:
a ``VirtualClock`` engine (replay) awaits it inline because virtual time is
frozen; a wall-clock engine polls the future on later ticks.

Window schedule, with ``s0`` the planner start, ``T = interval_seconds`` and
``G = grace_seconds``::

    W_k = [s0 + k*T, s0 + (k+1)*T)      L_k = s0 + (k+1)*T + G  (launch, as_of)
    ms(s) = trace_origin_ms + s * 1000 * arrival_speedup_ratio

A sweep launched at tick time ``t`` is decided at the first tick at or after
``t + max(D, scale_interval)`` at which it is done, so never in its launch
tick. ``D`` is ``decision_latency_seconds``, or for ``measured`` the sweep's
own ``wall_s`` (capped at the timeout) times ``decision_latency_scale``,
known once the sweep is done. In replay the old parameters therefore serve
traffic for the modeled sweep time; live, ``measured`` with scale 1 decides
on the first tick after the sweep finishes.

A live host (``retune.live``) passes hooks that replay never sets, so replay
behaviour is unchanged:

- ``start(start_s, phase_s)`` anchors ``s0`` at an epoch phase instead of
  the planner start, so windows and ``decision_id = retune-w<end_ms>``
  survive restarts and leader changes.
- ``launch_gate(k, window)`` is asked before a due window is launched or
  recorded ``busy``: ``not_leader``/``not_ready`` leave the window due (it
  is retried on a later tick), ``already_decided`` consumes it silently.
- ``harvest_allowed(generation)`` is asked before a finished sweep becomes
  a decision; a refused or abandoned (``NON_DECISION_FAILURES``) sweep
  records nothing and rewinds the window so it can be relaunched or adopted.
- ``drop_pending``/``rewind`` let the host abandon an in-flight sweep
  without consuming its window; ``restore_desired`` re-installs persisted
  router values.
- ``predecessor``/``running_future`` carry the single sweep thread across
  controller generations: a rebuilt controller's first launch queues behind
  the previous generation's still-running sweep, as a relaunch queues
  behind a dropped sweep within one controller.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, field
from typing import Any

from dynamo.llm import KvRouterConfig
from dynamo.planner.config.planner_config import PlannerConfig, RetuneConfig
from dynamo.planner.core.types import (
    RetuneDecision,
    RetuneInputs,
    RetuneOutcome,
    RetuneParams,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.clock import Clock, VirtualClock
from dynamo.planner.reconfiguration import (
    PREDICTOR_FIELDS,
    candidate_config,
    runtime_candidate_errors,
)
from dynamo.planner.retune.domains import (
    LOAD_SCALING_DOWN_SENSITIVITY,
    OVERLAP_SCORE_CREDIT,
    PredictorCandidate,
    expand_planner_domain,
    expand_router_domain,
    grid_points,
    required_history_seconds,
)
from dynamo.planner.retune.live.contracts import (
    NON_DECISION_FAILURES,
    GateResult,
    HarvestAllowed,
    LaunchGate,
)
from dynamo.planner.retune.sweep import (
    SweepRequest,
    SweepResult,
    objective_maximized,
    run_sweep,
    with_predictor_fields,
)
from dynamo.planner.simulation.load_predictor import predictor_fields

logger = logging.getLogger(__name__)

_TOP_SCORES = 5
_EPSILON = 1e-9
# Gate reasons that leave a due window due, to be retried on a later tick.
_GATE_RETRY_REASONS = frozenset({"not_leader", "not_ready"})

# ``sweep_fn(request, *, cancel)`` in replay; live hooks add ``abort``.
SweepFn = Callable[..., SweepResult]


@dataclass(frozen=True)
class RetuneTickResult:
    """What one ``on_tick`` call hands back to the engine.

    ``router_parameters`` is ``None`` iff the router domain is empty;
    otherwise it is always the full desired map, keyed by ``aggregated`` or
    ``prefill`` and ``decode``, each the resolved 8-key PATCH
    ``/router_policy`` parameters. ``audit_events`` are
    ``retune:<event> w=<k>`` strings.
    """

    decisions: list[RetuneDecision] = field(default_factory=list)
    router_parameters: dict[str, dict[str, float]] | None = None
    audit_events: list[str] = field(default_factory=list)


@dataclass
class _PendingSweep:
    request: SweepRequest
    future: Future[SweepResult]
    window_start_s: float
    window_end_s: float
    as_of_s: float
    launched_at_s: float
    # Both None while a ``measured`` sweep is still running.
    decision_latency_s: float | None
    ready_at_s: float | None
    # Set by the supervisor to stop this sweep (lost leadership, rebuild).
    abort: threading.Event | None = None
    # When submitted behind a dropped (or predecessor generation's) sweep
    # that was still running: ``(submitted_monotonic, [started_monotonic])``.
    # ``launched_at_s`` then moves to the moment the sweep really started.
    queued: tuple[float, list[float]] | None = None


class RetuneController:
    """Cadence, sweeps and decisions for ``PlannerConfig.retune``.

    Construction runs the checks that need the router bindings, the worker
    capabilities or the full candidate set, and raises ``ValueError`` on any
    failure, except that ``_feasible_grid`` drops the planner grid points
    that do not validate or fit the deployment. ``config`` is the live config
    object the host mutates; the controller reads the searched planner fields
    from it. Under ``config.advisory`` a decision that would apply is
    recorded as ``advisory`` and leaves the desired state at the deployed
    point.

    The keyword-only hooks are for a live host and default to replay
    behaviour: ``sweep_fn`` replaces ``run_sweep`` (and must accept
    ``abort=`` when ``launch_gate`` is set), ``launch_gate`` and
    ``harvest_allowed`` are the supervisor's predicates (module docstring),
    and ``generation`` identifies this controller among the ones a live host
    builds over its lifetime; every sweep request carries it.
    ``predecessor`` is the previous generation's ``running_future()``: this
    controller's first launch waits for it, so two generations never sweep
    at once.
    """

    def __init__(
        self,
        config: PlannerConfig,
        inputs: RetuneInputs,
        capabilities: WorkerCapabilities,
        clock: Clock,
        scale_interval_s: float,
        *,
        sweep_fn: SweepFn | None = None,
        launch_gate: LaunchGate | None = None,
        harvest_allowed: HarvestAllowed | None = None,
        generation: int = 0,
        predecessor: Future[SweepResult] | None = None,
    ) -> None:
        retune = config.retune
        if retune is None:
            raise ValueError("RetuneController requires config.retune")
        if not (math.isfinite(scale_interval_s) and scale_interval_s > 0):
            raise ValueError("scale_interval_s must be positive and finite")
        if launch_gate is not None and sweep_fn is None:
            # run_sweep takes no abort event; a gated launch passes one.
            raise ValueError("RetuneController launch_gate requires a sweep_fn")
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError("generation must be an integer")
        if predecessor is not None and not isinstance(predecessor, Future):
            raise ValueError("predecessor must be a concurrent.futures.Future")
        self._sweep_fn = sweep_fn
        self._launch_gate = launch_gate
        self._harvest_allowed = harvest_allowed
        self._generation = generation
        self._config = config
        self._retune = retune
        self._inputs = inputs
        self._capabilities = capabilities
        self._clock = clock
        self._scale_interval_s = float(scale_interval_s)
        self._worker_types: tuple[str, ...] = (
            ("prefill", "decode") if config.mode == "disagg" else ("aggregated",)
        )

        _check_engine_shape(config, inputs)
        _check_inner_planner_isolation(config)
        # The scoring SLA is frozen here; ttft_ms/itl_ms are never searched.
        self._sla_ttft_ms, self._sla_itl_ms = retune.scoring_sla(
            config.ttft_ms, config.itl_ms
        )
        self._maximize = objective_maximized(retune.objective)

        self._startup_router_json: dict[str, Any] = dict(
            inputs.engine.router_config or {}
        )
        self._router_maps: dict[str, dict[str, dict[str, float]]] = {}
        router_domain = self._router_domain(retune)
        self._router_domain_values = router_domain
        self._router_keys = tuple(sorted(router_domain))
        self._router_searched = bool(router_domain)

        planner_domain = expand_planner_domain(config)
        self._predictor_candidates: tuple[PredictorCandidate, ...] | None = (
            tuple(planner_domain.predictor_candidates)
            if planner_domain.predictor_candidates
            else None
        )
        # History before each window that warms the inner predictors; unset,
        # the minimum the current and candidate predictors need.
        history_s = retune.history_seconds
        if history_s is None:
            history_s = (
                required_history_seconds(config, planner_domain)
                if config.enable_throughput_scaling
                else 0.0
            )
        self._history_s = history_s
        planner_keys = set(planner_domain.grid)
        if self._predictor_candidates:
            planner_keys |= PREDICTOR_FIELDS
        self._planner_keys = tuple(sorted(planner_keys))
        for key in self._planner_keys:
            if self._config_value(key) is None:
                raise ValueError(
                    f"retune.planner.{key} is searched but has no current value; "
                    "set it explicitly in the planner config"
                )

        self._grid = self._feasible_grid(router_domain, planner_domain.grid)

        self._desired_router: dict[str, float] = self._startup_router_values()
        self._pending_planner: dict[str, Any] = {}
        self._startup_point = self.current_point()

        self._s0 = 0.0
        # Live: s0 is an epoch phase and decision ids carry the window end.
        self._phase_anchored = False
        self._started = False
        self._last_k = -1
        self._history: deque[tuple[float, dict[str, int]]] = deque()
        self._pending: _PendingSweep | None = None
        # A dropped sweep that may still hold the single sweep thread, or
        # the previous generation's sweep that a first launch waits for.
        self._dropped: Future[SweepResult] | None = (
            predecessor if predecessor is not None and not predecessor.done() else None
        )
        self._cancel = threading.Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="planner-retune"
        )
        logger.info(
            "retune enabled: interval=%ss grace=%ss %d grid points, router keys %s, "
            "planner keys %s, predictor candidates %s, objective %s, "
            "scoring SLA ttft=%sms itl=%sms, advisory=%s",
            retune.interval_seconds,
            retune.grace_seconds,
            len(self._grid),
            list(self._router_keys),
            list(self._planner_keys),
            [label for label, _ in self._predictor_candidates or ()],
            retune.objective,
            self._sla_ttft_ms,
            self._sla_itl_ms,
            config.advisory,
        )
        if LOAD_SCALING_DOWN_SENSITIVITY in planner_domain.grid:
            logger.info(
                "retune searches load_scaling_down_sensitivity %s; "
                "load_min_observations is fixed at startup and stays %s",
                planner_domain.grid[LOAD_SCALING_DOWN_SENSITIVITY],
                config.load_min_observations,
            )

    # ------------------------------------------------------------------ API

    def start(self, start_s: float, phase_s: float | None = None) -> None:
        """Anchor the window schedule.

        Replay anchors ``s0`` at the planner's first tick time ``start_s``.
        A live host passes ``phase_s``: ``s0`` is then that epoch phase, so
        every planner process and controller generation cuts the same
        windows, and decision ids become ``retune-w<window_end_ms>``.
        """

        if phase_s is None:
            self._s0 = float(start_s)
            self._phase_anchored = False
        else:
            if not math.isfinite(phase_s):
                raise ValueError("phase_s must be finite")
            self._s0 = float(phase_s)
            self._phase_anchored = True
        self._started = True
        self._last_k = -1

    async def on_tick(
        self, now_s: float, worker_counts: WorkerCounts | None
    ) -> RetuneTickResult:
        """Record state, harvest a finished sweep, launch a due window."""

        if not self._started:
            raise RuntimeError("RetuneController.start() must be called first")
        decisions: list[RetuneDecision] = []
        audit: list[str] = []

        self._record_counts(now_s, worker_counts)
        self._pending_planner = {
            key: value
            for key, value in self._pending_planner.items()
            if not _same_value(getattr(self._config, key), value)
        }

        pending = self._pending
        if pending is not None and pending.future.done():
            if pending.queued is not None:
                self._settle_queue_delay(pending)
            if pending.ready_at_s is None:
                pending.decision_latency_s = self._decision_latency_s(pending.future)
                pending.ready_at_s = pending.launched_at_s + pending.decision_latency_s
            # Same tolerance as _due_window: with D == interval the decision
            # must land in the tick that launches the next window.
            if now_s + _EPSILON * self._retune.interval_seconds >= pending.ready_at_s:
                self._pending = None
                harvested = self._harvest_or_rewind(pending, now_s)
                if harvested is None:
                    audit.append(f"retune:dropped w={pending.request.window_index}")
                else:
                    decisions.append(harvested)

        due = self._due_window(now_s)
        if due > self._last_k:
            # The gate runs before both branches: a window this process may
            # not work on is neither launched nor recorded busy.
            gate = self._gate(due)
            if gate is not None and gate.reason in _GATE_RETRY_REASONS:
                pass
            elif gate is not None and gate.reason == "already_decided":
                self._last_k = due
            else:
                self._last_k = due
                if self._pending is not None:
                    decisions.append(
                        self._skipped(due, now_s, "busy", "sweep in flight")
                    )
                else:
                    launched = await self._launch(due, now_s, gate)
                    if launched is None:
                        audit.append(f"retune:launch w={due}")
                    else:
                        decisions.append(launched)

        for decision in decisions:
            audit.append(f"retune:{decision.outcome} w={decision.window_index}")
            self._log_decision(decision)
        return RetuneTickResult(
            decisions=decisions,
            router_parameters=self._desired_router_map(),
            audit_events=audit,
        )

    def close(self) -> None:
        """Kill the sweep workers and release the sweep thread."""

        self._cancel.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def drop_pending(self) -> None:
        """Abandon the in-flight sweep without deciding its window.

        Sets the sweep's abort event, cancels it if it has not started, and
        forgets it, so a following ``rewind`` takes effect and the window is
        relaunched (or adopted by another leader) instead of recorded busy.
        A sweep that is already running keeps the single sweep thread until
        it sees the abort; a later launch queues behind it.
        """

        pending = self._pending
        if pending is None:
            return
        self._pending = None
        if pending.abort is not None:
            pending.abort.set()
        if not pending.future.cancel() and not pending.future.done():
            # Running, so any earlier dropped sweep has finished, or a
            # predecessor generation's sweep is still running and this one
            # waits for it: either way this one is now what a new launch
            # would queue behind.
            self._dropped = pending.future
        logger.info(
            "retune dropped the in-flight sweep of window %d",
            pending.request.window_index,
        )

    def rewind(self, last_decided_k: int) -> None:
        """Make every window after ``last_decided_k`` due again.

        A no-op while a sweep is pending (``drop_pending`` first): its
        window is consumed until it is harvested or dropped.
        """

        if self._pending is None:
            self._last_k = min(self._last_k, int(last_decided_k))

    def restore_desired(self, flat: Mapping[str, float]) -> None:
        """Re-install persisted desired router values (live re-election).

        ``flat`` maps live router knob names to values and must cover every
        searched key; extra keys are ignored. The startup point is not
        touched: it stays a candidate of every sweep.
        """

        try:
            values = {key: flat[key] for key in self._router_keys}
        except KeyError as error:
            raise ValueError(
                f"restore_desired needs every searched router key "
                f"{list(self._router_keys)}; missing {error.args[0]!r}"
            ) from None
        # Validates finite, non-negative values and a resolvable config, so
        # the per-tick desired map can never fail afterwards.
        router = RetuneParams(router=values).router
        if self._router_searched:
            self._router_map(router)
        self._desired_router = router

    def running_future(self) -> Future[SweepResult] | None:
        """The sweep that holds or will hold the sweep thread, if any.

        The pending sweep while it runs, else a dropped (or predecessor
        generation's) sweep that is still running, else a pending sweep
        that has not started; ``None`` once all of them are done. A host
        replacing this controller calls it after ``drop_pending`` and
        ``close`` and passes it as the successor's ``predecessor``.
        """

        pending = self._pending.future if self._pending is not None else None
        if pending is not None and pending.running():
            return pending
        for future in (self._dropped, pending):
            if future is not None and not future.done():
                return future
        return None

    @property
    def generation(self) -> int:
        """The generation the host built this controller with."""

        return self._generation

    @property
    def router_keys(self) -> tuple[str, ...]:
        """Searched router knobs, by live (PATCH) name, sorted."""

        return self._router_keys

    @property
    def router_domain(self) -> dict[str, list[float]]:
        """The searched router domain: live knob name -> candidate values."""

        return {key: list(values) for key, values in self._router_domain_values.items()}

    def current_point(self) -> RetuneParams:
        """The point in effect: desired router values and live planner values.

        Planner values come from the live config overlaid with the
        controller's own changes that the host has not applied yet.
        """

        return RetuneParams(
            router={key: self._desired_router[key] for key in self._router_keys},
            planner={
                key: self._pending_planner.get(key, self._config_value(key))
                for key in self._planner_keys
            },
        )

    # ------------------------------------------------- construction checks

    def _router_domain(self, retune: RetuneConfig) -> dict[str, list[float]]:
        if retune.router is None:
            return {}
        engine = self._inputs.engine
        startup = self._startup_router_json
        # Only the builtin default policy is modelled identically.
        if engine.router_mode != "kv_router":
            raise ValueError("retune.router requires engine router_mode='kv_router'")
        if startup.get("router_policy_config") is not None:
            raise ValueError(
                "retune.router requires the builtin default router policy "
                "(router_config.router_policy_config must be unset)"
            )
        if startup.get("overlap_score_weight") is not None:
            raise ValueError(
                "retune.router cannot be used with the deprecated "
                "overlap_score_weight, which overrides prefill_load_scale"
            )
        if startup.get("conditional_disagg_enabled"):
            raise ValueError(
                "retune.router does not support conditional disaggregation: "
                "replay and live resolve decode credit differently"
            )

        startup_maps = self._router_map({})
        # A worker type that starts with no overlap credit can never get it.
        zero_credit = {
            worker_type
            for worker_type, parameters in startup_maps.items()
            if worker_type != "decode" and parameters[OVERLAP_SCORE_CREDIT] == 0
        }
        # The schema keeps prefill_load_model.type at its startup value; the
        # serde form of the AIS load model is "aic".
        startup_load_model = json.loads(
            KvRouterConfig.from_json(json.dumps(startup)).to_json()
        )["router_prefill_load_model"]
        domain = expand_router_domain(
            retune,
            startup_load_model=startup_load_model,
            zero_startup_credit=bool(zero_credit),
        )
        # The schema admits only PolicyParameters names, with the bounds
        # KvRouterConfig checks; resolving every candidate also guards
        # against the two drifting apart.
        for point in grid_points(domain, {}):
            try:
                maps = self._router_map(point.router)
            except ValueError as error:
                raise ValueError(
                    f"retune.router candidate {point.router} is not a valid "
                    f"router configuration: {error}"
                ) from None
            for worker_type in zero_credit:
                if maps[worker_type][OVERLAP_SCORE_CREDIT] != 0:
                    raise ValueError(
                        f"retune.router candidate {point.router} gives the "
                        f"{worker_type} router overlap_score_credit > 0, but its "
                        "startup credit is 0 (a live router rejects that update)"
                    )
        return domain

    def _feasible_grid(
        self,
        router_domain: Mapping[str, Sequence[float]],
        planner_grid: Mapping[str, Sequence[Any]],
    ) -> list[RetuneParams]:
        """The grid without the planner points that cannot run here.

        Grid points and predictor candidates are checked apart, each on the
        startup config: no ``PlannerConfig`` validator relates a predictor
        field to another field, and ``_decide`` checks the combination it
        applies. A point that fails validation or does not fit the
        deployment is dropped, as recommend drops an infeasible candidate.
        An invalid predictor candidate, or a grid value that no point keeps
        (so also a grid without a valid point), raises.
        """

        # The retune checks never depend on a candidate; skip them.
        base = self._config.model_copy(update={"retune": None})
        # Defensive: retune.planner validation already rejects a predictor
        # candidate that PlannerConfig would (a non-positive or non-finite
        # value), so this raises only if the two schemas drift apart.
        for _, entry in self._predictor_candidates or ():
            updates = predictor_fields(entry)
            error = _candidate_error(base, updates, self._capabilities)
            if error is not None:
                raise ValueError(f"retune.planner candidate {updates} {error}")
        points = grid_points({}, planner_grid)
        rejected: dict[str, str] = {}
        for point in points:
            if point.planner:
                error = _candidate_error(base, point.planner, self._capabilities)
                if error is not None:
                    rejected[point.key()] = error
        if not rejected:
            return grid_points(router_domain, planner_grid)

        kept = [point for point in points if point.key() not in rejected]
        for key, values in planner_grid.items():
            for value in values:
                if not any(_same_value(p.planner[key], value) for p in kept):
                    point = next(
                        p for p in points if _same_value(p.planner[key], value)
                    )
                    raise ValueError(
                        f"no retune.planner candidate with {key}={value!r} fits; "
                        f"candidate {point.planner} {rejected[point.key()]}"
                    )
        first = next(p for p in points if p.key() in rejected)
        logger.warning(
            "retune drops %d of %d planner grid points that fail on this "
            "deployment, for example %s %s",
            len(rejected),
            len(points),
            first.planner,
            rejected[first.key()],
        )
        return [
            point
            for point in grid_points(router_domain, planner_grid)
            if RetuneParams(planner=point.planner).key() not in rejected
        ]

    def _startup_router_values(self) -> dict[str, float]:
        if not self._router_keys:
            return {}
        startup = KvRouterConfig.from_json(json.dumps(self._startup_router_json))
        serialized = json.loads(startup.to_json())
        resolved = startup.policy_parameters("aggregated")
        values: dict[str, float] = {}
        for key in self._router_keys:
            # Unset optional fields are omitted from the serde form; their
            # aggregated resolution is the value that leaves startup unchanged.
            value = serialized.get(key)
            values[key] = float(resolved[key] if value is None else value)
        return values

    # --------------------------------------------------------- tick helpers

    def _record_counts(self, now_s: float, worker_counts: WorkerCounts | None) -> None:
        if worker_counts is not None:
            if self._config.mode == "disagg":
                counts = {
                    "prefill": worker_counts.ready_num_prefill,
                    "decode": worker_counts.ready_num_decode,
                }
            else:
                counts = {"aggregated": worker_counts.ready_num_decode}
            if all(value is not None for value in counts.values()):
                self._history.append(
                    (now_s, {key: int(value or 0) for key, value in counts.items()})
                )
        # Keep enough to answer "ready counts at the start of the latest due
        # window": that start is at most 2T + G before now.
        horizon = now_s - 2 * self._retune.interval_seconds - self._retune.grace_seconds
        while len(self._history) > 1 and self._history[1][0] <= horizon:
            self._history.popleft()

    def _replicas_at(self, time_s: float) -> dict[str, int] | None:
        chosen: dict[str, int] | None = None
        for at_s, counts in self._history:
            if at_s <= time_s + _EPSILON or chosen is None:
                chosen = counts
            if at_s > time_s + _EPSILON:
                break
        if chosen is None or any(value < 1 for value in chosen.values()):
            return None
        return dict(chosen)

    def _due_window(self, now_s: float) -> int:
        """Latest window ``k`` whose launch time ``L_k`` is at or before now."""

        elapsed = now_s - self._s0 - self._retune.grace_seconds
        return math.floor(elapsed / self._retune.interval_seconds + _EPSILON) - 1

    def _window_times(self, k: int) -> tuple[float, float, float]:
        interval = self._retune.interval_seconds
        start = self._s0 + k * interval
        end = start + interval
        return start, end, end + self._retune.grace_seconds

    def _ms(self, time_s: float) -> float:
        inputs = self._inputs
        return inputs.trace_origin_ms + time_s * 1000.0 * inputs.arrival_speedup_ratio

    def _decision_id(self, k: int) -> str:
        if self._phase_anchored:
            _, end_s, _ = self._window_times(k)
            return f"retune-w{int(round(self._ms(end_s)))}"
        return f"retune-{int(self._s0)}-w{k}"

    def _gate(self, k: int) -> GateResult | None:
        """Ask the live host whether window ``k`` may launch now.

        ``None`` without a gate (replay). A gate that raises or answers an
        unknown reason holds the window like ``not_ready``: the window stays
        due and the gate is asked again on the next tick.
        """

        if self._launch_gate is None:
            return None
        try:
            gate = self._launch_gate(k, self._window_times(k))
        except Exception:
            logger.exception("retune launch gate failed for window %d", k)
            return GateResult(reason="not_ready")
        if gate.reason is not None and gate.reason not in (
            *_GATE_RETRY_REASONS,
            "already_decided",
        ):
            logger.warning(
                "retune launch gate returned unknown reason %r for window %d; "
                "holding the window",
                gate.reason,
                k,
            )
            return GateResult(reason="not_ready")
        return gate

    def _harvest_or_rewind(
        self, pending: _PendingSweep, now_s: float
    ) -> RetuneDecision | None:
        """Turn a finished sweep into a decision, or abandon its window.

        The window is abandoned (no decision, ``_last_k`` rewound so it is
        due again) when the host refuses the harvest, for instance because
        this process lost leadership or rebuilt its controller, or when the
        sweep reports a ``NON_DECISION_FAILURES`` reason.
        """

        k = pending.request.window_index
        allowed = True
        if self._harvest_allowed is not None:
            try:
                allowed = bool(self._harvest_allowed(pending.request.generation))
            except Exception:
                logger.exception("retune harvest_allowed failed for window %d", k)
                allowed = False
        if not allowed:
            logger.info("retune discards the sweep of window %d: harvest refused", k)
            self.rewind(k - 1)
            return None
        result = self._sweep_result(pending)
        if result.failure_reason in NON_DECISION_FAILURES:
            logger.info(
                "retune abandons window %d: sweep stopped (%s)",
                k,
                result.failure_reason,
            )
            self.rewind(k - 1)
            return None
        return self._decide(pending, result, now_s)

    def _settle_queue_delay(self, pending: _PendingSweep) -> None:
        """Move a queued sweep's launch time to when it actually started."""

        assert pending.queued is not None
        submitted, started = pending.queued
        pending.queued = None
        if not started:
            return  # never ran (cancelled at close); keep the submit time
        delay = max(0.0, started[0] - submitted)
        pending.launched_at_s += delay
        if pending.ready_at_s is not None:
            pending.ready_at_s += delay

    def _replicas_for_launch(
        self, start_s: float, end_s: float, spec_replicas: Mapping[str, int] | None
    ) -> tuple[dict[str, int] | None, bool, str | None]:
        """``(replicas, verified, mismatch)`` for a window launch.

        Without ``spec_replicas`` (replay), the ready counts at window start,
        verified. With them (live), every ready count recorded over the
        window, including the one in effect at its start, must equal the
        spec; a mismatch is reported. The spec replicas are used either way,
        unverified when no recorded count covers the window start (the
        window began before this controller started).
        """

        if spec_replicas is None:
            return self._replicas_at(start_s), True, None
        spec = {key: int(value) for key, value in spec_replicas.items()}
        covering: dict[str, int] | None = None
        observed: list[dict[str, int]] = []
        for at_s, counts in self._history:
            if at_s <= start_s + _EPSILON:
                covering = counts
            elif at_s < end_s - _EPSILON:
                observed.append(counts)
        if covering is not None:
            observed.insert(0, covering)
        for counts in observed:
            if counts != spec:
                return spec, covering is not None, json.dumps(counts, sort_keys=True)
        return spec, covering is not None, None

    async def _launch(
        self, k: int, now_s: float, gate: GateResult | None = None
    ) -> RetuneDecision | None:
        """Submit the sweep of window ``k``; a decision only if it cannot run."""

        start_s, end_s, as_of_s = self._window_times(k)
        spec_replicas = gate.spec_replicas if gate is not None else None
        replicas, verified, mismatch = self._replicas_for_launch(
            start_s, end_s, spec_replicas
        )
        if mismatch is not None:
            return self._skipped(
                k,
                now_s,
                "failed",
                f"ready worker counts {mismatch} differ from the DGD spec "
                f"replicas {json.dumps(spec_replicas, sort_keys=True)} during "
                "the window",
                failure_reason="replicas_changed",
                details={
                    "spec_replicas": dict(spec_replicas or {}),
                    "observed_replicas": json.loads(mismatch),
                },
            )
        if replicas is None:
            return self._skipped(
                k, now_s, "failed", "no ready worker counts recorded at window start"
            )
        baseline = self.current_point()
        candidates = tuple(
            sorted({*self._grid, self._startup_point, baseline}, key=RetuneParams.key)
        )
        base_planner = None
        history_start_ms = self._ms(start_s)
        if self._config.scaling_enabled():
            base_planner = self._config.model_dump(mode="json")
            base_planner.update(self._pending_planner)
            base_planner["retune"] = None
            history_start_ms = self._ms(start_s - self._history_s)
        request = SweepRequest(
            window_index=k,
            window_start_ms=self._ms(start_s),
            window_end_ms=self._ms(end_s),
            history_start_ms=history_start_ms,
            as_of_ms=self._ms(as_of_s),
            trace_paths=self._inputs.trace_paths,
            history_paths=self._inputs.history_paths,
            arrival_speedup_ratio=self._inputs.arrival_speedup_ratio,
            engine=self._inputs.engine,
            candidates=candidates,
            baseline=baseline,
            replicas=replicas,
            base_planner=base_planner,
            predictor_candidates=(
                self._predictor_candidates if base_planner is not None else None
            ),
            objective=self._retune.objective,
            sla_ttft_ms=self._sla_ttft_ms,
            sla_itl_ms=self._sla_itl_ms,
            max_parallel=self._retune.max_parallel,
            timeout_s=self._retune.effective_timeout_seconds,
            # 0 in replay, the field's default.
            generation=self._generation,
            **self._live_request_fields(as_of_s, spec_replicas, verified),
        )
        abort = gate.abort if gate is not None else None
        future, queued = self._submit(request, abort, gated=gate is not None)
        if isinstance(self._clock, VirtualClock):
            # Virtual time is frozen while the sweep runs; wait for it here.
            await asyncio.wait([asyncio.wrap_future(future)])
        latency = (
            None
            if self._retune.decision_latency_seconds == "measured" and not future.done()
            else self._decision_latency_s(future)
        )
        self._pending = _PendingSweep(
            request=request,
            future=future,
            window_start_s=start_s,
            window_end_s=end_s,
            as_of_s=as_of_s,
            launched_at_s=now_s,
            decision_latency_s=latency,
            ready_at_s=None if latency is None else now_s + latency,
            abort=abort,
            queued=queued,
        )
        logger.info(
            "retune launched window %d [%s, %s) with %d candidates, replicas %s%s",
            k,
            start_s,
            end_s,
            len(candidates),
            replicas,
            ""
            if verified
            else " (unverified: the window began before this controller)",
        )
        return None

    def _live_request_fields(
        self,
        as_of_s: float,
        spec_replicas: Mapping[str, int] | None,
        verified: bool,
    ) -> dict[str, Any]:
        """The ``SweepRequest`` fields only a gated (live) launch sets."""

        if self._launch_gate is None:
            return {}
        return {
            "router_domain": {
                key: tuple(float(value) for value in values)
                for key, values in self._router_domain_values.items()
            },
            "deadline_s": as_of_s + self._retune.effective_timeout_seconds,
            "spec_replicas": None if spec_replicas is None else dict(spec_replicas),
            "replicas_verified": verified,
        }

    def _submit(
        self,
        request: SweepRequest,
        abort: threading.Event | None,
        *,
        gated: bool,
    ) -> tuple[Future[SweepResult], tuple[float, list[float]] | None]:
        """Submit the sweep to the single sweep thread.

        ``run_sweep`` is looked up at call time so tests can substitute it.
        A gated launch also passes the gate's abort event. Behind a dropped
        sweep that still holds the thread, or a predecessor generation's
        sweep that still runs on its own controller's thread, the call waits
        for it and records when it really starts (``_settle_queue_delay``).
        """

        sweep_fn: SweepFn = self._sweep_fn if self._sweep_fn is not None else run_sweep
        kwargs: dict[str, Any] = {"cancel": self._cancel}
        if gated:
            kwargs["abort"] = abort
        if self._dropped is not None and self._dropped.done():
            self._dropped = None
        if self._dropped is None:
            return self._executor.submit(sweep_fn, request, **kwargs), None
        clock = self._clock
        blocker = self._dropped
        started: list[float] = []

        def queued_sweep() -> SweepResult:
            # A dropped sweep of this controller has already finished (one
            # sweep thread); a predecessor generation's may still be running.
            wait_futures((blocker,))
            started.append(clock.monotonic())
            return sweep_fn(request, **kwargs)

        future = self._executor.submit(queued_sweep)
        return future, (clock.monotonic(), started)

    def _decision_latency_s(self, future: Future[SweepResult]) -> float:
        """Delay from launch to decision; ``measured`` needs a done sweep."""
        latency = self._retune.decision_latency_seconds
        if latency == "measured":
            # A sweep that raised has no wall time; it decides one tick later.
            wall_s = 0.0 if future.exception() is not None else future.result().wall_s
            if not math.isfinite(wall_s):
                # NaN would make the decision time unreachable.
                wall_s = self._retune.effective_timeout_seconds
            latency = (
                min(wall_s, self._retune.effective_timeout_seconds)
                * self._retune.decision_latency_scale
            )
        return max(latency, self._scale_interval_s)

    @staticmethod
    def _sweep_result(pending: _PendingSweep) -> SweepResult:
        """The finished sweep's result; a raised exception is ``failed``."""

        error = pending.future.exception()
        if error is not None:
            return SweepResult(
                status="failed", error=f"{type(error).__name__}: {error}"
            )
        return pending.future.result()

    def _decide(
        self, pending: _PendingSweep, result: SweepResult, now_s: float
    ) -> RetuneDecision:
        request = pending.request
        baseline = request.baseline
        presweep_fields = (
            None
            if result.predictor_presweep is None
            else result.predictor_presweep.get("fields")
        )

        def record(outcome: RetuneOutcome, **overrides: Any) -> RetuneDecision:
            values: dict[str, Any] = {
                "decision_id": self._decision_id(request.window_index),
                "outcome": outcome,
                "window_index": request.window_index,
                "window_start_s": pending.window_start_s,
                "window_end_s": pending.window_end_s,
                "as_of_s": pending.as_of_s,
                "launched_at_s": pending.launched_at_s,
                "decided_at_s": now_s,
                "num_requests": result.num_requests,
                "num_candidates": len(request.candidates),
                "previous": baseline,
                "chosen": baseline,
                "baseline_score": None,
                "best_score": None,
                "predictor_presweep": result.predictor_presweep,
                "sweep_wall_s": result.wall_s,
                "decision_latency_s": pending.decision_latency_s,
                "failure_reason": result.failure_reason,
                "details": result.details,
            }
            values.update(overrides)
            return RetuneDecision(**values)

        if result.status == "empty":
            return record("kept", reason="empty window")
        if result.status == "timeout":
            return record("timeout", reason=result.error)
        if result.status == "failed":
            return record("failed", reason=result.error)
        if self.current_point() != baseline:
            return record(
                "stale", reason="searched parameters changed during the sweep"
            )
        baseline_score = result.scores.get(baseline.key())
        if baseline_score is None:
            error = result.metrics.get(baseline.key(), {}).get("error")
            return record("failed", reason=f"baseline replay failed: {error}")

        order = {
            candidate.key(): index for index, candidate in enumerate(request.candidates)
        }
        sign = 1.0 if self._maximize else -1.0
        ranked = sorted(
            (
                (score, candidate)
                for candidate in request.candidates
                if (score := result.scores.get(candidate.key())) is not None
            ),
            # Best score first (lowest for a minimized objective); ties go to
            # the baseline, then canonical order.
            key=lambda item: (
                -sign * item[0],
                item[1] != baseline,
                order[item[1].key()],
            ),
        )
        top: list[tuple[str, float]] = []
        for score, candidate in ranked:
            # Equivalent candidates share one evaluation; list it once.
            key = with_predictor_fields(candidate, baseline, presweep_fields).key()
            if len(top) < _TOP_SCORES and all(key != seen for seen, _ in top):
                top.append((key, score))
        best_score, best = ranked[0]
        scores = {
            "baseline_score": baseline_score,
            "best_score": best_score,
            "top": top,
        }
        if best == baseline or not self._improves(best_score, baseline_score):
            return record("kept", **scores)

        chosen = with_predictor_fields(best, baseline, presweep_fields)
        current = self.current_point()
        planner_changes = {
            key: value
            for key, value in chosen.planner.items()
            if not _same_value(current.planner.get(key), value)
        }
        router_changed = self._router_searched and self._router_map(
            chosen.router
        ) != self._router_map(current.router)
        if not planner_changes and not router_changed:
            return record(
                "kept",
                reason="best candidate equals the current configuration",
                **scores,
            )

        predictor_warmup = None
        if planner_changes:
            # _feasible_grid checked grid points and predictors apart; check the
            # combination the host will apply, so it never raises.
            updates = {**self._pending_planner, **planner_changes}
            error = _candidate_error(self._config, updates, self._capabilities)
            if error is not None:
                return record(
                    "failed", reason=f"planner candidate {updates} {error}", **scores
                )
        if planner_changes.keys() & PREDICTOR_FIELDS:
            # Hosts swap a predictor in only warmed; the sweep prepared the
            # winner on the same warmup.
            if not result.warmup_observations:
                return record(
                    "failed",
                    reason="predictor change without warmup observations",
                    **scores,
                )
            error = result.predictor_warmup_error
            if error is not None:
                return record(
                    "failed", reason=f"predictor warmup rejected: {error}", **scores
                )
            predictor_warmup = result.warmup_observations
        if self._config.advisory:
            # The decision an active planner would apply, after the same
            # feasibility checks; nothing is applied, so the desired router
            # and the pending planner changes stay at the deployed point and
            # the next sweep searches from it again.
            return record(
                "advisory", chosen=chosen, planner_changes=planner_changes, **scores
            )
        decision = record(
            "applied",
            chosen=chosen,
            planner_changes=planner_changes,
            predictor_warmup=predictor_warmup,
            **scores,
        )
        self._desired_router = {key: chosen.router[key] for key in self._router_keys}
        self._pending_planner.update(planner_changes)
        return decision

    def _improves(self, score: float, baseline_score: float) -> bool:
        """Gain in the objective's direction must exceed
        ``min_improvement_percentage`` of ``baseline_score``."""

        gain = score - baseline_score if self._maximize else baseline_score - score
        margin = baseline_score * self._retune.min_improvement_percentage / 100.0
        return gain > margin

    def _skipped(
        self,
        k: int,
        now_s: float,
        outcome: RetuneOutcome,
        reason: str,
        *,
        failure_reason: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> RetuneDecision:
        start_s, end_s, as_of_s = self._window_times(k)
        current = self.current_point()
        return RetuneDecision(
            decision_id=self._decision_id(k),
            outcome=outcome,
            window_index=k,
            window_start_s=start_s,
            window_end_s=end_s,
            as_of_s=as_of_s,
            launched_at_s=None,
            decided_at_s=now_s,
            num_requests=0,
            num_candidates=0,
            previous=current,
            chosen=current,
            baseline_score=None,
            best_score=None,
            reason=reason,
            failure_reason=failure_reason,
            details=dict(details or {}),
        )

    # ------------------------------------------------------- router helpers

    def _router_map(self, router: Mapping[str, float]) -> dict[str, dict[str, float]]:
        """Resolve ``startup ⊕ router`` to the per-worker-type parameters."""

        key = json.dumps(dict(router), sort_keys=True)
        cached = self._router_maps.get(key)
        if cached is None:
            config = KvRouterConfig.from_json(
                json.dumps({**self._startup_router_json, **router})
            )
            cached = {
                worker_type: dict(config.policy_parameters(worker_type))
                for worker_type in self._worker_types
            }
            self._router_maps[key] = cached
        return cached

    def _desired_router_map(self) -> dict[str, dict[str, float]] | None:
        if not self._router_searched:
            return None
        return {
            worker_type: dict(parameters)
            for worker_type, parameters in self._router_map(
                self._desired_router
            ).items()
        }

    def _config_value(self, key: str) -> Any:
        if key == "prefill_min_endpoint":
            return self._config.effective_prefill_min_endpoint
        if key == "decode_min_endpoint":
            return self._config.effective_decode_min_endpoint
        return getattr(self._config, key)

    @staticmethod
    def _log_decision(decision: RetuneDecision) -> None:
        logger.info(
            "retune decision %s: %s (baseline=%s best=%s requests=%d "
            "chosen=%s planner_changes=%s reason=%s failure_reason=%s)",
            decision.decision_id,
            decision.outcome,
            decision.baseline_score,
            decision.best_score,
            decision.num_requests,
            (
                decision.chosen.to_dict()
                if decision.outcome in ("applied", "advisory")
                else None
            ),
            decision.planner_changes,
            decision.reason,
            decision.failure_reason,
        )


def _check_engine_shape(config: PlannerConfig, inputs: RetuneInputs) -> None:
    """The pinned engine matches the planner mode."""

    if inputs.engine.is_disagg != (config.mode == "disagg"):
        raise ValueError(
            "retune engine must set prefill_engine_args and decode_engine_args "
            "for mode='disagg', and neither for mode='agg'"
        )


def _check_inner_planner_isolation(config: PlannerConfig) -> None:
    """Inner sweep planners must never call out to external plugins."""

    if not config.scaling_enabled():
        return
    if config.scheduling.external_plugins:
        raise ValueError(
            "retune with scaling enabled requires scheduling.external_plugins "
            "to be empty: inner sweep replays would call them"
        )
    if config.scheduling.gateway.enabled:
        raise ValueError(
            "retune with scaling enabled requires scheduling.gateway.enabled=false"
        )


def _candidate_error(
    config: PlannerConfig,
    updates: dict[str, Any],
    capabilities: WorkerCapabilities,
) -> str | None:
    """Why ``config`` with ``updates`` cannot run here; ``None`` if it can."""

    try:
        candidate = candidate_config(config, updates)
    except ValueError as error:
        return f"is invalid: {error}"
    errors = runtime_candidate_errors(candidate, capabilities)
    if errors:
        return f"cannot run on this deployment: {'; '.join(errors)}"
    return None


def _same_value(left: Any, right: Any) -> bool:
    """Equality that keeps bools distinct from 0/1."""

    return (isinstance(left, bool), left) == (isinstance(right, bool), right)
