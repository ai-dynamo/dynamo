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
``t + max(decision_latency_seconds, scale_interval)`` at which it is done, so
never in its launch tick.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from dynamo.llm import KvRouterConfig
from dynamo.planner.config.planner_config import PlannerConfig, RetuneConfig
from dynamo.planner.core.types import (
    RetuneDecision,
    RetuneFeedback,
    RetuneInputs,
    RetuneOutcome,
    RetuneParams,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.clock import Clock, VirtualClock
from dynamo.planner.reconfiguration import (
    PREDICTOR_FIELDS,
    candidate_config,
    prepare_predictor,
    runtime_candidate_errors,
)
from dynamo.planner.retune.domains import (
    LOAD_SCALING_DOWN_SENSITIVITY,
    OVERLAP_SCORE_CREDIT,
    PredictorCandidate,
    expand_planner_domain,
    expand_router_domain,
    grid_points,
)
from dynamo.planner.retune.sweep import (
    SweepRequest,
    SweepResult,
    run_sweep,
    with_predictor_fields,
)
from dynamo.planner.simulation.load_predictor import _PREDICTOR_ERRORS, predictor_fields

logger = logging.getLogger(__name__)

ROLLBACK_SUFFIX = "-rb"
_TOP_SCORES = 5
_EPSILON = 1e-9


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
    ready_at_s: float


class RetuneController:
    """Cadence, sweeps and decisions for ``PlannerConfig.retune``.

    Construction runs the stage-2 checks (E1-E7) that need the router
    bindings, the worker capabilities or the full candidate set, and raises
    ``ValueError`` on any failure. ``config`` is the live config object the
    host mutates; the controller reads the searched planner fields from it.
    """

    def __init__(
        self,
        config: PlannerConfig,
        inputs: RetuneInputs,
        capabilities: WorkerCapabilities,
        clock: Clock,
        scale_interval_s: float,
    ) -> None:
        retune = config.retune
        if retune is None:
            raise ValueError("RetuneController requires config.retune")
        if not (math.isfinite(scale_interval_s) and scale_interval_s > 0):
            raise ValueError("scale_interval_s must be positive and finite")
        self._config = config
        self._retune = retune
        self._inputs = inputs
        self._capabilities = capabilities
        self._clock = clock
        self._scale_interval_s = float(scale_interval_s)
        self._worker_types: tuple[str, ...] = (
            ("prefill", "decode") if config.mode == "disagg" else ("aggregated",)
        )

        _check_engine_shape(config, inputs)  # E5
        _check_inner_planner_isolation(config)  # E7
        # E6: the scoring SLA is frozen here; ttft_ms/itl_ms are never searched.
        self._sla_ttft_ms, self._sla_itl_ms = retune.scoring_sla(
            config.ttft_ms, config.itl_ms
        )

        self._startup_router_json: dict[str, Any] = dict(
            inputs.engine.router_config or {}
        )
        self._router_maps: dict[str, dict[str, dict[str, float]]] = {}
        router_domain = self._router_domain(retune)  # E1-E3
        self._router_keys = tuple(sorted(router_domain))
        self._router_searched = bool(router_domain)

        planner_domain = expand_planner_domain(config)
        self._predictor_candidates: tuple[PredictorCandidate, ...] | None = (
            tuple(planner_domain.predictor_candidates)
            if planner_domain.predictor_candidates
            else None
        )
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

        self._grid = grid_points(router_domain, planner_domain.grid)
        self._check_planner_candidates(planner_domain.grid)  # E4

        self._desired_router: dict[str, float] = self._startup_router_values()
        self._pending_planner: dict[str, Any] = {}
        self._startup_point = self.current_point()

        self._s0 = 0.0
        self._started = False
        self._last_k = -1
        self._history: deque[tuple[float, dict[str, int]]] = deque()
        self._pending: _PendingSweep | None = None
        self._last_actuated: RetuneDecision | None = None
        self._disabled = False
        self._cancel = threading.Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="planner-retune"
        )
        logger.info(
            "retune enabled: interval=%ss grace=%ss %d grid points, router keys %s, "
            "planner keys %s, predictor candidates %s, scoring SLA ttft=%sms itl=%sms",
            retune.interval_seconds,
            retune.grace_seconds,
            len(self._grid),
            list(self._router_keys),
            list(self._planner_keys),
            [label for label, _ in self._predictor_candidates or ()],
            self._sla_ttft_ms,
            self._sla_itl_ms,
        )
        if LOAD_SCALING_DOWN_SENSITIVITY in planner_domain.grid:
            logger.info(
                "retune searches load_scaling_down_sensitivity %s; "
                "load_min_observations is fixed at startup and stays %s",
                planner_domain.grid[LOAD_SCALING_DOWN_SENSITIVITY],
                config.load_min_observations,
            )

    # ------------------------------------------------------------------ API

    def start(self, start_s: float) -> None:
        """Anchor the window schedule at the planner's first tick time."""

        self._s0 = float(start_s)
        self._started = True
        self._last_k = -1

    async def on_tick(
        self,
        now_s: float,
        worker_counts: WorkerCounts | None,
        feedback: Sequence[RetuneFeedback],
    ) -> RetuneTickResult:
        """Record state, handle feedback, harvest a finished sweep, launch."""

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

        if not self._disabled:
            decisions.extend(self._handle_feedback(now_s, feedback))

        pending = self._pending
        if (
            not self._disabled
            and pending is not None
            and pending.future.done()
            and now_s >= pending.ready_at_s
        ):
            self._pending = None
            decisions.append(self._harvest(pending, now_s))

        due = self._due_window(now_s)
        if due > self._last_k:
            self._last_k = due
            if self._disabled:
                pass
            elif self._pending is not None:
                decisions.append(self._skipped(due, now_s, "busy", "sweep in flight"))
            else:
                launched = await self._launch(due, now_s)
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

    # ------------------------------------------------------ stage-2 checks

    def _router_domain(self, retune: RetuneConfig) -> dict[str, list[float]]:
        if retune.router is None:
            return {}
        engine = self._inputs.engine
        startup = self._startup_router_json
        # E1: only the builtin default policy is modelled identically.
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
        # E3: a worker type that starts with no overlap credit can never get it.
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
        # E2: the schema admits only PolicyParameters names; every candidate
        # must also be a valid router configuration (for example
        # host_cache_hit_weight <= 1).
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

    def _check_planner_candidates(self, grid: Mapping[str, Sequence[Any]]) -> None:
        """E4: every planner candidate validates and fits the deployment."""

        planner_parts: dict[str, dict[str, Any]] = {}
        for point in grid_points({}, grid):
            planner_parts[point.key()] = dict(point.planner)
        if self._predictor_candidates:
            for part in list(planner_parts.values()):
                for _, entry in self._predictor_candidates:
                    merged = {**part, **predictor_fields(entry)}
                    planner_parts[RetuneParams(planner=merged).key()] = merged
        for updates in planner_parts.values():
            if not updates:
                continue
            try:
                candidate = candidate_config(self._config, updates)
            except ValueError as error:
                raise ValueError(
                    f"retune.planner candidate {updates} is invalid: {error}"
                ) from None
            errors = runtime_candidate_errors(candidate, self._capabilities)
            if errors:
                raise ValueError(
                    f"retune.planner candidate {updates} cannot run on this "
                    f"deployment: {'; '.join(errors)}"
                )

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
        return f"retune-{int(self._s0)}-w{k}"

    async def _launch(self, k: int, now_s: float) -> RetuneDecision | None:
        """Submit the sweep of window ``k``; a decision only if it cannot run."""

        start_s, end_s, as_of_s = self._window_times(k)
        replicas = self._replicas_at(start_s)
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
            history_s = self._retune.effective_history_seconds
            history_start_ms = self._ms(start_s - history_s)
        request = SweepRequest(
            window_index=k,
            window_start_ms=self._ms(start_s),
            window_end_ms=self._ms(end_s),
            history_start_ms=history_start_ms,
            as_of_ms=self._ms(as_of_s),
            trace_paths=self._inputs.trace_paths,
            history_paths=self._inputs.history_paths,
            prune_by_mtime=self._inputs.prune_by_mtime,
            arrival_speedup_ratio=self._inputs.arrival_speedup_ratio,
            engine=self._inputs.engine,
            candidates=candidates,
            baseline=baseline,
            replicas=replicas,
            base_planner=base_planner,
            predictor_candidates=(
                self._predictor_candidates if base_planner is not None else None
            ),
            sla_ttft_ms=self._sla_ttft_ms,
            sla_itl_ms=self._sla_itl_ms,
            max_parallel=self._retune.max_parallel,
            timeout_s=self._retune.effective_timeout_seconds,
        )
        # Looked up at call time so tests can substitute the sweep function.
        future = self._executor.submit(run_sweep, request, cancel=self._cancel)
        if isinstance(self._clock, VirtualClock):
            # Virtual time is frozen while the sweep runs; wait for it here.
            await asyncio.wait([asyncio.wrap_future(future)])
        self._pending = _PendingSweep(
            request=request,
            future=future,
            window_start_s=start_s,
            window_end_s=end_s,
            as_of_s=as_of_s,
            launched_at_s=now_s,
            ready_at_s=now_s
            + max(self._retune.decision_latency_seconds, self._scale_interval_s),
        )
        logger.info(
            "retune launched window %d [%s, %s) with %d candidates, replicas %s",
            k,
            start_s,
            end_s,
            len(candidates),
            replicas,
        )
        return None

    def _harvest(self, pending: _PendingSweep, now_s: float) -> RetuneDecision:
        error = pending.future.exception()
        if error is not None:
            result = SweepResult(
                status="failed", error=f"{type(error).__name__}: {error}"
            )
        else:
            result = pending.future.result()
        return self._decide(pending, result, now_s)

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
        ranked = sorted(
            (
                (score, candidate)
                for candidate in request.candidates
                if (score := result.scores.get(candidate.key())) is not None
            ),
            # Score descending; ties go to the baseline, then canonical order.
            key=lambda item: (-item[0], item[1] != baseline, order[item[1].key()]),
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
        threshold = baseline_score * (
            1.0 + self._retune.min_improvement_percentage / 100.0
        )
        if best == baseline or best_score <= threshold:
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
        if self._retune.dry_run:
            return record("dry_run", chosen=chosen, **scores)

        predictor_warmup = None
        if planner_changes.keys() & PREDICTOR_FIELDS:
            if not result.warmup_observations:
                return record(
                    "failed",
                    reason="predictor change without warmup observations",
                    **scores,
                )
            error = self._predictor_warmup_error(
                planner_changes, result.warmup_observations
            )
            if error is not None:
                return record(
                    "failed",
                    reason=f"predictor warmup rejected: {error}",
                    **scores,
                )
            predictor_warmup = result.warmup_observations
        decision = record(
            "applied",
            chosen=chosen,
            planner_changes=planner_changes,
            predictor_warmup=predictor_warmup,
            **scores,
        )
        self._desired_router = {key: chosen.router[key] for key in self._router_keys}
        self._pending_planner.update(planner_changes)
        self._last_actuated = decision
        return decision

    def _predictor_warmup_error(
        self,
        planner_changes: Mapping[str, Any],
        warmup: Sequence[TrafficObservation],
    ) -> str | None:
        """Prepare the changed predictor as the host will; ``None`` if it can.

        Hosts swap a predictor in only warmed (``prepare_predictor`` raises
        on too few effective points or a bad first forecast), so an applied
        decision must carry a warmup its new predictor accepts. Before any
        history exists, the window alone can be too short for that predictor.
        """

        try:
            candidate = candidate_config(
                self._config, {**self._pending_planner, **planner_changes}
            )
            prepare_predictor(candidate, self._capabilities, warmup)
        except _PREDICTOR_ERRORS as error:
            return f"{type(error).__name__}: {error}"
        return None

    def _handle_feedback(
        self, now_s: float, feedback: Sequence[RetuneFeedback]
    ) -> list[RetuneDecision]:
        last = self._last_actuated
        if last is None:
            unmatched = list(feedback)
            matching: list[RetuneFeedback] = []
        else:
            matching = [
                item for item in feedback if item.decision_id == last.decision_id
            ]
            unmatched = [
                item for item in feedback if item.decision_id != last.decision_id
            ]
        for item in unmatched:
            logger.warning(
                "ignoring retune feedback for %s (%s): not the last actuated decision",
                item.decision_id,
                item.target,
            )
        if last is None or not matching:
            return []
        errors = "; ".join(f"{item.target}: {item.error}" for item in matching)
        current = self.current_point()

        if last.outcome == "rolled_back":
            self._disabled = True
            self._cancel.set()
            self._pending = None
            logger.error(
                "retune disabled: rollback %s was rejected (%s)",
                last.decision_id,
                errors,
            )
            return [
                RetuneDecision(
                    decision_id=f"{last.decision_id}-failed",
                    outcome="failed",
                    window_index=last.window_index,
                    window_start_s=last.window_start_s,
                    window_end_s=last.window_end_s,
                    as_of_s=last.as_of_s,
                    launched_at_s=None,
                    decided_at_s=now_s,
                    num_requests=last.num_requests,
                    num_candidates=last.num_candidates,
                    previous=current,
                    chosen=current,
                    baseline_score=None,
                    best_score=None,
                    reason=f"rollback rejected; retune disabled: {errors}",
                )
            ]

        if any(item.target == "planner" for item in matching):
            # The rejected planner change will never land.
            for key in last.planner_changes:
                self._pending_planner.pop(key, None)
            current = self.current_point()
        target = last.previous
        planner_changes = {
            key: value
            for key, value in target.planner.items()
            if not _same_value(current.planner.get(key), value)
        }
        rollback = RetuneDecision(
            decision_id=f"{last.decision_id}{ROLLBACK_SUFFIX}",
            outcome="rolled_back",
            window_index=last.window_index,
            window_start_s=last.window_start_s,
            window_end_s=last.window_end_s,
            as_of_s=last.as_of_s,
            launched_at_s=None,
            decided_at_s=now_s,
            num_requests=last.num_requests,
            num_candidates=last.num_candidates,
            previous=current,
            chosen=target,
            baseline_score=last.baseline_score,
            best_score=last.best_score,
            planner_changes=planner_changes,
            predictor_warmup=(
                last.predictor_warmup
                if planner_changes.keys() & PREDICTOR_FIELDS
                else None
            ),
            reason=f"actuation failed: {errors}",
        )
        self._desired_router = {key: target.router[key] for key in self._router_keys}
        self._pending_planner.update(planner_changes)
        self._last_actuated = rollback
        return [rollback]

    def _skipped(
        self, k: int, now_s: float, outcome: RetuneOutcome, reason: str
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
            "planner_changes=%s reason=%s)",
            decision.decision_id,
            decision.outcome,
            decision.baseline_score,
            decision.best_score,
            decision.num_requests,
            decision.planner_changes,
            decision.reason,
        )


def _check_engine_shape(config: PlannerConfig, inputs: RetuneInputs) -> None:
    """E5: the pinned engine matches the planner mode."""

    if inputs.engine.is_disagg != (config.mode == "disagg"):
        raise ValueError(
            "retune engine must set prefill_engine_args and decode_engine_args "
            "for mode='disagg', and neither for mode='agg'"
        )


def _check_inner_planner_isolation(config: PlannerConfig) -> None:
    """E7: inner sweep planners must never call out to external plugins."""

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


def _same_value(left: Any, right: Any) -> bool:
    """Equality that keeps bools distinct from 0/1."""

    return (isinstance(left, bool), left) == (isinstance(right, bool), right)
