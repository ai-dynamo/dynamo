# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay-backed deployment evaluator: a :class:`DeploymentPlan` -> a trace_report.

Three load shapes (``Workload``) x two deployment cases (static / planner) all route to
a single dynamo replay entrypoint that emits the same flat ``trace_report`` dict:

| load | entrypoint |
|---|---|
| **mooncake trace** | ``run_trace_replay(trace_path, ...)`` |
| **synthetic** (rate or concurrency) | ``run_synthetic_trace_replay(isl, osl, request_count, ...)`` |

``planner_config`` selects the case on either entrypoint: ``None`` -> a static
fixed-fleet replay (the mocker's event-driven ``run()`` loop, no scaling); a dict ->
planner-in-the-loop, where the Rust bridge owns the same loop and calls back into the
Python planner adapter once per ``PlannerTick``. Static returns the bare ``trace_report``
dict; planner-in-the-loop returns a ``ReplayPlannerReport`` whose ``.trace_report`` is the
identical flat dict (:func:`_unwrap` normalizes both). By construction
``plan.is_static == (plan.planner_config is None)``.

The closed-loop in-flight cap is ``replay_concurrency`` for a trace and ``concurrency``
for a synthetic workload (``Workload.effective_in_flight_cap``; a pareto sweep overrides
it per-trial via ``concurrency_override``).

The **goodput SLA** (``goal.sla``) is passed as ``sla_*_ms`` on every path (the mocker
classifies SLA-satisfying requests per-request to compute goodput); it is independent of
the planner's own scaling SLA. Under ``kv_router`` the searched router-weight knobs are
built into a real ``KvRouterConfig``; ``round_robin`` passes ``router_config=None``.

dynamo is imported lazily so ``import dynamo.profiler.spica`` stays light and unit tests can stub the
replay entrypoints.
"""

from __future__ import annotations

import inspect
import json

from .config import OptimizationGoal, Workload
from .deploy import DeploymentPlan


def _build_kv_router_config(payload: dict | None):
    """A dynamo ``KvRouterConfig`` from dynamo.profiler.spica's router-knob dict (its keys map 1:1 to
    KvRouterConfig kwargs), or ``None`` under round_robin (empty/None payload)."""
    if not payload:
        return None
    from dynamo.llm import KvRouterConfig

    return KvRouterConfig(**payload)


def _unwrap(report) -> dict[str, float]:
    """Normalize a replay result to the flat ``trace_report`` dict: a static replay
    returns that dict directly; planner-in-the-loop returns a ``ReplayPlannerReport``
    whose ``.trace_report`` carries the identical shape. Preserve the planner tick
    count so callers and e2e coverage can distinguish bridge initialization from an
    actual planner callback."""
    if not hasattr(report, "trace_report"):
        return report
    trace_report = dict(report.trace_report)
    if hasattr(report, "total_ticks"):
        trace_report["planner_total_ticks"] = float(report.total_ticks)
    return trace_report


def _replay_accepts_kw(func, name: str) -> bool:
    params = inspect.signature(func).parameters
    return name in params or any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
    )


def _replay_kwargs(func, kwargs: dict) -> dict:
    params = inspect.signature(func).parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in params}


def _check_planner_supported(func, planner_config: dict | None) -> None:
    if planner_config is not None and not _replay_accepts_kw(func, "planner_config"):
        raise RuntimeError(
            "installed Dynamo replay API does not accept planner_config; use static planner_scaling_policy='disabled'"
        )


def _run_trace_replay_compat(func, trace_path: str, kwargs: dict):
    _check_planner_supported(func, kwargs.get("planner_config"))
    if "trace_file" in inspect.signature(func).parameters:
        return func(trace_path, **_replay_kwargs(func, kwargs))
    kwargs = dict(kwargs, trace_files=trace_path)
    return func(**_replay_kwargs(func, kwargs))


def _run_synthetic_trace_replay_compat(func, kwargs: dict):
    _check_planner_supported(func, kwargs.get("planner_config"))
    return func(**_replay_kwargs(func, kwargs))


def _require_goodput_metric(
    report: dict[str, float], goal: OptimizationGoal
) -> dict[str, float]:
    """Fail closed when a goodput objective cannot be measured per request.

    Aggregate mean latency cannot recover goodput: a mixed population may have the
    same means while a very different fraction of requests satisfies the SLA. Older
    replay APIs that omit the metric are therefore incompatible with goodput goals.
    """
    if "goodput_output_throughput_tok_s" in report:
        return report
    target = goal.target.value
    objectives = (
        {item.value for item in goal.resolved_pareto_objectives}
        if goal.is_pareto
        else {target}
    )
    if not objectives.intersection({"goodput", "goodput_per_gpu"}):
        return report
    raise RuntimeError(
        "Dynamo replay did not emit goodput_output_throughput_tok_s for a goodput objective; "
        "install a replay version with per-request SLA accounting (aggregate mean latency "
        "cannot be used as a goodput fallback)"
    )


class ReplayEvaluator:
    """Evaluate one candidate by replaying the workload (trace or synthetic)."""

    def __init__(
        self,
        workload: Workload,
        goal: OptimizationGoal,
        *,
        trace_block_size: int = 512,
        benchmark_granularity: int = 8,
    ):
        self.workload = workload
        self.goal = goal
        self.trace_block_size = trace_block_size
        self.benchmark_granularity = benchmark_granularity

    def _goodput_sla_kwargs(self) -> dict[str, float | None]:
        sla = self.goal.sla
        if sla is None:
            return {}
        return {
            "sla_ttft_ms": sla.ttft_ms,
            "sla_itl_ms": sla.itl_ms,
            "sla_e2e_ms": sla.e2e_ms,
        }

    def _router_config(self, plan: DeploymentPlan):
        return _build_kv_router_config(plan.router_config)

    def _synthetic_kwargs(self, concurrency_override: int | None = None) -> dict:
        """The synthetic-workload params ``run_synthetic_trace_replay`` takes
        (``arrival_interval_ms`` is ignored in closed-loop mode).

        ``request_count`` is derived from ``num_request_ratio`` and the effective load
        (the per-trial KV-load-derived ``concurrency_override``, else the workload's fixed
        concurrency / request_rate)."""
        wl = self.workload
        return dict(
            input_tokens=wl.isl,
            output_tokens=wl.osl,
            request_count=wl.resolved_request_count(concurrency_override),
            arrival_interval_ms=wl.synthetic_arrival_interval_ms,
            turns_per_session=wl.turns_per_session,
            shared_prefix_ratio=wl.shared_prefix_ratio,
            num_prefix_groups=wl.num_prefix_groups,
            inter_turn_delay_ms=wl.inter_turn_delay_ms,
        )

    def evaluate(
        self, plan: DeploymentPlan, *, concurrency_override: int | None = None
    ) -> dict[str, float]:
        """Run one replay and return its trace_report dict.

        ``concurrency_override`` is the per-trial in-flight cap derived from KV load; it
        overrides a fixed workload ``concurrency`` for both the closed-loop cap and the
        ``num_request_ratio``-derived request count. Trace workloads ignore it (they cap
        via ``replay_concurrency``)."""
        if self.workload.is_trace_based:
            return _require_goodput_metric(self._evaluate_trace(plan), self.goal)
        return _require_goodput_metric(
            self._evaluate_synthetic(plan, concurrency_override=concurrency_override),
            self.goal,
        )

    # -- trace workloads -------------------------------------------------------

    def _evaluate_trace(self, plan: DeploymentPlan) -> dict[str, float]:
        from dynamo.mocker import MockEngineArgs
        from dynamo.replay.api import run_trace_replay

        wl = self.workload
        common = dict(
            router_mode=plan.router_mode,
            router_config=self._router_config(plan),
            arrival_speedup_ratio=wl.arrival_speedup_ratio,
            trace_block_size=self.trace_block_size,
            replay_concurrency=wl.effective_in_flight_cap(),  # None -> arrival timestamps
            planner_config=None if plan.is_static else plan.planner_config,
            benchmark_granularity=self.benchmark_granularity,
            **self._goodput_sla_kwargs(),
        )
        if plan.deployment_mode == "agg":
            extra = MockEngineArgs.from_json(json.dumps(plan.agg_engine_args))
            report = _run_trace_replay_compat(
                run_trace_replay,
                wl.trace_path,
                dict(extra_engine_args=extra, num_workers=plan.num_workers, **common),
            )
        else:
            prefill = MockEngineArgs.from_json(json.dumps(plan.prefill_engine_args))
            decode = MockEngineArgs.from_json(json.dumps(plan.decode_engine_args))
            report = _run_trace_replay_compat(
                run_trace_replay,
                wl.trace_path,
                dict(
                    prefill_engine_args=prefill,
                    decode_engine_args=decode,
                    num_prefill_workers=plan.num_prefill_workers,
                    num_decode_workers=plan.num_decode_workers,
                    **common,
                ),
            )
        return _unwrap(report)

    # -- synthetic workloads ---------------------------------------------------

    def _evaluate_synthetic(
        self, plan: DeploymentPlan, *, concurrency_override: int | None = None
    ) -> dict[str, float]:
        from dynamo.mocker import MockEngineArgs
        from dynamo.replay.api import run_synthetic_trace_replay

        common = dict(
            router_mode=plan.router_mode,
            router_config=self._router_config(plan),
            arrival_speedup_ratio=1.0,
            replay_concurrency=self.workload.effective_in_flight_cap(
                concurrency_override
            ),
            planner_config=None if plan.is_static else plan.planner_config,
            benchmark_granularity=self.benchmark_granularity,
            **self._goodput_sla_kwargs(),
            **self._synthetic_kwargs(concurrency_override),
        )
        if plan.deployment_mode == "agg":
            extra = MockEngineArgs.from_json(json.dumps(plan.agg_engine_args))
            report = _run_synthetic_trace_replay_compat(
                run_synthetic_trace_replay,
                dict(extra_engine_args=extra, num_workers=plan.num_workers, **common),
            )
        else:
            prefill = MockEngineArgs.from_json(json.dumps(plan.prefill_engine_args))
            decode = MockEngineArgs.from_json(json.dumps(plan.decode_engine_args))
            report = _run_synthetic_trace_replay_compat(
                run_synthetic_trace_replay,
                dict(
                    prefill_engine_args=prefill,
                    decode_engine_args=decode,
                    num_prefill_workers=plan.num_prefill_workers,
                    num_decode_workers=plan.num_decode_workers,
                    **common,
                ),
            )
        return _unwrap(report)
