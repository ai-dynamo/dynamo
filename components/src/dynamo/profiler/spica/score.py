# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score a replay ``trace_report`` against the optimization goal.

Three steps, mirroring the existing profiler replay optimizer
(``components/src/dynamo/profiler/utils/replay_optimize``) adapted to dynamo.profiler.spica's
``Candidate`` / ``OptimizationGoal`` and the merged replay report keys:

1. **objective** — map the goal target to a number from the report. The
   ``goodput_per_gpu`` / ``throughput_per_gpu`` targets divide ``goodput`` / ``throughput``
   (already a tok/s rate) by the **time-averaged provisioned GPU count**
   ``avg_gpu = gpu_hours / e2e_hours`` — units tok/s/gpu, matching a benchmark's
   "throughput per GPU". For a static deployment ``avg_gpu`` is the fixed GPU count
   (``gpu_hours = gpu_count * e2e_hours``); for a planner-scaled run it is the integral
   of provisioned GPUs over the run divided by its duration. (Dividing by ``gpu_hours``
   directly would be wrong — the rate already has time divided out.)
2. **feasibility** — within the GPU budget. SLA is intentionally *not* gated here:
   when the user cares about latency they pick a ``goodput`` / ``goodput_per_gpu``
   target, whose metric already counts only SLA-satisfying requests (the bridge's
   per-request goodput SLA). Over-budget candidates are dropped.
3. **rank** — feasible candidates best-first by score, ties broken toward fewer GPUs.

For a ``pareto`` goal the score is a *vector* instead: :func:`objective_vector` reads one
value per objective, :func:`make_candidate` stores them on the candidate, and
:func:`pareto_front` returns the non-dominated set (step 3 becomes Pareto dominance, not a
scalar rank). The default objectives are throughput-per-GPU vs per-user throughput — the
InferenceX tok/s/gpu vs tok/s/user frontier.
"""

from __future__ import annotations

import math

from .config import Candidate, OptimizationTarget

# trace_report keys the report always carries (goodput_* only when an SLA was
# supplied to the replay). Surfaced into Candidate.metrics for inspection.
_METRIC_KEYS = (
    "output_throughput_tok_s",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_e2e_latency_ms",
    "mean_output_token_throughput_per_user",
    "goodput_output_throughput_tok_s",
    "gpu_hours",
    "duration_ms",
    "planner_total_ticks",
)


def _avg_gpu(report: dict[str, float]) -> float:
    """Time-averaged provisioned GPU count = ``gpu_hours / e2e_hours`` (the integral of
    provisioned GPUs over the run, divided by its duration). For a static deployment this
    equals the fixed GPU count; for a planner-scaled run it averages over startup + serve +
    drain. Returns 0.0 when gpu_hours / duration are unavailable (guards divide-by-zero)."""
    gpu_hours = float(report.get("gpu_hours", 0.0))
    duration_ms = float(report.get("duration_ms", 0.0))
    if gpu_hours <= 0.0 or duration_ms <= 0.0:
        return 0.0
    return gpu_hours / (duration_ms / 3_600_000.0)


def objective_value(report: dict[str, float], target: OptimizationTarget) -> float:
    """The raw objective metric (NOT yet signed for direction)."""
    if target is OptimizationTarget.THROUGHPUT:
        return float(report.get("output_throughput_tok_s", 0.0))
    if target is OptimizationTarget.E2E_LATENCY:
        return float(report.get("mean_e2e_latency_ms", math.inf))
    if target is OptimizationTarget.GOODPUT:
        return float(report.get("goodput_output_throughput_tok_s", 0.0))
    if target is OptimizationTarget.GOODPUT_PER_GPU:
        avg_gpu = _avg_gpu(report)
        goodput = float(report.get("goodput_output_throughput_tok_s", 0.0))
        return goodput / avg_gpu if avg_gpu > 0.0 else 0.0
    if target is OptimizationTarget.THROUGHPUT_PER_GPU:
        avg_gpu = _avg_gpu(report)
        throughput = float(report.get("output_throughput_tok_s", 0.0))
        return throughput / avg_gpu if avg_gpu > 0.0 else 0.0
    if target is OptimizationTarget.THROUGHPUT_PER_USER:
        # per-user interactivity (tok/s/user): mean of per-token-gap 1000/itl. Already a
        # rate, so no GPU/time normalization — this is the InferenceX x-axis.
        return float(report.get("mean_output_token_throughput_per_user", 0.0))
    if target is OptimizationTarget.PARETO:
        raise ValueError(
            "'pareto' is multi-objective; use objective_vector / pareto_front, not objective_value"
        )
    raise ValueError(f"unknown optimization target: {target!r}")


def score_report(report: dict[str, float], target: OptimizationTarget) -> float:
    """Objective normalized so **higher is better** (minimized targets negated)."""
    value = objective_value(report, target)
    return value if target.maximize else -value


def is_feasible(used_gpus: int, gpu_budget: int) -> bool:
    """A candidate is feasible iff it fits the GPU budget.

    SLA is deliberately not a gate: the goodput targets already bake the SLA into
    their metric (the bridge counts only SLA-satisfying requests per-request), so an
    aggregate mean-latency gate here would double-count it and could drop a genuinely
    high-goodput config whose mean is dragged over by the tail.
    """
    return used_gpus <= gpu_budget


def objective_vector(
    report: dict[str, float], objectives: list[OptimizationTarget]
) -> dict[str, float]:
    """Raw value (natural units, NOT signed) for each Pareto objective, keyed by target
    value. Dominance uses each objective's own direction (``target.maximize``)."""
    return {t.value: objective_value(report, t) for t in objectives}


def _dominates(
    a: dict[str, float], b: dict[str, float], objectives: list[OptimizationTarget]
) -> bool:
    """True iff ``a`` Pareto-dominates ``b``: at least as good on every objective (in that
    objective's own direction) and strictly better on at least one."""
    strictly_better = False
    for t in objectives:
        av, bv = a[t.value], b[t.value]
        better = av > bv if t.maximize else av < bv
        worse = av < bv if t.maximize else av > bv
        if worse:
            return False
        if better:
            strictly_better = True
    return strictly_better


def pareto_front(
    candidates: list[Candidate], objectives: list[OptimizationTarget]
) -> list[Candidate]:
    """The non-dominated subset of ``candidates`` over ``objectives`` (each carrying an
    ``objectives`` vector), sorted by the **last** objective ascending — the x-axis — so the
    returned list traces the frontier left-to-right (e.g. low->high per-user throughput)."""
    pool = [c for c in candidates if c.objectives is not None]
    front = [
        c
        for c in pool
        if not any(
            _dominates(o.objectives, c.objectives, objectives)
            for o in pool
            if o is not c
        )
    ]
    x_axis = objectives[-1].value
    return sorted(front, key=lambda c: c.objectives.get(x_axis, 0.0))


def make_candidate(
    config: dict,
    report: dict[str, float],
    target: OptimizationTarget,
    *,
    pareto_objectives: list[OptimizationTarget] | None = None,
) -> Candidate:
    """Build a scored :class:`Candidate` from its config + replay report.

    Single-objective: ``score`` is the signed objective (higher=better). Under a
    ``pareto`` target, ``pareto_objectives`` must be given; the per-objective raw values are
    stored in ``Candidate.objectives`` (Pareto dominance reads these) and ``score`` carries
    the first objective's value as a headline number (it is not used for ranking)."""
    metrics = {key: float(report[key]) for key in _METRIC_KEYS if key in report}
    if target is OptimizationTarget.PARETO:
        if not pareto_objectives:
            raise ValueError("a pareto candidate needs pareto_objectives")
        objectives = objective_vector(report, pareto_objectives)
        return Candidate(
            config=config,
            used_gpus=int(config.get("used_gpus", 0)),
            score=objectives[pareto_objectives[0].value],
            metrics=metrics,
            objectives=objectives,
        )
    return Candidate(
        config=config,
        used_gpus=int(config.get("used_gpus", 0)),
        score=score_report(report, target),
        metrics=metrics,
    )


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """Best-first: highest score, ties broken toward fewer GPUs."""
    return sorted(candidates, key=lambda c: (-c.score, c.used_gpus))
