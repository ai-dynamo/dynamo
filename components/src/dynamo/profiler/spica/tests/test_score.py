# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoring / feasibility / ranking over a replay trace_report (pure)."""

import math

import pytest

from dynamo.profiler.spica.config import Candidate, OptimizationTarget
from dynamo.profiler.spica.score import (
    is_feasible,
    make_candidate,
    objective_value,
    objective_vector,
    pareto_front,
    rank,
    score_report,
)

# A representative trace_report (the keys the merged replay emits). duration_ms=0.5h
# with gpu_hours=2.0 -> avg_gpu = gpu_hours / e2e_hours = 2.0 / 0.5 = 4.0 (deliberately
# != gpu_hours so the per-gpu assertions discriminate rate/avg_gpu from a rate/gpu_hours regression).
REPORT = {
    "output_throughput_tok_s": 5000.0,
    "mean_ttft_ms": 800.0,
    "mean_tpot_ms": 20.0,
    "mean_e2e_latency_ms": 1200.0,
    "mean_output_token_throughput_per_user": 50.0,
    "goodput_output_throughput_tok_s": 4000.0,
    "gpu_hours": 2.0,
    "duration_ms": 1_800_000.0,  # 0.5 h
    "planner_total_ticks": 3.0,
}


def test_objective_per_target():
    assert objective_value(REPORT, OptimizationTarget.THROUGHPUT) == 5000.0
    assert objective_value(REPORT, OptimizationTarget.E2E_LATENCY) == 1200.0
    assert objective_value(REPORT, OptimizationTarget.GOODPUT) == 4000.0
    # avg_gpu = gpu_hours / e2e_hours = 2.0 / 0.5 = 4.0
    # goodput_per_gpu = goodput / avg_gpu = 4000 / 4 = 1000
    assert objective_value(REPORT, OptimizationTarget.GOODPUT_PER_GPU) == 1000.0
    # throughput_per_gpu = throughput / avg_gpu = 5000 / 4 = 1250
    assert objective_value(REPORT, OptimizationTarget.THROUGHPUT_PER_GPU) == 1250.0


def test_candidate_preserves_planner_tick_metric():
    candidate = make_candidate({"used_gpus": 4}, REPORT, OptimizationTarget.THROUGHPUT)
    assert candidate.metrics["planner_total_ticks"] == 3.0


def test_throughput_per_gpu_zero_when_avg_gpu_unavailable():
    # avg_gpu needs both gpu_hours>0 and duration_ms>0; either missing -> 0.0 (no divide-by-zero).
    assert (
        objective_value(
            {
                "output_throughput_tok_s": 5000.0,
                "gpu_hours": 0.0,
                "duration_ms": 1_800_000.0,
            },
            OptimizationTarget.THROUGHPUT_PER_GPU,
        )
        == 0.0
    )
    # gpu_hours present but duration_ms absent -> avg_gpu undefined -> 0.0
    assert (
        objective_value(
            {"output_throughput_tok_s": 5000.0, "gpu_hours": 2.0},
            OptimizationTarget.THROUGHPUT_PER_GPU,
        )
        == 0.0
    )
    assert (
        objective_value(
            {"output_throughput_tok_s": 5000.0}, OptimizationTarget.THROUGHPUT_PER_GPU
        )
        == 0.0
    )


def test_goodput_per_gpu_zero_when_no_gpu_hours():
    assert (
        objective_value(
            {
                "goodput_output_throughput_tok_s": 4000.0,
                "gpu_hours": 0.0,
                "duration_ms": 1_800_000.0,
            },
            OptimizationTarget.GOODPUT_PER_GPU,
        )
        == 0.0
    )


def test_goodput_per_gpu_zero_when_gpu_hours_missing():
    # gpu_hours absent -> avg_gpu defaults to 0.0, the guard avoids dividing by zero.
    assert (
        objective_value(
            {"goodput_output_throughput_tok_s": 4000.0},
            OptimizationTarget.GOODPUT_PER_GPU,
        )
        == 0.0
    )


def test_objective_defaults_on_missing_report_keys():
    # An empty report falls back to each target's neutral default.
    assert objective_value({}, OptimizationTarget.THROUGHPUT) == 0.0
    assert objective_value({}, OptimizationTarget.GOODPUT) == 0.0
    assert objective_value({}, OptimizationTarget.GOODPUT_PER_GPU) == 0.0
    # latency minimizes, so a missing report is the worst-possible +inf
    assert objective_value({}, OptimizationTarget.E2E_LATENCY) == math.inf


def test_score_defaults_on_missing_report_keys():
    # The latency default propagates through the sign flip to -inf (worst score).
    assert score_report({}, OptimizationTarget.E2E_LATENCY) == -math.inf
    assert score_report({}, OptimizationTarget.THROUGHPUT) == 0.0


def test_objective_value_unknown_target_raises():
    # A target that is none of the handled enum members hits the final guard.
    sentinel = object()
    with pytest.raises(ValueError, match="unknown optimization target"):
        objective_value(REPORT, sentinel)  # type: ignore[arg-type]


def test_score_sign():
    # maximized targets keep sign; e2e_latency is negated (higher score = lower latency)
    assert score_report(REPORT, OptimizationTarget.GOODPUT_PER_GPU) == 1000.0
    assert score_report(REPORT, OptimizationTarget.E2E_LATENCY) == -1200.0


def test_is_feasible_budget_only():
    # SLA is no longer a feasibility gate (the goodput targets bake it into the
    # metric); feasibility is purely the GPU budget.
    assert is_feasible(used_gpus=16, gpu_budget=32)
    assert is_feasible(used_gpus=32, gpu_budget=32)  # at the budget
    assert not is_feasible(used_gpus=64, gpu_budget=32)  # over budget


def test_throughput_per_user_objective():
    # the InferenceX x-axis: mean per-user output throughput (tok/s/user), a raw rate
    assert objective_value(REPORT, OptimizationTarget.THROUGHPUT_PER_USER) == 50.0
    assert objective_value({}, OptimizationTarget.THROUGHPUT_PER_USER) == 0.0


def test_pareto_target_is_not_a_scalar_objective():
    with pytest.raises(ValueError, match="multi-objective"):
        objective_value(REPORT, OptimizationTarget.PARETO)


def test_objective_vector_reads_each_objective_raw():
    objs = [
        OptimizationTarget.THROUGHPUT_PER_GPU,
        OptimizationTarget.THROUGHPUT_PER_USER,
    ]
    vec = objective_vector(REPORT, objs)
    # throughput_per_gpu = 5000 / avg_gpu(4) = 1250; throughput_per_user = 50 (raw)
    assert vec == {"throughput_per_gpu": 1250.0, "throughput_per_user": 50.0}


def _pareto_cand(
    tput_per_gpu: float, tput_per_user: float, used_gpus: int = 8
) -> Candidate:
    return Candidate(
        config={"used_gpus": used_gpus},
        used_gpus=used_gpus,
        score=tput_per_gpu,
        metrics={},
        objectives={
            "throughput_per_gpu": tput_per_gpu,
            "throughput_per_user": tput_per_user,
        },
    )


def test_pareto_front_drops_dominated_and_sorts_by_x_axis():
    objs = [
        OptimizationTarget.THROUGHPUT_PER_GPU,
        OptimizationTarget.THROUGHPUT_PER_USER,
    ]
    a = _pareto_cand(100.0, 10.0)  # high gpu, low user
    b = _pareto_cand(50.0, 20.0)  # low gpu, high user  -> a,b mutually non-dominated
    c = _pareto_cand(40.0, 8.0)  # dominated by a (worse on both)
    front = pareto_front([a, b, c], objs)
    assert c not in front and a in front and b in front
    # sorted by the last objective (x-axis = throughput_per_user) ascending
    assert front == [a, b]


def test_pareto_front_ignores_candidates_without_objectives():
    objs = [
        OptimizationTarget.THROUGHPUT_PER_GPU,
        OptimizationTarget.THROUGHPUT_PER_USER,
    ]
    a = _pareto_cand(100.0, 10.0)
    scalar = Candidate(config={}, used_gpus=8, score=1.0, metrics={})  # objectives=None
    assert pareto_front([a, scalar], objs) == [a]


def test_make_candidate_pareto_sets_objectives():
    objs = [
        OptimizationTarget.THROUGHPUT_PER_GPU,
        OptimizationTarget.THROUGHPUT_PER_USER,
    ]
    c = make_candidate(
        {"used_gpus": 8}, REPORT, OptimizationTarget.PARETO, pareto_objectives=objs
    )
    assert c.objectives == {"throughput_per_gpu": 1250.0, "throughput_per_user": 50.0}
    assert c.score == 1250.0  # headline = first objective
    assert c.metrics["mean_output_token_throughput_per_user"] == 50.0


def test_make_candidate_pareto_requires_objectives():
    with pytest.raises(ValueError, match="pareto_objectives"):
        make_candidate({"used_gpus": 8}, REPORT, OptimizationTarget.PARETO)


def test_make_candidate_and_rank():
    cfg = {"used_gpus": 16, "deployment_mode": "agg"}
    c = make_candidate(cfg, REPORT, OptimizationTarget.GOODPUT_PER_GPU)
    assert c.used_gpus == 16
    assert c.score == 1000.0  # goodput / avg_gpu = 4000 / 4
    assert (
        c.metrics["goodput_output_throughput_tok_s"] == 4000.0
        and c.metrics["gpu_hours"] == 2.0
    )

    # gpu_hours=4.0 over the same 0.5h -> avg_gpu = 8.0 -> goodput_per_gpu = 4000/8 = 500
    a = make_candidate(
        {"used_gpus": 8},
        {**REPORT, "gpu_hours": 4.0},
        OptimizationTarget.GOODPUT_PER_GPU,
    )  # 500
    b = make_candidate(
        {"used_gpus": 16}, REPORT, OptimizationTarget.GOODPUT_PER_GPU
    )  # 1000
    tie = make_candidate(
        {"used_gpus": 8}, REPORT, OptimizationTarget.GOODPUT_PER_GPU
    )  # 1000, fewer gpus
    ranked = rank([a, b, tie])
    assert (
        ranked[0] is tie and ranked[1] is b and ranked[2] is a
    )  # 1000(8gpu), 1000(16gpu), 500
