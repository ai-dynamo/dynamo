# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration against the REAL dynamo replay (no stubs).

Drives the full pipeline on a tiny mooncake trace: enumerate -> sample ->
build_deployment -> ReplayEvaluator (planner bridge) -> score, and the whole
``run_smart_search`` loop with the Vizier sampler.

Requires the dynamo bindings built with the ``aic-forward-pass`` Cargo feature
(the AIC perf model the mocker needs); skips otherwise. See README "Real replay".

Uses meta-llama/Meta-Llama-3.1-8B / gb200 / trtllm: a dense GQA model the AIC perf
DB fully covers (PASS in the gb200 support matrix). Bare ``deepseek-ai/DeepSeek-V3``
is *not* in that matrix and its MLA silicon grid has gaps, so it panics the perf
lookup -- a DB-coverage issue, unrelated to the pipeline under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("dynamo.mocker")

import dynamo._core as _core
import dynamo.replay.main as _replay_main
from dynamo.profiler.spica.config import (
    OptimizationTarget,
    SLATarget,
    SmartSearchConfig,
)
from dynamo.profiler.spica.deploy import build_deployment
from dynamo.profiler.spica.evaluator import ReplayEvaluator
from dynamo.profiler.spica.kv_estimate import resolve_backend_version
from dynamo.profiler.spica.sample import unroll_sample
from dynamo.profiler.spica.score import objective_value
from dynamo.profiler.spica.search import run_smart_search
from dynamo.profiler.spica.search_space import enumerate_branches

pytestmark = pytest.mark.filterwarnings(
    "ignore:\\[EXPERIMENTAL\\] Spica cannot apply KV-capacity filtering.*:UserWarning"
)

if not hasattr(_core, "RustEnginePerfModel"):
    pytest.skip(
        "dynamo bindings built without the aic-forward-pass feature (no AIC perf model)",
        allow_module_level=True,
    )

if not hasattr(_replay_main, "SyntheticWorkload"):
    pytest.skip(
        "installed dynamo predates the planner load-modes API (ai-dynamo/dynamo#10888); "
        "repin the [dynamo] dependency + rebuild to run these integration tests",
        allow_module_level=True,
    )

TRACE = str(Path(__file__).parent / "data" / "mooncake_tiny.jsonl")


def _config(**sweep_ov) -> SmartSearchConfig:
    # parallel_evals=2 exercises the real ProcessPool path (spawn + the real evaluator
    # in worker processes) end-to-end, not just the sequential fallback.
    sweep = {"max_rounds": 1, "candidates_per_round": 2, "parallel_evals": 2}
    sweep.update(sweep_ov)
    return SmartSearchConfig(
        search_space={
            "model_name": "meta-llama/Meta-Llama-3.1-8B",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 256,
        },
        workload={"trace_path": TRACE},
        sweep=sweep,
        goal={"target": "goodput_per_gpu", "sla": {"ttft_ms": 8000.0, "itl_ms": 200.0}},
    )


def test_real_bridge_emits_goodput_and_gpu_hours():
    """One scaling candidate through the real planner bridge -> goodput + gpu_hours."""
    cfg = _config()
    branch = enumerate_branches(cfg)[0]
    pc = branch.parallel_configs[0]
    bv = resolve_backend_version("gb200", "trtllm")
    selection = {
        "deployment_mode": "agg",
        "backend": "trtllm",
        "agg_max_num_batched_tokens": 16384,
        "agg_max_num_seqs": 512,
        "router_mode": "round_robin",
        "planner_scaling_policy": "load_180_5",  # load -> "easy mode", no FPM bootstrap
        "planner_fpm_sampling": "default",
        "planner_load_sensitivity": "default",
    }
    sample = unroll_sample(
        search_space=cfg.search_space, selection=selection, parallel_config=pc
    )
    plan = build_deployment(
        sample, backend_version=bv, planner_sla=SLATarget(ttft_ms=2000.0, itl_ms=30.0)
    )
    assert not plan.is_static  # scaling -> planner bridge path

    report = ReplayEvaluator(cfg.workload, cfg.goal).evaluate(plan)

    assert "goodput_output_throughput_tok_s" in report
    assert "gpu_hours" in report and report["gpu_hours"] > 0.0
    # score == goodput / avg_gpu, where avg_gpu = gpu_hours / e2e_hours (time-averaged
    # provisioned GPU count) -- computed from the real report.
    avg_gpu = report["gpu_hours"] / (report["duration_ms"] / 3_600_000.0)
    expected = report["goodput_output_throughput_tok_s"] / avg_gpu
    assert objective_value(report, OptimizationTarget.GOODPUT_PER_GPU) == pytest.approx(
        expected
    )


def test_static_path_emits_goodput():
    """A disabled/static candidate -> the plain run_trace_replay path, which (since
    dynamo#10849) also takes the goodput SLA -> still emits goodput + gpu_hours."""
    cfg = _config()
    pc = enumerate_branches(cfg)[0].parallel_configs[0]
    selection = {
        "deployment_mode": "agg",
        "backend": "trtllm",
        "agg_max_num_batched_tokens": 16384,
        "agg_max_num_seqs": 512,
        "router_mode": "round_robin",
        "planner_scaling_policy": "disabled",  # static -> plain replay (no planner)
        "planner_fpm_sampling": "default",
        "planner_load_sensitivity": "default",
    }
    sample = unroll_sample(
        search_space=cfg.search_space, selection=selection, parallel_config=pc
    )
    plan = build_deployment(
        sample,
        backend_version=resolve_backend_version("gb200", "trtllm"),
        planner_sla=cfg.goal.sla,
    )
    assert plan.is_static and plan.planner_config is None  # plain path

    report = ReplayEvaluator(cfg.workload, cfg.goal).evaluate(plan)
    assert report["goodput_output_throughput_tok_s"] > 0.0  # goodput on the static path
    assert report["gpu_hours"] > 0.0


def test_smart_search_returns_ranked_candidate():
    """The full loop (Vizier sampler + real replay) returns ranked candidates."""
    candidates = run_smart_search(_config())
    assert candidates, "expected at least one feasible candidate"
    # ranked best-first by goodput_per_gpu
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
    best = candidates[0]
    assert "goodput_output_throughput_tok_s" in best.metrics
    assert best.metrics["gpu_hours"] > 0.0
    assert best.used_gpus <= 256
    avg_gpu = best.metrics["gpu_hours"] / (best.metrics["duration_ms"] / 3_600_000.0)
    assert best.score == pytest.approx(
        best.metrics["goodput_output_throughput_tok_s"] / avg_gpu
    )
