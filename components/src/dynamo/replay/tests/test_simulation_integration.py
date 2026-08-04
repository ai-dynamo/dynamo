# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optional-dependency preflight must run before the simulation imports.
# ruff: noqa: E402

"""Real Replay integration coverage for the Spica adapter/runner boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip(
    "aisimulate.spica",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate.spica.adapter import CandidateContext, SweepContext
from aisimulate.spica.config import OptimizationTarget, SmartSearchConfig
from aisimulate.spica.deploy import build_backend_deployment
from aisimulate.spica.kv_estimate import resolve_backend_version
from aisimulate.spica.replay import ReplaySpec
from aisimulate.spica.sample import unroll_sample
from aisimulate.spica.score import objective_value
from aisimulate.spica.search_space import enumerate_branches
from dynamo.planner.simulation import create_adapter
from dynamo.replay.simulation import DynamoReplayRunnerFactory

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.planner,
    pytest.mark.pre_merge,
    pytest.mark.timeout(300),
    pytest.mark.filterwarnings("ignore:invalid escape sequence.*:SyntaxWarning"),
    pytest.mark.filterwarnings("ignore:invalid escape sequence.*:DeprecationWarning"),
]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_TRACE = str(_REPO_ROOT / "aisimulate/tests/spica/data/mooncake_tiny.jsonl")


def _config(*, scaling_policy: str) -> SmartSearchConfig:
    # Replay consumes model metadata and the AIC performance database; it does
    # not load model weights or require a GPU. This model/backend pair is kept
    # because it has complete coverage in the GB200 performance database.
    return SmartSearchConfig(
        search_space={
            "model_name": "meta-llama/Meta-Llama-3.1-8B",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 256,
        },
        adapters={
            "dynamo.planner": {
                "search_space": {
                    "scaling_policy": [scaling_policy],
                    "fpm_sampling": ["default"],
                    "load_sensitivity": ["default"],
                    "load_predictor_candidates": ["constant_last"],
                }
            }
        },
        # Slow the trace enough for load_180_5 to execute a Planner tick.
        workload={"trace_path": _TRACE, "arrival_speedup_ratio": 0.5},
        sweep={"max_rounds": 1, "candidates_per_round": 1, "parallel_evals": 1},
        goal={
            "target": "goodput_per_gpu",
            "sla": {"ttft_ms": 8000.0, "itl_ms": 200.0},
        },
    )


def _run_one(policy: str):
    config = _config(scaling_policy=policy)
    runner_factory = DynamoReplayRunnerFactory()
    branch = enumerate_branches(
        config,
        runner_capabilities=runner_factory.capabilities(),
    )[0]
    parallel_config = branch.parallel_configs[0]
    selection = {
        "deployment_mode": "agg",
        "backend": "trtllm",
        "agg_max_num_batched_tokens": 16384,
        "agg_max_num_seqs": 512,
    }
    sample = unroll_sample(
        search_space=config.search_space,
        selection=selection,
        parallel_config=parallel_config,
    )
    backend_version = resolve_backend_version("gb200", "trtllm")
    sample["backend_version"] = backend_version
    backend_deployment = build_backend_deployment(
        sample,
        backend_version=backend_version,
    )

    adapter = create_adapter()
    adapter_plan = adapter.generate_search_space(
        config.adapters["dynamo.planner"].search_space,
        SweepContext(
            core_search_space=config.search_space.model_dump(mode="json"),
            workload=config.workload.model_dump(mode="json"),
            goal=config.goal.model_dump(mode="json"),
            show_progress=False,
        ),
    )
    adapter_selection = {"scaling_policy": policy}
    if policy != "disabled":
        adapter_selection.update(
            fpm_sampling="default",
            load_sensitivity="default",
        )
    adapter_spec = adapter.materialize_replay(
        adapter_plan,
        adapter_selection,
        CandidateContext(
            sample=sample,
            backend_deployment=backend_deployment,
        ),
    )
    replay_spec = ReplaySpec(
        backend_deployment=backend_deployment,
        workload=config.workload.model_dump(mode="json"),
        goal=config.goal.model_dump(mode="json"),
        adapters={"dynamo.planner": adapter_spec},
    )

    runner = runner_factory.create(0)
    try:
        return runner.run(replay_spec), adapter_spec
    finally:
        runner.close()


def test_real_planner_bridge_preserves_goodput_and_gpu_hours() -> None:
    report, adapter_spec = _run_one("load_180_5")

    assert adapter_spec.runtime_hooks
    assert report.metrics["goodput_output_throughput_tok_s"] > 0.0
    assert report.metrics["gpu_hours"] > 0.0
    assert report.metrics["planner_total_ticks"] >= 1
    avg_gpu = report.metrics["gpu_hours"] / (
        report.metrics["duration_ms"] / 3_600_000.0
    )
    expected = report.metrics["goodput_output_throughput_tok_s"] / avg_gpu
    assert objective_value(
        report.metrics,
        OptimizationTarget.GOODPUT_PER_GPU,
    ) == pytest.approx(expected)


def test_real_static_path_preserves_goodput() -> None:
    report, adapter_spec = _run_one("disabled")

    assert adapter_spec.runtime_hooks == ()
    assert report.metrics["goodput_output_throughput_tok_s"] > 0.0
    assert report.metrics["gpu_hours"] > 0.0
    assert "planner_total_ticks" not in report.metrics
