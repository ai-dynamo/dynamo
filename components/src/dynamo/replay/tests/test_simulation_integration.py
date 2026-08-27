# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optional-dependency preflight must run before the simulation imports.
# ruff: noqa: E402

"""Real Replay integration coverage for the Sweeper adapter/runner boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate.sweeper.config import OptimizationTarget, SmartSearchConfig
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.kv_estimate import resolve_backend_version
from aisimulate.sweeper.provider import CandidateContext, SweepContext
from aisimulate.sweeper.replay import ReplaySpec
from aisimulate.sweeper.sample import unroll_sample
from aisimulate.sweeper.sampler import Suggestion
from aisimulate.sweeper.score import objective_value
from aisimulate.sweeper.search import Sweeper
from aisimulate.sweeper.search_space import enumerate_branches

from dynamo.planner.simulation import create_provider as create_planner_provider
from dynamo.replay.simulation import DynamoReplayRunnerFactory
from dynamo.router.simulation import create_provider as create_router_provider

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.planner,
    pytest.mark.filterwarnings("ignore:invalid escape sequence.*:SyntaxWarning"),
    pytest.mark.filterwarnings("ignore:invalid escape sequence.*:DeprecationWarning"),
]

_TRACE = str(Path(__file__).resolve().parent / "data/mooncake_tiny.jsonl")


def _config(
    *,
    scaling_policy: str,
    candidates_per_round: int = 1,
    parallel_evals: int = 1,
    include_router: bool = False,
) -> SmartSearchConfig:
    # Replay consumes model metadata and the AIC performance database; it does
    # not load model weights or require a GPU. This model/backend pair is kept
    # because it has complete coverage in the GB200 performance database.
    adapters = {
        "dynamo.planner": {
            "search_space": {
                "scaling_policy": {"preset": [scaling_policy]},
                "fpm_sampling": {"preset": ["default"]},
                "load_sensitivity": {"preset": ["default"]},
                "load_predictor": {"preset": ["constant_last"]},
            }
        }
    }
    if include_router:
        adapters["dynamo.router"] = {
            "search_space": {
                "mode": ["kv_router"],
                "overlap_score_credit": [0.5],
                "prefill_load_scale": [1.0],
                "temperature": [0.2],
            }
        }

    return SmartSearchConfig(
        search_space={
            "model_name": "meta-llama/Meta-Llama-3.1-8B",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 256,
        },
        adapters=adapters,
        # Slow the trace enough for load_180_5 to execute a Planner tick.
        workload={"trace_path": _TRACE, "arrival_speedup_ratio": 0.5},
        sweep={
            "max_rounds": 1,
            "candidates_per_round": candidates_per_round,
            "parallel_evals": parallel_evals,
            "max_eval_seconds": 240,
        },
        goal={
            "target": "goodput_per_gpu",
            "sla": {"ttft_ms": 8000.0, "itl_ms": 200.0},
        },
    )


class _TwoCandidateSampler:
    """Choose two deterministic valid candidates without depending on Vizier."""

    def __init__(self, branch, study_id, objectives=None):
        del study_id, objectives
        self.branch = branch

    def suggest(self, count):
        assert count == 2
        suggestions = []
        for index in range(count):
            selection = {
                name: values[min(index, len(values) - 1)]
                for name, values in self.branch.knob_choices.items()
            }
            selection["deployment_mode"] = self.branch.deployment_mode
            suggestions.append(
                Suggestion(
                    selection=selection,
                    parallel_config=self.branch.parallel_configs[
                        min(index, len(self.branch.parallel_configs) - 1)
                    ],
                    handle=index,
                )
            )
        return suggestions

    def observe(self, suggestion, metrics):
        del suggestion, metrics

    def observe_infeasible(self, suggestion, reason):
        pytest.fail(f"unexpected infeasible suggestion {suggestion}: {reason}")


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

    adapter = create_planner_provider()
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


@pytest.mark.pre_merge
@pytest.mark.timeout(300)
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


@pytest.mark.pre_merge
@pytest.mark.timeout(300)
def test_real_static_path_preserves_goodput() -> None:
    report, adapter_spec = _run_one("disabled")

    assert adapter_spec.runtime_hooks == ()
    assert report.metrics["goodput_output_throughput_tok_s"] > 0.0
    assert report.metrics["gpu_hours"] > 0.0
    assert "planner_total_ticks" not in report.metrics


@pytest.mark.pre_merge
@pytest.mark.post_merge
@pytest.mark.timeout(300)
def test_sweeper_runs_real_dynamo_replay_in_spawned_workers() -> None:
    config = _config(
        scaling_policy="load_180_5",
        candidates_per_round=2,
        parallel_evals=2,
        include_router=True,
    )

    result = Sweeper(
        runner_factory=DynamoReplayRunnerFactory(),
        providers={
            "dynamo.planner": create_planner_provider(),
            "dynamo.router": create_router_provider(),
        },
        sampler_factory=_TwoCandidateSampler,
        show_progress=False,
    ).run(config)
    candidates = result.selected_candidates

    assert len(candidates) == 2
    assert all(
        candidate.metrics["output_throughput_tok_s"] > 0.0 for candidate in candidates
    )
    assert all(candidate.metrics["gpu_hours"] > 0.0 for candidate in candidates)
    assert all(
        candidate.metrics["planner_total_ticks"] >= 1 for candidate in candidates
    )
    assert all(
        candidate.config["adapters"]["dynamo.router"]["mode"] == "kv_router"
        for candidate in candidates
    )
