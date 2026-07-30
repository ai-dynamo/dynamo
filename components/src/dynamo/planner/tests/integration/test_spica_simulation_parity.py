# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Replay parity gates for the Spica adapter/runner composition.

These are the non-KVBM goldens from the pre-refactor Spica integration suite,
rewired through ``ReplaySpec``, the published Planner adapter factory, and the
transitional Dynamo runner. The full-search case exercises real spawned workers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aisimulate.spica.adapter import CandidateContext, SweepContext
from aisimulate.spica.config import OptimizationTarget, SmartSearchConfig
from aisimulate.spica.deploy import build_backend_deployment
from aisimulate.spica.kv_estimate import resolve_backend_version
from aisimulate.spica.replay import ReplaySpec
from aisimulate.spica.sample import unroll_sample
from aisimulate.spica.score import objective_value
from aisimulate.spica.search import run_smart_search
from aisimulate.spica.search_space import enumerate_branches
from dynamo.planner.simulation import create_adapter
from dynamo.replay.simulation import DynamoReplayRunnerFactory

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.planner,
    pytest.mark.pre_merge,
    pytest.mark.filterwarnings("ignore:invalid escape sequence.*:SyntaxWarning"),
]

_REPO_ROOT = Path(__file__).resolve().parents[6]
TRACE = str(_REPO_ROOT / "aisimulate/tests/spica/data/mooncake_tiny.jsonl")


def _config(*, scaling_policy: list[str], **sweep_overrides) -> SmartSearchConfig:
    sweep = {"max_rounds": 1, "candidates_per_round": 2, "parallel_evals": 2}
    sweep.update(sweep_overrides)
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
                    "scaling_policy": scaling_policy,
                    "fpm_sampling": ["default"],
                    "load_sensitivity": ["default"],
                    "load_predictor_candidates": ["constant_last"],
                }
            }
        },
        workload={"trace_path": TRACE},
        sweep=sweep,
        goal={
            "target": "goodput_per_gpu",
            "sla": {"ttft_ms": 8000.0, "itl_ms": 200.0},
        },
    )


def _run_one(policy: str):
    config = _config(
        scaling_policy=[policy],
        candidates_per_round=1,
        parallel_evals=1,
    )
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


@pytest.mark.timeout(300)
def test_real_spawned_smart_search_returns_ranked_candidates() -> None:
    candidates = run_smart_search(
        _config(scaling_policy=["disabled", "load_180_5"]),
        runner_factory=DynamoReplayRunnerFactory(),
        show_progress=False,
    )

    assert candidates
    scores = [candidate.score for candidate in candidates]
    assert scores == sorted(scores, reverse=True)
    best = candidates[0]
    assert best.metrics["goodput_output_throughput_tok_s"] > 0.0
    assert best.metrics["gpu_hours"] > 0.0
    assert best.used_gpus <= 256
    avg_gpu = best.metrics["gpu_hours"] / (best.metrics["duration_ms"] / 3_600_000.0)
    assert best.score == pytest.approx(
        best.metrics["goodput_output_throughput_tok_s"] / avg_gpu
    )
