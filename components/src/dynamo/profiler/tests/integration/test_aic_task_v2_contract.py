# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the AIC Task-v2 boundary used by the Rapid profiler."""

from __future__ import annotations

import pytest
from aiconfigurator.cli.main import _execute_tasks, build_default_tasks
from aiconfigurator.sdk.task_v2 import Task

pytestmark = [
    pytest.mark.aic_full,
    pytest.mark.aiconfigurator,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.planner,
    pytest.mark.pre_merge,
    pytest.mark.timeout(60),
]


@pytest.fixture(autouse=True)
def _offline_aic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def test_rapid_task_v2_build_and_execute_contract() -> None:
    tasks = build_default_tasks(
        model_path="Qwen/Qwen3-32B",
        total_gpus=2,
        system="h200_sxm",
        backend="vllm",
        backend_version="0.14.0",
        isl=128,
        osl=8,
        ttft=100_000.0,
        tpot=100_000.0,
    )
    assert set(tasks) == {"agg", "disagg"}
    assert all(isinstance(task, Task) for task in tasks.values())

    agg = tasks["agg"]
    agg.pareto_sweep = False
    agg.agg_num_gpu_candidates = [1]
    agg.agg_tp_candidates = [1]
    agg.agg_pp_candidates = [1]
    agg.agg_dp_candidates = [1]
    agg.agg_moe_tp_candidates = [1]
    agg.agg_moe_ep_candidates = [1]
    agg.agg_cp_candidates = [1]

    chosen, best_configs, pareto_fronts, throughputs, latencies = _execute_tasks(
        {"agg": agg},
        mode="default",
        top_n=1,
    )

    assert chosen == "agg"
    assert not best_configs["agg"].empty
    assert not pareto_fronts["agg"].empty
    assert throughputs["agg"] > 0.0
    assert latencies["agg"]["ttft"] > 0.0
    assert latencies["agg"]["tpot"] > 0.0
