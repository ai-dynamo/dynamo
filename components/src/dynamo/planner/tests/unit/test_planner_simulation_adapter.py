# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for the Planner-owned Spica simulation adapter."""

from __future__ import annotations

import pytest

from aisimulate.spica.adapter import CandidateContext, SweepContext
from aisimulate.spica.replay import BackendDeploymentSpec
from dynamo.planner.simulation import create_adapter
from dynamo.planner.simulation.load_predictor import LoadPredictorResult

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _sweep_context(
    *,
    target: str = "goodput_per_gpu",
    sla: dict | None = None,
) -> SweepContext:
    return SweepContext(
        core_search_space={
            "deployment_mode": ["agg"],
            "gpu_budget": 32,
            "min_gpu_budget": 4,
        },
        workload={
            "trace_path": None,
            "isl": 512,
            "osl": 128,
            "concurrency": 16,
        },
        goal={
            "target": target,
            "sla": sla or {"ttft_ms": 2000.0, "itl_ms": 30.0},
        },
        show_progress=False,
    )


def _candidate_context() -> CandidateContext:
    sample = {
        "deployment_mode": "agg",
        "gpu_budget": 32,
        "min_gpu_budget": 4,
        "tp": 4,
        "attention_dp": 2,
    }
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.0",
        agg_engine_args={"engine_type": "vllm"},
        num_workers=2,
    )
    return CandidateContext(sample=sample, backend_deployment=deployment)


def test_non_goodput_filters_predictive_throughput_policies() -> None:
    adapter = create_adapter()

    plan = adapter.generate_search_space(
        {
            "scaling_policy": [
                "disabled",
                "throughput_180_5",
                "load_180_5",
                "hybrid_600_5",
            ]
        },
        _sweep_context(target="throughput"),
    )

    assert plan.fragment.choices_by_branch["agg"]["scaling_policy"] == [
        "disabled",
        "load_180_5",
    ]
    assert plan.diagnostics["dropped_scaling_policies"] == [
        "throughput_180_5",
        "hybrid_600_5",
    ]
    assert plan.potential_runtime_hooks[0].provider == "dynamo.planner"


def test_disabled_policy_materializes_no_runtime_hook() -> None:
    adapter = create_adapter()
    plan = adapter.generate_search_space(
        {"scaling_policy": ["disabled"]},
        _sweep_context(),
    )

    replay_spec = adapter.materialize_replay(
        plan,
        {"scaling_policy": "disabled"},
        _candidate_context(),
    )

    assert replay_spec.config == {
        "scaling_policy": "disabled",
        "enable_throughput_scaling": False,
        "enable_load_scaling": False,
        "throughput_adjustment_interval_seconds": None,
        "load_adjustment_interval_seconds": None,
    }
    assert replay_spec.runtime_hooks == ()


def test_scaling_policy_materializes_legacy_planner_payload() -> None:
    adapter = create_adapter()
    plan = adapter.generate_search_space(
        {
            "scaling_policy": ["throughput_180_5"],
            "fpm_sampling": ["fine"],
            "load_sensitivity": ["conservative"],
            "load_predictor_candidates": ["constant_last"],
            "min_endpoint": 2,
        },
        _sweep_context(),
    )

    replay_spec = adapter.materialize_replay(
        plan,
        {
            "scaling_policy": "throughput_180_5",
            "fpm_sampling": "fine",
            "load_sensitivity": "conservative",
        },
        _candidate_context(),
    )

    expected = {
        "mode": "agg",
        "optimization_target": "sla",
        "report_interval_hours": None,
        "live_dashboard_port": 0,
        "metric_pulling_prometheus_extra_query_params": None,
        "enable_throughput_scaling": True,
        "enable_load_scaling": False,
        "throughput_adjustment_interval_seconds": 180,
        "load_adjustment_interval_seconds": 5,
        "max_num_fpm_samples": 128,
        "fpm_sample_bucket_size": 64,
        "load_scaling_down_sensitivity": 90,
        "load_min_observations": 8,
        "load_predictor": "constant",
        "load_predictor_log1p": False,
        "max_gpu_budget": 32,
        "min_gpu_budget": 4,
        "min_endpoint": 2,
        "decode_engine_num_gpu": 8,
        "ttft_ms": 2000.0,
        "itl_ms": 30.0,
    }
    assert replay_spec.config == expected
    assert replay_spec.runtime_hooks[0].config == {"planner_config": expected}


def test_load_predictor_diagnostics_are_strict_json() -> None:
    state = LoadPredictorResult(
        losses={180: {"constant_last": float("inf")}}
    ).to_state()

    assert state["losses"] == {"180": {"constant_last": None}}
