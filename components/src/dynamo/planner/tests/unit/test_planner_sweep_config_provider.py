# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optional-dependency preflight must run before the simulation imports.
# ruff: noqa: E402

"""Unit tests for the Planner-owned Sweeper sweep configuration provider."""

from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

import dynamo.planner.simulation.provider as planner_provider_module
from aisimulate.sweeper.provider import CandidateContext, SweepContext
from aisimulate.sweeper.replay import BackendDeploymentSpec
from dynamo.planner.simulation import create_provider
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
    adapter = create_provider()

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


def test_pareto_goodput_uses_sla_planner_target() -> None:
    adapter = create_provider()
    context = replace(
        _sweep_context(target="pareto"),
        goal={
            "target": "pareto",
            "pareto_objectives": ["goodput_per_gpu", "throughput_per_user"],
            "sla": {"ttft_ms": 2000.0, "itl_ms": 30.0},
        },
    )

    plan = adapter.generate_search_space(
        {
            "scaling_policy": ["throughput_180_5"],
            "fpm_sampling": ["default"],
            "load_sensitivity": ["default"],
            "load_predictor_candidates": ["constant_last"],
        },
        context,
    )
    replay_spec = adapter.materialize_replay(
        plan,
        {
            "scaling_policy": "throughput_180_5",
            "fpm_sampling": "default",
            "load_sensitivity": "default",
        },
        _candidate_context(),
    )

    assert isinstance(plan.state, dict)
    assert plan.state["optimization_target"] == "sla"
    planner_config = replay_spec.runtime_hooks[0].config["planner_config"]
    assert isinstance(planner_config, dict)
    assert planner_config["optimization_target"] == "sla"
    assert planner_config["ttft_ms"] == 2000.0
    assert planner_config["itl_ms"] == 30.0


def test_default_pareto_keeps_throughput_planner_target() -> None:
    context = replace(
        _sweep_context(target="pareto"),
        goal={"target": "pareto", "pareto_objectives": None, "sla": None},
    )

    plan = create_provider().generate_search_space(
        {"scaling_policy": ["disabled"]},
        context,
    )

    assert isinstance(plan.state, dict)
    assert plan.state["optimization_target"] == "throughput"


def test_disabled_policy_materializes_no_runtime_hook() -> None:
    adapter = create_provider()
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
    adapter = create_provider()
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
    assert replay_spec.config == {
        "scaling_policy": "throughput_180_5",
        **expected,
    }
    assert replay_spec.runtime_hooks[0].config == {"planner_config": expected}


def test_custom_float_interval_resolves_predictor_and_preserves_selection() -> None:
    adapter = create_provider()
    custom_policy = {
        "enable_throughput_scaling": True,
        "enable_load_scaling": False,
        "throughput_adjustment_interval_seconds": 180.0,
        "load_adjustment_interval_seconds": 5,
    }
    plan = adapter.generate_search_space(
        {
            "scaling_policy": [custom_policy],
            "fpm_sampling": ["default"],
            "load_sensitivity": ["default"],
            "load_predictor_candidates": ["constant_last"],
        },
        _sweep_context(),
    )

    replay_spec = adapter.materialize_replay(
        plan,
        {
            "scaling_policy": custom_policy,
            "fpm_sampling": "default",
            "load_sensitivity": "default",
        },
        _candidate_context(),
    )

    assert replay_spec.config["scaling_policy"] == custom_policy
    assert replay_spec.config["throughput_adjustment_interval_seconds"] == 180
    assert replay_spec.config["load_predictor"] == "constant"
    planner_config = replay_spec.runtime_hooks[0].config["planner_config"]
    assert isinstance(planner_config, dict)
    assert "scaling_policy" not in planner_config
    assert planner_config["load_predictor"] == "constant"


def test_disaggregated_scaling_preserves_both_engine_gpu_counts() -> None:
    adapter = create_provider()
    plan = adapter.generate_search_space(
        {
            "scaling_policy": ["load_180_5"],
            "fpm_sampling": ["default"],
            "load_sensitivity": ["default"],
            "prefill_min_endpoint": 2,
            "decode_min_endpoint": 3,
        },
        SweepContext(
            core_search_space={
                "deployment_mode": ["disagg"],
                "gpu_budget": 32,
            },
            workload={"trace_path": "trace.jsonl"},
            goal={"target": "throughput"},
            show_progress=False,
        ),
    )
    context = CandidateContext(
        sample={
            "deployment_mode": "disagg",
            "gpu_budget": 32,
            "prefill_tp": 2,
            "prefill_attention_dp": 2,
            "decode_tp": 4,
            "decode_attention_dp": 2,
        },
        backend_deployment=BackendDeploymentSpec(
            deployment_mode="disagg",
            backend="vllm",
            backend_version="0.11.0",
        ),
    )

    replay_spec = adapter.materialize_replay(
        plan,
        {
            "scaling_policy": "load_180_5",
            "fpm_sampling": "default",
            "load_sensitivity": "default",
        },
        context,
    )

    assert replay_spec.config["mode"] == "disagg"
    assert replay_spec.config["prefill_engine_num_gpu"] == 4
    assert replay_spec.config["decode_engine_num_gpu"] == 8
    assert replay_spec.config["prefill_min_endpoint"] == 2
    assert replay_spec.config["decode_min_endpoint"] == 3


def test_load_predictor_diagnostics_are_strict_json() -> None:
    state = LoadPredictorResult(
        losses={180: {"constant_last": float("inf")}}
    ).to_state()

    assert state["losses"] == {"180": {"constant_last": None}}


def test_provider_owns_its_adapter_and_hook_abi_versions() -> None:
    adapter = create_provider()

    assert planner_provider_module._PROVIDER_API_VERSION == 1
    assert planner_provider_module._PLANNER_HOOK_API_VERSION == 1
    assert adapter.api_version == planner_provider_module._PROVIDER_API_VERSION


def test_policy_pruning_diagnostics_remain_visible(monkeypatch) -> None:
    messages = []
    monkeypatch.setattr(planner_provider_module.tqdm, "write", messages.append)

    create_provider().generate_search_space(
        {"scaling_policy": ["disabled", "throughput_180_5"]},
        replace(_sweep_context(target="throughput"), show_progress=True),
    )

    assert len(messages) == 1
    assert "dropped 1 throughput-scaling policy" in messages[0]
