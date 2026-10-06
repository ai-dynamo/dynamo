# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""Unit tests for the Planner-owned Sweeper sweep configuration provider."""

from __future__ import annotations

import warnings
from dataclasses import replace

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate.config_adapter import (
    PredictionAdapterContext,
    RecommendationAdapterContext,
)
from aisimulate.sweeper.provider import (
    CandidateContext,
    InfeasibleCandidate,
    SweepContext,
)
from aisimulate.sweeper.replay import BackendDeploymentSpec

import dynamo.planner.simulation.provider as planner_provider_module
from dynamo.planner.config.planner_config import PlannerConfig
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


def test_structured_preset_subitems_preserve_all_families() -> None:
    space = planner_provider_module.PlannerSearchSpace.model_validate(
        {
            "scaling_policy": {"preset": ["hybrid_180_5"]},
            "fpm_sampling": {"preset": ["fine"]},
            "load_sensitivity": {"preset": ["conservative"]},
            "load_predictor": {"preset": ["kalman_reactive_log1p"]},
        }
    )

    assert space.scaling_policy.preset == ["hybrid_180_5"]
    assert space.fpm_sampling.preset == ["fine"]
    assert space.load_sensitivity.preset == ["conservative"]
    assert space.load_predictor.preset == ["kalman_reactive_log1p"]


def test_custom_predictor_preset_is_completed_with_every_knob() -> None:
    space = planner_provider_module.PlannerSearchSpace.model_validate(
        {
            "load_predictor": {
                "preset": [{"load_predictor": "kalman"}],
            }
        }
    )

    entry = space.load_predictor.preset[0]
    assert isinstance(entry, dict)
    assert set(entry) == planner_provider_module._PREDICTOR_KEYS
    assert entry["load_predictor"] == "kalman"
    assert entry["prophet_window_size"] == 50
    assert entry["kalman_min_points"] == 5


def test_custom_predictor_preset_rejects_unknown_knob() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        planner_provider_module.PlannerSearchSpace.model_validate(
            {
                "load_predictor": {
                    "preset": [
                        {
                            "load_predictor": "constant",
                            "unknown": 1,
                        }
                    ]
                }
            }
        )


def test_structured_custom_preset_rejects_missing_subitem_knob() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        planner_provider_module.PlannerSearchSpace.model_validate(
            {
                "scaling_policy": {
                    "preset": [{"enable_throughput_scaling": True}],
                }
            }
        )


def test_legacy_flat_presets_warn_and_remain_compatible() -> None:
    with pytest.warns(FutureWarning, match="removed after the 1.5 release"):
        space = planner_provider_module.PlannerSearchSpace.model_validate(
            {
                "scaling_policy": ["throughput_180_5"],
                "fpm_sampling": ["default"],
                "load_sensitivity": ["default"],
                "load_predictor_candidates": ["constant_last"],
            }
        )

    assert space.scaling_policy.preset == ["throughput_180_5"]
    assert space.fpm_sampling.preset == ["default"]
    assert space.load_sensitivity.preset == ["default"]
    assert space.load_predictor.preset == ["constant_last"]


def test_legacy_warning_points_to_user_callsite() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_provider().generate_search_space(
            {"scaling_policy": ["disabled"]},
            _sweep_context(),
        )

    assert len(caught) == 1
    assert caught[0].category is FutureWarning
    assert caught[0].filename == __file__


def test_legacy_and_structured_inputs_generate_identical_plans() -> None:
    adapter = create_provider()
    structured = {
        "scaling_policy": {"preset": ["throughput_180_5"]},
        "fpm_sampling": {"preset": ["fine"]},
        "load_sensitivity": {"preset": ["conservative"]},
        "load_predictor": {"preset": ["constant_last"]},
        "min_endpoint": 2,
    }
    legacy = {
        "scaling_policy": ["throughput_180_5"],
        "fpm_sampling": ["fine"],
        "load_sensitivity": ["conservative"],
        "load_predictor_candidates": ["constant_last"],
        "min_endpoint": 2,
    }

    structured_plan = adapter.generate_search_space(structured, _sweep_context())
    with pytest.warns(FutureWarning, match="removed after the 1.5 release"):
        legacy_plan = adapter.generate_search_space(legacy, _sweep_context())

    assert legacy_plan == structured_plan
    selection = {
        "scaling_policy": "throughput_180_5",
        "fpm_sampling": "fine",
        "load_sensitivity": "conservative",
    }
    assert adapter.materialize_replay(
        legacy_plan,
        selection,
        _candidate_context(),
    ) == adapter.materialize_replay(
        structured_plan,
        selection,
        _candidate_context(),
    )


def test_pre_refactor_serialized_plan_state_still_materializes() -> None:
    adapter = create_provider()
    structured_plan = adapter.generate_search_space(
        {
            "scaling_policy": {"preset": ["throughput_180_5"]},
            "fpm_sampling": {"preset": ["fine"]},
            "load_sensitivity": {"preset": ["conservative"]},
            "load_predictor": {"preset": ["constant_last"]},
        },
        _sweep_context(),
    )
    legacy_state = dict(structured_plan.state)
    legacy_state["search_space"] = {
        "scaling_policy": ["throughput_180_5"],
        "fpm_sampling": ["fine"],
        "load_sensitivity": ["conservative"],
        "load_predictor_candidates": ["constant_last"],
    }
    legacy_plan = replace(structured_plan, state=legacy_state)
    selection = {
        "scaling_policy": "throughput_180_5",
        "fpm_sampling": "fine",
        "load_sensitivity": "conservative",
    }

    with pytest.warns(FutureWarning, match="removed after the 1.5 release"):
        legacy_spec = adapter.materialize_replay(
            legacy_plan,
            selection,
            _candidate_context(),
        )

    assert legacy_spec == adapter.materialize_replay(
        structured_plan,
        selection,
        _candidate_context(),
    )


def test_legacy_predictor_field_conflicts_with_structured_subitem() -> None:
    with pytest.raises(
        ValueError,
        match="load_predictor_candidates cannot be combined with load_predictor",
    ):
        planner_provider_module.PlannerSearchSpace.model_validate(
            {
                "load_predictor": {"preset": ["constant_last"]},
                "load_predictor_candidates": ["arima_raw"],
            }
        )


def test_legacy_partial_predictor_mapping_is_completed() -> None:
    with pytest.warns(FutureWarning, match="removed after the 1.5 release"):
        space = planner_provider_module.PlannerSearchSpace.model_validate(
            {
                "load_predictor_candidates": [
                    {
                        "load_predictor": "kalman",
                        "kalman_q_level": 3.0,
                    }
                ]
            }
        )

    entry = space.load_predictor.preset[0]
    assert isinstance(entry, dict)
    assert set(entry) == planner_provider_module._PREDICTOR_KEYS
    assert entry["kalman_q_level"] == 3.0
    assert entry["kalman_min_points"] == 5


def test_non_goodput_filters_predictive_throughput_policies() -> None:
    adapter = create_provider()

    plan = adapter.generate_search_space(
        {
            "scaling_policy": {
                "preset": [
                    "disabled",
                    "throughput_180_5",
                    "load_180_5",
                    "hybrid_600_5",
                ]
            }
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
            "scaling_policy": {"preset": ["throughput_180_5"]},
            "fpm_sampling": {"preset": ["default"]},
            "load_sensitivity": {"preset": ["default"]},
            "load_predictor": {"preset": ["constant_last"]},
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
        {"scaling_policy": {"preset": ["disabled"]}},
        context,
    )

    assert isinstance(plan.state, dict)
    assert plan.state["optimization_target"] == "throughput"


def test_disabled_policy_materializes_no_runtime_hook() -> None:
    adapter = create_provider()
    plan = adapter.generate_search_space(
        {"scaling_policy": {"preset": ["disabled"]}},
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
            "scaling_policy": {"preset": ["throughput_180_5"]},
            "fpm_sampling": {"preset": ["fine"]},
            "load_sensitivity": {"preset": ["conservative"]},
            "load_predictor": {"preset": ["constant_last"]},
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
        "max_throughput_scaling_replicas": 8,
        "max_num_fpm_samples": 128,
        "fpm_sample_bucket_size": 64,
        "load_predictor": "constant",
        "load_predictor_log1p": False,
        "max_gpu_budget": 32,
        "min_gpu_budget": 4,
        "min_endpoint": 2,
        "decode_engine_num_gpu": 8,
        "ttft_ms": 2000.0,
        "itl_ms": 30.0,
    }
    actual = replay_spec.runtime_hooks[0].config["planner_config"]
    assert PlannerConfig.model_validate(actual) == PlannerConfig.model_validate(
        expected
    )
    assert replay_spec.config == {"scaling_policy": "throughput_180_5", **actual}


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
            "scaling_policy": {"preset": [custom_policy]},
            "fpm_sampling": {"preset": ["default"]},
            "load_sensitivity": {"preset": ["default"]},
            "load_predictor": {"preset": ["constant_last"]},
            "max_throughput_scaling_replicas": 3,
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
    assert replay_spec.config["max_throughput_scaling_replicas"] == 3
    assert replay_spec.config["load_predictor"] == "constant"
    planner_config = replay_spec.runtime_hooks[0].config["planner_config"]
    assert isinstance(planner_config, dict)
    assert "scaling_policy" not in planner_config
    assert planner_config["load_predictor"] == "constant"
    assert planner_config["max_throughput_scaling_replicas"] == 3


def test_disaggregated_scaling_preserves_both_engine_gpu_counts() -> None:
    adapter = create_provider()
    plan = adapter.generate_search_space(
        {
            "scaling_policy": {"preset": ["load_180_5"]},
            "fpm_sampling": {"preset": ["default"]},
            "load_sensitivity": {"preset": ["default"]},
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
    assert adapter.config_adapter_api_version == 3
    assert adapter.section == "planner"


def test_public_planner_validation_is_owned_by_dynamo_adapter() -> None:
    adapter = create_provider()
    prediction_context = PredictionAdapterContext(engine={}, traffic={}, evaluation={})
    recommendation_context = RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={},
        optimization={},
        sweep=_sweep_context(target="goodput_per_gpu"),
    )

    assert adapter.compile_prediction({}, prediction_context).config == {
        "policy": "disabled"
    }
    plan = adapter.compile_recommendation({}, recommendation_context)
    assert plan.fragment.choices_by_branch["agg"]["policy"] == [
        "disabled",
        "enabled",
    ]

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        adapter.compile_prediction({"not_a_planner_knob": True}, prediction_context)
    with pytest.raises(ValueError, match="cannot be combined"):
        adapter.compile_recommendation(
            {
                "scaling_policy": {"preset": "default"},
                "enable_load_scaling": {"choices": [False, True]},
            },
            recommendation_context,
        )


def test_public_planner_prediction_requires_complete_sla() -> None:
    with pytest.raises(ValueError, match="Planner throughput scaling requires"):
        create_provider().compile_prediction(
            {
                "policy": "enabled",
                "target": "sla",
                "enable_throughput_scaling": True,
                "enable_load_scaling": False,
            },
            _prediction_context(),
        )


def test_planner_default_preset_conflicts_and_off_exposes_leaf_dimensions() -> None:
    adapter = create_provider()
    context = RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={"sla": {"ttft_ms": 2000.0, "itl_ms": 30.0}},
        optimization={"target": "goodput_per_gpu"},
        sweep=_sweep_context(target="goodput_per_gpu"),
    )
    with pytest.raises(ValueError, match="default preset"):
        adapter.compile_recommendation(
            {"enable_load_scaling": {"choices": [False, True]}}, context
        )

    plan = adapter.compile_recommendation(
        {
            "scaling_policy": {"preset": False},
            "enable_throughput_scaling": {"choices": [False, True]},
            "enable_load_scaling": {"choices": [False, True]},
            "throughput_adjustment_interval_seconds": {"choices": [180, 600]},
            "load_adjustment_interval_seconds": {"choices": [5, 10]},
        },
        context,
    )
    choices = plan.fragment.choices_by_branch["agg"]
    assert "scaling_policy" not in choices
    assert {
        "enable_throughput_scaling",
        "enable_load_scaling",
        "throughput_adjustment_interval_seconds",
        "load_adjustment_interval_seconds",
    }.issubset(choices)


def test_fpm_bucket_range_keeps_only_perfect_square_choices() -> None:
    context = RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={"sla": {"ttft_ms": 2000.0, "itl_ms": 30.0}},
        optimization={"target": "goodput"},
        sweep=_sweep_context(target="goodput"),
    )
    plan = create_provider().compile_recommendation(
        {
            "fpm_sampling": {"preset": False},
            "max_num_fpm_samples": 64,
            "fpm_sample_bucket_size": {"range": {"min": 4, "max": 64, "step": 4}},
        },
        context,
    )

    assert plan.fragment.choices_by_branch["agg"]["fpm_sample_bucket_size"] == [
        4,
        16,
        36,
        64,
    ]

    with pytest.raises(ValueError, match="contains no perfect squares"):
        create_provider().compile_recommendation(
            {
                "fpm_sampling": {"preset": False},
                "max_num_fpm_samples": 64,
                "fpm_sample_bucket_size": {"range": {"min": 5, "max": 8, "step": 1}},
            },
            context,
        )


def test_custom_fpm_preset_validation_ignores_prometheus_environment(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("PROMETHEUS_EXTRA_QUERY_PARAMS", "missing-equals")
    monkeypatch.setenv("PROMETHEUS_CA_BUNDLE", str(tmp_path / "missing-ca.pem"))
    context = RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={"sla": {"ttft_ms": 2000.0, "itl_ms": 30.0}},
        optimization={"target": "goodput"},
        sweep=_sweep_context(target="goodput"),
    )
    preset = {"max_num_fpm_samples": 64, "fpm_sample_bucket_size": 16}
    plan = create_provider().compile_recommendation(
        {"fpm_sampling": {"preset": [preset]}}, context
    )
    assert plan.fragment.choices_by_branch["agg"]["fpm_sampling"] == [preset]

    with pytest.raises(ValueError, match="must be a perfect square"):
        create_provider().compile_recommendation(
            {
                "fpm_sampling": {
                    "preset": [
                        {
                            "max_num_fpm_samples": 64,
                            "fpm_sample_bucket_size": 5,
                        }
                    ]
                }
            },
            context,
        )


def test_custom_disabled_scaling_preset_uses_null_inactive_intervals() -> None:
    context = RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={"sla": {"ttft_ms": 2000.0, "itl_ms": 30.0}},
        optimization={"target": "goodput"},
        sweep=_sweep_context(target="goodput"),
    )
    disabled = {
        "enable_throughput_scaling": False,
        "enable_load_scaling": False,
        "throughput_adjustment_interval_seconds": None,
        "load_adjustment_interval_seconds": None,
    }
    adapter = create_provider()
    plan = adapter.compile_recommendation(
        {"policy": "enabled", "scaling_policy": {"preset": [disabled]}}, context
    )
    assert plan.fragment.choices_by_branch["agg"]["scaling_policy"] == [disabled]
    spec = adapter.materialize_candidate(
        plan, {"policy": "enabled", "scaling_policy": disabled}, _candidate_context()
    )
    assert spec.config == {"policy": "disabled"}
    assert spec.runtime_hooks == ()

    invalid = dict(disabled)
    invalid["throughput_adjustment_interval_seconds"] = 180
    invalid["load_adjustment_interval_seconds"] = 5
    with pytest.raises(ValueError, match="requires null adjustment intervals"):
        create_provider().compile_recommendation(
            {"scaling_policy": {"preset": [invalid]}}, context
        )


@pytest.mark.parametrize("throughput_enabled", [False, True])
def test_preset_off_uses_real_planner_interval_validation(throughput_enabled) -> None:
    adapter = create_provider()
    context = RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={"sla": {"ttft_ms": 2000.0, "itl_ms": 30.0}},
        optimization={"target": "goodput"},
        sweep=_sweep_context(target="goodput"),
    )
    plan = adapter.compile_recommendation(
        {
            "scaling_policy": {"preset": False},
            "enable_throughput_scaling": throughput_enabled,
            "enable_load_scaling": {"choices": [False, True]},
            "throughput_adjustment_interval_seconds": {"choices": [20, 40]},
            "load_adjustment_interval_seconds": {"choices": [10, 30]},
        },
        context,
    )
    selection = {
        name: values[0]
        for name, values in plan.fragment.choices_by_branch["agg"].items()
    }
    selection.update(
        policy="enabled",
        enable_throughput_scaling=throughput_enabled,
        enable_load_scaling=True,
        throughput_adjustment_interval_seconds=20,
        load_adjustment_interval_seconds=30,
    )
    if throughput_enabled:
        with pytest.raises(InfeasibleCandidate, match="load_adjustment_interval"):
            adapter.materialize_candidate(plan, selection, _candidate_context())
    else:
        spec = adapter.materialize_candidate(plan, selection, _candidate_context())
        assert spec.config["load_adjustment_interval_seconds"] == 30


@pytest.mark.parametrize(
    "trace_format", ["mooncake-delta", "agentic_mooncake", "dynamo", "weka"]
)
def test_enabled_planner_prediction_keeps_public_config_separate_from_hook(
    trace_format,
) -> None:
    context = PredictionAdapterContext(
        engine={
            "mode": "aggregated",
            "workers": {"aggregated": {"parallelism": {"tensor": 1}}},
        },
        traffic={
            "source": {
                "type": "trace",
                "format": trace_format,
                "paths": ["unopened.jsonl"],
            }
        },
        evaluation={},
    )
    spec = create_provider().compile_prediction(
        {
            "policy": "enabled",
            "target": "load",
            "enable_throughput_scaling": False,
            "enable_load_scaling": True,
        },
        context,
    )
    assert "mode" not in spec.config
    assert "max_gpu_budget" not in spec.config
    assert spec.config["policy"] == "enabled"
    assert spec.runtime_hooks[0].config["planner_config"]["mode"] == "agg"


def test_planner_recommendation_target_is_derived_not_user_configurable() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        create_provider().compile_recommendation(
            {"target": "load"},
            RecommendationAdapterContext(
                engine={},
                traffic={},
                evaluation={},
                optimization={"target": "throughput"},
                sweep=_sweep_context(target="throughput"),
            ),
        )


def test_static_predictor_fallback_stays_within_configured_preset_list() -> None:
    plan = create_provider().compile_recommendation(
        {
            "scaling_policy": {"preset": ["throughput_180_5"]},
            "load_predictor": {"preset": ["arima_raw"]},
        },
        RecommendationAdapterContext(
            engine={},
            traffic={},
            evaluation={"sla": {"ttft_ms": 2000.0, "itl_ms": 30.0}},
            optimization={"target": "goodput"},
            sweep=_sweep_context(target="goodput"),
        ),
    )
    assert plan.state["load_predictor"]["best_by_interval"] == {"180": "arima_raw"}


@pytest.mark.parametrize(
    "trace_format", ["mooncake-delta", "agentic_mooncake", "dynamo", "weka"]
)
def test_enabled_planner_recommendation_accepts_native_trace_formats(
    trace_format,
) -> None:
    plan = create_provider().compile_recommendation(
        {"policy": "enabled", "scaling_policy": {"preset": ["load_180_5"]}},
        RecommendationAdapterContext(
            engine={},
            traffic={
                "source": {
                    "type": "trace",
                    "format": trace_format,
                    "paths": ["unopened.jsonl"],
                }
            },
            evaluation={},
            optimization={"target": "throughput"},
            sweep=_sweep_context(target="throughput"),
        ),
    )
    assert plan.fragment.choices_by_branch["agg"]["policy"] == ["enabled"]


def test_policy_pruning_diagnostics_remain_visible(monkeypatch) -> None:
    messages = []
    monkeypatch.setattr(planner_provider_module.tqdm, "write", messages.append)

    create_provider().generate_search_space(
        {"scaling_policy": {"preset": ["disabled", "throughput_180_5"]}},
        replace(_sweep_context(target="throughput"), show_progress=True),
    )

    assert len(messages) == 1
    assert "dropped 1 throughput-scaling policy" in messages[0]


def test_public_policy_and_min_workers_are_independent_dimensions() -> None:
    adapter = create_provider()
    plan = adapter.generate_search_space(
        {
            "policy": {"choices": ["disabled", "enabled"]},
            "scaling_policy": {"preset": ["load_180_5"]},
            "fpm_sampling": {"preset": ["default"]},
            "load_sensitivity": {"preset": ["default"]},
            "min_workers": {"choices": [0, 2]},
        },
        _sweep_context(target="throughput"),
    )
    choices = plan.fragment.choices_by_branch["agg"]
    assert choices["policy"] == ["disabled", "enabled"]
    assert choices["min_workers"] == [0, 2]

    common_selection = {
        "scaling_policy": "load_180_5",
        "fpm_sampling": "default",
        "load_sensitivity": "default",
        "min_workers": 0,
    }
    disabled = adapter.materialize_replay(
        plan,
        {**common_selection, "policy": "disabled"},
        _candidate_context(),
    )
    enabled = adapter.materialize_replay(
        plan,
        {**common_selection, "policy": "enabled"},
        _candidate_context(),
    )

    assert disabled.config == {"policy": "disabled"}
    assert disabled.runtime_hooks == ()
    assert enabled.config["policy"] == "enabled"
    assert enabled.config["min_workers"] == 0
    assert enabled.config["max_num_gpus"] == 8
    assert enabled.runtime_hooks[0].config["planner_config"]["min_endpoint"] == 0
    predicted = adapter.compile_prediction(enabled.config, _prediction_context())
    assert predicted.runtime_hooks == enabled.runtime_hooks


def test_public_custom_predictor_preset_requires_every_knob() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        planner_provider_module.PlannerSearchSpace.model_validate(
            {
                "policy": "enabled",
                "load_predictor": {
                    "preset": [{"load_predictor": "arima"}],
                },
            }
        )


def _prediction_context(sla=None) -> PredictionAdapterContext:
    return PredictionAdapterContext(
        engine={
            "mode": "aggregated",
            "workers": {
                "aggregated": {"parallelism": {"tensor": 4, "attention_data": 2}}
            },
        },
        traffic={},
        evaluation={"sla": sla} if sla else {},
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"throughput_adjustment_interval_seconds": None},
        {
            "target": "sla",
            "enable_load_scaling": False,
            "load_adjustment_interval_seconds": None,
        },
    ],
)
def test_prediction_defaults_and_inactive_nulls_match_real_planner(overrides) -> None:
    sla = {"ttft_ms": 2000.0, "itl_ms": 30.0}
    spec = create_provider().compile_prediction(
        {"policy": "enabled", **overrides}, _prediction_context(sla)
    )
    actual = PlannerConfig.model_validate(
        spec.runtime_hooks[0].config["planner_config"]
    )
    expected_fields = {
        key: value for key, value in overrides.items() if value is not None
    }
    expected_fields["optimization_target"] = expected_fields.pop("target", "throughput")
    if expected_fields["optimization_target"] == "sla":
        expected_fields.update(sla)
    expected = PlannerConfig(
        mode="agg",
        decode_engine_num_gpu=8,
        report_interval_hours=None,
        live_dashboard_port=0,
        metric_pulling_prometheus_extra_query_params=None,
        **expected_fields,
    )
    assert actual == expected
    assert spec.config["load_adjustment_interval_seconds"] is not None
    assert spec.config["throughput_adjustment_interval_seconds"] is not None
    assert not any("token" in key or "url" in key for key in spec.config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"load_adjustment_interval_seconds": None},
        {"target": "sla", "throughput_adjustment_interval_seconds": None},
    ],
)
def test_prediction_rejects_null_for_effective_active_interval(overrides) -> None:
    with pytest.raises(ValueError, match="cannot be null"):
        create_provider().compile_prediction(
            {"policy": "enabled", **overrides},
            _prediction_context({"ttft_ms": 2000.0, "itl_ms": 30.0}),
        )


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("target", ["throughput", "goodput"])
def test_default_search_filters_dimensions_and_roundtrips(
    target, independent, monkeypatch, tmp_path
) -> None:
    adapter = create_provider()
    context = _sweep_context(target=target)
    config = (
        {
            name: {"preset": False}
            for name in ("scaling_policy", "fpm_sampling", "load_sensitivity")
        }
        if independent
        else {}
    )
    plan = adapter.compile_recommendation(
        config,
        RecommendationAdapterContext(
            engine={},
            traffic={},
            evaluation={"sla": context.goal["sla"]},
            optimization={"target": target},
            sweep=context,
        ),
    )
    choices = plan.fragment.choices_by_branch["agg"]
    if target == "throughput":
        if independent:
            assert choices["enable_throughput_scaling"] == [False]
            assert "max_num_fpm_samples" not in choices
        else:
            assert choices["scaling_policy"] == [
                "disabled",
                "load_180_5",
                "load_180_10",
            ]
            assert "fpm_sampling" not in choices
    selection = {name: values[-1] for name, values in choices.items()}
    spec = adapter.materialize_candidate(plan, selection, _candidate_context())
    assert spec.config["policy"] == "enabled"
    for name, value in {
        "PROMETHEUS_ENDPOINT": "https://unused.invalid",
        "PROMETHEUS_TOKEN": "unused-token",
        "PROMETHEUS_TOKEN_FILE": str(tmp_path / "missing-token"),
        "PROMETHEUS_SSL_VERIFY": "true",
        "PROMETHEUS_EXTRA_QUERY_PARAMS": "missing-equals",
        "PROMETHEUS_CA_BUNDLE": str(tmp_path / "missing-ca.pem"),
        "DYN_PLANNER_PROMETHEUS_REQUEST_TIMEOUT_SECONDS": "invalid-timeout",
        "PLANNER_PROMETHEUS_PORT": "invalid-port",
    }.items():
        monkeypatch.setenv(name, value)
    assert adapter.materialize_candidate(plan, selection, _candidate_context()) == spec
    predicted = adapter.compile_prediction(
        spec.config, _prediction_context(context.goal["sla"])
    )
    assert predicted.runtime_hooks == spec.runtime_hooks


def test_explicit_subset_rejects_incompatible_policy_and_preserves_inactive_null() -> (
    None
):
    adapter = create_provider()
    context = RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={},
        optimization={"target": "throughput"},
        sweep=_sweep_context(target="throughput"),
    )
    with pytest.raises(ValueError, match="incompatible"):
        adapter.compile_recommendation(
            {"scaling_policy": {"preset": ["load_180_5", "throughput_180_5"]}},
            context,
        )
    custom = {
        "enable_throughput_scaling": False,
        "enable_load_scaling": True,
        "throughput_adjustment_interval_seconds": None,
        "load_adjustment_interval_seconds": 5,
    }
    plan = adapter.compile_recommendation(
        {
            "policy": "enabled",
            "scaling_policy": {"preset": [custom]},
            "load_sensitivity": {"preset": ["conservative"]},
        },
        context,
    )
    choices = plan.fragment.choices_by_branch["agg"]
    assert choices["scaling_policy"] == [custom]
    assert choices["load_sensitivity"] == ["conservative"]
    assert "fpm_sampling" not in choices
    spec = adapter.materialize_candidate(
        plan,
        {name: values[0] for name, values in choices.items()},
        _candidate_context(),
    )
    assert (
        spec.config["throughput_adjustment_interval_seconds"]
        == PlannerConfig.model_fields["throughput_adjustment_interval_seconds"].default
    )
    assert spec.config["load_scaling_down_sensitivity"] == 90
