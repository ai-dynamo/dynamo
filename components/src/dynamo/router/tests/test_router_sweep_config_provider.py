# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""Parity tests for the Router-owned Sweeper sweep configuration provider."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate.config_adapter import (
    PredictionAdapterContext,
    RecommendationAdapterContext,
)
from aisimulate.sweeper.provider import CandidateContext, SweepContext
from aisimulate.sweeper.replay import BackendDeploymentSpec

import dynamo.router.simulation.provider as router_provider_module
from dynamo.router.simulation import create_provider

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


def _sweep_context() -> SweepContext:
    return SweepContext(
        core_search_space={"deployment_mode": ["agg", "disagg"]},
        workload={"trace_path": "trace.jsonl"},
        goal={"target": "throughput"},
        show_progress=False,
    )


def _candidate_context() -> CandidateContext:
    deployment = BackendDeploymentSpec(
        deployment_mode="agg",
        backend="vllm",
        backend_version="0.11.0",
    )
    return CandidateContext(
        sample={"deployment_mode": "agg"},
        backend_deployment=deployment,
    )


def test_round_robin_only_has_no_inactive_router_dimensions() -> None:
    plan = create_provider().generate_search_space(
        {"mode": ["round_robin"]},
        _sweep_context(),
    )

    assert plan.fragment.choices_by_branch == {
        "agg": {"mode": ["round_robin"]},
        "disagg": {"mode": ["round_robin"]},
    }
    assert plan.potential_runtime_hooks == ()


def test_round_robin_materializes_no_dynamo_hook() -> None:
    adapter = create_provider()
    plan = adapter.generate_search_space(
        {"mode": ["round_robin"]},
        _sweep_context(),
    )

    replay_spec = adapter.materialize_replay(
        plan,
        {"mode": "round_robin"},
        _candidate_context(),
    )

    assert replay_spec.config == {"mode": "round_robin"}
    assert replay_spec.runtime_hooks == ()


def test_kv_router_materializes_current_router_config_without_kvbm() -> None:
    adapter = create_provider()
    plan = adapter.generate_search_space(
        {
            "mode": ["kv_router"],
            "overlap_score_credit": [0.5],
            "prefill_load_scale": [4.0],
            "temperature": [0.2],
        },
        _sweep_context(),
    )

    replay_spec = adapter.materialize_replay(
        plan,
        {
            "mode": "kv_router",
            "overlap_score_credit": 0.5,
            "prefill_load_scale": 4.0,
            "temperature": 0.2,
        },
        _candidate_context(),
    )

    assert replay_spec.config == {
        "mode": "kv_router",
        "overlap_score_credit": 0.5,
        "prefill_load_scale": 4.0,
        "router_temperature": 0.2,
    }
    hook = replay_spec.runtime_hooks[0]
    assert hook.provider == "dynamo.router"
    assert hook.kind == "placement_policy"
    assert hook.config == {
        "router_mode": "kv_router",
        "router_config": {
            "overlap_score_credit": 0.5,
            "prefill_load_scale": 4.0,
            "router_temperature": 0.2,
        },
    }
    assert "host_cache_hit_weight" not in hook.config["router_config"]
    assert "disk_cache_hit_weight" not in hook.config["router_config"]


def test_kv_router_rejects_admission_pins_until_replay_supports_them() -> None:
    with pytest.raises(ValueError, match="admission-control"):
        create_provider().generate_search_space(
            {
                "mode": ["kv_router"],
                "active_decode_blocks_threshold": 16,
            },
            _sweep_context(),
        )


def test_provider_owns_its_adapter_and_hook_abi_versions() -> None:
    adapter = create_provider()

    assert router_provider_module._PROVIDER_API_VERSION == 1
    assert router_provider_module._ROUTER_HOOK_API_VERSION == 1
    assert adapter.api_version == router_provider_module._PROVIDER_API_VERSION
    assert adapter.config_adapter_api_version == 3
    assert adapter.section == "router"


def test_public_router_validation_is_owned_by_dynamo_adapter() -> None:
    adapter = create_provider()
    prediction_context = PredictionAdapterContext(engine={}, traffic={}, evaluation={})
    recommendation_context = RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={},
        optimization={},
        sweep=_sweep_context(),
    )

    assert adapter.compile_prediction({}, prediction_context).config == {
        "policy": "round_robin",
        "prefill_load_model": {"type": "none"},
    }
    plan = adapter.compile_recommendation({}, recommendation_context)
    assert plan.fragment.choices_by_branch["agg"] == {
        "mode": ["round_robin", "kv_router"]
    }
    conditional = plan.fragment.conditional_by_branch["agg"][0]
    assert conditional.selector == "mode"
    assert conditional.values == ["kv_router"]
    assert conditional.choices["prefill_load_model_type"] == ["none", "aic"]
    assert conditional.choices["temperature"] == [0.0, 0.2, 0.5, 1.0]

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        adapter.compile_prediction({"not_a_router_knob": True}, prediction_context)
    with pytest.raises(ValueError, match="round_robin rejects"):
        adapter.compile_recommendation(
            {
                "policy": "round_robin",
                "prefill_load_model": {"type": "aic"},
            },
            recommendation_context,
        )


def test_public_router_search_accepts_custom_values_and_emits_public_config() -> None:
    adapter = create_provider()
    plan = adapter.generate_search_space(
        {
            "policy": {"choices": ["round_robin", "kv_router"]},
            "prefill_load_model": {"type": "none"},
            "overlap_score_credit": {"choices": [0.1]},
            "prefill_load_scale": {"choices": [3.0]},
            "temperature": {"choices": [0.1]},
        },
        _sweep_context(),
    )

    choices = plan.fragment.choices_by_branch["agg"]
    assert choices["mode"] == ["round_robin", "kv_router"]
    assert choices["temperature"] == [0.1]
    replay_spec = adapter.materialize_replay(
        plan,
        {
            "mode": "kv_router",
            "prefill_load_model_type": "none",
            "overlap_score_credit": 0.1,
            "prefill_load_scale": 3.0,
            "temperature": 0.1,
        },
        _candidate_context(),
    )

    assert replay_spec.config == {
        "policy": "kv_router",
        "prefill_load_model": {"type": "none"},
        "overlap_score_credit": 0.1,
        "prefill_load_scale": 3.0,
        "temperature": 0.1,
    }
    assert replay_spec.runtime_hooks[0].config["aic_perf_config"] is None


def test_public_router_prediction_materializes_runtime_hook() -> None:
    replay_spec = create_provider().compile_prediction(
        {
            "policy": "kv_router",
            "prefill_load_model": {"type": "none"},
            "temperature": 0.1,
        },
        PredictionAdapterContext(
            engine={},
            traffic={},
            evaluation={},
        ),
    )

    assert replay_spec.config["policy"] == "kv_router"
    assert replay_spec.runtime_hooks[0].config["router_config"] == {
        "overlap_score_credit": 1.0,
        "prefill_load_scale": 1.0,
        "router_temperature": 0.1,
        "router_prefill_load_model": "none",
    }


def test_public_router_aic_load_model_reaches_runtime_config() -> None:
    replay_spec = create_provider().compile_prediction(
        {
            "policy": "kv_router",
            "prefill_load_model": {"type": "aic"},
        },
        PredictionAdapterContext(
            engine={
                "mode": "aggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.11.0",
                "workers": {
                    "aggregated": {
                        "parallelism": {
                            "tensor": 1,
                            "attention_data": 1,
                            "moe_tensor": 1,
                            "moe_expert": 1,
                        }
                    }
                },
            },
            traffic={},
            evaluation={},
        ),
    )

    hook_config = replay_spec.runtime_hooks[0].config
    assert hook_config["router_config"]["router_prefill_load_model"] == "aic"
    assert hook_config["aic_perf_config"] == {
        "aic_backend": "vllm",
        "aic_system": "h200_sxm",
        "aic_model_path": "example/model",
        "aic_backend_version": "0.11.0",
        "aic_tp_size": 1,
        "aic_attention_dp_size": 1,
        "aic_moe_tp_size": None,
        "aic_moe_ep_size": None,
    }


def test_router_public_schema_rejects_internal_fields_and_supports_ranges() -> None:
    adapter = create_provider()
    context = RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={},
        optimization={},
        sweep=_sweep_context(),
    )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        adapter.compile_recommendation(
            {"policy": "kv_router", "mode": ["round_robin"]}, context
        )

    plan = adapter.compile_recommendation(
        {
            "policy": "kv_router",
            "temperature": {"range": {"min": 0.1, "max": 1.0}},
            "prefill_load_scale": {"range": {"min": 0.25, "max": 32.0, "scale": "log"}},
        },
        context,
    )
    assert plan.fragment.float_ranges_by_branch["agg"] == {
        "temperature": (0.1, 1.0),
        "prefill_load_scale": (0.25, 32.0),
    }
    assert plan.fragment.log_float_ranges_by_branch["agg"] == ["prefill_load_scale"]


def test_mixed_router_policy_preserves_continuous_and_log_ranges() -> None:
    adapter = create_provider()
    context = RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={},
        optimization={},
        sweep=_sweep_context(),
    )
    plan = adapter.compile_recommendation(
        {
            "policy": {"choices": ["round_robin", "kv_router"]},
            "temperature": {"range": {"min": 0.1, "max": 1.0}},
            "prefill_load_scale": {"range": {"min": 0.25, "max": 32.0, "scale": "log"}},
        },
        context,
    )

    conditional = plan.fragment.conditional_by_branch["agg"][0]
    assert conditional.float_ranges == {
        "temperature": (0.1, 1.0),
        "prefill_load_scale": (0.25, 32.0),
    }
    assert conditional.log_float_ranges == ["prefill_load_scale"]
    assert plan.fragment.choices_by_branch["agg"] == {
        "mode": ["round_robin", "kv_router"]
    }

    kv_router = adapter.materialize_candidate(
        plan,
        {
            "mode": "kv_router",
            "prefill_load_model_type": "none",
            "overlap_score_credit": 0.5,
            "prefill_load_scale": 3.25,
            "temperature": 0.75,
        },
        _candidate_context(),
    )
    assert kv_router.config["policy"] == "kv_router"
    assert kv_router.config["prefill_load_scale"] == 3.25
    assert kv_router.config["temperature"] == 0.75

    round_robin = adapter.materialize_candidate(
        plan,
        {"mode": "round_robin"},
        _candidate_context(),
    )
    assert round_robin.config == {
        "policy": "round_robin",
        "prefill_load_model": {"type": "none"},
    }
    assert round_robin.runtime_hooks == ()


@pytest.mark.parametrize(
    "config",
    [
        {"policy": "kv_router", "overlap_score_credit": None},
        {"policy": "kv_router", "prefill_load_scale": None},
        {"policy": "kv_router", "temperature": None},
    ],
)
def test_router_rejects_explicit_null_numeric_knobs(
    config: dict[str, object],
) -> None:
    adapter = create_provider()
    with pytest.raises(ValueError, match="do not accept null"):
        adapter.compile_prediction(
            config,
            PredictionAdapterContext(engine={}, traffic={}, evaluation={}),
        )
    with pytest.raises(ValueError, match="do not accept null"):
        adapter.compile_recommendation(
            config,
            RecommendationAdapterContext(
                engine={},
                traffic={},
                evaluation={},
                optimization={},
                sweep=_sweep_context(),
            ),
        )


def test_router_stepped_float_range_includes_exact_endpoint() -> None:
    plan = create_provider().compile_recommendation(
        {
            "policy": "kv_router",
            "temperature": {"range": {"min": 0.1, "max": 0.3, "step": 0.1}},
        },
        RecommendationAdapterContext(
            engine={},
            traffic={},
            evaluation={},
            optimization={},
            sweep=_sweep_context(),
        ),
    )

    assert plan.fragment.choices_by_branch["agg"]["temperature"] == [
        0.1,
        0.2,
        0.3,
    ]


def test_singleton_round_robin_policy_domain_uses_conditional_default() -> None:
    plan = create_provider().compile_recommendation(
        {"policy": {"choices": ["round_robin"]}},
        RecommendationAdapterContext(
            engine={},
            traffic={},
            evaluation={},
            optimization={},
            sweep=_sweep_context(),
        ),
    )

    assert plan.fragment.choices_by_branch == {
        "agg": {"mode": ["round_robin"]},
        "disagg": {"mode": ["round_robin"]},
    }


def test_kv_router_prediction_materializes_documented_defaults() -> None:
    spec = create_provider().compile_prediction(
        {"policy": "kv_router"},
        PredictionAdapterContext(engine={}, traffic={}, evaluation={}),
    )
    assert spec.config == {
        "policy": "kv_router",
        "prefill_load_model": {"type": "none"},
        "overlap_score_credit": 1.0,
        "prefill_load_scale": 1.0,
        "temperature": 0.0,
    }
