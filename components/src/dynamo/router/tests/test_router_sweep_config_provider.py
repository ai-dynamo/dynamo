# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optional-dependency preflight must run before the simulation imports.
# ruff: noqa: E402

"""Parity tests for the Router-owned Sweeper sweep configuration provider."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
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
