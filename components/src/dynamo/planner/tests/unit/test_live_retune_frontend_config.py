# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the retune imports.

"""Semantic Retune startup requires no authored or lowered Pod template."""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

pytest.importorskip(
    "aisim.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.llm import KvRouterConfig
from dynamo.planner.retune.live import frontend_config as fc
from dynamo.planner.retune.live.contracts import FrontendStartup, RetuneCapability
from dynamo.planner.tests.live_retune_fakes import (
    retune_launch_contract,
    retune_planner_status,
)
from dynamo.planner.tests.unit.test_live_retune_state import live_config

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def deployment():
    return {
        "metadata": {"name": "graph", "uid": "uid-1", "generation": 3},
        "spec": {
            "components": [
                {"name": "Frontend", "type": "frontend", "replicas": 2},
                {"name": "worker", "type": "worker", "replicas": 1},
                {"name": "Planner", "type": "planner", "replicas": 1},
            ]
        },
        "status": {
            "observedGeneration": 3,
            "components": {
                "Planner": retune_planner_status(),
                "worker": {"servedModelName": "served"},
            },
        },
    }


def capability():
    return RetuneCapability.from_dict(retune_launch_contract())


@pytest.mark.parametrize(
    "mode,worker_types", [("agg", ("aggregated",)), ("disagg", ("prefill", "decode"))]
)
def test_settings_resolve_complete_config_without_pod_templates(mode, worker_types):
    graph = deployment()
    launch = dataclasses.replace(
        capability(), overlap_score_credit=0.7, temperature=0.1
    )
    startup = fc.resolve_startup(graph, planner_mode=mode, capability=launch)
    assert isinstance(startup, FrontendStartup), startup
    canonical = KvRouterConfig(
        overlap_score_credit=0.7,
        prefill_load_scale=1.0,
        router_temperature=0.1,
        router_replica_sync=True,
    )
    assert startup.router_config == json.loads(canonical.to_json())
    assert startup.startup_maps == {
        wt: dict(canonical.policy_parameters(wt)) for wt in worker_types
    }
    assert startup.replica_sync and startup.replicas == 2
    assert startup.router_mode == "kv_router"
    before = copy.deepcopy(graph)
    for component in graph["spec"]["components"]:
        component["podTemplate"] = {
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "command": ["unknown-launcher"],
                        "args": ["$(UNKNOWN)"],
                        "envFrom": [{"configMapRef": {"name": "external"}}],
                    }
                ]
            }
        }
    assert fc.resolve_startup(graph, planner_mode=mode, capability=launch) == startup
    assert "roles" not in before["status"]["components"]["Planner"]


def test_target_acknowledgment_waits_for_current_hash_without_roles():
    graph = deployment()
    launch = capability()
    assert fc.retune_capability(graph, launch) is launch
    graph["metadata"]["generation"] += 1
    with pytest.raises(ValueError, match="current Planner"):
        fc.retune_capability(graph, launch)
    graph["status"]["observedGeneration"] += 1
    graph["status"]["components"]["Planner"]["retune"]["configHash"] = "next-target"
    with pytest.raises(ValueError, match="wait for its replacement"):
        fc.retune_capability(graph, launch)
    next_launch = dataclasses.replace(launch, config_hash="next-target")
    assert fc.retune_capability(graph, next_launch) is next_launch
    graph["status"]["components"]["Planner"].pop("retune")
    with pytest.raises(ValueError, match="current Planner"):
        fc.retune_capability(graph, next_launch)


def test_public_graph_and_search_constraints_remain():
    graph = deployment()
    launch = capability()
    config = live_config()
    assert fc.evaluate_premises(graph, planner_config=config, capability=launch).spec_ok
    graph["spec"]["components"][1]["scalingAdapter"] = {}
    report = fc.evaluate_premises(graph, planner_config=config, capability=launch)
    assert not report.spec_ok and any(
        "scalingAdapter" in reason for reason in report.reasons
    )
    graph["spec"]["components"].append({"name": "other", "type": "frontend"})
    report = fc.evaluate_premises(graph, planner_config=config, capability=launch)
    assert not report.p4_ok and any(
        "exactly one" in reason for reason in report.reasons
    )


def test_missing_launch_disables_only_retune():
    report = fc.evaluate_premises(deployment(), planner_config=live_config())
    assert not report.p4_ok and not report.spec_ok
    assert any("launch binding" in reason for reason in report.reasons)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("version", True),
        ("configHash", ""),
        ("router", None),
    ],
)
def test_invalid_or_unsupported_launch_contract(field, value):
    payload = retune_launch_contract()
    payload[field] = value
    with pytest.raises(ValueError):
        RetuneCapability.from_dict(payload)


@pytest.mark.parametrize(
    "parameter", ["overlapScoreCredit", "prefillLoadScale", "temperature"]
)
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "0.1", None])
def test_router_parameters_are_finite_nonnegative_numbers(parameter, value):
    payload = retune_launch_contract()
    payload["router"][parameter] = value
    with pytest.raises(ValueError, match="finite and non-negative"):
        RetuneCapability.from_dict(payload)


def test_storage_contract_roundtrip_and_legacy_assertions():
    launch = capability()
    assert RetuneCapability.from_dict(launch.to_dict()) == launch
    config = live_config().retune.dgdr
    resolved = fc.resolve_executor_config(config, launch)
    assert resolved.claim_name == launch.claim_name
    assert resolved is not config
    other = dataclasses.replace(launch, claim_name="other")
    with pytest.raises(ValueError, match="claim_name conflicts"):
        fc.resolve_executor_config(config, other)
