# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""``planner.retune`` and ``load_predictor_warmup_trace`` in ``aisimulate predict``."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate.config_adapter import (
    PredictionAdapterContext,
    RecommendationAdapterContext,
)
from aisimulate.sweeper.provider import SweepContext

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.simulation import create_provider

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

_RETUNE: dict[str, Any] = {
    "interval_seconds": 120,
    "grace_seconds": 1,
    "router": {"router_temperature": [0.0, 100.0]},
}
_RETUNE_ONLY: dict[str, Any] = {
    "policy": "enabled",
    "enable_load_scaling": False,
    "enable_throughput_scaling": False,
}
_SLA = {"ttft_ms": 60.0, "itl_ms": 100.0}


def _traffic(**load: Any) -> dict[str, Any]:
    return {
        "source": {"type": "trace", "format": "dynamo", "paths": ["trace.jsonl"]},
        "load": {"type": "trace_timestamps", **load},
    }


def _context(
    *, traffic: dict[str, Any] | None = None, sla: dict | None = None
) -> PredictionAdapterContext:
    return PredictionAdapterContext(
        engine={
            "mode": "aggregated",
            "workers": {"aggregated": {"parallelism": {"tensor": 1}}},
        },
        traffic=_traffic() if traffic is None else traffic,
        evaluation={"sla": sla} if sla is not None else {},
    )


def _hook_config(spec) -> dict[str, Any]:
    (hook,) = spec.runtime_hooks
    return hook.config["planner_config"]


def _host_config(spec) -> PlannerConfig:
    # The replay host parses the hook payload from JSON.
    return PlannerConfig.from_config_arg(json.dumps(_hook_config(spec)))


def test_retune_only_planner_hook_keeps_both_scaling_flags_explicit() -> None:
    spec = create_provider().compile_prediction(
        {**_RETUNE_ONLY, "retune": _RETUNE}, _context()
    )

    payload = _hook_config(spec)
    assert payload["enable_load_scaling"] is False
    assert payload["enable_throughput_scaling"] is False
    # The mapping reaches the host as written, not as a defaults-filled dump.
    assert payload["retune"] == _RETUNE
    config = _host_config(spec)
    assert {"enable_load_scaling", "enable_throughput_scaling"} <= (
        config.model_fields_set
    )
    assert config.retune is not None and not config.scaling_enabled()
    assert config.retune.router == {"router_temperature": [0.0, 100.0]}
    assert spec.config["policy"] == "enabled"
    assert spec.config["retune"] == _RETUNE


def test_retune_keeps_easy_mode_scaling_unless_both_flags_are_false() -> None:
    # Scenario 2 with a load-scaling planner: flags omitted, easy mode applies.
    retune = {**_RETUNE, "planner": {"load_scaling_down_sensitivity": [70, 90]}}
    spec = create_provider().compile_prediction(
        {"policy": "enabled", "retune": retune}, _context()
    )

    config = _host_config(spec)
    assert config.enable_load_scaling and not config.enable_throughput_scaling
    assert config.retune is not None
    assert config.retune.planner == {"load_scaling_down_sensitivity": [70, 90]}


@pytest.mark.parametrize(
    "retune, message",
    [
        ({**_RETUNE, "unknown_knob": 1}, "unknown_knob"),
        ({**_RETUNE, "interval_seconds": 7}, "multiple of"),
        (
            {**_RETUNE, "planner": {"load_scaling_down_sensitivity": [70]}},
            "requires load or throughput scaling",
        ),
    ],
    ids=["unknown-field", "interval-not-a-tick-multiple", "planner-domain-no-scaling"],
)
def test_retune_mapping_is_validated_by_planner_config(retune, message) -> None:
    with pytest.raises(ValueError, match=message):
        create_provider().compile_prediction(
            {**_RETUNE_ONLY, "retune": retune}, _context()
        )


def test_retune_scores_against_evaluation_sla_unless_set() -> None:
    provider = create_provider()

    defaulted = provider.compile_prediction(
        {**_RETUNE_ONLY, "retune": _RETUNE}, _context(sla=_SLA)
    )
    assert _hook_config(defaulted)["retune"] == {
        **_RETUNE,
        "sla_ttft_ms": 60.0,
        "sla_itl_ms": 100.0,
    }
    assert _host_config(defaulted).retune.scoring_sla(1.0, 1.0) == (60.0, 100.0)

    explicit = {**_RETUNE, "sla_ttft_ms": 30.0, "sla_itl_ms": None}
    kept = provider.compile_prediction(
        {**_RETUNE_ONLY, "retune": explicit}, _context(sla=_SLA)
    )
    assert _hook_config(kept)["retune"] == explicit


def test_disabled_policy_rejects_retune() -> None:
    with pytest.raises(ValueError, match="planner.policy: enabled"):
        create_provider().compile_prediction(
            {
                "policy": "disabled",
                "enable_load_scaling": False,
                "enable_throughput_scaling": False,
                "retune": _RETUNE,
            },
            _context(),
        )
    # An omitted policy is disabled too.
    with pytest.raises(ValueError, match="planner.policy: enabled"):
        create_provider().compile_prediction({"retune": _RETUNE}, _context())


@pytest.mark.parametrize(
    "traffic, message",
    [
        (
            {
                "source": {"type": "trace", "format": "mooncake", "paths": ["t"]},
                "load": {"type": "trace_timestamps"},
            },
            "format: dynamo",
        ),
        (
            {"source": {"type": "synthetic"}, "load": {"type": "constant_rate"}},
            "format: dynamo",
        ),
        (
            {**_traffic(), "load": {"type": "concurrency", "concurrency": 4}},
            "trace_timestamps",
        ),
        (_traffic(agentic_lanes=2), "agentic_lanes"),
        (_traffic(agentic_warmup=True), "agentic_warmup"),
        (
            {**_traffic(), "stop": {"max_virtual_time_seconds": 60.0}},
            "max_virtual_time_seconds",
        ),
    ],
    ids=[
        "mooncake",
        "synthetic",
        "closed-loop",
        "agentic-lanes",
        "agentic-warmup",
        "virtual-time-stop",
    ],
)
def test_retune_requires_open_loop_dynamo_trace_traffic(traffic, message) -> None:
    with pytest.raises(ValueError, match=message):
        create_provider().compile_prediction(
            {**_RETUNE_ONLY, "retune": _RETUNE}, _context(traffic=traffic)
        )


def test_warmup_trace_passes_through_resolved_against_the_working_directory(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "history.jsonl").write_text("")
    monkeypatch.chdir(tmp_path)
    expected = str(tmp_path / "history.jsonl")
    provider = create_provider()

    # Without retune: a throughput-scaling planner warms its predictor.
    spec = provider.compile_prediction(
        {
            "policy": "enabled",
            "target": "sla",
            "enable_throughput_scaling": True,
            "enable_load_scaling": False,
            "load_predictor_warmup_trace": "history.jsonl",
        },
        _context(sla=_SLA),
    )
    assert _hook_config(spec)["load_predictor_warmup_trace"] == expected
    assert _host_config(spec).load_predictor_warmup_trace == expected
    assert spec.config["load_predictor_warmup_trace"] == expected

    # Scenario 2: the warmup trace is also the retune history.
    spec = provider.compile_prediction(
        {
            "policy": "enabled",
            "target": "sla",
            "enable_throughput_scaling": True,
            "enable_load_scaling": True,
            "load_predictor_warmup_trace": "history.jsonl",
            "retune": {**_RETUNE, "planner": "default"},
        },
        _context(sla=_SLA),
    )
    config = _host_config(spec)
    assert config.load_predictor_warmup_trace == expected
    assert config.retune is not None and config.retune.planner == "default"

    with pytest.raises(ValueError, match="relative paths resolve against"):
        provider.compile_prediction(
            {**_RETUNE_ONLY, "load_predictor_warmup_trace": "missing.jsonl"},
            _context(),
        )


def test_recommend_rejects_retune() -> None:
    context = RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={},
        optimization={},
        sweep=SweepContext(
            core_search_space={"deployment_mode": ["agg"]},
            workload={"trace_path": None, "isl": 512, "osl": 128},
            goal={"target": "throughput"},
            show_progress=False,
        ),
    )

    with pytest.raises(ValueError, match="not supported by aisimulate recommend"):
        create_provider().compile_recommendation(
            {"policy": "enabled", "retune": _RETUNE}, context
        )
