# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""``planner.retune``, ``load_predictor_warmup_trace`` and ``planner.advisory``
in ``aisimulate predict``."""

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

# AISimulate recommend syntax with public names; credit and scale are pinned.
_ROUTER_DOMAIN: dict[str, Any] = {
    "overlap_score_credit": 1.0,
    "prefill_load_scale": 1.0,
    "temperature": {"choices": [0.0, 100.0]},
}
_RETUNE: dict[str, Any] = {
    "interval_seconds": 120,
    "grace_seconds": 1,
    "router": _ROUTER_DOMAIN,
}
_SENSITIVITY = {
    "load_sensitivity": {"preset": False},
    "load_scaling_down_sensitivity": {"choices": [70, 90]},
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
    assert config.retune.router == _ROUTER_DOMAIN
    assert spec.config["policy"] == "enabled"
    assert spec.config["retune"] == _RETUNE


def test_retune_keeps_easy_mode_scaling_unless_both_flags_are_false() -> None:
    # Scenario 2 with a load-scaling planner: flags omitted, easy mode applies.
    retune = {**_RETUNE, "planner": _SENSITIVITY}
    spec = create_provider().compile_prediction(
        {"policy": "enabled", "retune": retune}, _context()
    )

    config = _host_config(spec)
    assert config.enable_load_scaling and not config.enable_throughput_scaling
    assert config.retune is not None
    assert config.retune.planner == _SENSITIVITY


def test_retune_mapping_is_validated_by_planner_config() -> None:
    # PlannerConfig owns the retune checks; the provider passes its errors on.
    with pytest.raises(ValueError, match="unknown_knob"):
        create_provider().compile_prediction(
            {**_RETUNE_ONLY, "retune": {**_RETUNE, "unknown_knob": 1}}, _context()
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


@pytest.mark.parametrize("advisory", [True, False])
def test_advisory_passes_through_to_the_replay_host(advisory) -> None:
    spec = create_provider().compile_prediction(
        {"policy": "enabled", "advisory": advisory, "retune": _RETUNE}, _context()
    )

    assert _hook_config(spec)["advisory"] is advisory
    assert _host_config(spec).advisory is advisory
    assert spec.config["advisory"] is advisory


def test_advisory_is_omitted_unless_written() -> None:
    spec = create_provider().compile_prediction({"policy": "enabled"}, _context())

    assert "advisory" not in _hook_config(spec)
    assert _host_config(spec).advisory is False
    assert "advisory" not in spec.config


def test_disabled_policy_rejects_advisory() -> None:
    with pytest.raises(ValueError, match="planner.advisory requires planner.policy"):
        create_provider().compile_prediction(
            {"policy": "disabled", "advisory": True}, _context()
        )
    # ``advisory: false`` is a no-op, so a shared config may set it with any
    # policy, including the omitted (disabled) one.
    for config in ({"policy": "disabled", "advisory": False}, {"advisory": False}):
        spec = create_provider().compile_prediction(config, _context())
        assert spec.config == {"policy": "disabled"}


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


def _recommendation_context() -> RecommendationAdapterContext:
    return RecommendationAdapterContext(
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


@pytest.mark.parametrize(
    "field",
    [{"retune": _RETUNE}, {"advisory": True}, {"advisory": False}],
    ids=["retune", "advisory", "advisory-false"],
)
def test_recommend_rejects_prediction_only_fields(field) -> None:
    with pytest.raises(ValueError, match="not supported by aisimulate recommend"):
        create_provider().compile_recommendation(
            {"policy": "enabled", **field}, _recommendation_context()
        )
