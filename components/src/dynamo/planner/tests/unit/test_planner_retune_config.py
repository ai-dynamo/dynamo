# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""The Planner adapter's retune schema and its field classification."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from pydantic import ValidationError

from dynamo.planner.reconfiguration import PREDICTOR_FIELDS, RUNTIME_FIELDS
from dynamo.planner.simulation.config import (
    PLANNER_PRESET_GROUPS,
    PLANNER_PUBLIC_TO_CONFIG,
    PLANNER_RETUNE_EXTENSIONS,
    PLANNER_RETUNE_GROUPS,
    PLANNER_RETUNE_KNOBS,
    PLANNER_RETUNE_STARTUP_ONLY,
    PlannerRecommendationConfig,
    PlannerRetuneConfig,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_every_retune_field_is_classified_exactly_once() -> None:
    # A new recommendation knob fails here until it is classified for retune.
    # Group knobs a running planner can change are searched through their
    # group; the others (load_min_observations) must be startup-only.
    runtime_group_knobs = {
        knob
        for group in PLANNER_RETUNE_GROUPS
        for knob in PLANNER_PRESET_GROUPS[group]
        if knob in RUNTIME_FIELDS
    }
    classes = [
        PLANNER_RETUNE_STARTUP_ONLY,
        set(PLANNER_RETUNE_KNOBS),
        set(PLANNER_RETUNE_GROUPS),
        runtime_group_knobs,
    ]
    fields = set(PlannerRetuneConfig.model_fields)
    assert set().union(*classes) == fields
    assert sum(len(names) for names in classes) == len(fields)
    assert "load_min_observations" in PLANNER_RETUNE_STARTUP_ONLY


def test_retune_reaches_exactly_the_runtime_fields_except_the_sla() -> None:
    reachable = (
        set(PLANNER_RETUNE_KNOBS.values())
        | {"load_scaling_down_sensitivity"}
        | PREDICTOR_FIELDS
    )
    assert reachable == RUNTIME_FIELDS - {"ttft_ms", "itl_ms"}
    assert all(
        PLANNER_PUBLIC_TO_CONFIG.get(public, public) == live
        for public, live in PLANNER_RETUNE_KNOBS.items()
    )
    assert set(PLANNER_RETUNE_EXTENSIONS) <= set(PLANNER_RETUNE_KNOBS)


def test_default_retune_config_sets_nothing() -> None:
    public = PlannerRetuneConfig.model_validate({})
    assert public.model_fields_set == set()
    assert public.policy is None
    assert public.max_num_gpus is None
    assert public.max_throughput_scaling_replicas is None
    assert all(getattr(public, name) is None for name in PLANNER_RETUNE_KNOBS)
    # Recommend keeps its own defaults.
    recommendation = PlannerRecommendationConfig.model_validate({})
    assert recommendation.max_num_gpus == 8
    assert recommendation.policy.choices == ["disabled", "enabled"]


def test_explicit_retune_domains_parse_with_recommendation_syntax() -> None:
    public = PlannerRetuneConfig.model_validate(
        {
            "load_sensitivity": {"preset": False},
            "load_scaling_down_sensitivity": {"choices": [70, 80, 90]},
            "decode_min_workers": {"choices": [1, 2]},
            "load_predictor": {
                "preset": ["arima_log1p", "kalman_default_raw", "kalman_reactive_log1p"]
            },
            "max_num_gpus": {"choices": [-1, 16]},
            "min_num_gpus": {"range": {"min": 2, "max": 6, "step": 2}},
            "max_throughput_scaling_replicas": 3,
            "prefill_scale_down_queue_tokens": {"choices": [0, 64]},
            "decode_scale_up_kv_rate": {"range": {"min": 80, "max": 90, "step": 5}},
        }
    )
    assert public.model_fields_set == {
        "load_sensitivity",
        "load_scaling_down_sensitivity",
        "decode_min_workers",
        "load_predictor",
        "max_num_gpus",
        "min_num_gpus",
        "max_throughput_scaling_replicas",
        "prefill_scale_down_queue_tokens",
        "decode_scale_up_kv_rate",
    }


@pytest.mark.parametrize("name", sorted(PLANNER_RETUNE_STARTUP_ONLY))
def test_startup_only_knobs_are_rejected(name: str) -> None:
    with pytest.raises(ValidationError, match=f"cannot change \\['{name}'\\]"):
        PlannerRetuneConfig.model_validate({name: {"preset": "default"}})


@pytest.mark.parametrize(
    ("live", "public"),
    [
        ("min_endpoint", "min_workers"),
        ("prefill_min_endpoint", "prefill_min_workers"),
        ("decode_min_endpoint", "decode_min_workers"),
        ("max_gpu_budget", "max_num_gpus"),
        ("min_gpu_budget", "min_num_gpus"),
    ],
)
def test_live_names_point_to_the_public_name(live: str, public: str) -> None:
    with pytest.raises(ValidationError, match=f"{live} is not a public .* {public}"):
        PlannerRetuneConfig.model_validate({live: {"choices": [1, 2]}})


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"ttft_ms": 200.0}, "ttft_ms is the planner SLA .* retune.sla_ttft_ms"),
        ({"itl_ms": 20.0}, "itl_ms is the planner SLA .* retune.sla_itl_ms"),
        (
            {"load_scaling_down_sensitivity": {"choices": [60]}},
            "load_sensitivity uses the default preset",
        ),
        (
            {"load_predictor": {"preset": "default"}, "kalman_r": 1.0},
            "load_predictor.preset cannot be combined",
        ),
        ({"prefill_min_workers": {"choices": [1, None]}}, "does not accept null"),
        ({"min_workers": None}, "min_workers does not accept null; omit it"),
        ({"decode_scale_up_kv_rate": None}, "decode_scale_up_kv_rate does not accept"),
        ({"max_num_gpus": [8, 16]}, "max_num_gpus"),
        ({"max_num_gpus": True}, "max_num_gpus"),
        ({"max_throughput_scaling_replicas": 0}, "greater than 0"),
        ({"target": "sla"}, "Extra inputs are not permitted"),
    ],
)
def test_retune_config_rejections(config, match) -> None:
    with pytest.raises(ValidationError, match=match):
        PlannerRetuneConfig.model_validate(config)
