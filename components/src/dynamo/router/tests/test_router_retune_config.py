# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""The Router adapter's retune schema and its public-to-live name mapping."""

from __future__ import annotations

import math

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from pydantic import ValidationError

from dynamo.aisimulate.domains import finite_values
from dynamo.llm import KvRouterConfig
from dynamo.router.simulation.config import (
    ROUTER_KNOB_DEFAULTS,
    ROUTER_PUBLIC_TO_LIVE,
    ROUTER_RETUNE_EXTENSIONS,
    RouterRecommendationConfig,
    RouterRetuneConfig,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


def test_public_and_extension_names_cover_every_policy_parameter() -> None:
    # A new PolicyParameters field must be classified before retune can see it.
    policy_parameters = set(KvRouterConfig().policy_parameters("aggregated"))
    assert len(policy_parameters) == 8
    assert (
        set(ROUTER_PUBLIC_TO_LIVE.values()) | set(ROUTER_RETUNE_EXTENSIONS)
        == policy_parameters
    )
    assert not set(ROUTER_PUBLIC_TO_LIVE.values()) & set(ROUTER_RETUNE_EXTENSIONS)
    assert list(ROUTER_KNOB_DEFAULTS) == list(ROUTER_PUBLIC_TO_LIVE)


def test_default_retune_config_is_the_recommendation_default_domain() -> None:
    public = RouterRetuneConfig.model_validate({})
    assert public.model_fields_set == set()
    assert public.policy == "kv_router"
    domain = {
        ROUTER_PUBLIC_TO_LIVE[name]: finite_values(
            name, getattr(public, name), defaults, limit=512
        )
        for name, defaults in ROUTER_KNOB_DEFAULTS.items()
    }
    assert math.prod(len(values) for values in domain.values()) == 108
    assert set(domain) == {
        "overlap_score_credit",
        "prefill_load_scale",
        "router_temperature",
    }
    assert all(getattr(public, name) is None for name in ROUTER_RETUNE_EXTENSIONS)


def test_retune_config_accepts_recommendation_output_and_extension_domains() -> None:
    public = RouterRetuneConfig.model_validate(
        {
            "policy": "kv_router",
            "prefill_load_model": {"type": "ais"},
            "overlap_score_credit": 0.5,
            "temperature": {"range": {"min": 0.0, "max": 0.2, "step": 0.1}},
            "decode_active_request_weight": {"choices": [0.5, 1.0]},
            "host_cache_hit_weight": {
                "range": {"min": 0.25, "max": 0.75, "step": 0.25}
            },
        }
    )
    assert public.prefill_load_model.type == "ais"
    assert finite_values("t", public.temperature, None, limit=10) == [0.0, 0.1, 0.2]
    assert finite_values("w", public.host_cache_hit_weight, None, limit=10) == [
        0.25,
        0.5,
        0.75,
    ]
    assert public.model_fields_set == {
        "policy",
        "prefill_load_model",
        "overlap_score_credit",
        "temperature",
        "decode_active_request_weight",
        "host_cache_hit_weight",
    }
    assert RouterRetuneConfig.model_validate(
        {"policy": {"choices": ["kv_router"]}}
    ).policy.choices == ["kv_router"]


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            {"router_temperature": 0.5},
            "router_temperature is not a public .* temperature",
        ),
        ({"router_mode": "kv"}, "router_mode is not a public .* policy"),
        (
            {"router_prefill_load_model": "ais"},
            "write prefill_load_model.type instead",
        ),
        ({"policy": "round_robin"}, "policy is fixed at startup .* requires kv_router"),
        (
            {"policy": {"choices": ["round_robin", "kv_router"]}},
            "policy is fixed at startup",
        ),
        ({"overlap_score_credit": [0.5, 1.0]}, "overlap_score_credit"),
        ({"temperature": None}, "do not accept null"),
        ({"shared_cache_multiplier": None}, "do not accept null"),
        ({"disk_cache_hit_weight": -0.5}, "greater than or equal to 0"),
        (
            {
                "overlap_score_credit_decay": {
                    "range": {"min": -1.0, "max": 1.0, "step": 1.0}
                }
            },
            "overlap_score_credit_decay range requires min >= 0",
        ),
        ({"router_queue_threshold": 1.0}, "Extra inputs are not permitted"),
    ],
)
def test_retune_config_rejections(config, match) -> None:
    with pytest.raises(ValidationError, match=match):
        RouterRetuneConfig.model_validate(config)


@pytest.mark.parametrize("name", ROUTER_RETUNE_EXTENSIONS)
def test_extension_knobs_stay_out_of_the_recommendation_schema(name) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RouterRecommendationConfig.model_validate({name: 0.5})
