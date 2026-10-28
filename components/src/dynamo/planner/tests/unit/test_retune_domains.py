# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the retune imports.

"""Retune search-domain expansion: the default domains of each scaling mode,
and predictor candidates that match the running predictor."""

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import RetuneParams
from dynamo.planner.reconfiguration import PREDICTOR_FIELDS
from dynamo.planner.retune.domains import (
    expand_planner_domain,
    expand_router_domain,
    grid_size,
)
from dynamo.planner.retune.sweep import with_predictor_fields
from dynamo.planner.simulation.load_predictor import (
    LOAD_PREDICTOR_PRESETS,
    predictor_fields,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

SLA_THROUGHPUT = {"optimization_target": "sla", "enable_throughput_scaling": True}
LSD = "load_scaling_down_sensitivity"


def _config(planner="default", **overrides):
    retune = {
        "interval_seconds": 600,
        "history_seconds": 30000,
        "router": "default",
        "planner": planner,
    }
    return PlannerConfig(namespace="test-ns", mode="disagg", retune=retune, **overrides)


@pytest.mark.parametrize(
    ("overrides", "grid", "presets", "size"),
    [
        # Easy mode forces load scaling on.
        ({}, {LSD: [70, 80, 90]}, False, 324),
        (SLA_THROUGHPUT, {}, True, 108),
        (
            {"enable_load_scaling": True, **SLA_THROUGHPUT},
            {LSD: [70, 80, 90]},
            True,
            324,
        ),
    ],
    ids=["load", "sla-throughput", "hybrid"],
)
def test_default_domains_follow_the_scaling_mode(overrides, grid, presets, size):
    config = _config(**overrides)

    router = expand_router_domain(config.retune)
    planner = expand_planner_domain(config)

    # router: default is the AISimulate domain, 3 x 9 x 4 = 108 points.
    assert list(router) == [
        "overlap_score_credit",
        "prefill_load_scale",
        "router_temperature",
    ]
    assert planner.grid == grid
    labels = [label for label, _ in planner.predictor_candidates or ()]
    assert labels == (list(LOAD_PREDICTOR_PRESETS) if presets else [])
    # The predictor pre-sweep is not a grid dimension.
    assert grid_size(router, planner.grid) == size


def test_custom_predictor_mapping_values_take_their_field_types():
    # YAML writes kalman_r: 1; PlannerConfig holds 1.0. A candidate equal to
    # the running predictor must get the running config's key, or the sweep
    # replays the baseline configuration twice.
    integral = {
        "load_predictor": "kalman",
        "load_predictor_log1p": False,
        "prophet_window_size": 50,
        "kalman_q_level": 1,
        "kalman_q_trend": 1,
        "kalman_r": 1,
        "kalman_min_points": 5,
    }
    config = _config(
        planner={"load_predictor": {"preset": ["arima_raw", integral]}},
        load_predictor="kalman",
        load_predictor_log1p=False,
        kalman_q_level=1.0,
        kalman_q_trend=1.0,
        kalman_r=1.0,
        kalman_min_points=5,
        **SLA_THROUGHPUT,
    )

    label, entry = expand_planner_domain(config).predictor_candidates[1]

    assert label == "custom_1"
    assert type(entry["kalman_r"]) is float
    baseline = RetuneParams(
        planner={name: getattr(config, name) for name in PREDICTOR_FIELDS}
    )
    assert (
        with_predictor_fields(RetuneParams(), baseline, predictor_fields(entry))
        == baseline
    )
