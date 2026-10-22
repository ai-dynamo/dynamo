# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for retune search-domain expansion."""

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import RetuneParams
from dynamo.planner.retune.domains import (
    PlannerDomain,
    expand_planner_domain,
    expand_router_domain,
    grid_points,
    grid_size,
    required_history_seconds,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

RETUNE_ONLY = {"enable_load_scaling": False, "enable_throughput_scaling": False}
SLA_THROUGHPUT = {"optimization_target": "sla", "enable_throughput_scaling": True}


def _config(router="default", planner="default", **overrides):
    retune = {
        "interval_seconds": 600,
        "history_seconds": 9000,
        "router": router,
        "planner": planner,
    }
    return PlannerConfig(namespace="test-ns", mode="disagg", retune=retune, **overrides)


@pytest.fixture
def simulation_defaults():
    pytest.importorskip(
        "aisimulate.sweeper",
        reason="AI Simulate is an optional Dynamo simulation dependency",
    )
    from dynamo.planner.simulation.load_predictor import LOAD_PREDICTOR_PRESETS
    from dynamo.router.simulation.config import (
        OVERLAP_SCORE_CREDIT_DEFAULTS,
        PREFILL_LOAD_SCALE_DEFAULTS,
        TEMPERATURE_DEFAULTS,
    )

    return {
        "credit": list(OVERLAP_SCORE_CREDIT_DEFAULTS),
        "scale": list(PREFILL_LOAD_SCALE_DEFAULTS),
        "temperature": list(TEMPERATURE_DEFAULTS),
        "predictors": list(LOAD_PREDICTOR_PRESETS),
    }


def test_default_router_domain_uses_live_names_and_108_points(simulation_defaults):
    config = _config(planner={}, **RETUNE_ONLY)
    assert config.retune is not None

    domain = expand_router_domain(config.retune)

    assert domain == {
        "overlap_score_credit": simulation_defaults["credit"],
        "prefill_load_scale": simulation_defaults["scale"],
        "router_temperature": simulation_defaults["temperature"],
    }
    assert grid_size(domain) == 108
    points = grid_points(domain, {})
    assert len(points) == len(set(points)) == 108
    assert [point.key() for point in points] == sorted(point.key() for point in points)
    assert all(set(point.router) == set(domain) for point in points)


def test_default_router_domain_credit_hook(simulation_defaults):
    config = _config(planner={}, **RETUNE_ONLY)
    assert config.retune is not None

    domain = expand_router_domain(config.retune, zero_startup_credit=True)

    assert domain["overlap_score_credit"] == [0.0]
    assert grid_size(domain) == 36


def test_explicit_router_domain_is_copied_unchanged():
    router = {"overlap_score_credit": [0.0, 0.5], "router_temperature": [0.0]}
    config = _config(router=router, planner={}, **RETUNE_ONLY)
    assert config.retune is not None

    domain = expand_router_domain(config.retune, zero_startup_credit=True)

    assert domain == router
    domain["overlap_score_credit"].append(9.0)
    assert config.retune.router == router


def test_default_planner_domain_is_empty_without_scaling(simulation_defaults):
    config = _config(**RETUNE_ONLY)

    domain = expand_planner_domain(config)

    assert domain == PlannerDomain()
    assert domain.empty


def test_default_planner_domain_for_load_scaling(simulation_defaults):
    config = _config()  # easy mode: load scaling forced on
    assert config.enable_load_scaling and not config.enable_throughput_scaling

    domain = expand_planner_domain(config)

    assert domain.grid == {"load_scaling_down_sensitivity": [70, 80, 90]}
    assert domain.predictor_presets is None


def test_default_planner_domain_for_sla_throughput(simulation_defaults):
    config = _config(**SLA_THROUGHPUT)

    domain = expand_planner_domain(config)

    assert domain.grid == {}
    assert domain.predictor_presets == simulation_defaults["predictors"]
    assert len(domain.predictor_presets) == 11
    assert not domain.empty


def test_default_planner_domain_for_hybrid_scaling(simulation_defaults):
    config = _config(enable_load_scaling=True, **SLA_THROUGHPUT)

    domain = expand_planner_domain(config)

    assert domain.grid == {"load_scaling_down_sensitivity": [70, 80, 90]}
    assert domain.predictor_presets == simulation_defaults["predictors"]
    # The predictor pre-sweep is not a grid dimension.
    assert grid_size(expand_router_domain(config.retune), domain.grid) == 324


def test_explicit_planner_domain_is_the_grid():
    planner = {
        "load_predictor": ["arima", "kalman"],
        "max_throughput_scaling_replicas": [2, 4, 8],
    }
    config = _config(router={}, planner=planner, **SLA_THROUGHPUT)

    domain = expand_planner_domain(config)

    assert domain.grid == planner
    assert domain.predictor_presets is None
    assert grid_size(domain.grid) == 6


def test_expand_planner_domain_requires_retune():
    with pytest.raises(ValueError, match="no retune section"):
        expand_planner_domain(PlannerConfig(namespace="test-ns"))


def test_grid_size_and_points():
    router = {"overlap_score_credit": [1.0, 0.0], "prefill_load_scale": [2.0]}
    planner = {"decode_min_endpoint": [2, 1]}

    assert grid_size() == 1
    assert grid_size(router, planner) == 4
    assert grid_size(router, {"x": []}) == 0
    points = grid_points(router, planner)
    assert points == sorted(points, key=RetuneParams.key)
    assert {point.key() for point in points} == {
        RetuneParams(
            router={"overlap_score_credit": credit, "prefill_load_scale": 2.0},
            planner={"decode_min_endpoint": endpoint},
        ).key()
        for credit in (0.0, 1.0)
        for endpoint in (1, 2)
    }
    assert grid_points({}, {}) == [RetuneParams()]


def test_required_history_covers_current_and_candidate_predictors():
    config = _config(router={"overlap_score_credit": [0.0]}, planner={})
    # Defaults: 180 s throughput interval x prophet_window_size 50.
    assert required_history_seconds(config, PlannerDomain()) == 9000.0
    domain = PlannerDomain(
        grid={"prophet_window_size": [20, 80], "kalman_min_points": [3]}
    )
    assert required_history_seconds(config, domain) == 180.0 * 80
    domain = PlannerDomain(grid={"kalman_min_points": [120]})
    assert required_history_seconds(config, domain) == 180.0 * 120


def test_required_history_covers_predictor_presets(simulation_defaults):
    config = _config(
        router={"overlap_score_credit": [0.0]},
        planner={},
        prophet_window_size=10,
        kalman_min_points=2,
        **SLA_THROUGHPUT,
    )
    domain = PlannerDomain(predictor_presets=simulation_defaults["predictors"])
    # The largest preset window is prophet_w50 (50 intervals of 180 s).
    assert required_history_seconds(config, domain) == 180.0 * 50
