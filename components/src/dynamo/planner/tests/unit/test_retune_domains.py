# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for retune search-domain expansion."""

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import RetuneParams
from dynamo.planner.retune.domains import (
    PlannerDomain,
    describe_grid,
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
HYBRID = {"enable_load_scaling": True, **SLA_THROUGHPUT}
LSD = "load_scaling_down_sensitivity"
KALMAN_MAPPING = {
    "load_predictor": "kalman",
    "load_predictor_log1p": False,
    "prophet_window_size": 50,
    "kalman_q_level": 1.0,
    "kalman_q_trend": 0.1,
    "kalman_r": 10.0,
    "kalman_min_points": 120,
}


def _config(router="default", planner="default", mode="disagg", **overrides):
    retune = {
        "interval_seconds": 600,
        "history_seconds": 30000,
        "router": router,
        "planner": planner,
    }
    return PlannerConfig(namespace="test-ns", mode=mode, retune=retune, **overrides)


@pytest.fixture
def simulation_defaults():
    from dynamo.planner.simulation.load_predictor import (
        LOAD_PREDICTOR_PRESETS,
        distinct_predictor_candidates,
    )
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
        "candidates": distinct_predictor_candidates(list(LOAD_PREDICTOR_PRESETS)),
    }


def test_default_router_domain_uses_live_names_and_108_points(simulation_defaults):
    # Scenario 1: router: default, planner omitted.
    config = _config(planner=None, **RETUNE_ONLY)
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
    # planner: default without scaling expands to nothing, like no planner.
    assert expand_planner_domain(config) == PlannerDomain()
    assert expand_planner_domain(_config(**RETUNE_ONLY)).empty


@pytest.mark.parametrize(
    ("overrides", "grid", "presets", "size"),
    [
        # Easy mode forces load scaling on.
        ({}, {LSD: [70, 80, 90]}, False, 324),
        (SLA_THROUGHPUT, {}, True, 108),
        (HYBRID, {LSD: [70, 80, 90]}, True, 324),
    ],
    ids=["load", "sla-throughput", "hybrid"],
)
def test_default_planner_domain_for_hybrid_scaling(
    simulation_defaults, overrides, grid, presets, size
):
    config = _config(**overrides)

    domain = expand_planner_domain(config)

    assert domain.grid == grid
    if presets:
        # The 11 simulation presets, by name, as complete public mappings.
        assert domain.predictor_candidates == simulation_defaults["candidates"]
        assert [label for label, _ in domain.predictor_candidates] == (
            simulation_defaults["predictors"]
        )
    else:
        assert domain.predictor_candidates is None
    assert not domain.empty
    # The predictor pre-sweep is not a grid dimension.
    assert grid_size(expand_router_domain(config.retune), domain.grid) == size


def test_explicit_domains_use_the_recommendation_syntax():
    # Proposal example 2.5-A: disagg + sla + hybrid scaling.
    router = {
        "overlap_score_credit": {"choices": [0.5, 1.0, 2.0]},
        "prefill_load_scale": 1.0,
        "temperature": {"range": {"min": 0.0, "max": 0.2, "step": 0.1}},
        "decode_active_request_weight": {"choices": [0.5, 1.0]},
    }
    planner = {
        "load_sensitivity": {"preset": False},
        LSD: {"choices": [70, 80, 90]},
        "decode_min_workers": {"choices": [1, 2]},
        "load_predictor": {
            "preset": ["arima_log1p", "kalman_default_raw", "kalman_reactive_log1p"]
        },
    }
    config = _config(router, planner, **HYBRID)

    router_domain = expand_router_domain(config.retune)
    planner_domain = expand_planner_domain(config)

    assert router_domain == {
        "overlap_score_credit": [0.5, 1.0, 2.0],
        "prefill_load_scale": [1.0],
        # Decimal steps: no float drift at the endpoint.
        "router_temperature": [0.0, 0.1, 0.2],
        "decode_active_request_weight": [0.5, 1.0],
    }
    assert planner_domain.grid == {LSD: [70, 80, 90], "decode_min_endpoint": [1, 2]}
    assert grid_size(router_domain) == 18
    assert grid_size(planner_domain.grid) == 6
    assert grid_size(router_domain, planner_domain.grid) == 108
    assert [label for label, _ in planner_domain.predictor_candidates] == [
        "arima_log1p",
        "kalman_default_raw",
        "kalman_reactive_log1p",
    ]
    assert describe_grid(router_domain, planner_domain.grid) == (
        "router overlap_score_credit 3 x prefill_load_scale 1 x temperature 3 x "
        "decode_active_request_weight 2; planner load_scaling_down_sensitivity 3 x "
        "decode_min_workers 2"
    )


def test_predictor_preset_off_and_lists_give_distinct_candidates():
    # Proposal example 2.5-C: 4 x 2^5 = 128 combinations, 34 distinct.
    planner = {
        "load_predictor": {"preset": False, "type": {"choices": ["arima", "kalman"]}}
    }
    domain = expand_planner_domain(_config(planner=planner, **SLA_THROUGHPUT))
    assert len(domain.predictor_candidates) == 34
    assert {entry["load_predictor"] for _, entry in domain.predictor_candidates} == {
        "arima",
        "kalman",
    }
    labels = [label for label, _ in domain.predictor_candidates]
    assert len(set(labels)) == len(labels) and labels[0] == "custom_0"

    # A complete mapping equal to a named preset is the same predictor.
    arima_raw = {**KALMAN_MAPPING, "load_predictor": "arima", "kalman_min_points": 5}
    planner = {"load_predictor": {"preset": ["arima_raw", arima_raw, KALMAN_MAPPING]}}
    domain = expand_planner_domain(_config(planner=planner, **SLA_THROUGHPUT))
    assert domain.predictor_candidates == [
        (
            "arima_raw",
            {
                **arima_raw,
                "prophet_window_size": 50,
                "kalman_q_trend": 0.1,
                "kalman_r": 10.0,
            },
        ),
        ("custom_2", KALMAN_MAPPING),
    ]


def test_custom_predictor_mapping_values_take_their_field_types():
    # YAML writes kalman_r: 1; the named presets and PlannerConfig hold 1.0.
    # A candidate equal to the running predictor must get the running
    # config's key, or the sweep replays the baseline configuration twice.
    integral = {
        **KALMAN_MAPPING,
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
    assert {name: type(value) for name, value in entry.items()} == {
        "load_predictor": str,
        "load_predictor_log1p": bool,
        "prophet_window_size": int,
        "kalman_q_level": float,
        "kalman_q_trend": float,
        "kalman_r": float,
        "kalman_min_points": int,
    }

    from dynamo.planner.reconfiguration import PREDICTOR_FIELDS
    from dynamo.planner.retune.sweep import with_predictor_fields
    from dynamo.planner.simulation.load_predictor import predictor_fields

    baseline = RetuneParams(
        planner={name: getattr(config, name) for name in PREDICTOR_FIELDS}
    )
    grid_point = RetuneParams(router={}, planner={})
    assert with_predictor_fields(grid_point, baseline, predictor_fields(entry)) == (
        baseline
    )


def test_predictor_candidates_are_expanded_once_per_spec(monkeypatch):
    # candidate_config validates retune again for every planner candidate the
    # controller checks, so the predictor expansion is memoized by spec.
    from dynamo.planner.reconfiguration import candidate_config
    from dynamo.planner.retune import domains

    calls = []
    expand = domains._expand_predictor_candidates

    def counting(public, limit):
        calls.append(limit)
        return expand(public, limit)

    monkeypatch.setattr(domains, "_PREDICTOR_CANDIDATE_CACHE", {})
    monkeypatch.setattr(domains, "_expand_predictor_candidates", counting)
    planner = {
        "load_predictor": {"preset": False, "type": "kalman"},
        "kalman_r": {"choices": [1, 2]},
    }
    config = _config(router=None, planner=planner, **SLA_THROUGHPUT)
    first = expand_planner_domain(config)
    candidate_config(config, {"decode_min_endpoint": 2})
    second = expand_planner_domain(config)
    assert len(calls) == 1
    assert second == first and len(first.predictor_candidates) == 32
    # Callers get their own mappings.
    first.predictor_candidates[0][1]["kalman_r"] = 99.0
    assert expand_planner_domain(config).predictor_candidates[0][1]["kalman_r"] == 1.0
    # A different domain or limit is expanded again.
    _config(
        router=None,
        planner={**planner, "kalman_r": {"choices": [1, 3]}},
        **SLA_THROUGHPUT,
    )
    assert len(calls) == 2
    PlannerConfig(
        namespace="test-ns",
        mode="disagg",
        retune={**config.retune.model_dump(), "max_candidates": 100},
        **SLA_THROUGHPUT,
    )
    assert calls[-1] == 100 and len(calls) == 3


def test_explicit_planner_group_that_does_not_apply_is_an_error():
    with pytest.raises(ValueError, match="load_predictor has no effect here"):
        _config(planner={"load_predictor": {"preset": ["arima_raw"]}})
    with pytest.raises(ValueError, match="load_sensitivity has no effect here"):
        _config(
            planner={"load_sensitivity": {"preset": ["aggressive"]}}, **SLA_THROUGHPUT
        )
    with pytest.raises(ValueError, match="requires load or throughput scaling"):
        _config(planner={"load_predictor": {"preset": False}}, **RETUNE_ONLY)


def test_zero_startup_credit_collapses_only_an_omitted_credit():
    config = _config(planner=None, **RETUNE_ONLY)
    domain = expand_router_domain(config.retune, zero_startup_credit=True)
    assert domain["overlap_score_credit"] == [0.0]
    assert grid_size(domain) == 36

    explicit = _config(
        router={"overlap_score_credit": {"choices": [0.0, 1.0]}},
        planner=None,
        **RETUNE_ONLY,
    )
    # The controller rejects the positive credit per candidate (E3).
    assert expand_router_domain(explicit.retune, zero_startup_credit=True)[
        "overlap_score_credit"
    ] == [0.0, 1.0]


@pytest.mark.parametrize(
    ("load_model", "startup", "error"),
    [
        (None, "none", None),
        ({}, "aic", None),
        ({"type": "ais"}, "aic", None),
        ({"type": {"choices": ["ais"]}}, "aic", None),
        ({"type": "none"}, "none", None),
        ({"type": "none"}, "aic", r"keeps the running 'ais'; got \['none'\]"),
        ({"type": "ais"}, "none", r"keeps the running 'none'; got \['ais'\]"),
        ({"type": {"choices": ["none", "ais"]}}, "none", "fixed at startup"),
    ],
)
def test_prefill_load_model_must_equal_the_startup_model(load_model, startup, error):
    router = {"temperature": 0.0}
    if load_model is not None:
        router["prefill_load_model"] = load_model
    # Stage 1 (config validation) has no startup value and checks syntax only.
    config = _config(router=router, planner=None, **RETUNE_ONLY)

    def expand():
        return expand_router_domain(config.retune, startup_load_model=startup)

    if error is None:
        assert grid_size(expand()) == 27
    else:
        with pytest.raises(ValueError, match=error):
            expand()


def test_grid_size_and_points():
    router = {"overlap_score_credit": [1.0, 0.0], "prefill_load_scale": [2.0]}
    planner = {"decode_min_endpoint": [2, 1]}

    assert grid_size() == 1
    assert grid_size(router, planner) == 4
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


def test_required_history_covers_current_and_candidate_predictors(
    simulation_defaults,
):
    config = _config(router={"temperature": 0.0}, planner=None)
    # Defaults: 180 s throughput interval x prophet_window_size 50.
    assert required_history_seconds(config, PlannerDomain()) == 9000.0
    domain = PlannerDomain(predictor_candidates=[("custom_0", KALMAN_MAPPING)])
    assert required_history_seconds(config, domain) == 180.0 * 120

    config = _config(
        router={"temperature": 0.0},
        planner=None,
        prophet_window_size=10,
        kalman_min_points=2,
        **SLA_THROUGHPUT,
    )
    domain = PlannerDomain(predictor_candidates=simulation_defaults["candidates"])
    # The largest preset window is prophet_w50 (50 intervals of 180 s).
    assert required_history_seconds(config, domain) == 180.0 * 50
