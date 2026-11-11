# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the retune imports.

"""Retune search-domain expansion: the default domains of each scaling mode,
and predictor candidates that match the running predictor."""

import pytest

pytest.importorskip(
    "aisim.sweeper",
    reason="AISim is an optional Dynamo simulation dependency",
)

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import RetuneParams
from dynamo.planner.reconfiguration import PREDICTOR_FIELDS
from dynamo.planner.retune.domains import (
    expand_planner_domain,
    expand_router_domain,
    grid_size,
    required_history_seconds,
    throughput_intervals,
)
from dynamo.planner.retune.sweep import evaluated_params
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


# One router point, so the planner grid alone sizes the search.
PINNED_ROUTER = {
    "overlap_score_credit": 1.0,
    "prefill_load_scale": 1.0,
    "temperature": 0.0,
}


def _config(planner="default", router="default", **overrides):
    retune = {
        "interval_seconds": 600,
        "history_seconds": 30000,
        "router": router,
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

    # router: default is the AISim domain, 3 x 9 x 4 = 108 points.
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
    presweep = {"fields": predictor_fields(entry)}
    base = config.model_dump(mode="json")
    assert evaluated_params(RetuneParams(), baseline, presweep, base) == baseline


# --- scaling policy -------------------------------------------------------

THROUGHPUT = "enable_throughput_scaling"
LOAD = "enable_load_scaling"
TI = "throughput_adjustment_interval_seconds"
LI = "load_adjustment_interval_seconds"
SLA_HYBRID = {"enable_load_scaling": True, **SLA_THROUGHPUT}


def _points(domain, *keys):
    return sorted(tuple(point[key] for key in keys) for point in domain.points)


def test_scaling_policy_is_searched_only_when_written():
    domain = expand_planner_domain(_config(router=PINNED_ROUTER, **SLA_HYBRID))

    assert not set(domain.grid) & {THROUGHPUT, LOAD, TI, LI}
    assert _points(domain, LSD) == [(70,), (80,), (90,)]


def test_default_scaling_policies_fold_inactive_fields_and_deduplicate():
    # The six AISim presets that scale; "disabled" is never a candidate.
    domain = expand_planner_domain(
        _config(
            router=PINNED_ROUTER,
            planner={"scaling_policy": {"preset": "default"}},
            **SLA_HYBRID,
        )
    )

    assert domain.grid[THROUGHPUT] == [True, False]
    # Throughput-only points fold the load sensitivity to the running 80 and
    # load-only points the throughput interval to the running 180, so each
    # appears once per distinct active value: 2 + 2 x 3 + 2 x 3 points.
    assert _points(domain, THROUGHPUT, LOAD, TI, LI, LSD) == sorted(
        [
            (True, False, 180, 5, 80),
            (True, False, 600, 5, 80),
            *((False, True, 180, li, lsd) for li in (5, 10) for lsd in (70, 80, 90)),
            *((True, True, ti, 5, lsd) for ti in (180, 600) for lsd in (70, 80, 90)),
        ]
    )
    # The pre-sweep runs: some point scales on throughput.
    assert len(domain.predictor_candidates) == len(LOAD_PREDICTOR_PRESETS)


def test_independent_scaling_knobs_combine_and_drop_invalid_points():
    domain = expand_planner_domain(
        _config(
            router=PINNED_ROUTER,
            planner={
                "scaling_policy": {"preset": False},
                THROUGHPUT: True,
                LOAD: {"choices": [False, True]},
                TI: {"choices": [10, 180]},
                LI: {"choices": [5, 10]},
                "load_sensitivity": {"preset": False},
                LSD: 80,
            },
            **SLA_THROUGHPUT,
        )
    )

    # Hybrid points need the load interval below the throughput interval:
    # (10, 10) is dropped.
    assert _points(domain, LOAD, TI, LI) == sorted(
        [
            (False, 10, 5),
            (False, 10, 10),
            (False, 180, 5),
            (False, 180, 10),
            (True, 10, 5),
            (True, 180, 5),
            (True, 180, 10),
        ]
    )


@pytest.mark.parametrize(
    ("overrides", "policy", "match"),
    [
        (
            SLA_HYBRID,
            ["disabled"],
            "'disabled' turns both scaling modes off; retune never turns "
            "scaling off",
        ),
        (
            SLA_HYBRID,
            [{THROUGHPUT: True, LOAD: True, TI: 10, LI: 10}],
            "load_adjustment_interval_seconds 10 not shorter than "
            "throughput_adjustment_interval_seconds 10",
        ),
        (
            {**SLA_HYBRID, LI: 10, "scheduling": {"scale_interval_seconds": 10}},
            ["load_180_5"],
            "load_adjustment_interval_seconds 5, not a multiple of the planner's "
            r"scheduling.scale_interval_seconds 10 \(fixed at startup\)",
        ),
        # Other targets always scale on load only.
        ({}, ["hybrid_180_5"], "has no effect here"),
    ],
    ids=["disabled", "interval-order", "scale-interval", "easy-mode"],
)
def test_written_scaling_policies_that_cannot_run_are_rejected(
    overrides, policy, match
):
    with pytest.raises(ValueError, match=match):
        expand_planner_domain(
            _config(
                router=PINNED_ROUTER,
                planner={"scaling_policy": {"preset": policy}},
                **overrides,
            )
        )


def test_easy_mode_searches_the_load_interval_only():
    domain = expand_planner_domain(
        _config(router=PINNED_ROUTER, planner={"scaling_policy": {"preset": "default"}})
    )

    # load_180_5 and load_180_10, each with the three sensitivities; the
    # throughput interval folds away.
    assert _points(domain, THROUGHPUT, LOAD, TI, LI, LSD) == [
        (False, True, 180, li, lsd) for li in (5, 10) for lsd in (70, 80, 90)
    ]
    assert domain.predictor_candidates is None
    with pytest.raises(ValueError, match="require optimization_target='sla'"):
        expand_planner_domain(
            _config(
                router=PINNED_ROUTER,
                planner={"scaling_policy": {"preset": False}, THROUGHPUT: True},
            )
        )


def test_a_policy_that_adds_throughput_scaling_brings_the_presweep_and_history():
    # A load-only planner whose grid can turn throughput scaling on at 600 s.
    config = _config(
        router=PINNED_ROUTER,
        planner={"scaling_policy": {"preset": ["load_180_5", "hybrid_600_5"]}},
        optimization_target="sla",
        enable_throughput_scaling=False,
        enable_load_scaling=True,
    )

    domain = expand_planner_domain(config)

    assert len(domain.predictor_candidates) == len(LOAD_PREDICTOR_PRESETS)
    assert throughput_intervals(config, domain.points) == [600]
    # 600 s x the largest prophet_window_size / kalman_min_points.
    assert required_history_seconds(config, domain) % 600 == 0
    assert required_history_seconds(config, domain) >= 600 * config.prophet_window_size


def test_max_throughput_scaling_replicas_needs_a_throughput_point():
    planner = {"max_throughput_scaling_replicas": {"choices": [4, 8]}}
    with pytest.raises(ValueError, match="requires enable_throughput_scaling"):
        expand_planner_domain(
            _config(
                router=PINNED_ROUTER,
                planner=planner,
                optimization_target="sla",
                enable_throughput_scaling=False,
                enable_load_scaling=True,
            )
        )

    domain = expand_planner_domain(
        _config(
            router=PINNED_ROUTER,
            planner={
                **planner,
                "scaling_policy": {"preset": ["load_180_5", "hybrid_180_5"]},
            },
            optimization_target="sla",
            enable_throughput_scaling=False,
            enable_load_scaling=True,
        )
    )
    # Load-only points fold the replicas to the running 8.
    assert _points(
        domain, THROUGHPUT, "max_throughput_scaling_replicas", LSD
    ) == sorted(
        [
            *((False, 8, lsd) for lsd in (70, 80, 90)),
            *((True, replicas, lsd) for replicas in (4, 8) for lsd in (70, 80, 90)),
        ]
    )


def test_evaluated_params_fold_inactive_fields_to_the_baseline():
    base = PlannerConfig(namespace="test-ns", mode="disagg", **SLA_HYBRID).model_dump(
        mode="json"
    )
    baseline = RetuneParams(planner={THROUGHPUT: True, LSD: 70})
    # Throughput-only: its sensitivity is inert, so it is the baseline's.
    candidate = RetuneParams(planner={THROUGHPUT: True, LOAD: False, LSD: 90})

    evaluated = evaluated_params(candidate, baseline, None, base)

    assert evaluated.planner == {THROUGHPUT: True, LOAD: False, LSD: 70}
    # The pre-sweep winner of the candidate's interval only.
    presweep = {
        "by_interval": {
            "180": {"fields": {"load_predictor": "kalman"}},
            "600": {"fields": {"load_predictor": "arima"}},
        }
    }
    at_600 = RetuneParams(planner={THROUGHPUT: True, TI: 600, LSD: 70})
    assert (
        evaluated_params(at_600, baseline, presweep, base).planner["load_predictor"]
        == "arima"
    )
