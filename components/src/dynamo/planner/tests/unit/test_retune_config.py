# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the retune imports.

"""The planner retune config section: a valid round trip, a rejected
startup-only knob and the grid-size limit, plus the retune-only, decision
latency and history rules that only config validation enforces, and one
error per invalid knob."""

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from pydantic import ValidationError

from dynamo.planner.config.planner_config import PlannerConfig, RetuneConfig
from dynamo.planner.reconfiguration import candidate_config
from dynamo.planner.retune.domains import (
    expand_planner_domain,
    expand_router_domain,
    parse_retune_section,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

# AISimulate recommend syntax. An omitted AISimulate knob searches its default
# domain, so temperature is pinned with a scalar: a 2 x 2 grid.
ROUTER = {
    "overlap_score_credit": {"choices": [0.0, 1.0]},
    "prefill_load_scale": {"choices": [0.5, 1.0]},
    "temperature": 0.0,
}
ROUTER_GRID = {
    "overlap_score_credit": [0.0, 1.0],
    "prefill_load_scale": [0.5, 1.0],
    "router_temperature": [0.0],
}
LSD = "load_scaling_down_sensitivity"
SENSITIVITY_OFF = {"load_sensitivity": {"preset": False}}
RETUNE_ONLY = {"enable_load_scaling": False, "enable_throughput_scaling": False}
SLA_THROUGHPUT = {"optimization_target": "sla", "enable_throughput_scaling": True}
LOAD_TARGET = {
    "optimization_target": "load",
    "prefill_scale_up_queue_tokens": 1000,
    "prefill_scale_down_queue_tokens": 100,
    "decode_scale_up_kv_rate": 80,
    "decode_scale_down_kv_rate": 40,
}


def _retune(**overrides):
    return {"interval_seconds": 600, "router": ROUTER, **overrides}


def _config(mode="disagg", retune=None, **overrides):
    values = {"namespace": "test-ns", "mode": mode, **overrides}
    if retune is not None:
        values["retune"] = retune
    return PlannerConfig(**values)


@pytest.mark.parametrize(
    ("overrides", "retune", "updates", "router", "planner"),
    [
        # Without retune, the dump has no retune key.
        ({}, None, {"max_gpu_budget": 24}, None, None),
        # Router knobs expand to live names and floats; an extension knob is
        # a dimension only when written.
        (
            RETUNE_ONLY,
            _retune(
                router={
                    "temperature": {"choices": [0, 1]},
                    "prefill_load_scale": 1,
                    "decode_active_request_weight": {
                        "range": {"min": 0.5, "max": 1.0, "step": 0.25}
                    },
                }
            ),
            {"max_gpu_budget": 24},
            {
                "overlap_score_credit": [0.0, 0.5, 1.0],
                "prefill_load_scale": [1.0],
                "router_temperature": [0.0, 1.0],
                "decode_active_request_weight": [0.5, 0.75, 1.0],
            },
            {},
        ),
        # Planner knobs expand to PlannerConfig fields, in their field types.
        (
            LOAD_TARGET,
            _retune(
                planner={
                    "prefill_scale_up_queue_tokens": {"choices": [800, 1200]},
                    "decode_scale_down_kv_rate": {"choices": [30, 50.0]},
                    "decode_min_workers": 2,
                    "min_num_gpus": {"choices": [-1, 4]},
                }
            ),
            {"decode_scale_down_kv_rate": 50.0, LSD: 70},
            ROUTER_GRID,
            {
                LSD: [70, 80, 90],
                "prefill_scale_up_queue_tokens": [800, 1200],
                "decode_scale_down_kv_rate": [30.0, 50.0],
                "decode_min_endpoint": [2],
                "min_gpu_budget": [-1, 4],
            },
        ),
        # The load predictor goes to the pre-sweep, not the grid.
        (
            SLA_THROUGHPUT,
            _retune(
                planner={
                    "load_predictor": {"preset": False, "type": "kalman"},
                    "kalman_r": {"choices": [1, 2.5]},
                },
            ),
            {"load_predictor": "kalman", "kalman_r": 2.5},
            ROUTER_GRID,
            {},
        ),
    ],
    ids=["no-retune", "router", "planner", "predictor"],
)
def test_retune_round_trips_through_strict_runtime_reconfiguration(
    overrides, retune, updates, router, planner
):
    config = _config(retune=retune, **overrides)
    dumped = config.model_dump()

    if retune is None:
        assert "retune" not in dumped
    else:
        # Stored as written, expanded on use: validation never writes back
        # (history_seconds stays auto), so one section can back many configs.
        assert config.retune.model_fields_set == set(retune)
        assert config.retune.history_seconds == retune.get("history_seconds")
        assert config.retune.router == retune["router"]
        assert config.retune.planner == retune.get("planner")
        assert expand_router_domain(config.retune) == router
        grid = expand_planner_domain(config).grid
        assert grid == planner
        assert {key: [type(v) for v in values] for key, values in grid.items()} == {
            key: [type(v) for v in values] for key, values in planner.items()
        }
    restored = PlannerConfig.model_validate(dumped, strict=True)
    assert restored.model_dump() == dumped
    candidate = candidate_config(config, updates)
    assert candidate.retune == config.retune
    assert (candidate.enable_load_scaling, candidate.enable_throughput_scaling) == (
        config.enable_load_scaling,
        config.enable_throughput_scaling,
    )
    for field, value in updates.items():
        assert getattr(candidate, field) == value


@pytest.mark.parametrize(
    ("retune", "match"),
    [
        (
            {"planner": {"enable_load_scaling": {"choices": [True, False]}}},
            r"cannot change \['enable_load_scaling'\]: they are fixed when the "
            "planner starts",
        ),
        ({"router": {"policy": "round_robin"}}, "policy is fixed at startup"),
        # A preset mapping cannot change load_min_observations either.
        (
            {
                "planner": {
                    "load_sensitivity": {
                        "preset": [{LSD: 60, "load_min_observations": 3}]
                    }
                }
            },
            "load_min_observations, which is fixed at startup; it must equal the "
            "running value 5",
        ),
    ],
    ids=["planner-knob", "router-policy", "preset-mapping"],
)
def test_startup_only_knobs_are_rejected(retune, match):
    with pytest.raises(ValidationError, match=match):
        _config(retune=_retune(**retune))


@pytest.mark.parametrize(
    ("overrides", "retune", "match"),
    [
        # The product of all dimensions, each under its public name.
        (
            {},
            _retune(
                router={**ROUTER, "overlap_score_credit": {"choices": [0.0, 0.5, 1.0]}},
                planner={**SENSITIVITY_OFF, LSD: {"choices": [70, 80, 90]}},
                max_candidates=17,
            ),
            r"18 candidates, exceeding retune.max_candidates=17 \(router "
            "overlap_score_credit 3 x prefill_load_scale 2 x temperature 1; "
            r"planner load_scaling_down_sensitivity 3\)",
        ),
        # One dimension, sized before a value is built.
        (
            RETUNE_ONLY,
            _retune(
                router={
                    "temperature": {"range": {"min": 0.0, "max": 1e3, "step": 1e-6}}
                }
            ),
            "temperature expands to 1000000001 values, more than the limit of 512",
        ),
        # preset: false is 256 combinations of 40 distinct predictors.
        (
            SLA_THROUGHPUT,
            _retune(
                planner={"load_predictor": {"preset": False}},
                history_seconds=9000,
                max_candidates=39,
            ),
            "40 distinct predictor candidates, exceeding retune.max_candidates=39",
        ),
        (
            SLA_THROUGHPUT,
            _retune(
                planner={
                    "load_predictor": {"preset": False},
                    "kalman_q_level": {"choices": [float(i + 1) for i in range(64)]},
                    "kalman_q_trend": {"choices": [float(i + 1) for i in range(64)]},
                },
                history_seconds=9000,
            ),
            "expands to 262144 combinations, more than 65536",
        ),
        # A preset: false knob, sized before its combinations are built.
        (
            SLA_THROUGHPUT,
            _retune(
                planner={
                    "load_predictor": {"preset": False, "type": "kalman"},
                    "kalman_r": {"range": {"min": 0.1, "max": 1e3, "step": 1e-30}},
                }
            ),
            r"retune.planner.kalman_r expands to over 10\^28 values",
        ),
    ],
    ids=["grid", "dimension", "predictors", "preset-combinations", "preset-off-knob"],
)
def test_grid_size_is_limited(overrides, retune, match):
    with pytest.raises(ValidationError, match=match):
        _config(retune=retune, **overrides)


def test_retune_only_mode_needs_both_scaling_flags_explicitly_false():
    # Retune-only: no easy-mode forcing, and a load target needs no load
    # thresholds, since load scaling never runs.
    config = _config(
        mode="agg", optimization_target="load", retune=_retune(), **RETUNE_ONLY
    )
    assert (config.enable_load_scaling, config.enable_throughput_scaling) == (
        False,
        False,
    )
    # enable_load_scaling is false only by default: easy mode forces it on.
    config = _config(retune=_retune(), enable_throughput_scaling=False)
    assert (config.enable_load_scaling, config.enable_throughput_scaling) == (
        True,
        False,
    )
    # Without retune, both flags false is still an error.
    with pytest.raises(ValidationError, match="At least one scaling mode must be"):
        _config(optimization_target="sla", **RETUNE_ONLY)


def test_measured_decision_latency_must_fit_the_interval():
    # A measured latency is at most decision_latency_scale x timeout_seconds,
    # which must fit in the interval.
    retune = RetuneConfig(
        interval_seconds=600,
        timeout_seconds=300,
        decision_latency_seconds="measured",
        decision_latency_scale=2.0,
    )
    assert retune.decision_latency_scale == 2.0
    with pytest.raises(ValidationError, match=r"scale \* timeout_seconds <= interval"):
        RetuneConfig(
            interval_seconds=600,
            decision_latency_seconds="measured",
            decision_latency_scale=2.0,
        )


def test_dry_run_is_rejected_in_favor_of_advisory():
    # Recommend-only retune is the planner-wide advisory switch.
    with pytest.raises(ValidationError, match="dry_run\n.*Extra inputs"):
        RetuneConfig(interval_seconds=600, dry_run=True)


def test_history_must_warm_every_candidate_predictor():
    planner = {
        "load_predictor": {"preset": False, "type": "prophet"},
        "prophet_window_size": {"choices": [20, 100]},
    }
    # 180 s throughput interval x the largest candidate window, 100.
    with pytest.raises(ValidationError, match=r">= 18000s"):
        _config(retune=_retune(planner=planner, history_seconds=9000), **SLA_THROUGHPUT)
    _config(retune=_retune(planner=planner, history_seconds=18000), **SLA_THROUGHPUT)


INCOMPLETE_KALMAN = {
    "load_predictor": "kalman",
    "load_predictor_log1p": False,
    "prophet_window_size": 50,
    "kalman_q_level": 1.0,
    "kalman_q_trend": 0.1,
    "kalman_min_points": 5,
}


@pytest.mark.parametrize(
    ("section", "raw", "message"),
    [
        (
            "router",
            {"temperature": [0.5, 1.0]},
            "retune.router.temperature: write a scalar, {choices: [...]} or "
            "{range: {min, max, step}}",
        ),
        (
            "planner",
            {"kalman_r": {"choices": [1.0, -1.0]}},
            "retune.planner.kalman_r.choices.choices.1: Input should be greater than 0",
        ),
        # A preset mapping's errors are reported against the entry.
        (
            "planner",
            {"load_predictor": {"preset": [INCOMPLETE_KALMAN]}},
            "retune.planner: planner.load_predictor.preset entry "
            f"{INCOMPLETE_KALMAN}: kalman_r: Field required",
        ),
    ],
    ids=["bare-list", "one-form", "preset-entry"],
)
def test_an_invalid_knob_reports_one_error(section, raw, message):
    # Each knob union is tagged by the form the value is written in, so
    # pydantic reports that form's error, not one per union member.
    with pytest.raises(ValueError) as error:
        parse_retune_section(section, raw)
    assert str(error.value) == message
