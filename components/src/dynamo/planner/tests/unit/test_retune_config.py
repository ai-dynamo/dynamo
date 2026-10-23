# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the planner retune config section and its core types."""

import json

import pytest
from pydantic import ValidationError

from dynamo.planner.config.planner_config import (
    PlannerConfig,
    RetuneConfig,
    RetuneEngineSpec,
)
from dynamo.planner.core.types import RetuneParams
from dynamo.planner.reconfiguration import candidate_config
from dynamo.planner.retune.domains import expand_planner_domain, expand_router_domain

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
LSD = "load_scaling_down_sensitivity"
SENSITIVITY_OFF = {"load_sensitivity": {"preset": False}}
RETUNE_ONLY = {"enable_load_scaling": False, "enable_throughput_scaling": False}
SLA_THROUGHPUT = {"optimization_target": "sla", "enable_throughput_scaling": True}


def _retune(**overrides):
    return {"interval_seconds": 600, "router": ROUTER, **overrides}


def _config(mode="disagg", retune=None, **overrides):
    values = {"namespace": "test-ns", "mode": mode, **overrides}
    if retune is not None:
        values["retune"] = retune
    return PlannerConfig(**values)


# ---------------------------------------------------------------------------
# Defaults and schema strictness
# ---------------------------------------------------------------------------


def test_a_shared_retune_instance_is_never_modified():
    shared = RetuneConfig(interval_seconds=600, router=ROUTER)
    fields_set = set(shared.model_fields_set)

    _config(retune=shared, **RETUNE_ONLY)
    sla = _config(retune=shared, **SLA_THROUGHPUT)

    # history_seconds stays "auto" (the controller resolves it) and is not
    # dumped as set.
    assert shared.history_seconds is None
    assert shared.model_fields_set == fields_set
    assert "history_seconds" not in sla.model_dump(exclude_unset=True)["retune"]

    written = {**SENSITIVITY_OFF, LSD: {"choices": [70, 90]}}
    planner = RetuneConfig(interval_seconds=600, planner=written)
    config = _config(retune=planner)
    # Domains are stored as written and expanded on use, never written back.
    assert config.retune.planner == planner.planner == written
    assert expand_planner_domain(config).grid == {LSD: [70, 90]}


@pytest.mark.parametrize(
    ("model", "values"),
    [
        # A retired design field is just an unknown field.
        (RetuneConfig, {"interval_seconds": 600, "traffic_model": "open_loop"}),
        # Live-only inputs come with the live planner.
        (RetuneConfig, {"interval_seconds": 600, "live": {}}),
        (RetuneEngineSpec, {"unknown": 1}),
    ],
)
def test_retune_models_forbid_extra_fields(model, values):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model(**values)


# ---------------------------------------------------------------------------
# R1: retune-only mode and easy-mode forcing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        # Both flags explicitly false: retune-only, no forcing.
        (RETUNE_ONLY, (False, False)),
        ({**RETUNE_ONLY, "optimization_target": "sla"}, (False, False)),
        # Every other easy-target combination keeps today's forcing, including
        # the examples/deployments/dgdr/planner.yaml pattern.
        ({"enable_load_scaling": False}, (True, False)),
        # Both values are false, but enable_load_scaling is only the default.
        ({"enable_throughput_scaling": False}, (True, False)),
        (
            {"enable_load_scaling": False, "enable_throughput_scaling": True},
            (True, False),
        ),
        (SLA_THROUGHPUT, (False, True)),
    ],
)
def test_retune_only_mode_requires_both_flags_explicitly_false(overrides, expected):
    config = _config(retune=_retune(history_seconds=9000), **overrides)

    assert (config.enable_load_scaling, config.enable_throughput_scaling) == expected


def test_retune_only_load_target_needs_no_load_thresholds():
    # Load scaling never runs in retune-only mode.
    config = _config(
        mode="agg", optimization_target="load", retune=_retune(), **RETUNE_ONLY
    )
    assert config.decode_scale_up_kv_rate is None

    with pytest.raises(ValidationError, match="decode_scale_up_kv_rate"):
        _config(mode="agg", optimization_target="load", retune=_retune())


def test_legacy_both_flags_off_error_is_unchanged_without_retune():
    with pytest.raises(ValidationError) as error:
        _config(optimization_target="sla", **RETUNE_ONLY)
    messages = [item["msg"] for item in error.value.errors()]
    assert messages == [
        "Value error, At least one scaling mode must be enabled "
        "(enable_throughput_scaling or enable_load_scaling)"
    ]
    # With a retune section the same error adds the retune-only hint;
    # enable_load_scaling defaults to false but was not set explicitly.
    with pytest.raises(ValidationError, match="both flags to be set explicitly"):
        _config(
            retune=_retune(), optimization_target="sla", enable_throughput_scaling=False
        )


@pytest.mark.parametrize("overrides", [{}, RETUNE_ONLY])
def test_legacy_configs_without_retune_are_unchanged(overrides):
    config = _config(**overrides)

    assert config.retune is None
    dumped = config.model_dump()
    assert "retune" not in dumped
    assert "retune" not in json.loads(config.model_dump_json())
    assert PlannerConfig.model_validate(dumped).model_dump() == dumped
    assert _config(retune=None, **overrides).model_dump() == dumped


# ---------------------------------------------------------------------------
# R2-R6: mode, cadence, timing and domain syntax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_retune_requires_agg_or_disagg_mode(mode):
    with pytest.raises(ValidationError, match="retune requires mode='agg' or"):
        _config(mode=mode, retune=_retune(), **RETUNE_ONLY)


def test_retune_interval_must_be_multiple_of_scale_interval():
    with pytest.raises(ValidationError, match="multiple of scheduling"):
        _config(retune=_retune(interval_seconds=602), **RETUNE_ONLY)
    config = _config(
        retune=_retune(interval_seconds=630),
        scheduling={"scale_interval_seconds": 30},
        load_adjustment_interval_seconds=30,
        **RETUNE_ONLY,
    )
    assert config.retune is not None and config.retune.interval_seconds == 630


@pytest.mark.parametrize(
    "overrides",
    [
        {"timeout_seconds": 601},
        {"decision_latency_seconds": 601},
        {"timeout_seconds": 100, "decision_latency_seconds": 200},
    ],
)
def test_retune_timing_order_is_enforced(overrides):
    with pytest.raises(ValidationError, match="decision_latency_seconds <="):
        RetuneConfig(interval_seconds=600, **overrides)
    retune = RetuneConfig(
        interval_seconds=600, timeout_seconds=600, decision_latency_seconds=600
    )
    assert retune.effective_timeout_seconds == 600


def test_measured_decision_latency_is_accepted():
    retune = RetuneConfig(interval_seconds=600, decision_latency_seconds="measured")
    assert (retune.decision_latency_seconds, retune.decision_latency_scale) == (
        "measured",
        1.0,
    )
    scaled = RetuneConfig(
        interval_seconds=600,
        timeout_seconds=300,
        decision_latency_seconds="measured",
        decision_latency_scale=2.0,
    )
    assert scaled.decision_latency_scale == 2.0
    # An explicit default scale is fine with a fixed latency.
    fixed = RetuneConfig(
        interval_seconds=600, decision_latency_seconds=5, decision_latency_scale=1.0
    )
    assert fixed.decision_latency_seconds == 5


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        # timeout_seconds defaults to interval_seconds, so 2 * 600 > 600.
        (
            {"decision_latency_seconds": "measured", "decision_latency_scale": 2.0},
            r"decision_latency_scale \* timeout_seconds <= interval_seconds",
        ),
        (
            {
                "decision_latency_seconds": "measured",
                "timeout_seconds": 400,
                "decision_latency_scale": 2.0,
            },
            r"decision_latency_scale \* timeout_seconds <= interval_seconds",
        ),
        (
            {"decision_latency_seconds": "measured", "timeout_seconds": 601},
            r"timeout_seconds <= interval_seconds",
        ),
        (
            {"decision_latency_scale": 2.0},
            "decision_latency_scale other than 1.0 requires",
        ),
        (
            {"decision_latency_seconds": "measured", "decision_latency_scale": 0},
            "greater than 0",
        ),
        ({"decision_latency_seconds": "wall"}, "'measured'"),
        ({"decision_latency_seconds": -1.0}, "greater than or equal to 0"),
    ],
)
def test_decision_latency_mode_is_validated(overrides, match):
    with pytest.raises(ValidationError, match=match):
        RetuneConfig(interval_seconds=600, **overrides)


@pytest.mark.parametrize(
    ("router", "match"),
    [
        ({"overlap_score_credit": {"choices": []}}, "nonempty list"),
        (
            {"overlap_score_credit": {"choices": [float("nan")]}},
            r"overlap_score_credit.choices\[0\]: Input should be a finite number",
        ),
        ({"overlap_score_credit": {"choices": [-1.0]}}, "greater than or equal to 0"),
        ({"overlap_score_credit": {"choices": [1.0, 1]}}, "unique values"),
        ({"overlap_score_credit": True}, "overlap_score_credit: .*valid number"),
        # The list syntax of earlier drafts.
        (
            {"overlap_score_credit": [0.5, 1.0]},
            r"bare list; .*\{choices: \[0.5, 1.0\]\}",
        ),
        # A bare list under a name the schema rejects gets that rejection.
        ({"router_temperature": [0.0, 0.1]}, "write temperature instead"),
        ({"policy": ["round_robin", "kv_router"]}, "retune requires kv_router"),
        (
            {"prefill_load_model": ["ais"]},
            r"prefill_load_model is a bare list; write \{type: \.\.\.\}",
        ),
        (
            {"temperature": {"range": {"min": 0.0, "max": 1.0, "step": -0.5}}},
            "temperature.range.step: Input should be greater than 0",
        ),
        ({"prefill_load_model": {"type": "aic"}}, "Input should be 'none' or 'ais'"),
        (
            {"temperature": {"range": {"min": 0.0}}},
            r"retune.router.temperature.range.max: Field required \[",
        ),
        ({}, "retune.router is empty; write 'default'"),
        ("all", "default"),
    ],
)
def test_router_domain_syntax_is_the_recommendation_schema(router, match):
    with pytest.raises(ValidationError, match=match):
        RetuneConfig(interval_seconds=600, router=router)


def test_router_domain_expands_to_live_names_and_floats():
    config = _config(
        retune=_retune(
            router={"temperature": {"choices": [0, 1]}, "prefill_load_scale": 1}
        ),
        **RETUNE_ONLY,
    )
    assert config.retune.router == {
        "temperature": {"choices": [0, 1]},
        "prefill_load_scale": 1,
    }
    domain = expand_router_domain(config.retune)
    # An omitted AISimulate knob searches its AISimulate default domain.
    assert domain == {
        "overlap_score_credit": [0.0, 0.5, 1.0],
        "prefill_load_scale": [1.0],
        "router_temperature": [0.0, 1.0],
    }
    assert {type(value) for values in domain.values() for value in values} == {float}


@pytest.mark.parametrize(
    ("router", "match"),
    [
        (
            {"temperature": {"range": {"min": 0.01, "max": 1.0, "scale": "log"}}},
            "retune.router.temperature is a log range",
        ),
        (
            {"temperature": {"range": {"min": 0.0, "max": 1000.0, "step": 1.0}}},
            "temperature expands to 1001 values, more than the limit of 512",
        ),
    ],
    ids=["log", "too-many-values"],
)
def test_router_ranges_must_be_finite(router, match):
    with pytest.raises(ValidationError, match=match):
        _config(retune=_retune(router=router), **RETUNE_ONLY)


# ---------------------------------------------------------------------------
# R7 / R7b: planner domain keys, relevance, value types and history
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("planner", "overrides", "match"),
    [
        (
            {LSD: {"choices": [70]}},
            {},
            "load_sensitivity uses the default preset and cannot be combined",
        ),
        ({"max_num_gpus": {"choices": [8, 16]}}, RETUNE_ONLY, "requires load or"),
        ({}, {}, "retune.planner is empty; write 'default'"),
        (
            {"load_predictor": {"preset": [{"load_predictor": "kalman"}]}},
            {},
            # A preset mapping lists every knob, as in AISimulate recommend.
            r"kalman_r: Field required \(in \{'load_predictor': 'kalman'\}\)",
        ),
        # Bare lists (the syntax of earlier drafts): a preset group takes
        # preset names, and a name the schema rejects gets that rejection.
        (
            {"load_predictor": ["arima", "kalman"]},
            {},
            r"load_predictor is a bare list; load_predictor is a preset group: "
            r"write \{preset: \[\.\.\.\]\} listing preset names \(constant_last, "
            r"arima_raw, .*\{preset: false, type: \{choices: \[\.\.\.\]\}\}",
        ),
        (
            {"load_sensitivity": ["aggressive"]},
            {},
            r"preset names \(aggressive, default, conservative\) .*\{preset: false\} "
            r"with load_scaling_down_sensitivity",
        ),
        (
            {"min_workers": [1, 2]},
            {},
            r"min_workers is a bare list; write a scalar, \{choices: \[1, 2\]\}",
        ),
        ({"enable_load_scaling": [True, False]}, {}, "fixed when the planner starts"),
        ({"min_endpoint": [1, 2]}, {}, "write min_workers instead"),
        ({"ttft_ms": [100.0, 200.0]}, {}, "set retune.sla_ttft_ms"),
    ],
    ids=[
        "group-conflict",
        "requires-scaling",
        "empty",
        "incomplete-mapping",
        "bare-list-predictor-group",
        "bare-list-sensitivity-group",
        "bare-list-knob",
        "bare-list-startup-knob",
        "bare-list-live-name",
        "bare-list-sla",
    ],
)
def test_planner_domain_rejects_non_runtime_fields(planner, overrides, match):
    with pytest.raises(ValidationError, match=match):
        _config(retune=_retune(planner=planner), **overrides)


@pytest.mark.parametrize(
    ("mode", "overrides", "planner", "requirement"),
    [
        (
            "disagg",
            {
                "optimization_target": "sla",
                "enable_load_scaling": True,
                "enable_throughput_scaling": False,
            },
            {"load_predictor": {"preset": ["arima_raw"]}},
            "optimization_target='sla' with throughput scaling",
        ),
        (
            "disagg",
            {},
            {"max_throughput_scaling_replicas": {"choices": [2]}},
            "enable_throughput",
        ),
        (
            "disagg",
            {**SLA_THROUGHPUT, "enable_load_scaling": False},
            {**SENSITIVITY_OFF, LSD: {"choices": [70]}},
            "enable_load_scaling",
        ),
        (
            "disagg",
            {},
            {"decode_scale_up_kv_rate": {"choices": [80.0]}},
            "optimization_target='load'",
        ),
        (
            "agg",
            {
                "optimization_target": "load",
                "decode_scale_up_kv_rate": 80,
                "decode_scale_down_kv_rate": 40,
            },
            {"prefill_scale_up_queue_tokens": {"choices": [100]}},
            "with a prefill role",
        ),
        ("disagg", {}, {"min_workers": {"choices": [1, 2]}}, "requires mode='agg'"),
        (
            "agg",
            {},
            {"decode_min_workers": {"choices": [1, 2]}},
            "requires mode='disagg'",
        ),
    ],
)
def test_planner_domain_keys_must_affect_a_registered_builtin(
    mode, overrides, planner, requirement
):
    retune = _retune(planner=planner, history_seconds=9000)
    with pytest.raises(ValidationError, match="has no effect here") as error:
        _config(mode=mode, retune=retune, **overrides)
    assert requirement in str(error.value)
    # Errors name the public knob.
    assert f"retune.planner.{next(iter(planner))}" in str(error.value)


LOAD_TARGET = {
    "optimization_target": "load",
    "prefill_scale_up_queue_tokens": 1000,
    "prefill_scale_down_queue_tokens": 100,
    "decode_scale_up_kv_rate": 80,
    "decode_scale_down_kv_rate": 40,
}


@pytest.mark.parametrize(
    ("overrides", "planner", "grid"),
    [
        (
            {},
            {
                "min_num_gpus": {"choices": [-1, 4]},
                "max_num_gpus": {"range": {"min": 8, "max": 16, "step": 8}},
            },
            # The omitted load_sensitivity group searches its default domain.
            {LSD: [70, 80, 90], "min_gpu_budget": [-1, 4], "max_gpu_budget": [8, 16]},
        ),
        (
            LOAD_TARGET,
            {
                "prefill_scale_up_queue_tokens": {"choices": [800, 1200]},
                "decode_scale_down_kv_rate": {"choices": [30, 50.0]},
                "decode_min_workers": 2,
            },
            {
                LSD: [70, 80, 90],
                "prefill_scale_up_queue_tokens": [800, 1200],
                "decode_scale_down_kv_rate": [30.0, 50.0],
                "decode_min_endpoint": [2],
            },
        ),
    ],
)
def test_relevant_planner_domains_are_accepted(overrides, planner, grid):
    config = _config(retune=_retune(planner=planner, history_seconds=9000), **overrides)
    assert config.retune is not None and config.retune.planner == planner

    domain = expand_planner_domain(config)

    # Keyed by PlannerConfig field; float fields hold floats.
    assert domain.grid == grid
    assert domain.predictor_candidates is None
    for key, values in grid.items():
        assert [type(value) for value in domain.grid[key]] == [
            type(value) for value in values
        ]


@pytest.mark.parametrize(
    ("planner", "overrides", "match"),
    [
        ({**SENSITIVITY_OFF, LSD: {"choices": [70.0]}}, {}, "valid integer"),
        ({**SENSITIVITY_OFF, LSD: {"choices": [101]}}, {}, "must be <= 100"),
        (
            {"load_predictor": {"preset": False, "type": "linear"}},
            {},
            "'constant', 'arima', 'prophet' or 'kalman'",
        ),
        (
            {
                "load_predictor": {"preset": False},
                "load_predictor_log1p": {"choices": [True, 2]},
            },
            {},
            "valid boolean",
        ),
        (
            {"load_predictor": {"preset": ["arima_fast"]}},
            {},
            "invalid choice 'arima_fast'",
        ),
        (
            {"load_sensitivity": {"preset": ["aggressive", "relaxed"]}},
            {},
            "invalid choice 'relaxed'",
        ),
        # Bounds come from PlannerConfig: a GPU budget may be -1, not -2.
        ({"max_num_gpus": {"choices": [-2, 8]}}, {}, "greater than or equal to -1"),
        (
            {"decode_scale_up_kv_rate": {"choices": [80.0, 120.0]}},
            LOAD_TARGET,
            "decode_scale_up_kv_rate value 120.0 is invalid: .*less than or equal to 100",
        ),
    ],
)
def test_planner_domain_values_follow_strict_field_types(planner, overrides, match):
    overrides = overrides or {"enable_load_scaling": True, **SLA_THROUGHPUT}
    with pytest.raises(ValidationError, match=match):
        _config(retune=_retune(planner=planner, history_seconds=9000), **overrides)


@pytest.mark.parametrize(
    ("planner", "match"),
    [
        (
            {**SENSITIVITY_OFF, LSD: {"range": {"min": 60, "max": 90, "scale": "log"}}},
            "load_scaling_down_sensitivity is a log range",
        ),
        (
            {
                "load_predictor": {"preset": False},
                "kalman_r": {"range": {"min": 1.0, "max": 10.0}},
            },
            "kalman_r is a range without step",
        ),
        (
            {"min_workers": {"range": {"min": 1, "max": 4096, "step": 1}}},
            "min_workers expands to 4096 values, more than the limit of 512",
        ),
        (
            {
                "load_predictor": {"preset": False},
                "kalman_r": {"range": {"min": 1.0, "max": 100.0, "step": 0.01}},
            },
            "kalman_r expands to 9901 values",
        ),
        (
            {
                "load_predictor": {"preset": False},
                "kalman_q_level": {"choices": [float(i + 1) for i in range(64)]},
                "kalman_q_trend": {"choices": [float(i + 1) for i in range(64)]},
            },
            "expands to 262144 combinations, more than 65536",
        ),
    ],
    ids=["log", "no-step", "dimension-limit", "group-knob-limit", "combinations"],
)
def test_planner_ranges_must_be_finite_and_bounded(planner, match):
    with pytest.raises(ValidationError, match=match):
        _config(
            mode="agg",
            retune=_retune(planner=planner, history_seconds=9000),
            enable_load_scaling=True,
            **SLA_THROUGHPUT,
        )


def test_load_sensitivity_mappings_keep_the_startup_observations():
    mapping = {LSD: 60, "load_min_observations": 5}
    config = _config(
        retune=_retune(
            planner={"load_sensitivity": {"preset": ["aggressive", mapping]}}
        )
    )
    # A named preset is projected onto its runtime knob; the running
    # load_min_observations (5) stays.
    assert expand_planner_domain(config).grid == {LSD: [70, 60]}
    with pytest.raises(ValidationError, match="must equal the running value 5"):
        _config(
            retune=_retune(
                planner={
                    "load_sensitivity": {
                        "preset": [{LSD: 60, "load_min_observations": 3}]
                    }
                }
            )
        )
    # preset: false holds load_min_observations at its running value.
    config = _config(
        retune=_retune(planner={**SENSITIVITY_OFF, LSD: {"choices": [60, 70]}}),
        load_min_observations=8,
    )
    assert expand_planner_domain(config).grid == {LSD: [60, 70]}


def test_history_must_warm_every_candidate_predictor():
    planner = {
        "load_predictor": {"preset": False, "type": "prophet"},
        "prophet_window_size": {"choices": [20, 100]},
    }
    with pytest.raises(ValidationError, match=r">= 18000s"):
        _config(retune=_retune(planner=planner, history_seconds=9000), **SLA_THROUGHPUT)
    config = _config(
        retune=_retune(planner=planner, history_seconds=18000), **SLA_THROUGHPUT
    )
    # prophet x log1p {false, true} x window {20, 100}; the Kalman knobs do
    # not change a Prophet predictor.
    assert len(expand_planner_domain(config).predictor_candidates) == 4


# ---------------------------------------------------------------------------
# R6b / R8: default domains and grid size
# ---------------------------------------------------------------------------


def test_grid_size_is_capped_by_max_candidates():
    router = {
        "overlap_score_credit": {"choices": [0.0, 0.5, 1.0]},
        "prefill_load_scale": {"choices": [1.0, 2.0]},
        "temperature": 0.0,
    }
    planner = {**SENSITIVITY_OFF, LSD: {"choices": [70, 80, 90]}}
    retune = _retune(router=router, planner=planner, max_candidates=17)
    with pytest.raises(ValidationError, match="18 candidates, exceeding") as error:
        _config(retune=retune)
    # Each dimension's size, under its public name.
    assert (
        "(router overlap_score_credit 3 x prefill_load_scale 2 x temperature 1; "
        "planner load_scaling_down_sensitivity 3)"
    ) in str(error.value)
    retune["max_candidates"] = 18
    assert _config(retune=retune)


def test_omitted_router_knobs_search_their_default_domain():
    # Proposal example 2.5-B: credit 3 x scale 9 x temperature 2.
    retune = _retune(router={"temperature": {"choices": [0.0, 0.1]}}, max_candidates=53)
    with pytest.raises(ValidationError, match="54 candidates") as error:
        _config(retune=retune, **RETUNE_ONLY)
    assert "prefill_load_scale 9" in str(error.value)
    assert "pin knobs with a scalar" in str(error.value)


def test_predictor_candidates_are_capped_by_max_candidates():
    planner = {"load_predictor": {"preset": False}}
    # 256 combinations, 40 distinct predictors.
    with pytest.raises(ValidationError, match="40 distinct predictor candidates"):
        _config(
            retune=_retune(planner=planner, history_seconds=9000, max_candidates=39),
            **SLA_THROUGHPUT,
        )
    config = _config(
        retune=_retune(planner=planner, history_seconds=9000, max_candidates=40),
        **SLA_THROUGHPUT,
    )
    domain = expand_planner_domain(config)
    assert len(domain.predictor_candidates) == 40
    # The pre-sweep is not a grid dimension.
    assert domain.grid == {}


def test_default_planner_domain_that_expands_to_nothing_is_empty():
    with pytest.raises(ValidationError, match="non-empty router or planner"):
        _config(retune={"interval_seconds": 600, "planner": "default"}, **RETUNE_ONLY)


# ---------------------------------------------------------------------------
# Engine spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"prefill_engine_args": {}}, "must be set together"),
        (
            {
                "extra_engine_args": {},
                "prefill_engine_args": {},
                "decode_engine_args": {},
            },
            "cannot be combined",
        ),
    ],
)
def test_engine_spec_shape_is_validated(values, match):
    with pytest.raises(ValidationError, match=match):
        RetuneEngineSpec(**values)


def test_engine_spec_replay_kwargs_merge_router_overrides():
    pytest.importorskip("dynamo.llm")
    spec = RetuneEngineSpec(
        prefill_engine_args={"block_size": 64},
        decode_engine_args={"block_size": 64},
        router_config={"overlap_score_credit": 1.0, "prefill_load_scale": 2.0},
    )
    kwargs = spec.replay_kwargs({"overlap_score_credit": 0.5})

    # Every pinned field is a run_trace_replay kwarg.
    assert set(kwargs) == set(RetuneEngineSpec.model_fields)
    assert kwargs["router_mode"] == "kv_router"
    assert kwargs["router_config"].overlap_score_credit == 0.5
    assert kwargs["router_config"].prefill_load_scale == 2.0
    assert kwargs["prefill_engine_args"] == {"block_size": 64}
    assert kwargs["prefill_engine_args"] is not spec.prefill_engine_args
    # The startup snapshot itself is never modified.
    assert spec.router_config == {
        "overlap_score_credit": 1.0,
        "prefill_load_scale": 2.0,
    }
    assert RetuneEngineSpec().replay_kwargs()["router_config"] is None
    assert RetuneEngineSpec().replay_kwargs({"prefill_load_scale": 4.0})[
        "router_config"
    ].prefill_load_scale == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# R10: strict round trip through runtime reconfiguration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "retune", "updates"),
    [
        (RETUNE_ONLY, _retune(), {"max_gpu_budget": 24}),
        (
            {},
            _retune(
                planner={**SENSITIVITY_OFF, LSD: {"choices": [70, 90]}},
                timeout_seconds=300,
                sla_ttft_ms=200,
                dry_run=True,
            ),
            {"load_scaling_down_sensitivity": 70},
        ),
        (
            SLA_THROUGHPUT,
            _retune(
                planner={
                    "load_predictor": {"preset": False, "type": "kalman"},
                    "kalman_r": {"choices": [1, 2.5]},
                },
                history_seconds=9000,
            ),
            {"load_predictor": "kalman", "kalman_r": 2.5},
        ),
        (
            {"enable_load_scaling": True, **SLA_THROUGHPUT},
            _retune(router="default", planner="default", history_seconds=9000),
            {"load_scaling_down_sensitivity": 90},
        ),
    ],
)
def test_retune_survives_strict_runtime_reconfiguration(overrides, retune, updates):
    config = _config(retune=retune, **overrides)
    dumped = config.model_dump()

    restored = PlannerConfig.model_validate(dumped, strict=True)
    assert restored.model_dump() == dumped
    assert restored.retune == config.retune
    candidate = candidate_config(config, updates)
    assert candidate.retune == config.retune
    assert (candidate.enable_load_scaling, candidate.enable_throughput_scaling) == (
        config.enable_load_scaling,
        config.enable_throughput_scaling,
    )
    for field, value in updates.items():
        assert getattr(candidate, field) == value


# ---------------------------------------------------------------------------
# Core retune types
# ---------------------------------------------------------------------------


def test_retune_params_key_is_canonical():
    first = RetuneParams(
        router={"router_temperature": 0, "overlap_score_credit": 0.5},
        planner={"load_predictor": "kalman", "decode_min_endpoint": 2},
    )
    second = RetuneParams(
        router={"overlap_score_credit": 0.5, "router_temperature": 0.0},
        planner={"decode_min_endpoint": 2, "load_predictor": "kalman"},
    )

    assert first.key() == second.key()
    assert first.key() == (
        '{"planner":{"decode_min_endpoint":2,"load_predictor":"kalman"},'
        '"router":{"overlap_score_credit":0.5,"router_temperature":0.0}}'
    )
    assert first == second and hash(first) == hash(second)
    assert len({first, second}) == 1
    assert RetuneParams(planner={"x": True}) != RetuneParams(planner={"x": 1})
    assert RetuneParams().key() == '{"planner":{},"router":{}}'
    with pytest.raises(ValueError, match="must be numeric"):
        RetuneParams(router={"overlap_score_credit": True})
    with pytest.raises(ValueError, match="must be a bool, number or string"):
        RetuneParams(planner={"load_predictor": None})
