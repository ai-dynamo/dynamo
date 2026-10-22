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
    RetuneLiveConfig,
)
from dynamo.planner.core.types import RetuneParams
from dynamo.planner.reconfiguration import candidate_config

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

ROUTER = {"overlap_score_credit": [0.0, 1.0], "prefill_load_scale": [0.5, 1.0]}
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


def test_history_seconds_none_resolves_to_required_minimum():
    sla = _config(retune=_retune(), **SLA_THROUGHPUT)
    # Default throughput interval 180 s x default prophet_window_size 50.
    assert sla.retune.effective_history_seconds == 9000.0
    # The configured value stays "auto" and is not dumped as set.
    assert sla.retune.history_seconds is None
    assert "history_seconds" not in sla.model_dump(exclude_unset=True)["retune"]

    retune_only = _config(retune=_retune(), **RETUNE_ONLY)
    assert retune_only.retune.effective_history_seconds == 0.0

    explicit = _config(retune=_retune(history_seconds=345600), **SLA_THROUGHPUT)
    assert explicit.retune.history_seconds == 345600.0
    assert explicit.retune.effective_history_seconds == 345600.0


def test_auto_history_is_resolved_again_for_a_runtime_candidate():
    config = _config(retune=_retune(), **SLA_THROUGHPUT)

    candidate = candidate_config(config, {"prophet_window_size": 60})

    assert candidate.retune.history_seconds is None
    assert candidate.retune.effective_history_seconds == 180.0 * 60
    assert config.retune.effective_history_seconds == 9000.0


def test_a_shared_retune_instance_is_never_modified():
    shared = RetuneConfig(interval_seconds=600, router=ROUTER)
    fields_set = set(shared.model_fields_set)

    retune_only = _config(retune=shared, **RETUNE_ONLY)
    sla = _config(retune=shared, **SLA_THROUGHPUT)

    assert shared.history_seconds is None
    assert shared.model_fields_set == fields_set
    assert retune_only.retune is not shared and sla.retune is not shared
    assert retune_only.retune.effective_history_seconds == 0.0
    assert sla.retune.effective_history_seconds == 9000.0

    planner = RetuneConfig(
        interval_seconds=600,
        history_seconds=9000,
        planner={"kalman_r": [1, 2.5]},
    )
    config = _config(retune=planner, **SLA_THROUGHPUT)
    assert [type(value) for value in config.retune.planner["kalman_r"]] == [
        float,
        float,
    ]
    assert [type(value) for value in planner.planner["kalman_r"]] == [int, float]


@pytest.mark.parametrize(
    ("model", "values"),
    [
        # A retired design field is just an unknown field.
        (RetuneConfig, {"interval_seconds": 600, "traffic_model": "open_loop"}),
        (RetuneLiveConfig, {"unknown": 1}),
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
# R2-R6: mode, cadence, timing and router domain
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


@pytest.mark.parametrize(
    ("router", "match"),
    [
        ({"overlap_score_credit": []}, "at least one value"),
        ({"overlap_score_credit": [float("nan")]}, "finite and non-negative"),
        ({"overlap_score_credit": [1.0, 1]}, "distinct"),
        ({"overlap_score_credit": [True]}, "valid number"),
        ({"": [1.0]}, "non-empty"),
        ("all", "default"),
    ],
)
def test_router_domain_values_are_validated(router, match):
    with pytest.raises(ValidationError, match=match):
        RetuneConfig(interval_seconds=600, router=router)


def test_router_domain_accepts_ints_as_floats():
    retune = RetuneConfig(interval_seconds=600, router={"router_temperature": [0, 1]})
    assert retune.router == {"router_temperature": [0.0, 1.0]}
    assert {type(value) for value in retune.router["router_temperature"]} == {float}


# ---------------------------------------------------------------------------
# R7 / R7b: planner domain keys, relevance, value types and history
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("planner", "overrides", "match"),
    [
        ({"ttft_ms": [100.0, 200.0]}, {}, "define the fixed SLA"),
        ({"nope": [1]}, {}, "not a runtime-changeable"),
        ({"max_gpu_budget": [8, 16]}, RETUNE_ONLY, "retune.planner requires load or"),
    ],
    ids=["fixed-sla", "not-runtime", "requires-scaling"],
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
            {"kalman_r": [1.0]},
            "optimization_target='sla' with throughput scaling",
        ),
        ("disagg", {}, {"max_throughput_scaling_replicas": [2]}, "enable_throughput"),
        (
            "disagg",
            {**SLA_THROUGHPUT, "enable_load_scaling": False},
            {"load_scaling_down_sensitivity": [70]},
            "enable_load_scaling",
        ),
        (
            "disagg",
            {},
            {"decode_scale_up_kv_rate": [80.0]},
            "optimization_target='load'",
        ),
        (
            "agg",
            {
                "optimization_target": "load",
                "decode_scale_up_kv_rate": 80,
                "decode_scale_down_kv_rate": 40,
            },
            {"prefill_scale_up_queue_tokens": [100]},
            "with a prefill role",
        ),
        ("disagg", {}, {"min_endpoint": [1, 2]}, "active in mode='disagg'"),
    ],
)
def test_planner_domain_keys_must_affect_a_registered_builtin(
    mode, overrides, planner, requirement
):
    retune = _retune(planner=planner, history_seconds=9000)
    with pytest.raises(ValidationError, match="has no effect here") as error:
        _config(mode=mode, retune=retune, **overrides)
    assert requirement in str(error.value)


@pytest.mark.parametrize(
    ("overrides", "planner"),
    [
        ({}, {"min_gpu_budget": [-1, 4], "max_gpu_budget": [8, 16]}),
        (
            {
                "optimization_target": "load",
                "prefill_scale_up_queue_tokens": 1000,
                "prefill_scale_down_queue_tokens": 100,
                "decode_scale_up_kv_rate": 80,
                "decode_scale_down_kv_rate": 40,
            },
            {
                "prefill_scale_up_queue_tokens": [800, 1200],
                "decode_scale_down_kv_rate": [30.0, 50.0],
            },
        ),
    ],
)
def test_relevant_planner_domains_are_accepted(overrides, planner):
    config = _config(retune=_retune(planner=planner, history_seconds=9000), **overrides)
    assert config.retune is not None
    assert set(config.retune.planner) == set(planner)


@pytest.mark.parametrize(
    ("planner", "match"),
    [
        ({"load_scaling_down_sensitivity": [70.0]}, "valid integer"),
        ({"load_scaling_down_sensitivity": [101]}, "less than or equal to 100"),
        ({"load_predictor": ["linear"]}, "'constant', 'arima', 'prophet' or 'kalman'"),
        # True and 1 are distinct values; 1 then fails the strict bool type.
        ({"load_predictor_log1p": [True, 1]}, "valid boolean"),
    ],
)
def test_planner_domain_values_follow_strict_field_types(planner, match):
    with pytest.raises(ValidationError, match=match):
        _config(
            retune=_retune(planner=planner), enable_load_scaling=True, **SLA_THROUGHPUT
        )


def test_history_must_warm_every_candidate_predictor():
    planner = {"prophet_window_size": [20, 100], "load_predictor": ["prophet"]}
    with pytest.raises(ValidationError, match=r">= 18000s"):
        _config(retune=_retune(planner=planner, history_seconds=9000), **SLA_THROUGHPUT)
    assert _config(
        retune=_retune(planner=planner, history_seconds=18000), **SLA_THROUGHPUT
    )


# ---------------------------------------------------------------------------
# R6b / R8: default domains and grid size
# ---------------------------------------------------------------------------


def test_grid_size_is_capped_by_max_candidates():
    router = {"overlap_score_credit": [0.0, 0.5, 1.0], "prefill_load_scale": [1.0, 2.0]}
    planner = {"load_scaling_down_sensitivity": [70, 80, 90]}
    retune = _retune(router=router, planner=planner, max_candidates=17)
    with pytest.raises(ValidationError, match="18 candidates, exceeding"):
        _config(retune=retune)
    retune["max_candidates"] = 18
    assert _config(retune=retune)


def test_default_planner_domain_that_expands_to_nothing_is_empty():
    with pytest.raises(ValidationError, match="non-empty router or planner"):
        _config(retune={"interval_seconds": 600, "planner": "default"}, **RETUNE_ONLY)


# ---------------------------------------------------------------------------
# Live section and engine spec
# ---------------------------------------------------------------------------


def test_live_engine_shape_must_match_mode():
    agg_engine = {"extra_engine_args": {"block_size": 64}}
    disagg_engine = {
        "prefill_engine_args": {"block_size": 64},
        "decode_engine_args": {"block_size": 64},
    }
    with pytest.raises(ValidationError, match="retune.live.engine must set"):
        _config(retune=_retune(live={"engine": agg_engine}), **RETUNE_ONLY)
    with pytest.raises(ValidationError, match="retune.live.engine must set"):
        _config(mode="agg", retune=_retune(live={"engine": disagg_engine}))
    assert _config(retune=_retune(live={"engine": disagg_engine}), **RETUNE_ONLY)
    assert _config(mode="agg", retune=_retune(live={"engine": agg_engine}))


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


def test_live_config_validation():
    with pytest.raises(ValidationError, match="credentials"):
        RetuneLiveConfig(frontend_urls=["http://user:pw@10.0.0.5:8000"])


def test_engine_spec_replay_kwargs_merge_router_overrides():
    pytest.importorskip("dynamo.llm")
    spec = RetuneEngineSpec(
        prefill_engine_args={"block_size": 64},
        decode_engine_args={"block_size": 64},
        router_config={"overlap_score_credit": 1.0, "prefill_load_scale": 2.0},
    )
    kwargs = spec.replay_kwargs({"overlap_score_credit": 0.5})

    assert kwargs["router_mode"] == "kv_router"
    assert kwargs["router_config"].overlap_score_credit == 0.5
    assert kwargs["router_config"].prefill_load_scale == 2.0
    assert kwargs["prefill_engine_args"] == {"block_size": 64}
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
                planner={"load_scaling_down_sensitivity": [70, 90]},
                live={
                    "engine": {
                        "prefill_engine_args": {"block_size": 64},
                        "decode_engine_args": {"block_size": 64},
                        "router_config": {"overlap_score_credit": 1.0},
                    },
                    "trace_paths": ["/traces/frontend"],
                    "frontend_urls": ["http://10.0.0.5:8000"],
                },
                timeout_seconds=300,
                sla_ttft_ms=200,
                dry_run=True,
            ),
            {"load_scaling_down_sensitivity": 70},
        ),
        (
            SLA_THROUGHPUT,
            _retune(
                planner={"kalman_r": [1, 2.5], "load_predictor": ["arima", "kalman"]},
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
