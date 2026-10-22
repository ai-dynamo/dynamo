# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the planner retune config section and its core types."""

import json
import math

import pytest
from pydantic import ValidationError

from dynamo.planner.config.planner_config import (
    PlannerConfig,
    RetuneConfig,
    RetuneEngineSpec,
    RetuneLiveConfig,
)
from dynamo.planner.core.types import (
    PlannerEffects,
    RetuneDecision,
    RetuneFeedback,
    RetuneInputs,
    RetuneParams,
    TickInput,
    TrafficObservation,
)
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


def test_retune_config_defaults():
    retune = RetuneConfig(interval_seconds=600)

    assert retune.interval_seconds == 600.0
    assert retune.grace_seconds == 60.0
    assert retune.history_seconds is None
    assert retune.decision_latency_seconds == 0.0
    assert retune.timeout_seconds is None
    assert retune.effective_timeout_seconds == 600.0
    assert retune.sla_ttft_ms is None and retune.sla_itl_ms is None
    assert retune.min_improvement_percentage == 3.0
    assert retune.max_candidates == 512
    assert retune.max_parallel == 2
    assert retune.dry_run is False
    assert retune.router == {} and retune.planner == {}
    assert retune.live == RetuneLiveConfig()
    assert retune.live.reconcile_interval_seconds == 30.0
    assert retune.live.engine is None


def test_scoring_sla_defaults_to_planner_sla_and_accepts_overrides():
    assert RetuneConfig(interval_seconds=600).scoring_sla(500.0, 30.0) == (
        500.0,
        30.0,
    )
    retune = RetuneConfig(interval_seconds=600, sla_ttft_ms=200, sla_itl_ms=20)
    assert retune.scoring_sla(500.0, 30.0) == (200.0, 20.0)


def test_interval_seconds_is_required():
    with pytest.raises(ValidationError, match="interval_seconds"):
        RetuneConfig()


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (RetuneConfig, {"interval_seconds": 600, "unknown": 1}),
        (RetuneLiveConfig, {"unknown": 1}),
        (RetuneEngineSpec, {"unknown": 1}),
    ],
)
def test_retune_models_forbid_extra_fields(model, values):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model(**values)


@pytest.mark.parametrize(
    "field",
    [
        "traffic_model",
        "worker_memory_mb",
        "repeats",
        "min_window_requests",
        "max_window_requests",
        "min_improvement",
    ],
)
def test_retired_design_fields_are_rejected(field):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RetuneConfig(interval_seconds=600, **{field: 1})


def test_retune_section_is_rejected_inside_planner_config_when_malformed():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _config(retune=_retune(traffic_model="open_loop"), **RETUNE_ONLY)


# ---------------------------------------------------------------------------
# R1: retune-only mode and easy-mode forcing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        # Both flags explicitly false: retune-only, no forcing.
        (RETUNE_ONLY, (False, False)),
        ({**RETUNE_ONLY, "optimization_target": "sla"}, (False, False)),
        ({**RETUNE_ONLY, "optimization_target": "latency"}, (False, False)),
        # Every other easy-target combination keeps today's forcing, including
        # the examples/deployments/dgdr/planner.yaml pattern.
        ({}, (True, False)),
        ({"enable_load_scaling": False}, (True, False)),
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


def test_retune_only_mode_rejects_implicit_flag_on_sla_target():
    # enable_load_scaling defaults to false but was not set explicitly.
    with pytest.raises(ValidationError, match="both flags to be set explicitly"):
        _config(
            retune=_retune(),
            optimization_target="sla",
            enable_throughput_scaling=False,
        )


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


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        RETUNE_ONLY,
        {"enable_load_scaling": False, "enable_throughput_scaling": True},
        {"optimization_target": "sla"},
        {"optimization_target": "sla", "enable_load_scaling": True},
        {"optimization_target": "load", "mode": "agg"},
    ],
)
def test_legacy_configs_without_retune_are_unchanged(overrides):
    overrides = dict(overrides)
    if overrides.get("optimization_target") == "load":
        overrides.update(decode_scale_up_kv_rate=80, decode_scale_down_kv_rate=40)
    config = _config(**overrides)

    assert config.retune is None
    dumped = config.model_dump()
    assert "retune" not in dumped
    assert "retune" not in json.loads(config.model_dump_json())
    assert PlannerConfig.model_validate(dumped).model_dump() == dumped
    assert _config(retune=None, **overrides).model_dump() == dumped


# ---------------------------------------------------------------------------
# R2-R6: mode, cadence, timing, ranges and router domain
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


def test_retune_timing_accepts_boundaries():
    retune = RetuneConfig(
        interval_seconds=600, timeout_seconds=600, decision_latency_seconds=600
    )
    assert retune.effective_timeout_seconds == 600


@pytest.mark.parametrize(
    "overrides",
    [
        {"interval_seconds": 0},
        {"interval_seconds": float("inf")},
        {"grace_seconds": -1},
        {"history_seconds": -1},
        {"decision_latency_seconds": -1},
        {"timeout_seconds": 0},
        {"min_improvement_percentage": -0.1},
        {"min_improvement_percentage": 100},
        {"max_parallel": 0},
        {"max_candidates": 0},
        {"sla_ttft_ms": 0},
        {"sla_itl_ms": -1},
        {"sla_itl_ms": float("nan")},
    ],
)
def test_retune_ranges_are_enforced(overrides):
    with pytest.raises(ValidationError):
        RetuneConfig(**{"interval_seconds": 600, **overrides})


@pytest.mark.parametrize(
    ("router", "match"),
    [
        ({"overlap_score_credit": []}, "at least one value"),
        ({"overlap_score_credit": [float("nan")]}, "finite and non-negative"),
        ({"overlap_score_credit": [float("inf")]}, "finite and non-negative"),
        ({"overlap_score_credit": [-0.5]}, "finite and non-negative"),
        ({"overlap_score_credit": [1.0, 1]}, "distinct"),
        ({"overlap_score_credit": [True]}, "valid number"),
        ({"overlap_score_credit": ["0.5"]}, "valid number"),
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


# ---------------------------------------------------------------------------
# R7 / R7b: planner domain keys, relevance, value types and history
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["ttft_ms", "itl_ms"])
def test_sla_fields_are_never_searchable(key):
    with pytest.raises(ValidationError, match="define the fixed SLA"):
        _config(retune=_retune(planner={key: [100.0, 200.0]}))


@pytest.mark.parametrize(
    "key", ["throughput_adjustment_interval_seconds", "enable_load_scaling", "nope"]
)
def test_planner_domain_rejects_non_runtime_fields(key):
    with pytest.raises(ValidationError, match="not a runtime-changeable"):
        _config(retune=_retune(planner={key: [1]}))


def test_planner_domain_requires_scaling():
    with pytest.raises(ValidationError, match="retune.planner requires load or"):
        _config(retune=_retune(planner={"max_gpu_budget": [8, 16]}), **RETUNE_ONLY)


@pytest.mark.parametrize(
    ("mode", "overrides", "planner", "requirement"),
    [
        ("disagg", {}, {"load_predictor": ["arima"]}, "optimization_target='sla'"),
        (
            "disagg",
            {
                "optimization_target": "sla",
                "enable_load_scaling": True,
                "enable_throughput_scaling": False,
            },
            {"kalman_r": [1.0]},
            "optimization_target='sla'",
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
        ("agg", {}, {"decode_min_endpoint": [1, 2]}, "active in mode='agg'"),
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
    ("mode", "overrides", "planner"),
    [
        ("disagg", {}, {"min_gpu_budget": [-1, 4], "max_gpu_budget": [8, 16]}),
        ("disagg", {}, {"load_scaling_down_sensitivity": [70, 90]}),
        ("disagg", {}, {"prefill_min_endpoint": [1, 2], "decode_min_endpoint": [1]}),
        ("agg", {}, {"min_endpoint": [1, 2]}),
        (
            "disagg",
            SLA_THROUGHPUT,
            {
                "max_throughput_scaling_replicas": [2, 4],
                "load_predictor": ["constant", "kalman"],
                "load_predictor_log1p": [False, True],
                "kalman_r": [1, 2.5],
            },
        ),
        (
            "disagg",
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
def test_relevant_planner_domains_are_accepted(mode, overrides, planner):
    config = _config(
        mode=mode, retune=_retune(planner=planner, history_seconds=9000), **overrides
    )
    assert config.retune is not None
    assert set(config.retune.planner) == set(planner)


@pytest.mark.parametrize(
    ("planner", "match"),
    [
        ({"load_scaling_down_sensitivity": [70.0]}, "valid integer"),
        ({"load_scaling_down_sensitivity": [101]}, "less than or equal to 100"),
        ({"prefill_min_endpoint": [0]}, "greater than or equal to 1"),
        ({"max_gpu_budget": ["8"]}, "valid integer"),
        ({"load_scaling_down_sensitivity": [70, 70]}, "distinct"),
        ({"load_scaling_down_sensitivity": []}, "at least one value"),
    ],
)
def test_planner_domain_values_follow_strict_field_types(planner, match):
    with pytest.raises(ValidationError, match=match):
        _config(retune=_retune(planner=planner))


@pytest.mark.parametrize(
    ("planner", "match"),
    [
        ({"load_predictor": ["linear"]}, "'constant', 'arima', 'prophet' or 'kalman'"),
        ({"load_predictor_log1p": [True, 1]}, "valid boolean"),
        ({"kalman_r": [float("nan")]}, "finite"),
    ],
)
def test_predictor_domain_values_follow_strict_field_types(planner, match):
    with pytest.raises(ValidationError, match=match):
        _config(retune=_retune(planner=planner, history_seconds=9000), **SLA_THROUGHPUT)


def test_planner_domain_values_are_stored_as_field_types():
    config = _config(
        retune=_retune(planner={"kalman_r": [1, 2.5]}, history_seconds=9000),
        **SLA_THROUGHPUT,
    )
    assert config.retune is not None
    values = config.retune.planner["kalman_r"]
    assert values == [1.0, 2.5]
    assert all(isinstance(value, float) for value in values)


def test_history_must_warm_the_current_throughput_predictor():
    # Defaults: 180 s throughput interval x prophet_window_size 50.
    with pytest.raises(ValidationError, match=r"history_seconds must be >= 9000s"):
        _config(retune=_retune(history_seconds=8999), **SLA_THROUGHPUT)
    assert _config(retune=_retune(history_seconds=9000), **SLA_THROUGHPUT)


def test_history_must_warm_every_candidate_predictor():
    planner = {"prophet_window_size": [20, 100], "load_predictor": ["prophet"]}
    with pytest.raises(ValidationError, match=r">= 18000s"):
        _config(retune=_retune(planner=planner, history_seconds=9000), **SLA_THROUGHPUT)
    assert _config(
        retune=_retune(planner=planner, history_seconds=18000), **SLA_THROUGHPUT
    )


def test_history_bound_has_no_warmup_interval_cap():
    # 5,000 throughput intervals is above the HTTP warmup cap of 4,096.
    config = _config(retune=_retune(history_seconds=180 * 5000), **SLA_THROUGHPUT)
    assert config.retune is not None


def test_history_bound_does_not_apply_without_throughput_scaling():
    config = _config(retune=_retune(history_seconds=0), **RETUNE_ONLY)
    assert config.retune is not None and config.retune.history_seconds == 0


# ---------------------------------------------------------------------------
# R6b / R8: default domains and grid size
# ---------------------------------------------------------------------------


def test_retune_requires_a_non_empty_domain():
    with pytest.raises(ValidationError, match="non-empty router or planner"):
        _config(retune={"interval_seconds": 600}, **RETUNE_ONLY)


def test_grid_size_is_capped_by_max_candidates():
    router = {"overlap_score_credit": [0.0, 0.5, 1.0], "prefill_load_scale": [1.0, 2.0]}
    planner = {"load_scaling_down_sensitivity": [70, 80, 90]}
    retune = _retune(router=router, planner=planner, max_candidates=17)
    with pytest.raises(ValidationError, match="18 candidates, exceeding"):
        _config(retune=retune)
    retune["max_candidates"] = 18
    assert _config(retune=retune)


def test_default_planner_domain_that_expands_to_nothing_is_empty():
    pytest.importorskip("aisimulate.sweeper")
    with pytest.raises(ValidationError, match="non-empty router or planner"):
        _config(retune={"interval_seconds": 600, "planner": "default"}, **RETUNE_ONLY)


def test_default_router_domain_counts_toward_max_candidates():
    pytest.importorskip("aisimulate.sweeper")
    retune = {"interval_seconds": 600, "router": "default", "max_candidates": 107}
    with pytest.raises(ValidationError, match="108 candidates"):
        _config(retune=retune, **RETUNE_ONLY)
    retune["max_candidates"] = 108
    assert _config(retune=retune, **RETUNE_ONLY)


def test_use_case_configs_validate():
    pytest.importorskip("aisimulate.sweeper")
    scenario_1 = _config(
        retune={
            "interval_seconds": 21600,
            "router": "default",
            "planner": {},
            "max_parallel": 8,
        },
        **RETUNE_ONLY,
    )
    assert (scenario_1.enable_load_scaling, scenario_1.enable_throughput_scaling) == (
        False,
        False,
    )
    scenario_2 = _config(
        retune={
            "interval_seconds": 21600,
            "history_seconds": 345600,
            "router": "default",
            "planner": "default",
            "max_parallel": 8,
        },
        enable_load_scaling=True,
        **SLA_THROUGHPUT,
    )
    assert scenario_2.retune is not None
    assert scenario_2.retune.planner == "default"


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
    # Default engine args are an aggregated replay engine.
    assert _config(mode="agg", retune=_retune(live={"engine": {}}))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"prefill_engine_args": {}}, "must be set together"),
        ({"decode_engine_args": {}}, "must be set together"),
        (
            {
                "extra_engine_args": {},
                "prefill_engine_args": {},
                "decode_engine_args": {},
            },
            "cannot be combined",
        ),
        ({"router_mode": "random"}, "round_robin"),
        ({"kv_event_lag_ms": -1}, "greater than or equal to 0"),
        ({"benchmark_granularity": 0}, "greater than 0"),
        ({"model_name": ""}, "at least 1 character"),
    ],
)
def test_engine_spec_shape_and_ranges(values, match):
    with pytest.raises(ValidationError, match=match):
        RetuneEngineSpec(**values)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"frontend_urls": ["10.0.0.5:8000"]}, "absolute http"),
        ({"frontend_urls": ["http://user:pw@10.0.0.5:8000"]}, "credentials"),
        ({"trace_paths": [""]}, "non-empty"),
        ({"reconcile_interval_seconds": 0}, "greater than 0"),
    ],
)
def test_live_config_validation(values, match):
    with pytest.raises(ValidationError, match=match):
        RetuneLiveConfig(**values)


def test_engine_spec_replay_kwargs_merge_router_overrides():
    pytest.importorskip("dynamo.llm")
    spec = RetuneEngineSpec(
        prefill_engine_args={"block_size": 64},
        decode_engine_args={"block_size": 64},
        router_config={"overlap_score_credit": 1.0, "prefill_load_scale": 2.0},
        model_name="model",
        kv_event_lag_ms=5.0,
    )
    kwargs = spec.replay_kwargs({"overlap_score_credit": 0.5})

    assert set(kwargs) == {
        "extra_engine_args",
        "prefill_engine_args",
        "decode_engine_args",
        "router_mode",
        "router_config",
        "ais_perf_config",
        "performance_model_metadata",
        "model_name",
        "kv_event_lag_ms",
        "benchmark_granularity",
    }
    assert kwargs["router_mode"] == "kv_router"
    assert kwargs["router_config"].overlap_score_credit == 0.5
    assert kwargs["router_config"].prefill_load_scale == 2.0
    assert kwargs["ais_perf_config"] is None
    assert kwargs["extra_engine_args"] is None
    assert kwargs["prefill_engine_args"] == {"block_size": 64}
    kwargs["prefill_engine_args"]["block_size"] = 16
    assert spec.prefill_engine_args == {"block_size": 64}
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


def test_default_domains_survive_strict_runtime_reconfiguration():
    pytest.importorskip("aisimulate.sweeper")
    config = _config(
        retune={
            "interval_seconds": 600,
            "history_seconds": 9000,
            "router": "default",
            "planner": "default",
        },
        enable_load_scaling=True,
        **SLA_THROUGHPUT,
    )
    candidate = candidate_config(config, {"load_scaling_down_sensitivity": 90})
    assert candidate.retune == config.retune


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


@pytest.mark.parametrize(
    ("router", "planner"),
    [
        ({"overlap_score_credit": -1.0}, {}),
        ({"overlap_score_credit": math.nan}, {}),
        ({"overlap_score_credit": True}, {}),
        ({}, {"kalman_r": math.inf}),
        ({}, {"load_predictor": None}),
    ],
)
def test_retune_params_reject_invalid_values(router, planner):
    with pytest.raises(ValueError):
        RetuneParams(router=router, planner=planner)


def test_retune_decision_to_dict_replaces_warmup_by_length():
    previous = RetuneParams(router={"overlap_score_credit": 1.0})
    chosen = RetuneParams(router={"overlap_score_credit": 0.5})
    warmup = tuple(
        TrafficObservation(duration_s=180, num_req=10, isl=100, osl=20)
        for _ in range(3)
    )
    decision = RetuneDecision(
        decision_id="retune-0-w1",
        outcome="applied",
        window_index=1,
        window_start_s=600.0,
        window_end_s=1200.0,
        as_of_s=1260.0,
        launched_at_s=1260.0,
        decided_at_s=1265.0,
        num_requests=42,
        num_candidates=4,
        previous=previous,
        chosen=chosen,
        baseline_score=1.0,
        best_score=1.2,
        planner_changes={"load_predictor": "kalman"},
        predictor_presweep={"winner": "kalman_default_raw", "losses": {"a": 1.0}},
        predictor_warmup=warmup,
        sweep_wall_s=3.5,
        top=[(chosen.key(), 1.2), (previous.key(), 1.0)],
    )

    record = decision.to_dict()
    assert record["predictor_warmup"] == 3
    assert record["previous"] == {
        "router": {"overlap_score_credit": 1.0},
        "planner": {},
    }
    assert record["chosen"]["router"] == {"overlap_score_credit": 0.5}
    assert record["top"] == [[chosen.key(), 1.2], [previous.key(), 1.0]]
    assert record["reason"] is None
    assert json.loads(json.dumps(record)) == record
    kept = RetuneDecision(
        decision_id="retune-0-w2",
        outcome="kept",
        window_index=2,
        window_start_s=1200.0,
        window_end_s=1800.0,
        as_of_s=1860.0,
        launched_at_s=1860.0,
        decided_at_s=1865.0,
        num_requests=0,
        num_candidates=4,
        previous=chosen,
        chosen=chosen,
        baseline_score=None,
        best_score=None,
        reason="empty window",
    )
    assert kept.to_dict()["predictor_warmup"] is None
    assert kept.planner_changes == {} and kept.top == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"outcome": "small_window"},
        {"decision_id": ""},
        {"window_index": -1},
        {"num_requests": True},
    ],
)
def test_retune_decision_rejects_invalid_records(overrides):
    params = RetuneParams()
    values = {
        "decision_id": "retune-0-w0",
        "outcome": "kept",
        "window_index": 0,
        "window_start_s": 0.0,
        "window_end_s": 600.0,
        "as_of_s": 660.0,
        "launched_at_s": None,
        "decided_at_s": 665.0,
        "num_requests": 0,
        "num_candidates": 1,
        "previous": params,
        "chosen": params,
        "baseline_score": None,
        "best_score": None,
        **overrides,
    }
    with pytest.raises(ValueError):
        RetuneDecision(**values)


def test_retune_feedback_and_inputs_validation():
    assert RetuneFeedback("retune-0-w1", "router", "422 invalid").target == "router"
    with pytest.raises(ValueError):
        RetuneFeedback("retune-0-w1", "frontend", "422")
    with pytest.raises(ValueError):
        RetuneFeedback("", "planner", "conflict")

    engine = RetuneEngineSpec()
    inputs = RetuneInputs(engine=engine, trace_paths=["/traces/a.jsonl"])
    assert inputs.trace_paths == ("/traces/a.jsonl",)
    assert inputs.history_paths == ()
    assert (inputs.trace_origin_ms, inputs.arrival_speedup_ratio) == (0.0, 1.0)
    assert inputs.prune_by_mtime is False
    with pytest.raises(ValueError, match="at least one trace"):
        RetuneInputs(engine=engine, trace_paths=())
    with pytest.raises(ValueError):
        RetuneInputs(engine=engine, trace_paths=("a",), arrival_speedup_ratio=0)
    with pytest.raises(ValueError):
        RetuneInputs(engine=engine, trace_paths=("a",), trace_origin_ms=math.nan)


def test_tick_input_and_effects_default_to_no_retune_data():
    assert TickInput(now_s=0.0).retune_feedback == []
    effects = PlannerEffects()
    assert effects.router_parameters is None
    assert effects.retune_decisions == []
