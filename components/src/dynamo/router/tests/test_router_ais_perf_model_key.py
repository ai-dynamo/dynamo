# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""The ``ais_perf_model`` key of the ``router`` section of an AISimulate config."""

from __future__ import annotations

import json
import re
import warnings
from copy import deepcopy

import pytest
from pydantic import ValidationError

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate.config_adapter import (
    PredictionAdapterContext,
    RecommendationAdapterContext,
)
from aisimulate.sweeper.provider import (
    CandidateContext,
    InfeasibleCandidate,
    SweepContext,
)
from aisimulate.sweeper.replay import BackendDeploymentSpec

from dynamo.router.simulation import create_provider
from dynamo.router.simulation.config import RouterPredictionConfig
from tests.utils.ais_perf_model import (
    AGGREGATED_ENTRY,
    IDENTITY,
    PREFILL_ENTRY,
    exactly,
    normalized,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]

_KNOBS = {"overlap_score_credit": 0.1, "prefill_load_scale": 3.0, "temperature": 0.1}
_KNOB_CHOICES = {name: {"choices": [value]} for name, value in _KNOBS.items()}
# The keys a custom engine timing config has only for KV capacity.
_CAPACITY = {"gpu_memory_utilization": 0.9, "mem_fraction_static": 0.8}
_AIS = {"policy": "kv_router", "prefill_load_model": {"type": "ais"}}
# A search that can also pick round robin or no load model.
_AIS_AMONG_CHOICES = {
    "policy": {"choices": ["round_robin", "kv_router"]},
    "prefill_load_model": {"type": {"choices": ["none", "ais"]}},
}
_USE_ENGINE = {"use": "engine"}
_OLD_FORM_WARNING = (
    "router.prefill_load_model.type: ais without router.ais_perf_model is "
    "deprecated and will be removed in v1.9; use router.ais_perf_model: "
    "{use: engine} (or a perf model config when the engine does not use an AIS "
    "perf model)."
)
# A pattern for the error for the key not_a_field: after the prefix comes
# what the perf model config class says, which is not Dynamo's.
_UNKNOWN_KEY_ERROR = "ais_perf_model: .*not_a_field.*"
_NO_ENGINE_PERF_MODEL = (
    "router.ais_perf_model: {{use: engine}} needs an AIS perf model on the "
    "{worker_type} engine; it has none"
)
# Router sections that write no load model type: the search takes the default
# load models, none and ais.
_DEFAULT_LOAD_MODELS = [
    pytest.param({}, id="empty"),
    pytest.param({"policy": "kv_router"}, id="kv_router"),
    pytest.param(
        {"policy": "kv_router", "prefill_load_model": {}}, id="empty load model"
    ),
]
_FIXED_TIMING = {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}


def _prediction_context(mode: str = "aggregated") -> PredictionAdapterContext:
    worker = "prefill" if mode == "disaggregated" else "aggregated"
    return PredictionAdapterContext(
        engine={
            "mode": mode,
            "model": "example/model",
            "hardware": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.11.0",
            "workers": {
                worker: {
                    "parallelism": {
                        "tensor": 1,
                        "attention_data": 1,
                        "moe_tensor": 1,
                        "moe_expert": 1,
                    }
                }
            },
        },
        traffic={},
        evaluation={},
    )


def _recommendation_context(*modes: str) -> RecommendationAdapterContext:
    return RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={},
        optimization={},
        sweep=_sweep_context(*modes),
    )


def _sweep_context(*modes: str) -> SweepContext:
    return SweepContext(
        core_search_space={"deployment_mode": list(modes or ("agg", "disagg"))},
        workload={"trace_path": "trace.jsonl"},
        goal={"target": "throughput"},
        show_progress=False,
    )


def _candidate(mode: str = "agg", **sample) -> CandidateContext:
    return CandidateContext(
        sample={
            "deployment_mode": mode,
            "backend": "vllm",
            "backend_version": "0.11.0",
            "hardware_sku": "h200_sxm",
            "model_name": "example/model",
            "tp": 1,
            "attention_dp": 1,
            **sample,
        },
        backend_deployment=BackendDeploymentSpec(
            deployment_mode=mode,
            backend="vllm",
            backend_version="0.11.0",
        ),
    )


def _selection(load_model: str = "ais") -> dict:
    return {"mode": "kv_router", "prefill_load_model_type": load_model, **_KNOBS}


def _hook_config(result) -> dict:
    (hook,) = result.runtime_hooks
    return hook.config


def _compile_old_form_prediction(provider):
    return provider.compile_prediction(_AIS, _prediction_context())


def _compile_old_form_recommendation(provider):
    return provider.compile_recommendation(
        {**_AIS, **_KNOB_CHOICES}, _recommendation_context()
    )


def test_predict_written_config_reaches_hook():
    result = create_provider().compile_prediction(
        {**_AIS, "ais_perf_model": IDENTITY}, _prediction_context()
    )

    hook_config = _hook_config(result)
    assert hook_config["ais_perf_config"] == normalized(AGGREGATED_ENTRY)
    assert "ais_perf_config_from_engine" not in hook_config
    assert result.config["ais_perf_model"] == IDENTITY


@pytest.mark.parametrize(
    ("mode", "entry"),
    [("aggregated", AGGREGATED_ENTRY), ("disaggregated", PREFILL_ENTRY)],
)
def test_predict_written_list_selects_by_engine_mode(mode, entry):
    written = [PREFILL_ENTRY, AGGREGATED_ENTRY]

    result = create_provider().compile_prediction(
        {**_AIS, "ais_perf_model": written}, _prediction_context(mode)
    )

    assert _hook_config(result)["ais_perf_config"] == normalized(entry)
    assert result.config["ais_perf_model"] == written


def test_predict_use_engine_sets_marker():
    result = create_provider().compile_prediction(
        {**_AIS, "ais_perf_model": _USE_ENGINE}, _prediction_context()
    )

    hook_config = _hook_config(result)
    assert hook_config["ais_perf_config"] is None
    assert hook_config["ais_perf_config_from_engine"] is True
    assert result.config["ais_perf_model"] == _USE_ENGINE


def test_recommend_use_engine_takes_candidate_estimator():
    provider = create_provider()
    plan = provider.compile_recommendation(
        {**_AIS, **_KNOB_CHOICES, "ais_perf_model": _USE_ENGINE},
        _recommendation_context(),
    )
    estimator = {**AGGREGATED_ENTRY, "systems_paths": ["/first"]}
    prefill_estimator = {**PREFILL_ENTRY, "systems_paths": ["/second"]}
    timing = {
        "type": "external",
        "provider": "ais",
        "config": {**AGGREGATED_ENTRY, **_CAPACITY},
    }

    from_estimator = provider.materialize_replay(
        plan,
        _selection(),
        _candidate(forward_pass_estimators={"agg": {"config": estimator}}),
    )
    from_prefill_estimator = provider.materialize_replay(
        plan,
        _selection(),
        _candidate(
            "disagg",
            forward_pass_estimators={"prefill": {"config": prefill_estimator}},
        ),
    )
    from_timing = provider.materialize_replay(
        plan, _selection(), _candidate(agg_timing_model=timing)
    )

    assert _hook_config(from_estimator)["ais_perf_config"] == estimator
    assert _hook_config(from_prefill_estimator)["ais_perf_config"] == prefill_estimator
    assert _hook_config(from_timing)["ais_perf_config"] == AGGREGATED_ENTRY
    for result in (from_estimator, from_prefill_estimator, from_timing):
        assert "ais_perf_config_from_engine" not in _hook_config(result)
        assert result.config == {**_AIS, **_KNOBS, "ais_perf_model": _USE_ENGINE}


@pytest.mark.parametrize(
    "search", [_AIS, _AIS_AMONG_CHOICES], ids=["ais only", "ais among choices"]
)
def test_recommend_written_config_is_resolved_once_per_role(search, tmp_path):
    template = tmp_path / "template.json"
    template.write_text(json.dumps([PREFILL_ENTRY, AGGREGATED_ENTRY]))
    written = [
        {"from": str(template), "worker_type": "prefill"},
        {"from": str(template), "worker_type": "aggregated"},
    ]
    provider = create_provider()
    plan = provider.compile_recommendation(
        {**search, **_KNOB_CHOICES, "ais_perf_model": written},
        _recommendation_context(),
    )
    # The check before the search read the file for every searched mode and
    # the candidates reuse what it read: the changed file is not read again.
    template.write_text(
        json.dumps(
            [
                {**entry, "system": "another_system"}
                for entry in (PREFILL_ENTRY, AGGREGATED_ENTRY)
            ]
        )
    )

    aggregated = [
        provider.materialize_candidate(deepcopy(plan), _selection(), _candidate())
        for _ in range(3)
    ]
    disaggregated = [
        provider.materialize_candidate(
            deepcopy(plan), _selection(), _candidate("disagg")
        )
        for _ in range(2)
    ]

    for result in aggregated:
        assert _hook_config(result)["ais_perf_config"] == normalized(AGGREGATED_ENTRY)
    for result in disaggregated:
        assert _hook_config(result)["ais_perf_config"] == normalized(PREFILL_ENTRY)
    for result in aggregated + disaggregated:
        assert result.config == {**_AIS, **_KNOBS, "ais_perf_model": written}
    # Each candidate gets its own mapping.
    first, second = aggregated[:2]
    assert _hook_config(first)["ais_perf_config"] is not (
        _hook_config(second)["ais_perf_config"]
    )


def test_recommend_reads_written_value_again_for_each_search(tmp_path):
    template = tmp_path / "template.json"
    template.write_text(json.dumps(AGGREGATED_ENTRY))
    section = {
        **_AIS,
        **_KNOB_CHOICES,
        "ais_perf_model": {"from": str(template)},
    }
    context = _recommendation_context("agg")
    provider = create_provider()

    first_plan = provider.compile_recommendation(section, context)
    template.write_text(json.dumps({**AGGREGATED_ENTRY, "system": "another_system"}))
    second_plan = provider.compile_recommendation(section, context)

    def candidate_config(plan):
        result = provider.materialize_candidate(
            deepcopy(plan), _selection(), _candidate()
        )
        return _hook_config(result)["ais_perf_config"]

    # Each search has what it read; a later search does not change an
    # earlier plan.
    assert candidate_config(second_plan) == normalized(
        {**AGGREGATED_ENTRY, "system": "another_system"}
    )
    assert candidate_config(first_plan) == normalized(AGGREGATED_ENTRY)

    # The check before a search reads the file again too.
    template.unlink()
    with pytest.raises(ValueError):
        provider.compile_recommendation(section, context)
    assert candidate_config(first_plan) == normalized(AGGREGATED_ENTRY)


def test_recommend_candidate_without_ais_ignores_key():
    provider = create_provider()
    plan = provider.compile_recommendation(
        {**_AIS_AMONG_CHOICES, **_KNOB_CHOICES, "ais_perf_model": IDENTITY},
        _recommendation_context(),
    )

    without_load_model = provider.materialize_replay(
        plan, _selection("none"), _candidate()
    )
    round_robin = provider.materialize_replay(
        plan, {"mode": "round_robin"}, _candidate()
    )

    assert _hook_config(without_load_model)["ais_perf_config"] is None
    assert without_load_model.config == {
        "policy": "kv_router",
        "prefill_load_model": {"type": "none"},
        **_KNOBS,
    }
    assert round_robin.config == {
        "policy": "round_robin",
        "prefill_load_model": {"type": "none"},
    }
    assert round_robin.runtime_hooks == ()


@pytest.mark.parametrize(
    ("written", "pattern"),
    [
        pytest.param(
            {**IDENTITY, "not_a_field": 1}, _UNKNOWN_KEY_ERROR, id="unknown key"
        ),
        pytest.param(
            {"backend": "vllm", "system": "h200_sxm"},
            exactly("ais_perf_model requires model"),
            id="missing model",
        ),
        pytest.param(
            {**IDENTITY, "system": ""},
            exactly("ais_perf_model requires system"),
            id="empty system",
        ),
    ],
)
def test_recommend_written_value_is_checked_before_search(written, pattern):
    provider = create_provider()

    with pytest.raises(ValueError) as recommended:
        provider.compile_recommendation(
            {**_AIS, **_KNOB_CHOICES, "ais_perf_model": written},
            _recommendation_context(),
        )
    # A prediction with the same value ends in the same error.
    with pytest.raises(ValueError) as predicted:
        provider.compile_prediction(
            {**_AIS, "ais_perf_model": written}, _prediction_context()
        )

    assert type(recommended.value) is ValueError
    assert re.fullmatch(pattern, str(recommended.value))
    assert type(predicted.value) is ValueError
    assert str(predicted.value) == str(recommended.value)


@pytest.mark.parametrize(
    ("written", "perf_configs"),
    [
        pytest.param(_USE_ENGINE, {}, id="use engine"),
        pytest.param(
            IDENTITY,
            {
                "aggregated": normalized(AGGREGATED_ENTRY),
                "prefill": normalized({**IDENTITY, "worker_type": "prefill"}),
            },
            id="one config",
        ),
        pytest.param(
            [PREFILL_ENTRY, AGGREGATED_ENTRY],
            {
                "aggregated": normalized(AGGREGATED_ENTRY),
                "prefill": normalized(PREFILL_ENTRY),
            },
            id="list",
        ),
    ],
)
def test_plan_state_carries_written_value_and_its_configs(written, perf_configs):
    provider = create_provider()
    context = _recommendation_context()

    round_robin_only = provider.compile_recommendation(
        {"policy": "round_robin"}, context
    )
    mixed = provider.compile_recommendation(
        {
            "policy": {"choices": ["round_robin", "kv_router"]},
            "ais_perf_model": written,
        },
        context,
    )
    kv_router_only = provider.compile_recommendation(
        {**_AIS, "ais_perf_model": written}, context
    )

    # Only a search with the AIS load model may write the key.
    assert round_robin_only.state["ais_perf_model"] is None
    assert "ais_perf_configs" not in round_robin_only.state
    assert mixed.state["conditional_public"] is True
    for plan in (mixed, kv_router_only):
        assert plan.state["ais_perf_model"] == written
        assert plan.state["ais_perf_model"] is not written
        assert plan.state["ais_perf_model_by_default"] is False
        # One config for each searched worker type, read when the plan is made.
        assert plan.state["ais_perf_configs"] == perf_configs
    for plan in (round_robin_only, mixed, kv_router_only):
        assert json.loads(json.dumps(plan.state)) == plan.state


@pytest.mark.parametrize("written", [IDENTITY, _USE_ENGINE])
def test_predict_result_config_validates_again(written):
    result = create_provider().compile_prediction(
        {**_AIS, "ais_perf_model": written}, _prediction_context()
    )

    again = RouterPredictionConfig.model_validate(result.config)

    assert again.model_dump(mode="json", exclude_none=True) == result.config


def test_key_rejects_string():
    provider = create_provider()
    config = {**_AIS, "ais_perf_model": "/models/perf.yaml"}
    text = re.escape(
        "router.ais_perf_model must be {use: engine}, a perf model config or a "
        "list of configs, not a string"
    )

    with pytest.raises(ValueError, match=text):
        provider.compile_prediction(config, _prediction_context())
    with pytest.raises(ValueError, match=text):
        provider.compile_recommendation(config, _recommendation_context())


@pytest.mark.parametrize("written", [{"use": "router"}, {**_USE_ENGINE, "model": "m"}])
def test_key_rejects_use_with_other_content(written):
    text = re.escape(
        "router.ais_perf_model: write {use: engine} alone, or a perf model "
        "config without the key use"
    )

    with pytest.raises(ValueError, match=text):
        create_provider().compile_prediction(
            {**_AIS, "ais_perf_model": written}, _prediction_context()
        )


@pytest.mark.parametrize(
    "written",
    [
        [_USE_ENGINE],
        [{**_USE_ENGINE, "worker_type": "prefill"}],
        [AGGREGATED_ENTRY, {**_USE_ENGINE, "worker_type": "prefill"}],
        [AGGREGATED_ENTRY, {"use": "router", "worker_type": "prefill"}],
    ],
)
def test_key_rejects_use_in_list_entry(written):
    provider = create_provider()
    message = (
        "Value error, router.ais_perf_model: write {use: engine} alone, or a "
        "perf model config without the key use"
    )

    with pytest.raises(ValidationError) as predicted:
        provider.compile_prediction(
            {**_AIS, "ais_perf_model": written}, _prediction_context()
        )
    with pytest.raises(ValidationError) as recommended:
        provider.compile_recommendation(
            {**_AIS, "ais_perf_model": written}, _recommendation_context()
        )

    for raised in (predicted, recommended):
        (error,) = raised.value.errors()
        assert error["loc"] == ("ais_perf_model",)
        assert error["msg"] == message


@pytest.mark.parametrize(
    "written", [{**IDENTITY, "systems_paths": ["sys"]}, {"from": "template.yaml"}]
)
def test_key_rejects_relative_paths(written):
    text = re.escape(
        "router.ais_perf_model: relative paths are not allowed here; write "
        "absolute paths"
    )

    with pytest.raises(ValueError, match=text):
        create_provider().compile_prediction(
            {**_AIS, "ais_perf_model": written}, _prediction_context()
        )


def test_key_requires_ais_load_model():
    provider = create_provider()
    predict_text = re.escape(
        "router.ais_perf_model requires router.prefill_load_model.type: ais"
    )

    with pytest.raises(ValueError, match=predict_text):
        provider.compile_prediction(
            {
                "policy": "kv_router",
                "prefill_load_model": {"type": "none"},
                "ais_perf_model": IDENTITY,
            },
            _prediction_context(),
        )
    with pytest.raises(ValueError, match=predict_text):
        provider.compile_prediction(
            {"policy": "round_robin", "ais_perf_model": IDENTITY},
            _prediction_context(),
        )
    # The older check of a round_robin section still comes first.
    with pytest.raises(ValueError, match="round_robin requires") as raised:
        provider.compile_prediction(
            {"policy": "round_robin", "temperature": 0.5, "ais_perf_model": IDENTITY},
            _prediction_context(),
        )
    assert "ais_perf_model requires" not in str(raised.value)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "router.ais_perf_model requires ais among the "
            "router.prefill_load_model.type choices"
        ),
    ):
        provider.compile_recommendation(
            {
                "policy": "kv_router",
                "prefill_load_model": {"type": {"choices": ["none"]}},
                "ais_perf_model": IDENTITY,
            },
            _recommendation_context(),
        )


@pytest.mark.parametrize(
    "written",
    [
        # An empty config or list is a written value, not a missing key.
        pytest.param({}, id="empty config"),
        pytest.param([], id="empty list"),
        pytest.param([IDENTITY], id="list entry without worker type"),
    ],
)
def test_key_syntax_errors_are_loader_errors(written):
    with pytest.raises(ValidationError) as raised:
        create_provider().compile_prediction(
            {**_AIS, "ais_perf_model": written}, _prediction_context()
        )

    (error,) = raised.value.errors()
    assert (error["loc"], error["type"]) == (("ais_perf_model",), "value_error")
    # The loader's own error class, not one of the checks of this section.
    assert isinstance(error["ctx"]["error"], ValueError)
    assert type(error["ctx"]["error"]) is not ValueError


@pytest.mark.parametrize(
    ("mode", "worker_type"), [("agg", "aggregated"), ("disagg", "prefill")]
)
def test_recommend_use_engine_without_engine_perf_model_error(mode, worker_type):
    provider = create_provider()
    plan = provider.compile_recommendation(
        {**_AIS, **_KNOB_CHOICES, "ais_perf_model": _USE_ENGINE},
        _recommendation_context(),
    )
    text = re.escape(_NO_ENGINE_PERF_MODEL.format(worker_type=worker_type))
    role = "prefill" if mode == "disagg" else "agg"

    with pytest.raises(ValueError, match=f"^{text}$"):
        provider.materialize_replay(plan, _selection(), _candidate(mode))
    with pytest.raises(ValueError, match=f"^{text}$") as raised:
        provider.materialize_replay(
            plan,
            _selection(),
            _candidate(mode, **{f"{role}_timing_model": {"type": "polynomial"}}),
        )
    # The section asked for the engine's perf model, so this is not a
    # candidate the search may skip.
    assert type(raised.value) is ValueError


@pytest.mark.parametrize("section", _DEFAULT_LOAD_MODELS)
def test_recommend_default_load_models_read_engine_perf_model(section):
    provider = create_provider()
    context = _recommendation_context()
    estimator = {**AGGREGATED_ENTRY, "systems_paths": ["/first"]}
    timing = {
        "type": "external",
        "provider": "ais",
        "config": {**AGGREGATED_ENTRY, **_CAPACITY},
    }
    candidates = [
        _candidate(forward_pass_estimators={"agg": {"config": estimator}}),
        _candidate(agg_timing_model=timing),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = provider.compile_recommendation({**section, **_KNOB_CHOICES}, context)
        with_ais = [
            provider.materialize_candidate(deepcopy(plan), _selection(), candidate)
            for candidate in candidates
        ]
        without_load_model = provider.materialize_candidate(
            deepcopy(plan), _selection("none"), _candidate()
        )
        # A recommendation compiles a prediction for the candidate it selects.
        predicted = provider.compile_prediction(
            with_ais[0].config, _prediction_context()
        )
    with pytest.warns(FutureWarning):
        written_plan = provider.compile_recommendation(
            {**_AIS, **_KNOB_CHOICES}, context
        )

    assert plan.state["ais_perf_model"] == _USE_ENGINE
    assert plan.state["ais_perf_model_by_default"] is True
    from_estimator, from_timing = with_ais
    assert _hook_config(from_estimator)["ais_perf_config"] == estimator
    assert _hook_config(from_timing)["ais_perf_config"] == AGGREGATED_ENTRY
    for result, candidate in zip(with_ais, candidates):
        # The candidate says where its router perf model comes from.
        assert result.config == {**_AIS, **_KNOBS, "ais_perf_model": _USE_ENGINE}
        # The same config as from a section that writes type: ais.
        written = provider.materialize_candidate(
            deepcopy(written_plan), _selection(), candidate
        )
        assert _hook_config(result) == _hook_config(written)
    assert _hook_config(without_load_model)["ais_perf_config"] is None
    assert without_load_model.config == {
        "policy": "kv_router",
        "prefill_load_model": {"type": "none"},
        **_KNOBS,
    }
    assert _hook_config(predicted)["ais_perf_config_from_engine"] is True


@pytest.mark.parametrize(
    ("mode", "worker_type"), [("agg", "aggregated"), ("disagg", "prefill")]
)
@pytest.mark.parametrize("section", _DEFAULT_LOAD_MODELS)
def test_recommend_default_load_models_skip_ais_without_engine_perf_model(
    section, mode, worker_type
):
    provider = create_provider()
    plan = provider.compile_recommendation(
        {**section, **_KNOB_CHOICES}, _recommendation_context()
    )
    role = "prefill" if mode == "disagg" else "agg"
    text = (
        "router.prefill_load_model.type: ais needs an AIS perf model on the "
        f"{worker_type} engine; it has none"
    )

    for timing_model in (_FIXED_TIMING, {"type": "polynomial"}):
        candidate = _candidate(mode, **{f"{role}_timing_model": timing_model})
        # Infeasible, not failed: the search goes on with the other candidates.
        with pytest.raises(InfeasibleCandidate) as raised:
            provider.materialize_replay(deepcopy(plan), _selection(), candidate)
        assert str(raised.value) == text
        without_load_model = provider.materialize_replay(
            deepcopy(plan), _selection("none"), candidate
        )
        assert _hook_config(without_load_model)["ais_perf_config"] is None


def test_written_list_without_router_entry_error():
    decode_only = [{**IDENTITY, "worker_type": "decode"}]

    # The loader reports the missing entry, in its own words.
    with pytest.raises(ValueError):
        create_provider().compile_prediction(
            {**_AIS, "ais_perf_model": decode_only}, _prediction_context()
        )


def test_recommend_written_value_must_cover_searched_modes():
    provider = create_provider()

    # A list without a prefill entry, and one config written for another
    # worker type: a search ends in the error a prediction of a
    # disaggregated engine gives, which is the loader's.
    for written in ([AGGREGATED_ENTRY], AGGREGATED_ENTRY):
        with pytest.raises(ValueError) as recommended:
            provider.compile_recommendation(
                {**_AIS, "ais_perf_model": written},
                _recommendation_context("agg", "disagg"),
            )
        with pytest.raises(ValueError) as predicted:
            provider.compile_prediction(
                {**_AIS, "ais_perf_model": written},
                _prediction_context("disaggregated"),
            )
        assert type(recommended.value) is type(predicted.value)
        assert str(recommended.value) == str(predicted.value)

    provider.compile_recommendation(
        {**_AIS, "ais_perf_model": [AGGREGATED_ENTRY]},
        _recommendation_context("agg"),
    )
    # One config without a worker type serves every worker type.
    provider.compile_recommendation(
        {**_AIS, "ais_perf_model": IDENTITY},
        _recommendation_context("agg", "disagg"),
    )
    provider.compile_recommendation(
        {"policy": "round_robin"}, _recommendation_context("agg", "disagg")
    )


def test_hook_key_and_versions_are_unchanged():
    provider = create_provider()
    predicted = provider.compile_prediction(
        {**_AIS, "ais_perf_model": IDENTITY}, _prediction_context()
    )
    plan = provider.compile_recommendation(
        {**_AIS, **_KNOB_CHOICES, "ais_perf_model": IDENTITY},
        _recommendation_context(),
    )
    recommended = provider.materialize_candidate(plan, _selection(), _candidate())

    assert provider.config_adapter_api_version == 3
    for result in (predicted, recommended):
        (hook,) = result.runtime_hooks
        assert (hook.provider, hook.kind, hook.api_version) == (
            "dynamo.router",
            "placement_policy",
            1,
        )
        assert set(hook.config) == {"router_mode", "router_config", "ais_perf_config"}
        assert hook.config["ais_perf_config"] == normalized(AGGREGATED_ENTRY)


def test_search_space_entry_rejects_key():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        create_provider().generate_search_space(
            {"policy": ["kv_router"], "ais_perf_model": _USE_ENGINE},
            _sweep_context(),
        )


def test_search_space_plan_with_ais_candidate_is_unchanged_and_silent(tmp_path):
    provider = create_provider()
    estimator = {**IDENTITY, "worker_type": "aggregated", "systems_paths": ["/first"]}
    timing = {
        "type": "external",
        "provider": "ais",
        "config": {**IDENTITY, "worker_type": "aggregated", **_CAPACITY},
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = provider.generate_search_space(
            {
                "policy": {"choices": ["kv_router"]},
                "prefill_load_model": {"type": "ais"},
                **_KNOB_CHOICES,
            },
            _sweep_context(),
        )
        plan.state["core_search_space"]["systems_paths"] = [str(tmp_path)]
        from_estimator = provider.materialize_replay(
            plan,
            _selection(),
            _candidate(forward_pass_estimators={"agg": {"config": estimator}}),
        )
        from_timing = provider.materialize_replay(
            plan, _selection(), _candidate(agg_timing_model=timing)
        )
        from_request = provider.materialize_replay(plan, _selection(), _candidate())

    assert "ais_perf_model" not in plan.state
    assert _hook_config(from_estimator)["ais_perf_config"] == estimator
    assert _hook_config(from_timing)["ais_perf_config"] == {
        **IDENTITY,
        "worker_type": "aggregated",
    }
    request = _hook_config(from_request)["ais_perf_config"]
    assert request["worker_type"] == "aggregated"
    assert request["model"] == "example/model"
    assert request["system"] == "h200_sxm"
    assert request["backend"] == "vllm"
    assert request["systems_paths"] == [str(tmp_path)]
    assert from_request.config == {
        "policy": "kv_router",
        "prefill_load_model": {"type": "ais"},
        **_KNOBS,
    }


def test_absolute_paths_give_same_result_from_any_working_directory(
    tmp_path, monkeypatch
):
    systems = tmp_path / "systems"
    systems.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    written = {**IDENTITY, "systems_paths": [str(systems)]}
    results = []

    for directory in (tmp_path, elsewhere):
        monkeypatch.chdir(directory)
        result = create_provider().compile_prediction(
            {**_AIS, "ais_perf_model": written}, _prediction_context()
        )
        results.append(_hook_config(result)["ais_perf_config"])

    assert results[0] == results[1]
    assert results[0] == normalized({**written, "worker_type": "aggregated"})
    assert results[0]["systems_paths"] == [str(systems)]


def test_predict_without_key_warns_with_exact_text():
    with pytest.warns(FutureWarning) as records:
        create_provider().compile_prediction(_AIS, _prediction_context())

    (record,) = records
    assert str(record.message) == _OLD_FORM_WARNING
    assert record.filename == __file__


def test_recommend_without_key_warns_once_per_provider():
    provider = create_provider()
    context = _recommendation_context()

    with pytest.warns(FutureWarning) as records:
        plan = provider.compile_recommendation({**_AIS, **_KNOB_CHOICES}, context)
        # What a recommendation run does next with the same object.
        provider.compile_recommendation({**_AIS, **_KNOB_CHOICES}, context)
        for _ in range(2):
            provider.materialize_candidate(deepcopy(plan), _selection(), _candidate())
        provider.compile_prediction(_AIS, _prediction_context())
    (record,) = records
    assert str(record.message) == _OLD_FORM_WARNING
    assert record.filename == __file__

    with pytest.warns(FutureWarning) as records:
        # ais written as one of the choices is the old form too.
        create_provider().compile_recommendation(_AIS_AMONG_CHOICES, context)
    (record,) = records
    assert str(record.message) == _OLD_FORM_WARNING

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # No load model type written: ais is only among the default choices.
        create_provider().compile_recommendation({"policy": "kv_router"}, context)
        create_provider().compile_recommendation({}, context)
        create_provider().compile_recommendation(
            {**_AIS, "ais_perf_model": _USE_ENGINE}, context
        )
        create_provider().compile_recommendation(
            {"policy": "kv_router", "prefill_load_model": {"type": "none"}}, context
        )
        create_provider().compile_recommendation({"policy": "round_robin"}, context)


@pytest.mark.parametrize(
    "compile_old_form",
    [_compile_old_form_prediction, _compile_old_form_recommendation],
    ids=["predict", "recommend"],
)
def test_old_form_warning_turned_into_error_is_raised_again(compile_old_form):
    provider = create_provider()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            with pytest.raises(FutureWarning) as raised:
                compile_old_form(provider)
            assert type(raised.value) is FutureWarning
            assert str(raised.value) == _OLD_FORM_WARNING

    # Nobody saw the warning yet, so the same object still shows it, once.
    with pytest.warns(FutureWarning) as records:
        compile_old_form(provider)
        compile_old_form(provider)
    (record,) = records
    assert str(record.message) == _OLD_FORM_WARNING
    assert record.filename == __file__


def test_error_comes_before_warning(tmp_path):
    fpm_path = str(tmp_path / "forward.parquet")
    engine = {
        "mode": "disaggregated",
        "model": "example/model",
        "hardware": "h200_sxm",
        "backend": "vllm",
        "workers": {
            "prefill": {
                "timing": {
                    "estimation_mode": "fpm_interpolation",
                    "fpm_parquet_path": fpm_path,
                    "estimator_config": {
                        "fpm_interpolation": {
                            "fpm_parquet_path": str(tmp_path / "other.parquet")
                        }
                    },
                },
            }
        },
    }
    context = PredictionAdapterContext(engine=engine, traffic={}, evaluation={})

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="conflicting fpm_parquet_path"):
            create_provider().compile_prediction(_AIS, context)

    assert records == []
