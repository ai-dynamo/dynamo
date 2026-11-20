# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""The ``ais_perf_model`` key of the ``router`` section of an AISimulate config."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

pytest.importorskip(
    "aisim.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisim.config_adapter import PredictionAdapterContext, RecommendationAdapterContext
from aisim.sweeper.provider import CandidateContext, InfeasibleCandidate, SweepContext
from aisim.sweeper.replay import BackendDeploymentSpec

from dynamo.router.simulation import create_provider
from tests.utils.ais_perf_model import (
    AGGREGATED_ENTRY,
    IDENTITY,
    PREFILL_ENTRY,
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
_AIS = {"policy": "kv_router", "prefill_load_model": {"type": "ais"}}
# A search that can also pick round robin or no load model.
_AIS_AMONG_CHOICES = {
    "policy": {"choices": ["round_robin", "kv_router"]},
    "prefill_load_model": {"type": {"choices": ["none", "ais"]}},
}
_USE_ENGINE = {"use": "engine"}
_BOTH_ENTRIES = [PREFILL_ENTRY, AGGREGATED_ENTRY]
# Router sections that write no load model type: the search takes the default
# load models, none and ais.
_DEFAULT_LOAD_MODELS = [
    pytest.param({}, id="empty"),
    pytest.param(
        {"policy": "kv_router", "prefill_load_model": {}}, id="empty load model"
    ),
]
_FIXED_TIMING = {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
_USE_ALONE = r"router\.ais_perf_model: write \{use: engine\} alone"
_HOOK_KEYS = {"router_mode", "router_config", "ais_perf_config"}


def _prediction_context(mode: str = "aggregated") -> PredictionAdapterContext:
    worker = "prefill" if mode == "disaggregated" else "aggregated"
    return PredictionAdapterContext(
        engine={
            "mode": mode,
            "model": "example/model",
            "hardware": "h200_sxm",
            "backend": "vllm",
            "workers": {worker: {}},
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
        sweep=SweepContext(
            core_search_space={"deployment_mode": list(modes or ("agg", "disagg"))},
            workload={"trace_path": "trace.jsonl"},
            goal={"target": "throughput"},
            show_progress=False,
        ),
    )


def _candidate(mode: str = "agg", **sample) -> CandidateContext:
    return CandidateContext(
        sample={"deployment_mode": mode, **sample},
        backend_deployment=BackendDeploymentSpec(
            deployment_mode=mode, backend="vllm", backend_version="0.11.0"
        ),
    )


def _selection(load_model: str = "ais") -> dict:
    return {"mode": "kv_router", "prefill_load_model_type": load_model, **_KNOBS}


def _hook_config(result) -> dict:
    (hook,) = result.runtime_hooks
    return hook.config


@pytest.mark.parametrize(
    ("written", "mode", "entry"),
    [
        pytest.param(IDENTITY, "aggregated", AGGREGATED_ENTRY, id="one config"),
        pytest.param(_BOTH_ENTRIES, "aggregated", AGGREGATED_ENTRY, id="list, agg"),
        pytest.param(_BOTH_ENTRIES, "disaggregated", PREFILL_ENTRY, id="list, disagg"),
    ],
)
def test_predict_written_value_selects_by_engine_mode(written, mode, entry):
    result = create_provider().compile_prediction(
        {**_AIS, "ais_perf_model": written}, _prediction_context(mode)
    )

    hook_config = _hook_config(result)
    assert hook_config["ais_perf_config"] == normalized(entry)
    assert set(hook_config) == _HOOK_KEYS
    assert result.config["ais_perf_model"] == written


def test_predict_use_engine_sets_marker():
    result = create_provider().compile_prediction(
        {**_AIS, "ais_perf_model": _USE_ENGINE}, _prediction_context()
    )

    hook_config = _hook_config(result)
    assert hook_config["ais_perf_config"] is None
    assert hook_config["ais_perf_config_from_engine"] is True
    assert result.config["ais_perf_model"] == _USE_ENGINE


@pytest.mark.parametrize(
    "section",
    [
        pytest.param({**_AIS, "ais_perf_model": _USE_ENGINE}, id="use engine"),
        *_DEFAULT_LOAD_MODELS,
    ],
)
def test_recommend_ais_candidate_reads_engine_perf_model(section):
    provider = create_provider()
    estimator = {**AGGREGATED_ENTRY, "systems_paths": ["/first"]}
    prefill_estimator = {**PREFILL_ENTRY, "systems_paths": ["/second"]}
    # A custom engine timing config also has keys that only KV capacity uses.
    timing = {
        "type": "external",
        "provider": "ais",
        "config": {**AGGREGATED_ENTRY, "gpu_memory_utilization": 0.9},
    }
    candidates = [
        (_candidate(forward_pass_estimators={"agg": {"config": estimator}}), estimator),
        (
            _candidate(
                "disagg",
                forward_pass_estimators={"prefill": {"config": prefill_estimator}},
            ),
            prefill_estimator,
        ),
        (_candidate(agg_timing_model=timing), AGGREGATED_ENTRY),
    ]

    plan = provider.compile_recommendation(
        {**section, **_KNOB_CHOICES}, _recommendation_context()
    )

    for candidate, perf_config in candidates:
        result = provider.materialize_candidate(plan, _selection(), candidate)
        assert _hook_config(result)["ais_perf_config"] == perf_config
        assert "ais_perf_config_from_engine" not in _hook_config(result)
        # The candidate says where its router perf model comes from.
        assert result.config == {**_AIS, **_KNOBS, "ais_perf_model": _USE_ENGINE}


@pytest.mark.parametrize(
    ("written", "prefill_entry"),
    [
        pytest.param(IDENTITY, {**IDENTITY, "worker_type": "prefill"}, id="one config"),
        pytest.param(_BOTH_ENTRIES, PREFILL_ENTRY, id="list"),
    ],
)
def test_recommend_written_value_selects_by_candidate_mode(written, prefill_entry):
    provider = create_provider()
    plan = provider.compile_recommendation(
        {**_AIS_AMONG_CHOICES, **_KNOB_CHOICES, "ais_perf_model": written},
        _recommendation_context(),
    )

    aggregated = provider.materialize_candidate(plan, _selection(), _candidate())
    disaggregated = provider.materialize_candidate(
        plan, _selection(), _candidate("disagg")
    )
    without_load_model = provider.materialize_candidate(
        plan, _selection("none"), _candidate()
    )

    assert _hook_config(aggregated)["ais_perf_config"] == normalized(AGGREGATED_ENTRY)
    assert _hook_config(disaggregated)["ais_perf_config"] == normalized(prefill_entry)
    for result in (aggregated, disaggregated):
        assert set(_hook_config(result)) == _HOOK_KEYS
        assert result.config == {**_AIS, **_KNOBS, "ais_perf_model": written}
    # A candidate without the AIS load model neither reads nor writes the key.
    assert _hook_config(without_load_model)["ais_perf_config"] is None
    assert "ais_perf_model" not in without_load_model.config
    assert json.loads(json.dumps(plan.state)) == plan.state

    # The key without a written load model type is still a written value.
    plan = provider.compile_recommendation(
        {
            "policy": {"choices": ["round_robin", "kv_router"]},
            **_KNOB_CHOICES,
            "ais_perf_model": written,
        },
        _recommendation_context(),
    )
    result = provider.materialize_candidate(plan, _selection(), _candidate())
    assert _hook_config(result)["ais_perf_config"] == normalized(AGGREGATED_ENTRY)
    assert result.config["ais_perf_model"] == written


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


def test_written_value_is_checked_for_each_mode_before_the_run():
    provider = create_provider()
    aggregated_only = {**_AIS, "ais_perf_model": [AGGREGATED_ENTRY]}

    # No config for prefill: a prediction of a disaggregated engine fails, and
    # so does a search that includes disagg, before any candidate runs.
    with pytest.raises(ValueError, match="prefill"):
        provider.compile_prediction(
            aggregated_only, _prediction_context("disaggregated")
        )
    with pytest.raises(ValueError, match="prefill"):
        provider.compile_recommendation(
            aggregated_only, _recommendation_context("agg", "disagg")
        )
    provider.compile_recommendation(aggregated_only, _recommendation_context("agg"))
    # A config that the read rejects ends a search before it starts too.
    with pytest.raises(ValueError, match="^ais_perf_model requires model"):
        provider.compile_recommendation(
            {**_AIS, "ais_perf_model": {"backend": "vllm", "system": "h200_sxm"}},
            _recommendation_context(),
        )


@pytest.mark.parametrize(
    ("written", "text"),
    [
        ("/models/perf.yaml", r"router\.ais_perf_model must be .*not a string"),
        ({"use": "router"}, _USE_ALONE),
        ({**_USE_ENGINE, "model": "m"}, _USE_ALONE),
        # An entry that no mode selects is checked too.
        ([AGGREGATED_ENTRY, {**_USE_ENGINE, "worker_type": "decode"}], _USE_ALONE),
    ],
    ids=["string", "use: router", "use with a key", "use in an entry"],
)
def test_key_rejects_wrong_value(written, text):
    provider = create_provider()
    section = {**_AIS, "ais_perf_model": written}

    with pytest.raises(ValueError, match=text):
        provider.compile_prediction(section, _prediction_context())
    with pytest.raises(ValueError, match=text):
        provider.compile_recommendation(section, _recommendation_context())


def test_key_requires_ais_load_model():
    provider = create_provider()
    text = r"router\.ais_perf_model requires"

    for section in (
        {"policy": "kv_router", "prefill_load_model": {"type": "none"}},
        {"policy": "round_robin"},
    ):
        with pytest.raises(ValueError, match=text):
            provider.compile_prediction(
                {**section, "ais_perf_model": IDENTITY}, _prediction_context()
            )
    with pytest.raises(ValueError, match=text):
        provider.compile_recommendation(
            {
                "policy": "kv_router",
                "prefill_load_model": {"type": {"choices": ["none"]}},
                "ais_perf_model": IDENTITY,
            },
            _recommendation_context(),
        )


@pytest.mark.parametrize(
    ("mode", "worker_type"), [("agg", "aggregated"), ("disagg", "prefill")]
)
def test_recommend_use_engine_without_engine_perf_model_error(mode, worker_type):
    provider = create_provider()
    plan = provider.compile_recommendation(
        {**_AIS, **_KNOB_CHOICES, "ais_perf_model": _USE_ENGINE},
        _recommendation_context(),
    )
    role = "prefill" if mode == "disagg" else "agg"
    candidate = _candidate(mode, **{f"{role}_timing_model": {"type": "polynomial"}})

    with pytest.raises(ValueError) as raised:
        provider.materialize_replay(plan, _selection(), candidate)

    # The section asked for the engine's perf model, so this is not a
    # candidate the search may skip.
    assert type(raised.value) is ValueError
    assert str(raised.value) == (
        "router.ais_perf_model: {use: engine} needs an AIS perf model on the "
        f"{worker_type} engine; it has none"
    )


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


def test_search_space_plan_with_ais_candidate_keeps_old_sources():
    provider = create_provider()
    estimator = {**AGGREGATED_ENTRY, "systems_paths": ["/first"]}

    # A plan made by generate_search_space has no ais_perf_model in its state.
    plan = provider.generate_search_space(
        {
            "policy": {"choices": ["kv_router"]},
            "prefill_load_model": {"type": "ais"},
            **_KNOB_CHOICES,
        },
        _recommendation_context().sweep,
    )
    result = provider.materialize_replay(
        plan,
        _selection(),
        _candidate(forward_pass_estimators={"agg": {"config": estimator}}),
    )

    assert "ais_perf_model" not in plan.state
    assert _hook_config(result)["ais_perf_config"] == estimator
    # The candidate keeps the old sources and does not write the key.
    assert result.config == {**_AIS, **_KNOBS}


def test_predict_without_key_warns_with_exact_text():
    with pytest.warns(FutureWarning) as records:
        create_provider().compile_prediction(_AIS, _prediction_context())

    (record,) = records
    assert str(record.message) == (
        "router.prefill_load_model.type: ais without router.ais_perf_model is "
        "deprecated and will be removed in v1.9; use router.ais_perf_model: "
        "{use: engine} (or a perf model config when the engine does not use an AIS "
        "perf model)."
    )
    assert record.filename == __file__


def test_recommend_without_key_warns_once_per_provider():
    provider = create_provider()
    context = _recommendation_context()
    old_form = "without router.ais_perf_model is deprecated"

    with pytest.warns(FutureWarning, match=old_form) as records:
        plan = provider.compile_recommendation({**_AIS, **_KNOB_CHOICES}, context)
        # What a recommendation run does next with the same object.
        provider.compile_recommendation({**_AIS, **_KNOB_CHOICES}, context)
        provider.materialize_candidate(
            plan,
            _selection(),
            _candidate(forward_pass_estimators={"agg": {"config": AGGREGATED_ENTRY}}),
        )
        provider.compile_prediction(_AIS, _prediction_context())
    (record,) = records
    assert record.filename == __file__

    # ais written as one of the choices is the old form too.
    with pytest.warns(FutureWarning, match=old_form):
        create_provider().compile_recommendation(_AIS_AMONG_CHOICES, context)
    # A written load model type other than ais is not.
    create_provider().compile_recommendation(
        {"policy": "kv_router", "prefill_load_model": {"type": "none"}}, context
    )


@pytest.mark.filterwarnings("error::FutureWarning")
def test_old_form_warning_turned_into_error_is_raised_again():
    provider = create_provider()

    # A warning that a filter turns into an error was not shown: the next
    # compile of the same object raises it again.
    for _ in range(2):
        with pytest.raises(FutureWarning):
            provider.compile_prediction(_AIS, _prediction_context())
        with pytest.raises(FutureWarning):
            provider.compile_recommendation(_AIS, _recommendation_context())
