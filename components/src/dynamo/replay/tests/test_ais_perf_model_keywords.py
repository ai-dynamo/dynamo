# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight runs before the Dynamo imports.

"""The AIS perf model keywords of the two public replay functions."""

from __future__ import annotations

import contextlib
import json

import pytest

pytest.importorskip(
    "aisim_core.sdk",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.common.ais_perf_model import timing_model
from dynamo.llm import AisPerfConfig, KvRouterConfig
from dynamo.replay import ReplayReport, api
from tests.utils.ais_perf_model import (
    AGGREGATED_ENTRY,
    IDENTITY,
    PREFILL_ENTRY,
    normalized,
)

# The type marker is on each test: the ones that run the native replay are
# integration tests.
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]

_DECODE_ENTRY = {**IDENTITY, "worker_type": "decode", "backend_version": "0.12.0"}
# For the native replays: AISimulate ships the timing data of this model, so
# nothing is looked up online, and a replay loads no weights.
_KNOWN_MODEL_CONFIG = {**IDENTITY, "model": "Qwen/Qwen3-32B"}
# Report fields measured on the wall clock, not on the simulated one.
_WALL_CLOCK_FIELDS = (
    "wall_time_ms",
    "processed_tokens_per_s",
    "processed_output_tokens_per_s",
)
_OLD_KEYWORD_WARNING = (
    "ais_perf_config= is deprecated and will be removed in v1.9; use "
    "router_ais_perf_model=."
)
# A planner config that writes ais_perf_model with the deprecated roles key.
_OLD_PLANNER_CONFIG = {
    "mode": "agg",
    "ais_perf_model": {"roles": {"aggregated": IDENTITY}},
}
_OLD_PLANNER_CONFIG_WARNING = (
    "ais_perf_model.roles is deprecated and will be removed in v1.9; use "
    "ais_perf_model with one perf model config or a list of configs (one per "
    "worker_type)."
)
_AIS_LOAD_MODEL_ERROR = (
    "router_prefill_load_model='ais' requires router_ais_perf_model="
)


def _trace_replay(**kwargs):
    return api.run_trace_replay(["trace.jsonl"], **kwargs)


def _synthetic_replay(**kwargs):
    return api.run_synthetic_trace_replay(16, 4, 1, **kwargs)


def _native_synthetic_replay(**kwargs) -> ReplayReport:
    """A replay the native engine really runs; four requests keep it short."""
    return api.run_synthetic_trace_replay(256, 16, 4, replay_concurrency=2, **kwargs)


def _simulated(report: ReplayReport) -> dict:
    """The summary of a report without its wall-clock fields."""
    return {
        name: value
        for name, value in report.summary.items()
        if name not in _WALL_CLOCK_FIELDS
    }


@pytest.fixture
def native_calls(monkeypatch) -> list[dict]:
    """The keyword arguments of each native replay call; the stand-in raises."""
    calls: list[dict] = []

    def record(*_args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("recorded")

    monkeypatch.setattr(api, "_run_mocker_trace_replay", record)
    monkeypatch.setattr(api, "_run_mocker_synthetic_trace_replay", record)
    return calls


@pytest.fixture
def planner_adapter_calls(monkeypatch) -> list[dict]:
    """The keyword arguments of each planner adapter call; no planner is built."""
    calls: list[dict] = []

    def adapter(**kwargs):
        calls.append(kwargs)
        return contextlib.nullcontext()

    monkeypatch.setattr(api, "_planner_replay_adapter", lambda: adapter)
    return calls


@pytest.fixture
def replay(request, native_calls):
    """Call a public replay function and return what its native call got.

    The trace function, for a test of the code that both functions call. A
    test of what each function does itself takes both with _BOTH_FUNCTIONS.
    """
    function = getattr(request, "param", _trace_replay)

    def call(**kwargs) -> dict:
        with pytest.raises(RuntimeError, match="recorded"):
            function(**kwargs)
        return native_calls[-1]

    return call


# Each public function hands the keywords to the shared code and the results
# to its native call, and the warnings name the line that called it.
_BOTH_FUNCTIONS = pytest.mark.parametrize(
    "replay",
    [_trace_replay, _synthetic_replay],
    ids=["trace", "synthetic"],
    indirect=True,
)


def _kv_router():
    return {
        "router_mode": "kv_router",
        "router_config": KvRouterConfig(router_prefill_load_model="ais"),
    }


def _disaggregated_engine_args():
    return {
        "prefill_engine_args": {"engine": {"worker_type": "prefill"}},
        "decode_engine_args": {"engine": {"worker_type": "decode"}},
    }


@pytest.mark.unit
@_BOTH_FUNCTIONS
def test_old_router_keyword_reaches_native_call(replay):
    perf_config = AisPerfConfig(config=AGGREGATED_ENTRY)
    engine_args = {"engine": {"num_gpu_blocks": 1024}}
    router = _kv_router()

    with pytest.warns(FutureWarning) as records:
        call = replay(
            extra_engine_args=engine_args, ais_perf_config=perf_config, **router
        )

    # One warning, for the line that called the public function.
    (record,) = records
    assert str(record.message) == _OLD_KEYWORD_WARNING
    assert record.filename == __file__
    assert call["ais_perf_config"] is perf_config
    assert call["router_config"] is router["router_config"]
    assert call["extra_engine_args"] is engine_args
    assert engine_args == {"engine": {"num_gpu_blocks": 1024}}
    assert call["prefill_engine_args"] is None
    assert call["decode_engine_args"] is None


@pytest.mark.unit
def test_router_keyword_reads_file_and_inline_json(replay, tmp_path):
    written_file = tmp_path / "perf_model.json"
    written_file.write_text(json.dumps(IDENTITY))

    # Dynamo's type gate lets a path object, a path string and inline JSON through.
    for written in (written_file, str(written_file), json.dumps(IDENTITY)):
        call = replay(router_ais_perf_model=written, **_kv_router())

        assert isinstance(call["ais_perf_config"], AisPerfConfig)
        assert call["ais_perf_config"].to_dict() == normalized(AGGREGATED_ENTRY)


@pytest.mark.unit
def test_router_keyword_uses_prefill_entry_when_disaggregated(replay):
    written = [PREFILL_ENTRY, _DECODE_ENTRY, AGGREGATED_ENTRY]
    router = _kv_router()

    disaggregated = replay(
        router_ais_perf_model=written, **_disaggregated_engine_args(), **router
    )
    aggregated = replay(router_ais_perf_model=written, **router)
    one_config = replay(
        router_ais_perf_model=IDENTITY, **_disaggregated_engine_args(), **router
    )

    assert disaggregated["ais_perf_config"].to_dict() == normalized(PREFILL_ENTRY)
    assert aggregated["ais_perf_config"].to_dict() == normalized(AGGREGATED_ENTRY)
    assert one_config["ais_perf_config"].to_dict() == normalized(
        {**IDENTITY, "worker_type": "prefill"}
    )


@pytest.mark.unit
def test_engine_keyword_creates_aggregated_engine_args(replay):
    # The engine gets the config as resolved, without the defaults that its
    # normalized form adds.
    assert AGGREGATED_ENTRY != normalized(AGGREGATED_ENTRY)
    timing = {"type": "external", "provider": "ais", "config": AGGREGATED_ENTRY}
    expected = {"engine": {"timing_model": timing}}

    from_keyword = replay(ais_perf_model=IDENTITY)
    from_engine_args = replay(extra_engine_args=expected)

    # Nothing else changes: the keyword does not configure the router.
    assert from_keyword == {**from_engine_args, "extra_engine_args": expected}
    assert from_keyword["ais_perf_config"] is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "worker_type"),
    [
        ("extra_engine_args", "aggregated"),
        ("prefill_engine_args", "prefill"),
        ("decode_engine_args", "decode"),
    ],
)
def test_engine_keyword_sets_timing_model_on_given_args(replay, name, worker_type):
    engine_args = {"engine": {"num_gpu_blocks": 1024}}

    call = replay(ais_perf_model=IDENTITY, **{name: engine_args})

    config = {**IDENTITY, "worker_type": worker_type}
    assert call[name] == {
        "engine": {"num_gpu_blocks": 1024, "timing_model": timing_model(config)}
    }
    # Engine args are not created for a role that has none; the native replay
    # then reports that prefill and decode must be given together.
    others = {"extra_engine_args", "prefill_engine_args", "decode_engine_args"} - {name}
    assert [call[other] for other in others] == [None, None]
    # The caller's engine args are copied, not changed.
    assert engine_args == {"engine": {"num_gpu_blocks": 1024}}


@pytest.mark.unit
@_BOTH_FUNCTIONS
def test_engine_keyword_reaches_planner_adapter(replay, planner_adapter_calls):
    call = replay(planner_config={"mode": "agg"}, ais_perf_model=IDENTITY)

    (adapter_call,) = planner_adapter_calls
    assert call["extra_engine_args"] == {
        "engine": {"timing_model": timing_model(AGGREGATED_ENTRY)}
    }
    assert adapter_call["extra_engine_args"] is call["extra_engine_args"]


@pytest.mark.unit
def test_router_and_engine_keywords_reach_retune_inputs(replay, planner_adapter_calls):
    perf_config = AisPerfConfig(config=AGGREGATED_ENTRY)
    engine = {"planner_config": {"mode": "agg"}, "ais_perf_model": IDENTITY}

    new = replay(router_ais_perf_model=IDENTITY, **engine, **_kv_router())
    # The old router keyword only excludes the new router keyword.
    with pytest.warns(FutureWarning, match="ais_perf_config= is deprecated"):
        old = replay(ais_perf_config=perf_config, **engine, **_kv_router())

    assert new["ais_perf_config"].to_dict() == normalized(AGGREGATED_ENTRY)
    assert old["ais_perf_config"] is perf_config
    # The retune inputs get the objects the native call gets.
    for call, adapter_call in zip((new, old), planner_adapter_calls, strict=True):
        assert call["extra_engine_args"] == {
            "engine": {"timing_model": timing_model(AGGREGATED_ENTRY)}
        }
        retune_inputs = adapter_call["retune_inputs_factory"].keywords
        assert retune_inputs["extra_engine_args"] is call["extra_engine_args"]
        assert retune_inputs["ais_perf_config"] is call["ais_perf_config"]


@pytest.mark.unit
def test_old_and_new_router_keywords_error(replay, native_calls):
    perf_config = AisPerfConfig(config=AGGREGATED_ENTRY)

    with pytest.raises(
        ValueError,
        match="^ais_perf_config= cannot be combined with router_ais_perf_model=$",
    ):
        replay(
            ais_perf_config=perf_config,
            router_ais_perf_model=IDENTITY,
            **_kv_router(),
        )

    assert native_calls == []


@pytest.mark.unit
def test_engine_keyword_with_written_timing_model_error(replay, native_calls):
    engine_args = {"engine": {"timing_model": {"type": "polynomial"}}}

    with pytest.raises(
        ValueError,
        match="^extra_engine_args already sets engine.timing_model; it cannot "
        "be combined with ais_perf_model$",
    ):
        replay(ais_perf_model=IDENTITY, extra_engine_args=engine_args)

    assert native_calls == []


@pytest.mark.unit
@pytest.mark.parametrize("keyword", ["router_ais_perf_model", "ais_perf_model"])
def test_new_keywords_reject_other_objects(replay, keyword, native_calls):
    # The object the old keyword takes is one of them.
    for value in (AisPerfConfig(config=AGGREGATED_ENTRY), 3):
        with pytest.raises(
            ValueError, match=f"^{keyword}= takes .*, not {type(value).__name__}$"
        ):
            replay(**{keyword: value}, **_kv_router())

    assert native_calls == []


@pytest.mark.unit
@_BOTH_FUNCTIONS
def test_router_keyword_requires_kv_router_and_router_config(replay, native_calls):
    needs_router_config = (
        "^router_ais_perf_model= requires router_config with "
        "router_prefill_load_model='ais'$"
    )
    without_load_model = {"router_mode": "kv_router", "router_config": KvRouterConfig()}

    for router, message in (
        ({}, "^router_ais_perf_model= requires router_mode='kv_router'$"),
        ({"router_mode": "kv_router"}, needs_router_config),
        # A router config that does not select the load model is the same error.
        (without_load_model, needs_router_config),
    ):
        with pytest.raises(ValueError, match=message):
            replay(router_ais_perf_model=IDENTITY, **router)

    assert native_calls == []


@pytest.mark.unit
@pytest.mark.parametrize("load_model", ["ais", "aic"])
def test_ais_load_model_requires_router_keyword(replay, load_model, native_calls):
    router = {
        "router_mode": "kv_router",
        "router_config": KvRouterConfig(router_prefill_load_model=load_model),
    }

    # The engine keyword does not configure the router.
    for keywords in ({}, {"ais_perf_model": IDENTITY}):
        with pytest.raises(ValueError) as raised:
            replay(**keywords, **router)
        assert type(raised.value) is ValueError
        assert str(raised.value) == _AIS_LOAD_MODEL_ERROR

    assert native_calls == []


@pytest.mark.unit
def test_load_model_checks_left_to_native_replay(replay, native_calls):
    # Round robin does not read the router config.
    replay(router_config=KvRouterConfig(router_prefill_load_model="ais"))
    # The old keyword keeps the native errors, which name it.
    with pytest.warns(FutureWarning, match="ais_perf_config= is deprecated"):
        replay(
            ais_perf_config=AisPerfConfig(config=AGGREGATED_ENTRY),
            router_mode="kv_router",
            router_config=KvRouterConfig(),
        )

    assert len(native_calls) == 2


@pytest.mark.unit
def test_router_keyword_error_names_keyword(replay, native_calls):
    with pytest.raises(ValueError) as raised:
        replay(router_ais_perf_model=[PREFILL_ENTRY], **_kv_router())

    # The shared module's error, whatever its text, with the keyword in front.
    assert type(raised.value) is ValueError
    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value) == f"router_ais_perf_model: {raised.value.__cause__}"
    assert native_calls == []


@pytest.mark.unit
def test_engine_keyword_missing_role_entry_error(replay, native_calls):
    # The loader reports the missing entry, in its own words.
    with pytest.raises(ValueError):
        replay(ais_perf_model=[PREFILL_ENTRY, _DECODE_ENTRY])
    with pytest.raises(ValueError):
        replay(ais_perf_model=[AGGREGATED_ENTRY], **_disaggregated_engine_args())

    assert native_calls == []


@pytest.mark.unit
def test_engine_keyword_identity_value_error(replay, native_calls):
    with pytest.raises(ValueError) as raised:
        replay(ais_perf_model={"backend": "vllm", "system": "h200_sxm"})

    assert type(raised.value) is ValueError
    assert str(raised.value) == "ais_perf_model requires model"
    assert native_calls == []


@pytest.mark.unit
@pytest.mark.parametrize("wrong_role", ["prefill", "decode"])
def test_engine_keyword_config_error_in_one_disaggregated_entry(
    replay, wrong_role, native_calls
):
    entries = {"prefill": PREFILL_ENTRY, "decode": _DECODE_ENTRY}
    entries[wrong_role] = {**entries[wrong_role], "not_a_field": 1}

    # Each worker type's entry is checked, not only the first one.
    with pytest.raises(ValueError, match="^ais_perf_model: .*not_a_field"):
        replay(ais_perf_model=list(entries.values()), **_disaggregated_engine_args())

    assert native_calls == []


@pytest.mark.unit
@_BOTH_FUNCTIONS
def test_same_list_for_engines_and_router(replay):
    written = [PREFILL_ENTRY, _DECODE_ENTRY]

    call = replay(
        ais_perf_model=written,
        router_ais_perf_model=written,
        **_disaggregated_engine_args(),
        **_kv_router(),
    )

    assert call["ais_perf_config"].to_dict() == normalized(PREFILL_ENTRY)
    for worker_type, entry in (("prefill", PREFILL_ENTRY), ("decode", _DECODE_ENTRY)):
        assert call[f"{worker_type}_engine_args"]["engine"]["timing_model"] == (
            timing_model(entry)
        )
    assert written == [PREFILL_ENTRY, _DECODE_ENTRY]


@pytest.mark.unit
@_BOTH_FUNCTIONS
def test_old_planner_config_warns_once_and_names_caller(replay, planner_adapter_calls):
    with pytest.warns(FutureWarning) as records:
        replay(planner_config=_OLD_PLANNER_CONFIG)

    (record,) = records
    assert str(record.message) == _OLD_PLANNER_CONFIG_WARNING
    assert record.filename == __file__
    # The planner adapter gets the config that was read, with the old keys.
    (adapter_call,) = planner_adapter_calls
    assert set(adapter_call["planner_config_arg"].ais_perf_model.roles) == {
        "aggregated"
    }


@pytest.mark.unit
def test_old_inputs_already_warned_silences_both_old_input_warnings(
    replay, planner_adapter_calls
):
    old_inputs = {
        "planner_config": _OLD_PLANNER_CONFIG,
        "ais_perf_config": AisPerfConfig(config=AGGREGATED_ENTRY),
        **_kv_router(),
    }

    # The suite turns a warning into an error: inside the block there is none.
    with pytest.raises(RuntimeError, match="inside"):
        with api._old_inputs_already_warned():
            replay(**old_inputs)
            raise RuntimeError("inside")
    # Outside, both warn again, also after an error inside.
    with pytest.raises(FutureWarning, match="ais_perf_config= is deprecated"):
        replay(**old_inputs)
    with pytest.raises(FutureWarning, match="ais_perf_model.roles is deprecated"):
        replay(planner_config=_OLD_PLANNER_CONFIG)


@pytest.mark.integration
@pytest.mark.timeout(15)  # ~4 s; the first load of the perf model's data dominates
def test_engine_keyword_runs_native_replay_like_engine_args():
    engine_args = {
        "engine": {
            "timing_model": {
                "type": "external",
                "provider": "ais",
                "config": {**_KNOWN_MODEL_CONFIG, "worker_type": "aggregated"},
            }
        }
    }

    from_keyword = _native_synthetic_replay(ais_perf_model=_KNOWN_MODEL_CONFIG)
    from_engine_args = _native_synthetic_replay(extra_engine_args=engine_args)
    without_perf_model = _native_synthetic_replay()

    assert from_keyword.summary["completed_requests"] == 4
    # Both replays simulate the same engine, so the simulated values are equal.
    assert _simulated(from_keyword) == _simulated(from_engine_args)
    # The perf model decides the timing: without it the latency is another one.
    assert (
        from_keyword.summary["mean_ttft_ms"]
        != without_perf_model.summary["mean_ttft_ms"]
    )


@pytest.mark.integration
@pytest.mark.timeout(15)  # ~3.5 s alone, ~1 s once the perf model's data is loaded
def test_router_and_engine_keywords_run_native_kv_router_replay():
    report = _native_synthetic_replay(
        num_workers=2,
        router_ais_perf_model=_KNOWN_MODEL_CONFIG,
        ais_perf_model=_KNOWN_MODEL_CONFIG,
        **_kv_router(),
    )

    assert isinstance(report, ReplayReport)
    assert report.summary["completed_requests"] == 4
