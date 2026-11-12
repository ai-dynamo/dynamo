# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the AISimulate imports.

"""The AIS perf model keywords of the two public replay functions."""

from __future__ import annotations

import contextlib
import json
import re
import warnings
from copy import deepcopy

import pytest

pytest.importorskip(
    "aisimulate_core.sdk",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate_core.sdk import ForwardPassPerfModelConfig

from dynamo.common.ais_perf_model import (
    load_document,
    resolve_entry,
    router_config,
    timing_model,
)
from dynamo.llm import AisPerfConfig, KvRouterConfig
from dynamo.replay import api

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]

_CONFIG = {"backend": "vllm", "system": "h200_sxm", "model": "example/model"}
_PREFILL_ENTRY = {**_CONFIG, "worker_type": "prefill", "backend_version": "0.11.0"}
_DECODE_ENTRY = {**_CONFIG, "worker_type": "decode", "backend_version": "0.12.0"}
_AGGREGATED_ENTRY = {**_CONFIG, "worker_type": "aggregated"}
_OLD_KEYWORD_WARNING = (
    "ais_perf_config= is deprecated and will be removed in v1.9; use "
    "router_ais_perf_model=."
)
# What the perf model config class says about a key it does not take.
_UNKNOWN_KEY_ERROR = (
    "ForwardPassPerfModelConfig.__init__() got an unexpected keyword argument "
    "'not_a_field'"
)


class _NativeCallRecorded(Exception):
    """Raised by the stand-in native call, so no report is built."""


def _trace_replay(**kwargs):
    return api.run_trace_replay(["trace.jsonl"], **kwargs)


def _synthetic_replay(**kwargs):
    return api.run_synthetic_trace_replay(16, 4, 1, **kwargs)


_BOTH_FUNCTIONS = pytest.mark.parametrize(
    "replay", [_trace_replay, _synthetic_replay], ids=["trace", "synthetic"]
)


@pytest.fixture
def native_calls(monkeypatch):
    """The keyword arguments of every native replay call the test makes."""
    calls: list[dict] = []

    def record(*_args, **kwargs):
        calls.append(kwargs)
        raise _NativeCallRecorded

    monkeypatch.setattr(api, "_run_mocker_trace_replay", record)
    monkeypatch.setattr(api, "_run_mocker_synthetic_trace_replay", record)
    return calls


@pytest.fixture
def planner_adapter_calls(monkeypatch):
    """The keyword arguments of every planner adapter the test asks for."""
    calls: list[dict] = []

    def record(**kwargs):
        calls.append(kwargs)
        raise _NativeCallRecorded

    monkeypatch.setattr(api, "_planner_replay_adapter", lambda: record)
    return calls


@pytest.fixture
def planner_replay_calls(monkeypatch, native_calls):
    """The planner adapter calls and the native calls of a replay with a planner."""
    adapter_calls: list[dict] = []

    @contextlib.contextmanager
    def adapter(**kwargs):
        adapter_calls.append(kwargs)
        yield None

    monkeypatch.setattr(api, "_planner_replay_adapter", lambda: adapter)
    return adapter_calls, native_calls


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


def _resolved(value, worker_type):
    """The config the engines of one worker type get from a written value."""
    return resolve_entry(
        load_document(value).select(worker_type), worker_type=worker_type
    )


def _normalized(config):
    return ForwardPassPerfModelConfig(**config).to_dict()


@_BOTH_FUNCTIONS
def test_old_router_keyword_reaches_native_call(replay, native_calls):
    perf_config = AisPerfConfig(config={**_CONFIG, "worker_type": "aggregated"})
    engine_args = {"engine": {"num_gpu_blocks": 1024}}
    router = _kv_router()

    with pytest.warns(FutureWarning, match=r"ais_perf_config= is deprecated"):
        with pytest.raises(_NativeCallRecorded):
            replay(extra_engine_args=engine_args, ais_perf_config=perf_config, **router)

    (call,) = native_calls
    assert call["ais_perf_config"] is perf_config
    assert call["router_config"] is router["router_config"]
    assert call["extra_engine_args"] is engine_args
    assert engine_args == {"engine": {"num_gpu_blocks": 1024}}
    assert call["prefill_engine_args"] is None
    assert call["decode_engine_args"] is None


@_BOTH_FUNCTIONS
def test_old_router_keyword_unknown_key_keeps_type_error(replay, native_calls):
    unknown_key = {**_AGGREGATED_ENTRY, "not_a_field": 1}

    # The object rejects the key when the caller builds it, before the replay
    # function runs: there is no warning, as before the new keywords existed.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError) as raised:
            replay(ais_perf_config=AisPerfConfig(config=unknown_key), **_kv_router())

    assert type(raised.value) is TypeError
    assert str(raised.value) == _UNKNOWN_KEY_ERROR
    assert raised.value.__cause__ is None
    assert native_calls == []

    # A value that is not that object is not checked here: the warning comes
    # first and the native call gets the value as given and rejects it.
    with pytest.warns(FutureWarning) as records:
        with pytest.raises(_NativeCallRecorded):
            replay(ais_perf_config=unknown_key, **_kv_router())

    (record,) = records
    assert str(record.message) == _OLD_KEYWORD_WARNING
    (call,) = native_calls
    assert call["ais_perf_config"] is unknown_key


@_BOTH_FUNCTIONS
def test_no_perf_model_keyword_passes_none(replay, native_calls):
    with pytest.raises(_NativeCallRecorded):
        replay()

    (call,) = native_calls
    assert call["ais_perf_config"] is None
    assert call["extra_engine_args"] is None
    assert call["prefill_engine_args"] is None
    assert call["decode_engine_args"] is None


def test_old_router_keyword_reaches_retune_inputs(planner_adapter_calls):
    perf_config = AisPerfConfig(config={**_CONFIG, "worker_type": "aggregated"})

    with pytest.warns(FutureWarning, match=r"ais_perf_config= is deprecated"):
        with pytest.raises(_NativeCallRecorded):
            api.run_trace_replay(
                ["window.jsonl"],
                trace_format="dynamo",
                planner_config={"mode": "agg"},
                ais_perf_config=perf_config,
                **_kv_router(),
            )

    (call,) = planner_adapter_calls
    assert call["retune_inputs_factory"].keywords["ais_perf_config"] is perf_config


@_BOTH_FUNCTIONS
def test_router_keyword_builds_native_config(replay, native_calls, tmp_path):
    written_file = tmp_path / "perf_model.json"
    written_file.write_text(json.dumps(_CONFIG))
    written_forms = [
        _CONFIG,
        [_PREFILL_ENTRY, _AGGREGATED_ENTRY],
        str(written_file),
        written_file,
        json.dumps(_CONFIG),
    ]

    for written in written_forms:
        with pytest.raises(_NativeCallRecorded):
            replay(router_ais_perf_model=written, **_kv_router())

    assert len(native_calls) == len(written_forms)
    for written, call in zip(written_forms, native_calls):
        assert isinstance(call["ais_perf_config"], AisPerfConfig)
        assert call["ais_perf_config"].to_dict() == router_config(written, "aggregated")
        assert call["ais_perf_config"].to_dict() == _normalized(_AGGREGATED_ENTRY)


@_BOTH_FUNCTIONS
def test_router_keyword_uses_prefill_entry_when_disaggregated(replay, native_calls):
    written = [_PREFILL_ENTRY, _DECODE_ENTRY, _AGGREGATED_ENTRY]

    with pytest.raises(_NativeCallRecorded):
        replay(
            router_ais_perf_model=written,
            **_disaggregated_engine_args(),
            **_kv_router(),
        )
    with pytest.raises(_NativeCallRecorded):
        replay(router_ais_perf_model=written, **_kv_router())
    with pytest.raises(_NativeCallRecorded):
        replay(
            router_ais_perf_model=_CONFIG,
            **_disaggregated_engine_args(),
            **_kv_router(),
        )

    disaggregated, aggregated, one_config = native_calls
    assert disaggregated["ais_perf_config"].to_dict() == _normalized(_PREFILL_ENTRY)
    assert aggregated["ais_perf_config"].to_dict() == _normalized(_AGGREGATED_ENTRY)
    assert one_config["ais_perf_config"].to_dict() == _normalized(
        {**_CONFIG, "worker_type": "prefill"}
    )


def test_router_keyword_reaches_retune_inputs(planner_replay_calls):
    adapter_calls, native_calls = planner_replay_calls

    with pytest.raises(_NativeCallRecorded):
        api.run_trace_replay(
            ["window.jsonl"],
            trace_format="dynamo",
            planner_config={"mode": "agg"},
            router_ais_perf_model=_CONFIG,
            **_kv_router(),
        )

    (adapter_call,) = adapter_calls
    (native_call,) = native_calls
    retune_inputs = adapter_call["retune_inputs_factory"].keywords
    assert retune_inputs["ais_perf_config"] is native_call["ais_perf_config"]
    assert native_call["ais_perf_config"].to_dict() == _normalized(_AGGREGATED_ENTRY)


@_BOTH_FUNCTIONS
def test_engine_keyword_creates_aggregated_engine_args(replay, native_calls):
    expected = {
        "engine": {"timing_model": timing_model(_resolved(_CONFIG, "aggregated"))}
    }

    with pytest.raises(_NativeCallRecorded):
        replay(ais_perf_model=_CONFIG)
    with pytest.raises(_NativeCallRecorded):
        replay(extra_engine_args=expected)

    from_keyword, from_engine_args = native_calls
    assert from_keyword["extra_engine_args"] == expected
    assert from_keyword == {**from_engine_args, "extra_engine_args": expected}
    assert (
        from_keyword["extra_engine_args"]["engine"]["timing_model"]["config"][
            "worker_type"
        ]
        == "aggregated"
    )


@_BOTH_FUNCTIONS
def test_engine_keyword_sets_timing_model_on_given_args(replay, native_calls):
    as_mapping = {"engine": {"num_gpu_blocks": 1024}, "planner_profile_data": None}
    as_text = json.dumps(as_mapping)
    disaggregated = _disaggregated_engine_args()
    given = deepcopy([as_mapping, disaggregated])

    with pytest.raises(_NativeCallRecorded):
        replay(extra_engine_args=as_mapping, ais_perf_model=_CONFIG)
    with pytest.raises(_NativeCallRecorded):
        replay(extra_engine_args=as_text, ais_perf_model=_CONFIG)
    with pytest.raises(_NativeCallRecorded):
        replay(ais_perf_model=[_PREFILL_ENTRY, _DECODE_ENTRY], **disaggregated)
    with pytest.raises(_NativeCallRecorded):
        replay(ais_perf_model=_CONFIG, **disaggregated)

    from_mapping, from_text, from_list, from_one_config = native_calls
    aggregated = {
        "engine": {
            "num_gpu_blocks": 1024,
            "timing_model": timing_model(_resolved(_CONFIG, "aggregated")),
        },
        "planner_profile_data": None,
    }
    assert from_mapping["extra_engine_args"] == aggregated
    assert from_text["extra_engine_args"] == aggregated
    for call, written in (
        (from_list, [_PREFILL_ENTRY, _DECODE_ENTRY]),
        (from_one_config, _CONFIG),
    ):
        assert call["extra_engine_args"] is None
        for worker_type in ("prefill", "decode"):
            config = _resolved(written, worker_type)
            assert config["worker_type"] == worker_type
            assert call[f"{worker_type}_engine_args"] == {
                "engine": {
                    "worker_type": worker_type,
                    "timing_model": timing_model(config),
                }
            }
    assert from_list["prefill_engine_args"] != from_list["decode_engine_args"]
    # The caller's engine args are copied, not changed.
    assert [as_mapping, disaggregated] == given
    assert as_text == json.dumps(as_mapping)


def test_engine_keyword_reaches_planner_adapter_and_retune_inputs(
    planner_replay_calls,
):
    adapter_calls, native_calls = planner_replay_calls
    expected = {
        "engine": {"timing_model": timing_model(_resolved(_CONFIG, "aggregated"))}
    }

    with pytest.raises(_NativeCallRecorded):
        api.run_trace_replay(
            ["window.jsonl"],
            trace_format="dynamo",
            planner_config={"mode": "agg"},
            ais_perf_model=_CONFIG,
        )
    with pytest.raises(_NativeCallRecorded):
        api.run_synthetic_trace_replay(
            16, 4, 1, planner_config={"mode": "agg"}, ais_perf_model=_CONFIG
        )

    trace_adapter, synthetic_adapter = adapter_calls
    trace_native, synthetic_native = native_calls
    assert trace_adapter["extra_engine_args"] == expected
    assert trace_adapter["extra_engine_args"] is trace_native["extra_engine_args"]
    retune_inputs = trace_adapter["retune_inputs_factory"].keywords
    assert retune_inputs["extra_engine_args"] is trace_native["extra_engine_args"]
    assert retune_inputs["ais_perf_config"] is None
    assert synthetic_adapter["extra_engine_args"] == expected
    assert (
        synthetic_adapter["extra_engine_args"] is synthetic_native["extra_engine_args"]
    )


@_BOTH_FUNCTIONS
def test_router_and_engine_keywords_together(replay, native_calls):
    with pytest.raises(_NativeCallRecorded):
        replay(
            router_ais_perf_model=_CONFIG,
            ais_perf_model={**_CONFIG, "backend_version": "0.11.0"},
            **_kv_router(),
        )

    (call,) = native_calls
    assert call["ais_perf_config"].to_dict() == _normalized(_AGGREGATED_ENTRY)
    assert call["extra_engine_args"] == {
        "engine": {
            "timing_model": timing_model(
                {**_AGGREGATED_ENTRY, "backend_version": "0.11.0"}
            )
        }
    }


@_BOTH_FUNCTIONS
def test_internal_keyword_passes_object_without_warning(replay, native_calls):
    perf_config = AisPerfConfig(config=_AGGREGATED_ENTRY)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(_NativeCallRecorded):
            replay(_native_ais_perf_config=perf_config, **_kv_router())

    (call,) = native_calls
    assert call["ais_perf_config"] is perf_config
    assert "_native_ais_perf_config" not in call


@_BOTH_FUNCTIONS
def test_old_and_new_router_keywords_error(replay, native_calls):
    perf_config = AisPerfConfig(config=_AGGREGATED_ENTRY)

    with pytest.raises(
        ValueError,
        match="^ais_perf_config= cannot be combined with router_ais_perf_model=$",
    ):
        replay(
            ais_perf_config=perf_config,
            router_ais_perf_model=_CONFIG,
            **_kv_router(),
        )

    assert native_calls == []


@pytest.mark.parametrize("public", ["ais_perf_config", "router_ais_perf_model"])
@_BOTH_FUNCTIONS
def test_internal_keyword_with_public_router_keyword_error(
    replay, public, native_calls
):
    perf_config = AisPerfConfig(config=_AGGREGATED_ENTRY)
    value = perf_config if public == "ais_perf_config" else _CONFIG

    with pytest.raises(
        ValueError,
        match=re.escape(
            "_native_ais_perf_config is internal; it cannot be combined with "
            "ais_perf_config= or router_ais_perf_model="
        ),
    ):
        replay(_native_ais_perf_config=perf_config, **{public: value}, **_kv_router())

    assert native_calls == []


@pytest.mark.parametrize(
    "name", ["extra_engine_args", "prefill_engine_args", "decode_engine_args"]
)
@_BOTH_FUNCTIONS
def test_engine_keyword_with_written_timing_model_error(replay, name, native_calls):
    engine_args = (
        {"extra_engine_args": {"engine": {}}}
        if name == "extra_engine_args"
        else _disaggregated_engine_args()
    )
    engine_args[name]["engine"]["timing_model"] = {"type": "polynomial"}

    with pytest.raises(
        ValueError,
        match=f"^{name} already sets engine.timing_model; it cannot be combined "
        "with ais_perf_model$",
    ):
        replay(ais_perf_model=_CONFIG, **engine_args)

    assert native_calls == []


@pytest.mark.parametrize("keyword", ["router_ais_perf_model", "ais_perf_model"])
@_BOTH_FUNCTIONS
def test_new_keywords_reject_other_objects(replay, keyword, native_calls):
    other_objects = [
        AisPerfConfig(config=_AGGREGATED_ENTRY),
        3,
        ForwardPassPerfModelConfig(**_AGGREGATED_ENTRY),
    ]

    for value in other_objects:
        with pytest.raises(
            ValueError,
            match=f"^{keyword}= takes a file path, inline JSON, a config or a list "
            f"of configs, not {type(value).__name__}$",
        ):
            replay(**{keyword: value}, **_kv_router())

    assert native_calls == []


@_BOTH_FUNCTIONS
def test_router_keyword_requires_kv_router_and_router_config(replay, native_calls):
    with pytest.raises(
        ValueError,
        match="^router_ais_perf_model= requires router_mode='kv_router'$",
    ):
        replay(router_ais_perf_model=_CONFIG)
    with pytest.raises(
        ValueError,
        match="^router_ais_perf_model= requires router_config with "
        "router_prefill_load_model='ais'$",
    ):
        replay(router_ais_perf_model=_CONFIG, router_mode="kv_router")

    assert native_calls == []


@_BOTH_FUNCTIONS
def test_router_keyword_error_names_keyword(replay, native_calls):
    missing_entry = (
        "ais_perf_model: no entry for worker type 'aggregated'; it has: prefill"
    )
    with pytest.raises(ValueError) as from_shared_module:
        router_config([_PREFILL_ENTRY], "aggregated")

    with pytest.raises(ValueError) as raised:
        replay(router_ais_perf_model=[_PREFILL_ENTRY], **_kv_router())

    assert type(raised.value) is ValueError
    assert str(raised.value) == f"router_ais_perf_model: {missing_entry}"
    cause = raised.value.__cause__
    assert type(cause) is type(from_shared_module.value)
    assert str(cause) == str(from_shared_module.value) == missing_entry
    assert native_calls == []


@pytest.mark.parametrize(
    ("written", "message"),
    [
        pytest.param(
            {**_CONFIG, "not_a_field": 1},
            f"ais_perf_model: {_UNKNOWN_KEY_ERROR}",
            id="unknown key",
        ),
        pytest.param(
            {"backend": "vllm", "system": "h200_sxm"},
            "ais_perf_model requires model",
            id="missing model",
        ),
    ],
)
@pytest.mark.parametrize("disaggregated", [False, True], ids=["agg", "disagg"])
@_BOTH_FUNCTIONS
def test_router_keyword_config_error_names_keyword(
    replay, disaggregated, written, message, native_calls
):
    engine_args = _disaggregated_engine_args() if disaggregated else {}

    with pytest.raises(ValueError) as raised:
        replay(router_ais_perf_model=written, **engine_args, **_kv_router())

    # The config check raises a plain ValueError, not the class the loader
    # uses; the keyword is named for both.
    assert type(raised.value) is ValueError
    assert str(raised.value) == f"router_ais_perf_model: {message}"
    assert type(raised.value.__cause__) is ValueError
    assert str(raised.value.__cause__) == message
    assert native_calls == []


@_BOTH_FUNCTIONS
def test_engine_keyword_missing_role_entry_error(replay, native_calls):
    with pytest.raises(
        ValueError,
        match="ais_perf_model: no entry for worker type 'aggregated'; it has: "
        "prefill, decode",
    ):
        replay(ais_perf_model=[_PREFILL_ENTRY, _DECODE_ENTRY])
    with pytest.raises(
        ValueError,
        match="ais_perf_model: no entry for worker type 'prefill'; it has: aggregated",
    ):
        replay(ais_perf_model=[_AGGREGATED_ENTRY], **_disaggregated_engine_args())
    with pytest.raises(
        ValueError,
        match="ais_perf_model: no entry for worker type 'decode'; it has: prefill",
    ):
        replay(ais_perf_model=[_PREFILL_ENTRY], **_disaggregated_engine_args())

    assert native_calls == []


@_BOTH_FUNCTIONS
def test_engine_keyword_unknown_key_error(replay, native_calls):
    with pytest.raises(ValueError) as raised:
        replay(ais_perf_model={**_CONFIG, "not_a_field": 1})

    assert type(raised.value) is ValueError
    assert str(raised.value) == f"ais_perf_model: {_UNKNOWN_KEY_ERROR}"
    assert type(raised.value.__cause__) is TypeError
    assert str(raised.value.__cause__) == _UNKNOWN_KEY_ERROR
    assert native_calls == []


@pytest.mark.parametrize(
    ("written", "message"),
    [
        pytest.param(
            {"backend": "vllm", "system": "h200_sxm"},
            "ais_perf_model requires model",
            id="missing model",
        ),
        pytest.param(
            {**_CONFIG, "system": ""},
            "ais_perf_model requires system",
            id="empty system",
        ),
    ],
)
@_BOTH_FUNCTIONS
def test_engine_keyword_identity_value_error(replay, written, message, native_calls):
    with pytest.raises(ValueError) as raised:
        replay(ais_perf_model=written)

    assert type(raised.value) is ValueError
    assert str(raised.value) == message
    assert native_calls == []


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        pytest.param(
            {"not_a_field": 1},
            f"ais_perf_model: {_UNKNOWN_KEY_ERROR}",
            id="unknown key",
        ),
        pytest.param(
            {"system": ""},
            "ais_perf_model requires system",
            id="empty system",
        ),
    ],
)
@pytest.mark.parametrize("wrong_role", ["prefill", "decode"])
@_BOTH_FUNCTIONS
def test_engine_keyword_config_error_in_one_disaggregated_entry(
    replay, wrong_role, changed, message, native_calls
):
    entries = {"prefill": _PREFILL_ENTRY, "decode": _DECODE_ENTRY}
    entries[wrong_role] = {**entries[wrong_role], **changed}

    with pytest.raises(ValueError) as raised:
        replay(
            ais_perf_model=list(entries.values()),
            **_disaggregated_engine_args(),
        )

    # Each worker type's entry is checked, not only the first one.
    assert type(raised.value) is ValueError
    assert str(raised.value) == message
    assert native_calls == []


@_BOTH_FUNCTIONS
def test_engine_keyword_passes_config_on_as_resolved(replay, native_calls):
    written = {**_CONFIG, "backend_version": "0.11.0", "tp": 2}
    as_resolved = {**written, "worker_type": "aggregated"}

    with pytest.raises(_NativeCallRecorded):
        replay(ais_perf_model=written)

    (call,) = native_calls
    assert call["extra_engine_args"] == {
        "engine": {
            "timing_model": {
                "type": "external",
                "provider": "ais",
                "config": as_resolved,
            }
        }
    }
    # The value is checked as the perf model config class reads it, but the
    # engine gets the keys it wrote, not that class's defaults.
    assert as_resolved != _normalized(as_resolved)
    assert written == {**_CONFIG, "backend_version": "0.11.0", "tp": 2}


@pytest.mark.parametrize(
    ("given", "not_given"), [("prefill", "decode"), ("decode", "prefill")]
)
@_BOTH_FUNCTIONS
def test_engine_keyword_with_engine_args_of_one_role(
    replay, given, not_given, native_calls
):
    engine_args = {f"{given}_engine_args": {"engine": {"worker_type": given}}}

    with pytest.raises(_NativeCallRecorded):
        replay(ais_perf_model=_CONFIG, **engine_args)

    # Engine args are not created for the role that has none; the native
    # replay then reports that both must be given together.
    (call,) = native_calls
    assert call[f"{given}_engine_args"] == {
        "engine": {
            "worker_type": given,
            "timing_model": timing_model(_resolved(_CONFIG, given)),
        }
    }
    assert call[f"{not_given}_engine_args"] is None
    assert call["extra_engine_args"] is None


@_BOTH_FUNCTIONS
def test_error_comes_before_warning(replay, native_calls):
    perf_config = AisPerfConfig(config=_AGGREGATED_ENTRY)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="cannot be combined"):
            replay(
                ais_perf_config=perf_config,
                router_ais_perf_model=_CONFIG,
                **_kv_router(),
            )
        with pytest.raises(ValueError, match="already sets engine.timing_model"):
            replay(
                ais_perf_config=perf_config,
                ais_perf_model=_CONFIG,
                extra_engine_args={"engine": {"timing_model": {"type": "polynomial"}}},
                **_kv_router(),
            )

    assert records == []
    assert native_calls == []


@_BOTH_FUNCTIONS
def test_engine_keyword_does_not_configure_router(replay, native_calls):
    with pytest.raises(_NativeCallRecorded):
        replay(ais_perf_model=_CONFIG)

    (call,) = native_calls
    assert call["ais_perf_config"] is None
    assert call["router_config"] is None
    assert call["router_mode"] == "round_robin"


@_BOTH_FUNCTIONS
def test_router_keyword_does_not_change_engine_args(replay, native_calls):
    engine_args = {"engine": {"num_gpu_blocks": 1024}}
    disaggregated = _disaggregated_engine_args()

    with pytest.raises(_NativeCallRecorded):
        replay(
            router_ais_perf_model=_CONFIG, extra_engine_args=engine_args, **_kv_router()
        )
    with pytest.raises(_NativeCallRecorded):
        replay(router_ais_perf_model=_CONFIG, **disaggregated, **_kv_router())

    aggregated_call, disaggregated_call = native_calls
    assert aggregated_call["extra_engine_args"] is engine_args
    assert engine_args == {"engine": {"num_gpu_blocks": 1024}}
    assert aggregated_call["prefill_engine_args"] is None
    assert aggregated_call["decode_engine_args"] is None
    assert disaggregated_call["extra_engine_args"] is None
    for name, value in disaggregated.items():
        assert disaggregated_call[name] is value
    assert disaggregated == _disaggregated_engine_args()


@_BOTH_FUNCTIONS
def test_same_list_for_engines_and_router(replay, native_calls):
    written = [_PREFILL_ENTRY, _DECODE_ENTRY]

    with pytest.raises(_NativeCallRecorded):
        replay(
            ais_perf_model=written,
            router_ais_perf_model=written,
            **_disaggregated_engine_args(),
            **_kv_router(),
        )

    (call,) = native_calls
    assert call["ais_perf_config"].to_dict() == _normalized(_PREFILL_ENTRY)
    for worker_type, entry in (("prefill", _PREFILL_ENTRY), ("decode", _DECODE_ENTRY)):
        assert call[f"{worker_type}_engine_args"]["engine"]["timing_model"] == (
            timing_model(entry)
        )
    assert written == [_PREFILL_ENTRY, _DECODE_ENTRY]


@_BOTH_FUNCTIONS
def test_relative_file_starts_at_working_directory(
    replay, native_calls, tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    other_dir = tmp_path / "other"
    for directory in (run_dir, other_dir):
        directory.mkdir()
    (run_dir / "cfg.yaml").write_text(json.dumps(_CONFIG))

    monkeypatch.chdir(run_dir)
    with pytest.raises(_NativeCallRecorded):
        replay(router_ais_perf_model="cfg.yaml", **_kv_router())
    with pytest.raises(_NativeCallRecorded):
        replay(ais_perf_model="cfg.yaml")
    monkeypatch.chdir(other_dir)
    missing = re.escape(f"cannot read {other_dir / 'cfg.yaml'}")
    with pytest.raises(ValueError, match=missing):
        replay(router_ais_perf_model="cfg.yaml", **_kv_router())
    with pytest.raises(ValueError, match=missing):
        replay(ais_perf_model="cfg.yaml")

    router_call, engine_call = native_calls
    assert router_call["ais_perf_config"].to_dict() == _normalized(_AGGREGATED_ENTRY)
    assert engine_call["extra_engine_args"] == {
        "engine": {"timing_model": timing_model(_AGGREGATED_ENTRY)}
    }


@_BOTH_FUNCTIONS
def test_relative_paths_inside_file_start_at_file_folder(
    replay, native_calls, tmp_path, monkeypatch
):
    conf = tmp_path / "conf"
    run_dir = tmp_path / "run"
    for directory in (conf / "sys", run_dir):
        directory.mkdir(parents=True)
    written_file = conf / "cfg.yaml"
    written_file.write_text(json.dumps({**_CONFIG, "systems_paths": ["sys"]}))
    monkeypatch.chdir(run_dir)

    with pytest.raises(_NativeCallRecorded):
        replay(router_ais_perf_model=str(written_file), **_kv_router())
    with pytest.raises(_NativeCallRecorded):
        replay(ais_perf_model=str(written_file))

    router_call, engine_call = native_calls
    assert router_call["ais_perf_config"].to_dict()["systems_paths"] == [
        str(conf / "sys")
    ]
    engine_config = engine_call["extra_engine_args"]["engine"]["timing_model"]["config"]
    assert engine_config["systems_paths"] == [str(conf / "sys")]


@_BOTH_FUNCTIONS
def test_relative_paths_in_mapping_start_at_working_directory(
    replay, native_calls, tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    (run_dir / "sys").mkdir(parents=True)
    written = {**_CONFIG, "systems_paths": ["sys"]}
    monkeypatch.chdir(run_dir)

    with pytest.raises(_NativeCallRecorded):
        replay(router_ais_perf_model=written, **_kv_router())
    with pytest.raises(_NativeCallRecorded):
        replay(ais_perf_model=written)

    router_call, engine_call = native_calls
    assert router_call["ais_perf_config"].to_dict()["systems_paths"] == [
        str(run_dir / "sys")
    ]
    engine_config = engine_call["extra_engine_args"]["engine"]["timing_model"]["config"]
    assert engine_config["systems_paths"] == [str(run_dir / "sys")]
    assert written == {**_CONFIG, "systems_paths": ["sys"]}


@_BOTH_FUNCTIONS
def test_old_keyword_warning_names_caller(replay, native_calls):
    perf_config = AisPerfConfig(config=_AGGREGATED_ENTRY)

    with pytest.warns(FutureWarning) as records:
        with pytest.raises(_NativeCallRecorded):
            replay(ais_perf_config=perf_config, **_kv_router())

    (record,) = records
    assert str(record.message) == _OLD_KEYWORD_WARNING
    assert record.filename == __file__


@_BOTH_FUNCTIONS
def test_new_keywords_emit_no_warning(replay, native_calls):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(_NativeCallRecorded):
            replay(router_ais_perf_model=_CONFIG, **_kv_router())
        with pytest.raises(_NativeCallRecorded):
            replay(ais_perf_model=_CONFIG)

    assert len(native_calls) == 2
