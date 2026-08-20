# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecation contracts for Dynamo's engine-only replay compatibility path."""

import warnings
from types import SimpleNamespace

import pytest

from dynamo.replay import api as replay_api
from dynamo.replay import deprecation
from dynamo.replay import main as replay_main
from dynamo.replay import ReplayReport

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


@pytest.fixture(autouse=True)
def _fresh_warning_state():
    deprecation._warn_engine_only_replay_once.cache_clear()
    try:
        yield
    finally:
        deprecation._warn_engine_only_replay_once.cache_clear()


def _native_report():
    return SimpleNamespace(summary={}, per_request=None, coverage={})


def _stub_cli_dependencies(monkeypatch, seen: dict) -> None:
    monkeypatch.setattr(replay_main, "_load_engine_args", lambda value: value)
    monkeypatch.setattr(replay_main, "_load_router_config", lambda config, policy: None)
    monkeypatch.setattr(replay_main, "_load_aic_perf_config", lambda args: None)
    monkeypatch.setattr(
        replay_main,
        "run_synthetic_trace_replay",
        lambda *args, **kwargs: ReplayReport(
            summary={}, per_request=None, coverage={}, planner=None
        ),
    )
    monkeypatch.setattr(
        replay_main,
        "write_report_json",
        lambda payload, path: seen.setdefault("report", path or "report.json"),
    )
    monkeypatch.setattr(replay_main, "format_report_table", lambda summary: "table")


def test_engine_only_cli_warns_once_at_the_caller(monkeypatch) -> None:
    seen: dict = {}
    _stub_cli_dependencies(monkeypatch, seen)
    argv = [
        "--input-tokens",
        "8",
        "--output-tokens",
        "4",
        "--request-count",
        "1",
        "--replay-concurrency",
        "1",
    ]

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert replay_main.main(argv) == 0
        assert replay_main.main(argv) == 0

    assert len(captured) == 1
    warning = captured[0]
    assert warning.category is FutureWarning
    assert "`python -m dynamo.replay`" in str(warning.message)
    assert "`python -m aisimulate.replay with the same base arguments`" in str(
        warning.message
    )
    assert "Dynamo 1.6.0" in str(warning.message)
    assert "Dynamo 1.5.0 retains this compatibility path" in str(warning.message)
    assert warning.filename == __file__


def test_engine_only_python_api_warns_with_sdk_replacement(monkeypatch) -> None:
    monkeypatch.setattr(
        replay_api,
        "_run_mocker_synthetic_trace_replay",
        lambda *args, **kwargs: _native_report(),
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        replay_api.run_synthetic_trace_replay(
            8,
            4,
            1,
            replay_concurrency=1,
        )

    assert len(captured) == 1
    warning = captured[0]
    assert "dynamo.replay.run_synthetic_trace_replay()" in str(warning.message)
    assert "aisimulate.EngineReplayRunnerFactory" in str(warning.message)
    assert warning.filename == __file__


def test_engine_only_trace_api_warns_with_sdk_replacement(monkeypatch) -> None:
    monkeypatch.setattr(
        replay_api,
        "_run_mocker_trace_replay",
        lambda *args, **kwargs: _native_report(),
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        replay_api.run_trace_replay(
            "trace.jsonl",
            replay_concurrency=1,
        )

    assert len(captured) == 1
    warning = captured[0]
    assert "dynamo.replay.run_trace_replay()" in str(warning.message)
    assert "aisimulate.EngineReplayRunnerFactory" in str(warning.message)
    assert warning.filename == __file__


@pytest.mark.parametrize(
    "overrides",
    [
        {"replay_mode": "online"},
        {"router_mode": "kv_router"},
        {"router_config": object()},
        {"aic_perf_config": object()},
        {"planner_config": object()},
        {"model_name": "test-model"},
    ],
)
def test_dynamo_owned_paths_do_not_warn(overrides) -> None:
    values = {
        "replay_mode": "offline",
        "router_mode": "round_robin",
        "router_config": None,
        "aic_perf_config": None,
        "planner_config": None,
        "model_name": None,
        **overrides,
    }

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert deprecation.uses_dynamo_integration(**values)

    assert captured == []
