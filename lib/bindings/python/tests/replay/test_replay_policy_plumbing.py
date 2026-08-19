# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from types import SimpleNamespace

import pytest

import dynamo.replay.api as replay_api
from dynamo.llm import KvRouterConfig
from dynamo.replay import ReplayReport

from .replay_utils import _require_aisimulate_distribution

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_replay_api_routes_trace_file_lists(monkeypatch):
    api_calls = []

    def capture_api(*args, **kwargs):
        api_calls.append((args, kwargs))
        return SimpleNamespace(summary={}, per_request=None, coverage={})

    monkeypatch.setattr(replay_api, "_run_mocker_trace_replay", capture_api)
    replay_api.run_trace_replay("mooncake.jsonl")
    replay_api.run_trace_replay(
        ["request-trace.0001.jsonl.gz", "request-trace.0002.jsonl.gz"],
        trace_format="dynamo",
    )

    assert api_calls[0][0][0] == ["mooncake.jsonl"]
    assert api_calls[0][1]["trace_block_size"] is None
    assert api_calls[1][0][0] == [
        "request-trace.0001.jsonl.gz",
        "request-trace.0002.jsonl.gz",
    ]
    assert api_calls[1][1]["trace_format"] == "dynamo"


@pytest.mark.planner
def test_replay_cli_routes_trace_file_lists(monkeypatch):
    _require_aisimulate_distribution()
    replay_main = importlib.import_module("dynamo.replay.main")
    cli_calls = []
    monkeypatch.setattr(
        replay_main,
        "run_trace_replay",
        lambda trace_files, **kwargs: cli_calls.append((trace_files, kwargs))
        or ReplayReport(summary={}, per_request=None, coverage={}, planner=None),
    )
    monkeypatch.setattr(replay_main, "format_report_table", lambda report: "")
    monkeypatch.setattr(
        replay_main, "write_report_json", lambda report, path: "report.json"
    )

    assert (
        replay_main.main(
            [
                "request-trace.0001.jsonl.gz",
                "request-trace.0002.jsonl.gz",
                "--trace-format",
                "dynamo",
            ]
        )
        == 0
    )
    assert cli_calls[0][0] == [
        "request-trace.0001.jsonl.gz",
        "request-trace.0002.jsonl.gz",
    ]
    assert cli_calls[0][1]["trace_block_size"] is None


def test_planner_replay_rejects_empty_dynamo_trace_list():
    with pytest.raises(
        ValueError,
        match="trace_format='dynamo' requires at least one trace file",
    ):
        replay_api.run_trace_replay(
            [],
            trace_format="dynamo",
            planner_config={"mode": "agg"},
        )


def test_router_config_from_json_validates_policy_file(tmp_path):
    policy_path = tmp_path / "invalid-policy.yaml"
    policy_path.write_text("not: [valid", encoding="utf-8")

    with pytest.raises(ValueError, match="failed to parse router policy config"):
        KvRouterConfig.from_json(json.dumps({"router_policy_config": str(policy_path)}))
