# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

import dynamo.replay.api as replay_api
import dynamo.replay.main as replay_main
from dynamo.llm import KvRouterConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_replay_api_forwards_policy_model_name(monkeypatch):
    calls = []

    def capture_trace(*args, **kwargs):
        calls.append(("trace", args, kwargs))
        return {}

    def capture_synthetic(*args, **kwargs):
        calls.append(("synthetic", args, kwargs))
        return {}

    monkeypatch.setattr(replay_api, "_run_mocker_trace_replay", capture_trace)
    monkeypatch.setattr(
        replay_api,
        "_run_mocker_synthetic_trace_replay",
        capture_synthetic,
    )

    replay_api.run_trace_replay("trace.jsonl", model_name="model-a")
    replay_api.run_synthetic_trace_replay(64, 8, 2, model_name="model-b")

    assert calls[0][2]["model_name"] == "model-a"
    assert calls[1][2]["model_name"] == "model-b"


def test_replay_api_and_cli_route_trace_file_lists(monkeypatch):
    api_calls = []

    def capture_api(*args, **kwargs):
        api_calls.append((args, kwargs))
        return {}

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

    cli_calls = []
    monkeypatch.setattr(
        replay_main,
        "run_trace_replay",
        lambda trace_files, **kwargs: cli_calls.append((trace_files, kwargs)) or {},
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


def test_router_config_from_json_validates_policy_file(tmp_path):
    policy_path = tmp_path / "invalid-policy.yaml"
    policy_path.write_text("not: [valid", encoding="utf-8")

    with pytest.raises(ValueError, match="failed to parse router policy config"):
        KvRouterConfig.from_json(json.dumps({"router_policy_config": str(policy_path)}))
