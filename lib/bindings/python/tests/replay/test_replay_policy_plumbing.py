# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

import dynamo.replay.api as replay_api
from dynamo.llm import KvRouterConfig

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


def test_replay_api_requires_target_model_for_weka():
    with pytest.raises(
        ValueError,
        match="agentic execution requires a configured target model",
    ):
        replay_api.run_trace_replay("published-weka", trace_format="weka")


def test_replay_api_reports_agentic_model_projection(monkeypatch):
    api_calls = []

    def capture_api(*args, **kwargs):
        api_calls.append((args, kwargs))
        return SimpleNamespace(
            summary={
                "completed_requests": 2,
                "agentic_graph": {
                    "source_models": ["source-a", "source-b"],
                },
            },
            per_request=None,
            coverage={},
        )

    monkeypatch.setattr(replay_api, "_run_mocker_trace_replay", capture_api)

    report = replay_api.run_trace_replay(
        "published-weka",
        trace_format="weka",
        execution_model=" target-model ",
    )

    assert api_calls[0][1]["execution_model"] == "target-model"
    assert report.summary["agentic_model_projection"] == {
        "policy": "project_to_configured_target",
        "source_models": ["source-a", "source-b"],
        "target_model": "target-model",
    }


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
