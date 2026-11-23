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
        return SimpleNamespace(
            summary={}, per_request=None, coverage={}, telemetry=None
        )

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


@pytest.mark.parametrize("basis", ["auto", "absolute", "relative"])
def test_replay_api_rejects_weka_timestamp_basis_for_other_formats(basis):
    with pytest.raises(ValueError, match="requires trace_format='weka'"):
        replay_api.run_trace_replay("unused.jsonl", weka_nested_timestamp_basis=basis)


@pytest.mark.parametrize(
    "option",
    [
        {"weka_client": "agentx"},
        {
            "weka_chat_template": {
                "request_tokens": 5,
                "user_tokens": 7,
                "assistant_tokens": 6,
                "tail_tokens": 2,
                "append_tail_tokens": 1,
            }
        },
    ],
)
def test_replay_api_rejects_weka_import_options_for_other_formats(option):
    (name,) = option
    with pytest.raises(ValueError, match=f"{name} requires trace_format='weka'"):
        replay_api.run_trace_replay("unused.jsonl", **option)


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ({"weka_client": "AgentX"}, "weka_client: "),
        ({"weka_chat_template": {"request_tokens": 5}}, "weka_chat_template: "),
    ],
)
def test_replay_api_names_an_invalid_weka_import_option(option, message):
    with pytest.raises(ValueError, match=message):
        replay_api.run_trace_replay(
            "unused.jsonl",
            trace_format="weka",
            execution_model="target-model",
            **option,
        )


def test_replay_api_reports_a_misordered_weka_chat_template_at_trace_load(tmp_path):
    # AISimulate checks the token ordering when it loads the trace, so the
    # error names its chat_template rather than weka_chat_template.
    trace = tmp_path / "weka.jsonl"
    request = {
        "t": 0.0,
        "type": "s",
        "model": "source-model",
        "in": 4,
        "out": 1,
        "api_time": 0.5,
        "hash_ids": [1],
    }
    play = {"id": "play", "block_size": 4, "hash_id_scope": "local"}
    trace.write_text(json.dumps({**play, "requests": [request]}) + "\n")
    with pytest.raises(
        Exception,
        match="chat_template needs append_tail_tokens <= tail_tokens <= request_tokens",
    ):
        replay_api.run_trace_replay(
            str(trace),
            trace_format="weka",
            execution_model="target-model",
            weka_chat_template={
                "request_tokens": 2,
                "user_tokens": 1,
                "assistant_tokens": 1,
                "tail_tokens": 5,
                "append_tail_tokens": 1,
            },
        )


@pytest.mark.parametrize(
    ("basis", "error"),
    [
        ("unknown", ValueError),
        ("Relative", ValueError),
        (True, TypeError),
        (1, TypeError),
    ],
)
def test_replay_api_rejects_invalid_weka_timestamp_basis(basis, error):
    with pytest.raises(error, match="weka_nested_timestamp_basis must be"):
        replay_api.run_trace_replay(
            "unused.jsonl",
            trace_format="weka",
            execution_model="target-model",
            weka_nested_timestamp_basis=basis,
        )


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
