# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import dynamo.replay.main as replay_main
from dynamo.mocker import MockEngineArgs
from dynamo.replay import run_trace_replay
from dynamo.replay.exgentic import prepare_trace

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _span(start, end, input_tokens, output_tokens, model, status=1):
    return {
        "start_time": start,
        "end_time": end,
        "type": "llm_call",
        "status": {"code": status},
        "attributes": {
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
        },
    }


def _write_trace(tmp_path):
    source = tmp_path / "trace.parquet"
    row = {
        "harness": "claude_code",
        "models": ["openai/Azure/gpt-4.1", "Azure/gpt-4.1"],
        "session_id": "session-1",
        "spans": [
            _span(
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:10Z",
                600,
                10,
                "openai/Azure/gpt-4.1",
            ),
            _span(
                "2026-01-01T00:00:03Z",
                "2026-01-01T00:00:04Z",
                0,
                0,
                "openai/Azure/gpt-4.1",
                2,
            ),
            _span(
                "2026-01-01T00:00:05Z",
                "2026-01-01T00:00:08Z",
                700,
                20,
                "Azure/gpt-4.1",
            ),
            _span(
                "2026-01-01T00:00:20Z",
                "2026-01-01T00:00:21Z",
                100,
                5,
                "Azure/gpt-4.1",
            ),
        ],
    }
    other = {
        "harness": "tool_calling",
        "models": ["gcp/gemini-3-pro-preview"],
        "session_id": "session-2",
        "spans": [
            _span(
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:01Z",
                256,
                8,
                "gcp/gemini-3-pro-preview",
            )
        ],
    }
    pq.write_table(pa.Table.from_pylist([row, other]), source)
    return source


def test_converts_selected_exgentic_parquet_to_session_mooncake(tmp_path):
    _write_trace(tmp_path)

    with prepare_trace(tmp_path, 512, harness="claude_code", model="gpt-4.1") as output:
        converted = [json.loads(line) for line in output.read_text().splitlines()]

    assert [item["input_length"] for item in converted] == [600, 700, 100]
    assert [item["output_length"] for item in converted] == [10, 20, 5]
    assert [item["hash_ids"] for item in converted] == [[1, 2], [1, 2], [3]]
    assert converted[0]["timestamp"] == 1_767_225_600_000
    assert converted[1]["delay"] == 0
    assert converted[2]["delay"] == 12_000


def test_run_trace_replay_accepts_exgentic_with_selectors(tmp_path):
    source = _write_trace(tmp_path)

    report = run_trace_replay(
        source,
        trace_format="exgentic",
        trace_harness="tool_calling",
        trace_model="gemini-3-pro-preview",
        trace_block_size=512,
        replay_concurrency=1,
        extra_engine_args=MockEngineArgs(block_size=64, speedup_ratio=1000.0),
    )

    assert report["num_requests"] == 1
    assert report["completed_requests"] == 1
    assert report["total_input_tokens"] == 256
    assert report["total_output_tokens"] == 8


def test_planner_receives_converted_exgentic_trace(tmp_path, monkeypatch):
    source = _write_trace(tmp_path)
    captured = {}

    def fake_planner_replay(**kwargs):
        trace_file = Path(kwargs["trace_file"])
        assert trace_file.is_file()
        captured["rows"] = [
            json.loads(line) for line in trace_file.read_text().splitlines()
        ]
        captured["path"] = trace_file
        return SimpleNamespace(
            trace_report={}, scaling_events=[], total_ticks=1, html_report_path=None
        )

    monkeypatch.setattr(replay_main, "_run_planner_replay", fake_planner_replay)
    monkeypatch.setattr(replay_main, "format_report_table", lambda report: "")
    monkeypatch.setattr(
        replay_main,
        "write_report_json",
        lambda report, output: tmp_path / "report.json",
    )

    assert (
        replay_main.main(
            [
                str(source),
                "--trace-format",
                "exgentic",
                "--trace-harness",
                "tool_calling",
                "--trace-model",
                "gemini-3-pro-preview",
                "--planner-config",
                "{}",
            ]
        )
        == 0
    )
    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["input_length"] == 256
    assert not captured["path"].exists()
