# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Dynamo SGLang processor replay adapter.

These run without a frontend-crates checkout. They check that the adapter
drives the production post-processor and reports its rows in the harness format.
"""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("sglang", reason="SGLang is required for the SGLang chat processor")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynamo_sglang_rows import stream_rows  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]

TOOLS = [
    {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]


def _replay(chunks, **kwargs):
    return stream_rows(
        reasoning_parser_name="qwen3",
        tool_call_parser_name="qwen25",
        chunks=chunks,
        tools=TOOLS,
        **kwargs,
    )


def test_reasoning_and_text_rows_follow_harness_order():
    rows = _replay(["<think>plan</think>Sure. "])
    assert rows == [
        [{"kind": "reasoning", "text": "plan"}, {"kind": "text", "text": "Sure. "}]
    ]


def test_split_tool_call_is_reported_once_with_complete_arguments():
    # Which row carries the call is processor policy; the replay must report
    # exactly what the processor emitted, once, with the full arguments.
    rows = _replay(
        [
            '<tool_call>\n{"name": "get_weather", ',
            '"arguments": {"city": "Paris"}}\n</tool_call>',
        ]
    )
    assert len(rows) == 2
    calls = [delta for row in rows for delta in row if delta["kind"] == "tool_call"]
    assert [(call["name"], json.loads(call["arguments"])) for call in calls] == [
        ("get_weather", {"city": "Paris"})
    ]


def test_terminal_step_adds_one_empty_finishing_row():
    rows = _replay(["<think>a</think>hello"], terminal_step=True)
    assert rows == [
        [{"kind": "reasoning", "text": "a"}, {"kind": "text", "text": "hello"}],
        [],
    ]
