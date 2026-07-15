# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def coalescer():
    from dynamo.openai_backend.worker import _ToolCallCoalescer

    return _ToolCallCoalescer()


def _chunk(choices, usage=None):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": choices,
        "usage": usage,
    }


def _tool_delta(arguments, index=0, name=None, call_id=None):
    function = {"arguments": arguments}
    if name is not None:
        function["name"] = name
    tool_call = {"index": index, "function": function}
    if call_id is not None:
        tool_call["id"] = call_id
    return {
        "index": 0,
        "delta": {"tool_calls": [tool_call]},
        "finish_reason": None,
    }


def test_coalesces_tool_call_deltas_into_finish_chunk(coalescer):
    assert (
        coalescer.push(
            _chunk([_tool_delta('{"ci', name="get_weather", call_id="call_1")])
        )
        == []
    )
    assert coalescer.push(_chunk([_tool_delta('ty": "Paris"}')])) == []

    finish = _chunk([{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])
    output = coalescer.push(finish)

    assert len(output) == 1
    (choice,) = output[0]["choices"]
    assert choice["finish_reason"] == "tool_calls"
    (tool_call,) = choice["delta"]["tool_calls"]
    assert tool_call["id"] == "call_1"
    assert tool_call["function"]["name"] == "get_weather"
    assert tool_call["function"]["arguments"] == '{"city": "Paris"}'


def test_passes_through_terminal_usage_chunk_with_empty_choices(coalescer):
    coalescer.push(_chunk([_tool_delta("{}", name="get_weather")]))
    coalescer.push(_chunk([{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]))

    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    usage_chunk = _chunk([], usage=usage)

    assert coalescer.push(usage_chunk) == [usage_chunk]


def test_preserves_usage_attached_to_a_fully_buffered_delta_chunk(coalescer):
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    delta_with_usage = _chunk(
        [_tool_delta("{}", name="get_weather")],
        usage=usage,
    )

    output = coalescer.push(delta_with_usage)

    assert len(output) == 1
    assert output[0]["choices"] == []
    assert output[0]["usage"] == usage


def test_drops_fully_buffered_delta_chunks_without_usage(coalescer):
    assert coalescer.push(_chunk([_tool_delta("{}", name="get_weather")])) == []
