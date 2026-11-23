# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from responses_adapter import (  # noqa: E402
    build_responses_payload,
    parse_nonstream,
    parse_stream,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def function_item(arguments='{"city":"Rome"}', **updates):
    return {
        "id": "fc_item",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_public",
        "name": "lookup",
        "namespace": "weather",
        "arguments": arguments,
        **updates,
    }


def response(output, **updates):
    return {
        "id": "resp_1",
        "model": "served-model",
        "created_at": 123,
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 6,
            "total_tokens": 16,
            "output_tokens_details": {"reasoning_tokens": 2},
        },
        **updates,
    }


def event(kind, **values):
    return {"type": kind, **values}


def sse(events):
    for value in events:
        yield f"event: {value['type']}\r\n".encode()
        yield f"data: {json.dumps(value, ensure_ascii=False)}\r\n".encode()
        yield b"\r\n"


def call_events(arguments='{"city":"Rome"}'):
    initial = function_item(arguments="", status="in_progress")
    complete = function_item(arguments=arguments)
    return [
        event("response.output_item.added", output_index=0, item=initial),
        event(
            "response.function_call_arguments.delta",
            output_index=0,
            item_id="fc_item",
            delta=arguments[:8],
        ),
        event(
            "response.function_call_arguments.delta",
            output_index=0,
            item_id="fc_item",
            delta=arguments[8:],
        ),
        event(
            "response.function_call_arguments.done",
            output_index=0,
            item_id="fc_item",
            arguments=arguments,
        ),
        event("response.output_item.done", output_index=0, item=complete),
        event("response.completed", response=response([complete])),
    ]


def kinds(result):
    return {error["kind"] for error in result["parse_errors"]}


def test_request_translation_preserves_full_catalog_choice_and_history():
    payload = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "Use the tools."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "public",
                        "type": "function",
                        "namespace": "weather",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "public",
                "content": [
                    {"type": "text", "text": "result"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,abc",
                            "detail": "low",
                        },
                    },
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {"type": "object"},
                    "strict": False,
                },
            }
            for name in ("lookup", "delete")
        ],
        "tool_choice": {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "required",
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            },
        },
        "max_tokens": 200,
        "stream_options": {"include_usage": True},
    }
    original = copy.deepcopy(payload)
    result = build_responses_payload(payload)
    assert payload == original
    assert [tool["name"] for tool in result["tools"]] == ["lookup", "delete"]
    assert result["tools"][0]["strict"] is False
    assert result["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "function", "name": "lookup"}],
    }
    assert result["max_output_tokens"] == 200
    assert "stream_options" not in result and "max_tokens" not in result
    assert result["input"][1] == {
        "type": "function_call",
        "name": "lookup",
        "namespace": "weather",
        "call_id": "public",
        "arguments": "{}",
    }
    assert result["input"][2] == {
        "type": "function_call_output",
        "call_id": "public",
        "output": [
            {"type": "input_text", "text": "result"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,abc",
                "detail": "low",
            },
        ],
    }


@pytest.mark.parametrize("choice", ["none", "auto", "required", None])
def test_request_translation_keeps_non_named_choice(choice):
    result = build_responses_payload(
        {"messages": [], "tool_choice": choice, "max_completion_tokens": 32}
    )
    assert result["tool_choice"] == choice
    assert result["max_output_tokens"] == 32


def test_request_translation_named_choice_and_response_format():
    result = build_responses_payload(
        {
            "messages": [],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {"type": "object"},
                },
            },
        }
    )
    assert result["tool_choice"] == {"type": "function", "name": "lookup"}
    assert result["text"]["format"] == {
        "type": "json_schema",
        "name": "answer",
        "strict": True,
        "schema": {"type": "object"},
    }


def test_unary_keeps_public_call_identity_namespace_and_native_usage():
    raw = response([function_item()])
    result = parse_nonstream(raw)
    assert result["parse_errors"] == []
    assert result["finish_reason"] == "tool_calls"
    assert result["api"] == "responses" and result["response_status"] == "completed"
    call = result["tool_calls"][0]
    assert call["id"] == "call_public" and call["item_id"] == "fc_item"
    assert call["namespace"] == "weather"
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2
    assert result["raw_response"] is raw


def test_explicit_responses_budget_is_not_overwritten_by_chat_default():
    payload = build_responses_payload(
        {"messages": [], "max_tokens": 4096, "max_output_tokens": 16}
    )
    assert payload["max_output_tokens"] == 16


def test_argument_event_requires_item_identity_and_keeps_malformed_raw_data():
    events = call_events()
    del events[1]["item_id"]
    lines = [b"data: not-json\n", b"\n", *sse(events)]
    result = parse_stream(lines)
    assert "responses_shape" in kinds(result)
    assert result["raw_response"][0]["data"] == "not-json"


def test_stream_assembles_arguments_once_and_keeps_native_events():
    events = call_events()
    result = parse_stream(sse(events))
    assert result["parse_errors"] == []
    assert result["tool_calls"][0]["function"]["arguments"] == '{"city":"Rome"}'
    assert result["tool_calls"][0]["id"] == "call_public"
    assert result["chunk_count"] == len(events)
    assert result["raw_response"][-1]["data"] == events[-1]


def test_stream_matches_unicode_text_reasoning_and_multiline_sse_data():
    items = [
        {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Think ✓"}],
        },
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Zürich"}],
        },
    ]
    events = [
        event("response.output_item.added", output_index=i, item=item)
        for i, item in enumerate(items)
    ] + [
        event(
            "response.reasoning_summary_text.delta",
            output_index=0,
            summary_index=0,
            delta="Think ✓",
        ),
        event(
            "response.output_text.delta",
            output_index=1,
            content_index=0,
            delta="Zürich",
        ),
        event(
            "response.output_text.done", output_index=1, content_index=0, text="Zürich"
        ),
        *[
            event("response.output_item.done", output_index=i, item=item)
            for i, item in enumerate(items)
        ],
        event("response.completed", response=response(items)),
    ]
    lines = list(sse(events[:-1]))
    lines += [
        b"event: response.completed\n",
        b'data: {"type":"response.completed",\n',
        f'data: "response":{json.dumps(events[-1]["response"])}}}\n'.encode(),
        b"\n",
    ]
    result = parse_stream(lines)
    assert not result["parse_errors"]
    assert result["content"] == "Zürich" and result["reasoning_content"] == "Think ✓"


@pytest.mark.parametrize("terminal_event", ["error", "response.failed"])
def test_error_after_completed_call_cannot_be_hidden_by_success(terminal_event):
    events = call_events()
    values = {"error": {"message": "outside allowed_tools"}}
    if terminal_event == "response.failed":
        values = {"response": response([function_item()], status="failed", **values)}
    events.append(event(terminal_event, **values))
    assert "responses_error" in kinds(parse_stream(sse(events)))


def test_eof_after_valid_call_is_not_success():
    result = parse_stream(sse(call_events()[:-1]))
    assert result["finish_reason"] is None
    assert "responses_stream_incomplete" in kinds(result)
    assert result["tool_calls"][0]["function"]["arguments"] == '{"city":"Rome"}'


@pytest.mark.parametrize("where", ["arguments_done", "item_done", "terminal"])
def test_stream_rejects_disagreement_between_argument_representations(where):
    events = call_events()
    if where == "arguments_done":
        events[3]["arguments"] = "{}"
    elif where == "item_done":
        events[4]["item"]["arguments"] = "{}"
    else:
        events[-1]["response"]["output"][0] = function_item(arguments="{}")
    assert "responses_arguments_mismatch" in kinds(parse_stream(sse(events)))


@pytest.mark.parametrize("field", ["id", "call_id", "name", "namespace"])
def test_stream_rejects_changed_function_identity(field):
    events = call_events()
    events[4]["item"][field] = "changed"
    assert "responses_item_mismatch" in kinds(parse_stream(sse(events)))


def test_stream_accepts_identity_arriving_after_argument_fragments():
    events = call_events()
    initial = events.pop(0)
    events.insert(2, initial)
    assert parse_stream(sse(events))["parse_errors"] == []


def test_interleaved_functions_keep_output_index_call_id_and_namespace():
    first = call_events('{"city":"Rome"}')
    second = call_events('{"city":"Lima"}')
    for value in second[:-1]:
        value["output_index"] = 1
        if "item_id" in value:
            value["item_id"] = "fc_other"
        if "item" in value:
            value["item"].update(
                id="fc_other", call_id="call_other", namespace="billing"
            )
    second_item = second[4]["item"]
    events = [
        second[0],
        first[0],
        first[1],
        second[1],
        second[2],
        first[2],
        *second[3:5],
        *first[3:5],
        event("response.completed", response=response([first[4]["item"], second_item])),
    ]
    result = parse_stream(sse(events))
    assert result["parse_errors"] == []
    assert [call["id"] for call in result["tool_calls"]] == [
        "call_public",
        "call_other",
    ]
    assert [call["namespace"] for call in result["tool_calls"]] == [
        "weather",
        "billing",
    ]
    assert [
        json.loads(call["function"]["arguments"]) for call in result["tool_calls"]
    ] == [
        {"city": "Rome"},
        {"city": "Lima"},
    ]


def test_incomplete_budget_preserves_partial_call_and_native_status():
    partial = function_item(arguments='{"city":"Ro', status="incomplete")
    raw = response(
        [partial],
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
    )
    result = parse_nonstream(raw)
    assert (
        result["finish_reason"] == "length"
        and result["response_status"] == "incomplete"
    )
    assert result["parse_errors"] == []
    stream = [
        event("response.output_item.added", output_index=0, item=partial),
        event("response.incomplete", response=raw),
    ]
    assert parse_stream(sse(stream))["parse_errors"] == []


def test_sse_type_mismatch_and_snapshot_only_calls_are_visible():
    lines = [
        b"event: response.failed\n",
        f"data: {json.dumps(call_events()[-1])}\n".encode(),
        b"\n",
    ]
    result = parse_stream(lines)
    assert {"responses_event_mismatch", "responses_stream_incomplete"} <= kinds(result)


def test_duplicate_done_event_and_text_disagreement_are_visible():
    events = call_events()
    events.insert(4, copy.deepcopy(events[3]))
    assert "responses_duplicate_event" in kinds(parse_stream(sse(events)))
    events = [
        event(
            "response.output_text.delta",
            output_index=0,
            content_index=0,
            delta="before",
        ),
        event(
            "response.completed",
            response=response(
                [
                    {
                        "id": "msg",
                        "type": "message",
                        "content": [{"type": "output_text", "text": "after"}],
                    }
                ]
            ),
        ),
    ]
    assert "responses_event_mismatch" in kinds(parse_stream(sse(events)))


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"status": "completed", "output": "bad"},
        response([function_item(arguments={})]),
    ],
)
def test_invalid_response_shape_is_visible(raw):
    assert "responses_shape" in kinds(parse_nonstream(raw))


def test_dynamo_reasoning_and_empty_call_without_argument_deltas():
    reasoning = {
        "id": "rs_1",
        "type": "reasoning",
        "status": "completed",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "Checking the clock."}],
    }
    initial_reasoning = {**reasoning, "status": "in_progress", "content": []}
    complete = function_item(arguments="{}")
    events = [
        event("response.created", response=response([], status="in_progress")),
        event("response.in_progress", response=response([], status="in_progress")),
        event("response.output_item.added", output_index=0, item=initial_reasoning),
        event(
            "response.content_part.added",
            output_index=0,
            item_id="rs_1",
            content_index=0,
            part={"type": "reasoning_text", "text": ""},
        ),
        event(
            "response.reasoning_text.delta",
            output_index=0,
            item_id="rs_1",
            content_index=0,
            delta="Checking the clock.",
        ),
        event(
            "response.reasoning_text.done",
            output_index=0,
            item_id="rs_1",
            content_index=0,
            text="Checking the clock.",
        ),
        event(
            "response.content_part.done",
            output_index=0,
            item_id="rs_1",
            content_index=0,
            part=reasoning["content"][0],
        ),
        event("response.output_item.done", output_index=0, item=reasoning),
        event(
            "response.output_item.added",
            output_index=1,
            item=function_item(arguments="", status="in_progress"),
        ),
        event(
            "response.function_call_arguments.done",
            output_index=1,
            item_id="fc_item",
            name="lookup",
            arguments="{}",
        ),
        event("response.output_item.done", output_index=1, item=complete),
        event("response.completed", response=response([reasoning, complete])),
    ]
    result = parse_stream(sse(events))
    assert result["parse_errors"] == []
    assert result["reasoning_content"] == "Checking the clock."
    assert result["tool_calls"][0]["function"]["arguments"] == "{}"


def test_summary_and_raw_reasoning_have_independent_fragment_channels():
    reasoning = {
        "id": "rs_1",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "brief"}],
        "content": [{"type": "reasoning_text", "text": "long thought"}],
    }
    events = [
        event(
            "response.output_item.added",
            output_index=0,
            item={**reasoning, "content": [], "summary": []},
        ),
        event(
            "response.reasoning_text.delta",
            output_index=0,
            content_index=0,
            delta="long thought",
        ),
        event(
            "response.reasoning_summary_text.delta",
            output_index=0,
            summary_index=0,
            delta="brief",
        ),
        event("response.output_item.done", output_index=0, item=reasoning),
        event("response.completed", response=response([reasoning])),
    ]
    result = parse_stream(sse(events))
    assert result["parse_errors"] == []
    assert result["reasoning_content"] == "long thought"


@pytest.mark.parametrize("with_items", [False, True])
def test_terminal_text_cannot_replace_missing_streamed_text(with_items):
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "READY"}],
    }
    events = []
    if with_items:
        events += [
            event(
                "response.output_item.added",
                output_index=0,
                item={**message, "content": []},
            ),
            event("response.output_item.done", output_index=0, item=message),
        ]
    events.append(event("response.completed", response=response([message])))
    assert "responses_stream_incomplete" in kinds(parse_stream(sse(events)))


@pytest.mark.parametrize(
    "part",
    [
        {"type": "output_text", "text": ""},
        {"type": "refusal", "refusal": "Cannot help"},
    ],
)
def test_empty_text_and_refusal_do_not_require_nonempty_text_delta(part):
    message = {"id": "msg_1", "type": "message", "content": [part]}
    events = [event("response.completed", response=response([message]))]
    assert parse_stream(sse(events))["parse_errors"] == []


def test_negative_output_index_and_events_after_terminal_cannot_be_hidden():
    events = call_events()
    events[1]["output_index"] = -1
    assert "responses_shape" in kinds(parse_stream(sse(events)))
    events = call_events() + [
        event(
            "response.output_text.delta", output_index=0, content_index=0, delta="late"
        )
    ]
    assert "responses_event_mismatch" in kinds(parse_stream(sse(events)))


def test_arguments_done_name_is_checked_and_reasoning_effort_is_translated():
    events = call_events()
    events[3]["name"] = "other"
    assert "responses_item_mismatch" in kinds(parse_stream(sse(events)))
    translated = build_responses_payload(
        {"messages": [], "reasoning_effort": "high", "reasoning": {"summary": "auto"}}
    )
    assert translated["reasoning"] == {"effort": "high", "summary": "auto"}
    assert "reasoning_effort" not in translated
