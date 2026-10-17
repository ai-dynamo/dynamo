# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tests/frontend/test_tool_calling_sglang.py, ported to the component harness.

The original is 1036 lines, of which roughly 320 are process management: two
ManagedProcess subclasses, a runtime-services fixture, straggler cleanup, log
directories, and a topology parametrisation. All of that exists to get a
frontend listening on a port. It can only run where the test process is allowed
to spawn Dynamo.

Here that is the harness's job. What remains is the part that was always about
the tool-calling contract: the tool schemas, the streaming assertions, and the
multi-turn conversations. The same file runs against a container the test
deployed or one somebody else is already running.

Two behaviours are asserted through the component rather than the wire:
reassembling streamed ``tool_calls`` deltas is protocol detail and lives in
``Frontend.stream_chat``; and ``tool_choice: "required"`` is guarded by
``dynamo.require(Capability.CONSTRAINED_DECODING)`` instead of being assumed --
on a backend with no grammar engine that request returns HTTP 200 and is
silently ignored.

The tool schemas below are copied verbatim from the original.
"""

import json
from typing import Any, Dict, List

import pytest

jsonschema = pytest.importorskip("jsonschema")
from dynamo_harness import Capability  # noqa: E402
from jsonschema import Draft7Validator  # noqa: E402

pytestmark = [pytest.mark.e2e]


TOOLS_WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city"],
                "additionalProperties": True,
            },
        },
    }
]

TOOLS_SEARCH = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": True,
            },
        },
    }
]

TOOLS_CALCULATOR = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
                "additionalProperties": True,
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Assertion helpers. These are about the tool-calling contract, not transport,
# so they stay test-side rather than moving into the harness.
# ---------------------------------------------------------------------------
def tool_schema_map(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}


def parse_and_validate_tool_call(
    call: Dict[str, Any],
    schema_by_name: Dict[str, Dict[str, Any]],
    *,
    expected_name: str = None,
) -> Dict[str, Any]:
    assert call["type"] == "function", f"unexpected tool type: {call['type']!r}"
    assert call["id"], "tool call id must be non-empty"
    name = call["function"]["name"]
    assert name, "tool call function name must be non-empty"
    if expected_name is not None:
        assert name == expected_name, f"expected {expected_name!r}, got {name!r}"
    assert name in schema_by_name, f"unknown tool name {name!r}"

    raw = call["function"]["arguments"]
    assert raw, "tool call arguments must be non-empty"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"arguments are not valid JSON: {raw!r}") from exc
    assert isinstance(
        args, dict
    ), f"arguments must decode to an object, got {type(args)}"

    errors = sorted(
        Draft7Validator(schema_by_name[name]).iter_errors(args),
        key=lambda err: list(err.path),
    )
    if errors:
        rendered = "; ".join(f"path={list(e.path)} message={e.message}" for e in errors)
        raise AssertionError(f"arguments failed schema validation: {rendered}")
    return args


def assert_finish_reason(result, allowed: set) -> None:
    assert (
        result.finish_reason in allowed
    ), f"unexpected finish_reason={result.finish_reason!r}, allowed={sorted(allowed)}"


# ---------------------------------------------------------------------------
# Protocol / contract
# ---------------------------------------------------------------------------
def test_stream_has_required_chunk_shape(dynamo):
    """Every streamed chunk carries the OpenAI envelope fields."""
    result = dynamo.frontend.stream_chat(
        [{"role": "user", "content": "What's the weather in Berlin?"}],
        tools=TOOLS_WEATHER,
        max_tokens=256,
        temperature=0,
        seed=0,
    )
    assert result.chunks > 0
    saw_finish = False
    for chunk in result.raw_chunks:
        assert chunk.get("id")
        assert chunk.get("object") == "chat.completion.chunk"
        assert chunk.get("created", 0) > 0
        assert len(chunk.get("choices") or []) >= 1
        for choice in chunk["choices"]:
            assert choice.get("index") == 0
            if choice.get("finish_reason") is not None:
                saw_finish = True
                assert choice["finish_reason"] in {"stop", "tool_calls", "length"}
    assert saw_finish, "stream never emitted a finish_reason"


def test_single_tool_call_schema_valid(dynamo):
    dynamo.require(Capability.TOOL_CALLING)
    result = dynamo.frontend.stream_chat(
        [{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=TOOLS_WEATHER,
        temperature=0,
        seed=0,
    )
    assert_finish_reason(result, {"tool_calls"})
    assert len(result.tool_calls) >= 1
    args = parse_and_validate_tool_call(
        result.tool_calls[0],
        tool_schema_map(TOOLS_WEATHER),
        expected_name="get_weather",
    )
    assert isinstance(args.get("city"), str) and args["city"]


def test_tool_choice_required_forces_a_tool_call(dynamo):
    """Only meaningful where a grammar backend enforces it.

    Without constrained decoding this request is accepted with HTTP 200 and
    ignored, so the requirement is declared rather than assumed.
    """
    dynamo.require(Capability.CONSTRAINED_DECODING)
    result = dynamo.frontend.stream_chat(
        [{"role": "user", "content": "Hello there."}],
        tools=TOOLS_WEATHER,
        tool_choice="required",
        temperature=0,
        seed=0,
    )
    assert_finish_reason(result, {"tool_calls"})
    assert len(result.tool_calls) >= 1
    # The prompt does not warrant a tool call, so a small model may invent
    # values for optional fields. Assert the call is well formed and required
    # fields are present; do not enforce the full schema.
    schema = tool_schema_map(TOOLS_WEATHER)
    for call in result.tool_calls:
        assert call["type"] == "function" and call["id"]
        name = call["function"]["name"]
        assert name in schema, f"unknown tool name {name!r}"
        args = json.loads(call["function"]["arguments"])
        assert isinstance(args, dict)
        for required in schema[name].get("required", []):
            assert required in args, f"{name} missing required field {required!r}"


def test_tool_choice_none_suppresses_tool_calls(dynamo):
    dynamo.require(Capability.TOOL_CALLING)
    result = dynamo.frontend.stream_chat(
        [{"role": "user", "content": "What's the weather in Paris?"}],
        tools=TOOLS_WEATHER,
        tool_choice="none",
        temperature=0,
        seed=0,
    )
    assert_finish_reason(result, {"stop"})
    assert result.tool_calls == []
    assert result.content.strip()


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_named_tool_choice_forces_specific_function(dynamo):
    """Known model flake: a misspelled enum value ("celcius") fails schema
    validation at temperature 0. Reruns absorb most occurrences."""
    dynamo.require(Capability.CONSTRAINED_DECODING)
    result = dynamo.frontend.stream_chat(
        [{"role": "user", "content": "What's the weather in Paris?"}],
        tools=TOOLS_WEATHER,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert_finish_reason(result, {"tool_calls"})
    assert len(result.tool_calls) >= 1
    schema = tool_schema_map(TOOLS_WEATHER)
    for call in result.tool_calls:
        parse_and_validate_tool_call(call, schema, expected_name="get_weather")


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_parallel_multi_tool_request_includes_all_expected_tools(dynamo):
    dynamo.require(Capability.TOOL_CALLING)
    tools = TOOLS_WEATHER + TOOLS_SEARCH + TOOLS_CALCULATOR
    result = dynamo.frontend.stream_chat(
        [
            {
                "role": "user",
                "content": (
                    "Do all three of these with tools: "
                    "1) weather in Paris, "
                    "2) search the web for latest Python release, "
                    "3) calculate 15 * 23 + 7."
                ),
            }
        ],
        tools=tools,
        parallel_tool_calls=True,
    )
    assert_finish_reason(result, {"tool_calls"})
    # Models sometimes batch only a subset and emit follow-ups in later turns,
    # so require at least 2 distinct tools rather than all 3.
    schemas = tool_schema_map(tools)
    names = set()
    for call in result.tool_calls:
        parse_and_validate_tool_call(call, schemas)
        names.add(call["function"]["name"])
    assert len(names) >= 2, f"expected at least 2 distinct tools, got {names}"


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_array_argument_schema_valid(dynamo):
    dynamo.require(Capability.CONSTRAINED_DECODING)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "send_emails",
                "description": "Send emails",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipients": {"type": "array", "items": {"type": "string"}},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["recipients", "subject", "body"],
                },
            },
        }
    ]
    result = dynamo.frontend.stream_chat(
        [
            {
                "role": "user",
                "content": (
                    "Send an email with subject 'Team Update' and body "
                    "'Meeting at 3pm' to alice@example.com, bob@example.com, "
                    "and carol@example.com."
                ),
            }
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "send_emails"}},
    )
    assert_finish_reason(result, {"tool_calls"})
    args = parse_and_validate_tool_call(
        result.tool_calls[0], tool_schema_map(tools), expected_name="send_emails"
    )
    assert isinstance(args["recipients"], list)
    assert len(args["recipients"]) >= 3


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_no_tools_is_plain_text(dynamo):
    result = dynamo.frontend.stream_chat(
        [{"role": "user", "content": "What is the capital of France?"}]
    )
    assert_finish_reason(result, {"stop"})
    assert result.tool_calls == []
    assert result.content.strip()


# ---------------------------------------------------------------------------
# Multi-turn contract
# ---------------------------------------------------------------------------
@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_tool_result_is_consumed_and_final_answer_is_text(dynamo):
    dynamo.require(Capability.TOOL_CALLING)
    schemas = tool_schema_map(TOOLS_WEATHER)
    question = {"role": "user", "content": "What is the weather in London?"}

    first = dynamo.frontend.stream_chat([question], tools=TOOLS_WEATHER)
    assert_finish_reason(first, {"tool_calls"})
    assert len(first.tool_calls) >= 1
    parse_and_validate_tool_call(
        first.tool_calls[0], schemas, expected_name="get_weather"
    )

    second = dynamo.frontend.stream_chat(
        [
            question,
            first.assistant_message(),
            {
                "role": "tool",
                "tool_call_id": first.tool_calls[0]["id"],
                "content": json.dumps(
                    {"temperature": 15, "unit": "celsius", "condition": "cloudy"}
                ),
            },
        ],
        tools=TOOLS_WEATHER,
    )
    assert_finish_reason(second, {"stop"})
    assert second.tool_calls == []
    assert second.content.strip()
    assert "15" in second.content or "cloud" in second.content.lower()


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_chained_tool_use_search_then_calculate(dynamo):
    dynamo.require(Capability.TOOL_CALLING)
    tools = TOOLS_SEARCH + TOOLS_CALCULATOR
    schemas = tool_schema_map(tools)
    messages = [
        {
            "role": "user",
            "content": (
                "Search the web for the population of Tokyo, "
                "then calculate what 10% of that number is."
            ),
        }
    ]

    step1 = dynamo.frontend.stream_chat(messages, tools=tools)
    assert_finish_reason(step1, {"tool_calls"})
    assert len(step1.tool_calls) >= 1
    parse_and_validate_tool_call(step1.tool_calls[0], schemas)

    messages.append(step1.assistant_message())
    messages.append(
        {
            "role": "tool",
            "tool_call_id": step1.tool_calls[0]["id"],
            "content": json.dumps(
                {"results": [{"title": "Tokyo population", "snippet": "13,960,000"}]}
            ),
        }
    )

    step2 = dynamo.frontend.stream_chat(messages, tools=tools)
    # Small models sometimes short-circuit and do the arithmetic in their
    # reasoning instead of chaining a second call. Accept either path.
    assert_finish_reason(step2, {"tool_calls", "stop"})
    if step2.finish_reason == "tool_calls":
        assert len(step2.tool_calls) >= 1
        args = parse_and_validate_tool_call(step2.tool_calls[0], schemas)
        assert step2.tool_calls[0]["function"]["name"] == "calculate"
        expression = args["expression"].replace(",", "")
        assert "13960000" in expression or "1396000" in expression

        messages.append(step2.assistant_message())
        messages.append(
            {
                "role": "tool",
                "tool_call_id": step2.tool_calls[0]["id"],
                "content": "1396000",
            }
        )
        step3 = dynamo.frontend.stream_chat(messages, tools=tools)
        assert_finish_reason(step3, {"stop"})
        assert step3.tool_calls == []
        assert "1396000" in step3.content.replace(",", "")
    else:
        assert step2.tool_calls == []
        assert "1396000" in step2.content.replace(",", "")


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_multiple_prior_tool_results_synthesize_to_text(dynamo):
    """Prior tool results are supplied by the test, so no tool call is needed."""
    result = dynamo.frontend.stream_chat(
        [
            {"role": "user", "content": "Get the weather in Tokyo and Paris."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Tokyo"}),
                        },
                    },
                    {
                        "id": "call_002",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": json.dumps(
                    {"temperature": 22, "unit": "celsius", "condition": "sunny"}
                ),
            },
            {
                "role": "tool",
                "tool_call_id": "call_002",
                "content": json.dumps(
                    {"temperature": 18, "unit": "celsius", "condition": "rainy"}
                ),
            },
        ],
        tools=TOOLS_WEATHER,
    )
    assert_finish_reason(result, {"stop"})
    assert result.tool_calls == []
    assert result.content.strip()
    lowered = result.content.lower()
    assert "tokyo" in lowered or "paris" in lowered
