#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit the qwen3_coder-format hybrid reasoning models declarative case profile.

Derived from gen_qwen3_6_profile.py. the target model uses the same qwen3_coder
``<parameter=NAME>value</parameter>`` XML markup (untyped, so each parser must
recover types from the JSON schema), but its chat template differs from
Qwen3.6 in one structural way: thinking is unconditionally on. The generation
prompt always ends with ``<|im_start|>assistant\\n<think>\\n`` and there is no
``enable_thinking``/``preserve_thinking`` kwarg, so the thinking-off and
drop-history families have no analogue here. Assistant turns always re-render
``<think>reasoning_content</think>``, which makes reasoning_content ingress a
first-class parity surface (family F9).
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = (
    Path(__file__).resolve().parents[1]
    / "custom/configs/case_profiles/qwen3_coder_xml.json"
)

# Qwen3.6 control tokens and tool markup. None of these may survive into
# visible content or reasoning once parsing is correct.
FORBIDDEN = [
    "<parameter=",
    "</parameter>",
    "</function>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_response>",
    "</tool_response>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
]

# This template family exposes NO thinking kwargs at all: thinking is
# always on and the generation prompt pre-seeds "<think>\n". Every case runs
# under the template default.
PRESETS = {
    "template_default": {},
}


def fn(name, desc, props, required, additional=None):
    params = {"type": "object", "properties": props, "required": required}
    if additional is not None:
        params["additionalProperties"] = additional
    return {
        "type": "function",
        "function": {"name": name, "description": desc, "parameters": params},
    }


S = {"type": "string"}
INTEGER = {"type": "integer"}
N = {"type": "number"}
B = {"type": "boolean"}


TOOLS = {
    "add_numbers": fn(
        "add_numbers", "Add two integers.", {"a": INTEGER, "b": INTEGER}, ["a", "b"]
    ),
    "add_numbers_2": fn(
        "add_numbers_2",
        "Add two integers (second registry entry).",
        {"a": INTEGER, "b": INTEGER},
        ["a", "b"],
    ),
    "multiply_numbers": fn(
        "multiply_numbers",
        "Multiply two integers.",
        {"a": INTEGER, "b": INTEGER},
        ["a", "b"],
    ),
    "get_weather": fn(
        "get_weather",
        "Get the current weather for a city.",
        {"city": S, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        ["city", "unit"],
    ),
    "get_weather_2": fn(
        "get_weather_2",
        "Get the current weather for a city (second registry entry).",
        {"city": S, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        ["city", "unit"],
    ),
    "set_threshold": fn(
        "set_threshold", "Set a floating point threshold.", {"value": N}, ["value"]
    ),
    "set_mode": fn(
        "set_mode", "Enable or disable a mode.", {"enabled": B}, ["enabled"]
    ),
    "set_coordinate": fn(
        "set_coordinate", "Set an x/y coordinate.", {"x": N, "y": N}, ["x", "y"]
    ),
    "record_scalars": fn(
        "record_scalars",
        "Record a mixed-type scalar tuple.",
        {"count": INTEGER, "ratio": N, "enabled": B, "label": S},
        ["count", "ratio", "enabled", "label"],
    ),
    "store_text": fn(
        "store_text", "Store a text value verbatim.", {"text": S}, ["text"]
    ),
    "submit_json": fn(
        "submit_json",
        "Submit a JSON document encoded as a string.",
        {"payload": S},
        ["payload"],
    ),
    "copy_file": fn(
        "copy_file",
        "Copy a file from source to destination.",
        {"source": S, "destination": S},
        ["source", "destination"],
    ),
    "sum_values": fn(
        "sum_values",
        "Sum a list of integers.",
        {"values": {"type": "array", "items": INTEGER}},
        ["values"],
    ),
    "search_docs": fn(
        "search_docs",
        "Search documents with a query, result limit, and tag filter.",
        {"query": S, "limit": INTEGER, "tags": {"type": "array", "items": S}},
        ["query", "limit", "tags"],
    ),
    "register_address": fn(
        "register_address",
        "Register a user with a nested address object.",
        {
            "user": {
                "type": "object",
                "properties": {
                    "name": S,
                    "address": {
                        "type": "object",
                        "properties": {"city": S, "zip": S},
                        "required": ["city", "zip"],
                    },
                },
                "required": ["name", "address"],
            }
        },
        ["user"],
    ),
    "ping": fn("ping", "Health check that takes no arguments.", {}, []),
    "translate_text": fn(
        "translate_text",
        "Translate text into a target language.",
        {"text": S, "target_language": S},
        ["text", "target_language"],
    ),
    "lookup_inventory": fn(
        "lookup_inventory",
        "Look up inventory for a SKU in a warehouse.",
        {"sku": S, "warehouse": S},
        ["sku", "warehouse"],
    ),
    "lookup_inventory_2": fn(
        "lookup_inventory_2",
        "Look up inventory for a SKU in a warehouse (second registry entry).",
        {"sku": S, "warehouse": S},
        ["sku", "warehouse"],
    ),
    "send_email": fn(
        "send_email",
        "Send an email.",
        {"to": S, "subject": S, "body": S},
        ["to", "subject", "body"],
    ),
    "create_task": fn(
        "create_task",
        "Create a task. The due_date field is optional.",
        {
            "title": S,
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "due_date": S,
        },
        ["title", "priority"],
    ),
    "strict_profile": fn(
        "strict_profile",
        "Set a profile. No properties beyond those declared are permitted.",
        {"handle": S, "tier": INTEGER},
        ["handle", "tier"],
        additional=False,
    ),
    "get_time": fn(
        "get_time", "Get the current time in a timezone.", {"timezone": S}, ["timezone"]
    ),
    "read_file": fn("read_file", "Read a file by path.", {"path": S}, ["path"]),
    "fetch_url": fn("fetch_url", "Fetch a URL.", {"url": S}, ["url"]),
    "book_trip": fn(
        "book_trip",
        "Book a trip between two cities.",
        {"origin": S, "destination": S, "passengers": INTEGER},
        ["origin", "destination", "passengers"],
    ),
    "append_note": fn(
        "append_note",
        "Append a note that may span multiple lines.",
        {"note": S},
        ["note"],
    ),
}


CASES: list[dict] = []

# Qwen3.6 measured 292-1516 completion tokens with thinking on; the target model
# always thinks and its reasoning length on these prompts is unmeasured, so
# keep the 2048 budget everywhere. A budget below ~2k truncates mid-reasoning
# and every downstream assertion fails on finish_reason=length rather than on
# the behaviour under test.
BUDGET = {"template_default": 2048}


def case(
    case_id,
    description,
    prompt,
    *,
    preset="template_default",
    tools=(),
    tool_choice=None,
    expect_reasoning=None,
    finish=("tool_calls",),
    expected_tool_calls=None,
    expected_tool_names=None,
    exact_tool_calls=None,
    min_tool_calls=None,
    no_tools=False,
    expected_content=None,
    content_pattern=None,
    expected_json=None,
    max_tokens=None,
    system=None,
    history=None,
    followup=None,
    overrides=None,
    validate_schema=None,
    normalize_argument_strings=False,
):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    entry = {
        "case_id": case_id,
        "description": description,
        "messages": messages,
        "tools": list(tools),
        "tool_choice": tool_choice,
        "request_preset": preset,
        "max_tokens": max_tokens if max_tokens is not None else BUDGET[preset],
        "expected_finish_reasons": list(finish),
        "expect_no_tool_calls": no_tools,
        "min_tool_calls": 0
        if no_tools
        else (min_tool_calls if min_tool_calls is not None else 1),
        "exact_tool_calls": 0 if no_tools else exact_tool_calls,
        "expected_tool_names": list(expected_tool_names or []),
        "expected_tool_calls": list(expected_tool_calls or []),
        "expected_content": expected_content,
        "content_pattern": content_pattern,
        "expect_reasoning": expect_reasoning,
        "normalize_argument_strings": normalize_argument_strings,
    }
    if expected_json is not None:
        entry["expected_json"] = expected_json
    if followup is not None:
        entry["scripted_followup"] = followup
    if overrides is not None:
        entry["request_overrides"] = overrides
    if validate_schema is not None:
        entry["validate_schema"] = validate_schema
    CASES.append(entry)


def tc(name, arguments):
    return {"name": name, "arguments": arguments}


# ---------------------------------------------------------------------------
# F1  Core tool_choice behaviour. The template has no thinking switch, so the
#     qwen3_6 three-mode sweep collapses to the single always-thinking default.
# ---------------------------------------------------------------------------
for suffix, preset, reasoning in (("default", "template_default", True),):
    case(
        f"qx_core_required_single_{suffix}",
        f"tool_choice=required, single call, thinking={suffix}",
        "Use add_numbers exactly once with a=17 and b=19. Do not calculate the answer yourself.",
        preset=preset,
        tools=["add_numbers"],
        tool_choice="required",
        exact_tool_calls=1,
        expected_tool_names=["add_numbers"],
        expected_tool_calls=[tc("add_numbers", {"a": 17, "b": 19})],
        expect_reasoning=reasoning,
    )
    case(
        f"qx_core_auto_call_{suffix}",
        f"tool_choice=auto with an obvious call, thinking={suffix}",
        "What is the weather in Paris right now? Report it in celsius. Use the tool.",
        preset=preset,
        tools=["get_weather"],
        tool_choice="auto",
        exact_tool_calls=1,
        expected_tool_names=["get_weather"],
        expected_tool_calls=[tc("get_weather", {"city": "Paris", "unit": "celsius"})],
        expect_reasoning=reasoning,
    )
    case(
        f"qx_core_auto_no_call_{suffix}",
        f"tool_choice=auto where no call applies, thinking={suffix}",
        "Reply with exactly the word BLUE. Do not call any function.",
        preset=preset,
        tools=["get_weather"],
        tool_choice="auto",
        no_tools=True,
        finish=("stop",),
        expected_content="BLUE",
        expect_reasoning=reasoning,
    )
    case(
        f"qx_core_none_{suffix}",
        f"tool_choice=none must suppress calls, thinking={suffix}",
        "What is the weather in Paris? Answer in one short sentence without calling anything.",
        preset=preset,
        tools=["get_weather"],
        tool_choice="none",
        no_tools=True,
        finish=("stop",),
        expect_reasoning=reasoning,
    )

# ---------------------------------------------------------------------------
# F2  Argument type recovery. The wire format is untyped, so each parser must
#     rebuild the JSON type from the tool schema. Strict equality here means
#     17 != "17" and 1 != 1.0 — exactly the failures we want surfaced.
# ---------------------------------------------------------------------------
case(
    "qx_type_integer_scalars",
    "integers must not come back as strings",
    "Call add_numbers with a=17 and b=19.",
    tools=["add_numbers"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("add_numbers", {"a": 17, "b": 19})],
)
case(
    "qx_type_float_scalar",
    "a float must stay a float",
    "Call set_threshold with value 0.75.",
    tools=["set_threshold"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("set_threshold", {"value": 0.75})],
)
case(
    "qx_type_negative_floats",
    "negative floats keep their sign and type",
    "Call set_coordinate with x=-12.5 and y=3.25.",
    tools=["set_coordinate"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("set_coordinate", {"x": -12.5, "y": 3.25})],
)
case(
    "qx_type_boolean_true",
    "boolean true must not become the string 'true'",
    "Call set_mode with enabled set to true.",
    tools=["set_mode"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("set_mode", {"enabled": True})],
)
case(
    "qx_type_boolean_false",
    "boolean false must not become the string 'false' or be dropped",
    "Call set_mode with enabled set to false.",
    tools=["set_mode"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("set_mode", {"enabled": False})],
)
case(
    "qx_type_mixed_scalars",
    "int, float, bool and string in one call",
    "Call record_scalars with count=3, ratio=0.5, enabled=true, label=alpha.",
    tools=["record_scalars"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[
        tc(
            "record_scalars",
            {"count": 3, "ratio": 0.5, "enabled": True, "label": "alpha"},
        )
    ],
)
case(
    "qx_type_numeric_string_stays_string",
    "a numeric-looking value declared as string must stay a string",
    "Call store_text with text set to the four characters 1234. It is a string, not a number.",
    tools=["store_text"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("store_text", {"text": "1234"})],
)
case(
    "qx_type_array_of_integers",
    "an integer array must decode as a list of ints",
    "Call sum_values with values [1, 2, 3].",
    tools=["sum_values"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("sum_values", {"values": [1, 2, 3]})],
)
case(
    "qx_type_array_of_strings",
    "a string array plus sibling scalars",
    'Call search_docs with query=kv cache, limit=5, and tags ["gpu", "memory"].',
    tools=["search_docs"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[
        tc("search_docs", {"query": "kv cache", "limit": 5, "tags": ["gpu", "memory"]})
    ],
)
case(
    "qx_type_nested_object",
    "a two-level nested object must survive the parameter block",
    "Call register_address for a user named Ada whose address has city=Paris and zip=75001.",
    tools=["register_address"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[
        tc(
            "register_address",
            {"user": {"name": "Ada", "address": {"city": "Paris", "zip": "75001"}}},
        )
    ],
)
case(
    "qx_type_unicode_string",
    "non-ASCII argument text round-trips without escaping damage",
    "Call translate_text with text set to 你好世界 and target_language set to English.",
    tools=["translate_text"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[
        tc("translate_text", {"text": "你好世界", "target_language": "English"})
    ],
)
case(
    "qx_type_windows_path_backslashes",
    "backslashes must not be consumed as escapes",
    r"Call copy_file with source C:\data\input.txt and destination D:\backup\input.txt.",
    tools=["copy_file"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[
        tc(
            "copy_file",
            {"source": r"C:\data\input.txt", "destination": r"D:\backup\input.txt"},
        )
    ],
)
case(
    "qx_type_embedded_json_string",
    "a JSON document carried inside a string argument stays a string",
    'Call submit_json with payload set to exactly {"a":1,"b":2}',
    tools=["submit_json"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("submit_json", {"payload": '{"a":1,"b":2}'})],
)

# ---------------------------------------------------------------------------
# F3  Schema-shape stress.
# ---------------------------------------------------------------------------
case(
    "qx_schema_empty_object",
    "a no-argument tool must produce {} and not an empty string",
    "Call ping once.",
    tools=["ping"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("ping", {})],
)
case(
    "qx_schema_enum_required",
    "enum-valued argument",
    "Call get_weather for Tokyo using fahrenheit.",
    tools=["get_weather"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("get_weather", {"city": "Tokyo", "unit": "fahrenheit"})],
)
case(
    "qx_schema_optional_omitted",
    "an unmentioned optional property must be absent, not null",
    "Call create_task with title=Ship release and priority=high. Do not set a due date.",
    tools=["create_task"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[
        tc("create_task", {"title": "Ship release", "priority": "high"})
    ],
)
case(
    "qx_schema_additional_properties_false",
    "no invented properties when additionalProperties is false",
    "Call strict_profile with handle=ada and tier=2.",
    tools=["strict_profile"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("strict_profile", {"handle": "ada", "tier": 2})],
)
case(
    "qx_schema_all_required_fields",
    "every required field present on a three-field tool",
    "Call send_email to ada@example.com with subject Status and body All green.",
    tools=["send_email"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[
        tc(
            "send_email",
            {"to": "ada@example.com", "subject": "Status", "body": "All green"},
        )
    ],
    # The model deterministically writes "All green." — prose-valued args
    # tolerate case and one trailing mark; typed args stay strict.
    normalize_argument_strings=True,
)
case(
    "qx_schema_multiline_value",
    "a parameter value spanning lines must not truncate at the newline",
    "Call append_note with note set to exactly these two lines:\nfirst line\nsecond line",
    tools=["append_note"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("append_note", {"note": "first line\nsecond line"})],
)
case(
    "qx_schema_angle_brackets_in_value",
    "angle brackets inside a value must not be mistaken for markup",
    "Call store_text with text set to exactly a < b > c",
    tools=["store_text"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("store_text", {"text": "a < b > c"})],
)
case(
    "qx_schema_quotes_in_value",
    "embedded double quotes survive the parameter block",
    'Call store_text with text set to exactly she said "hi"',
    tools=["store_text"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("store_text", {"text": 'she said "hi"'})],
)

# ---------------------------------------------------------------------------
# F4  Parallel calls.
# ---------------------------------------------------------------------------
case(
    "qx_parallel_two_different_tools",
    "two distinct tools in one turn",
    "Call get_weather for Paris in celsius and add_numbers with a=2 and b=3. Use both tools.",
    tools=["get_weather", "add_numbers"],
    tool_choice="required",
    exact_tool_calls=2,
    expected_tool_calls=[
        tc("get_weather", {"city": "Paris", "unit": "celsius"}),
        tc("add_numbers", {"a": 2, "b": 3}),
    ],
)
case(
    "qx_parallel_same_tool_twice",
    "the same tool called twice with different arguments",
    "Call get_weather twice: once for Paris in celsius and once for Tokyo in celsius.",
    tools=["get_weather"],
    tool_choice="required",
    exact_tool_calls=2,
    expected_tool_calls=[
        tc("get_weather", {"city": "Paris", "unit": "celsius"}),
        tc("get_weather", {"city": "Tokyo", "unit": "celsius"}),
    ],
)
case(
    "qx_parallel_three_calls",
    "three calls in a single turn",
    "Call add_numbers with a=1 b=2, then add_numbers with a=3 b=4, then add_numbers with a=5 b=6.",
    tools=["add_numbers"],
    tool_choice="required",
    exact_tool_calls=3,
    expected_tool_calls=[
        tc("add_numbers", {"a": 1, "b": 2}),
        tc("add_numbers", {"a": 3, "b": 4}),
        tc("add_numbers", {"a": 5, "b": 6}),
    ],
)
case(
    "qx_parallel_distinct_registry_tools",
    "parallel calls across near-identical tool names",
    "Call add_numbers with a=1 b=2 and add_numbers_2 with a=3 b=4.",
    tools=["add_numbers", "add_numbers_2"],
    tool_choice="required",
    exact_tool_calls=2,
    expected_tool_calls=[
        tc("add_numbers", {"a": 1, "b": 2}),
        tc("add_numbers_2", {"a": 3, "b": 4}),
    ],
)

# ---------------------------------------------------------------------------
# F5  Multi-turn lifecycle. The second turn only runs when the first is clean.
# ---------------------------------------------------------------------------
case(
    "qx_lifecycle_single_result",
    "one tool result then a fixed final answer",
    "Call get_weather for Paris in celsius. After the tool result, return exactly PARIS_SUNNY.",
    preset="template_default",
    tools=["get_weather"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("get_weather", {"city": "Paris", "unit": "celsius"})],
    expect_reasoning=True,
    followup={
        "tool_results": ['{"condition":"sunny","temperature_c":21}'],
        "reverse_results": False,
        "tool_choice": "none",
        "expected_finish_reasons": ["stop"],
        "expect_no_tool_calls": True,
        "exact_tool_calls": 0,
        "min_tool_calls": 0,
        "expected_content": "PARIS_SUNNY",
        "expect_reasoning": True,
    },
)
case(
    "qx_lifecycle_parallel_results_in_order",
    "two results delivered in call order",
    "Call get_weather for Paris and for Tokyo, both in celsius. After both results, reply with exactly DONE.",
    preset="template_default",
    tools=["get_weather"],
    tool_choice="required",
    exact_tool_calls=2,
    expected_tool_names=["get_weather"],
    expect_reasoning=True,
    followup={
        "tool_results": [
            '{"condition":"sunny","temperature_c":21}',
            '{"condition":"rain","temperature_c":14}',
        ],
        "reverse_results": False,
        "tool_choice": "none",
        "expected_finish_reasons": ["stop"],
        "expect_no_tool_calls": True,
        "exact_tool_calls": 0,
        "min_tool_calls": 0,
        "expected_content": "DONE",
        "expect_reasoning": True,
    },
)
case(
    "qx_lifecycle_parallel_results_out_of_order",
    "the same two results appended in reverse order",
    "Call get_weather for Paris and for Tokyo, both in celsius. After both results, reply with exactly DONE.",
    preset="template_default",
    tools=["get_weather"],
    tool_choice="required",
    exact_tool_calls=2,
    expected_tool_names=["get_weather"],
    expect_reasoning=True,
    followup={
        "tool_results": [
            '{"condition":"sunny","temperature_c":21}',
            '{"condition":"rain","temperature_c":14}',
        ],
        "reverse_results": True,
        "tool_choice": "none",
        "expected_finish_reasons": ["stop"],
        "expect_no_tool_calls": True,
        "exact_tool_calls": 0,
        "min_tool_calls": 0,
        "expected_content": "DONE",
        "expect_reasoning": True,
    },
)
case(
    "qx_lifecycle_error_result",
    "an error payload must still reach a clean final answer",
    "Call lookup_inventory for sku=ABC in warehouse=east. If the tool reports an error, reply with exactly LOOKUP_FAILED.",
    preset="template_default",
    tools=["lookup_inventory"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("lookup_inventory", {"sku": "ABC", "warehouse": "east"})],
    expect_reasoning=True,
    followup={
        "tool_results": ['{"error":"warehouse unavailable"}'],
        "reverse_results": False,
        "tool_choice": "none",
        "expected_finish_reasons": ["stop"],
        "expect_no_tool_calls": True,
        "exact_tool_calls": 0,
        "min_tool_calls": 0,
        "expected_content": "LOOKUP_FAILED",
        "expect_reasoning": True,
    },
)
case(
    "qx_lifecycle_unicode_result",
    "non-ASCII tool output consumed on the second turn",
    "Call translate_text with text 你好 and target_language English. After the tool result, reply with exactly the translation the tool returned.",
    preset="template_default",
    tools=["translate_text"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[
        tc("translate_text", {"text": "你好", "target_language": "English"})
    ],
    expect_reasoning=True,
    followup={
        "tool_results": ['{"translation":"Hello"}'],
        "reverse_results": False,
        "tool_choice": "none",
        "expected_finish_reasons": ["stop"],
        "expect_no_tool_calls": True,
        "exact_tool_calls": 0,
        "min_tool_calls": 0,
        "expected_content": "Hello",
        "expect_reasoning": True,
    },
)

# ---------------------------------------------------------------------------
# F6  Reasoning. Qwen3.6's template pre-seeds "<think>\n" into the generation
#     prompt, so the model emits only the closing tag. A parser that waits for
#     an opening tag before entering reasoning mode will mis-split here.
# ---------------------------------------------------------------------------
_REASON = [
    ("multiply_17_19", "Compute 17 times 19. Return only the integer.", "323"),
    (
        "compare_fractions",
        "Which is larger, 5/8 or 7/12? Reply with exactly the larger fraction in the form a/b.",
        "5/8",
    ),
    (
        "sequence_next",
        "Continue the sequence 2, 6, 12, 20, 30. Return only the next integer.",
        "42",
    ),
    (
        "minutes_to_seconds",
        "How many seconds are in 7 minutes and 15 seconds? Return only the integer.",
        "435",
    ),
    (
        "sort_integers",
        "Sort 8, 3, 11, 5 ascending. Return only the integers separated by single spaces.",
        "3 5 8 11",
    ),
    (
        "parity_expression",
        "Is 3^7 + 2 even or odd? Reply with exactly one word, even or odd.",
        "odd",
    ),
    (
        "set_intersection",
        "Given A={1,2,3,4} and B={3,4,5}, list the intersection ascending, comma separated with no spaces.",
        "3,4",
    ),
    ("chinese_arithmetic", "用中文思考：23 乘以 4 等于多少？只返回这个整数。", "92"),
    ("unicode_symbol_math", "Evaluate ⌈7.2⌉ + ⌊3.9⌋. Return only the integer.", "11"),
]
for name, prompt, answer in _REASON:
    case(
        f"qx_reasoning_{name}",
        f"reasoning: {name}",
        prompt,
        preset="template_default",
        no_tools=True,
        finish=("stop",),
        expected_content=answer,
        expect_reasoning=True,
    )

case(
    "qx_reasoning_spanish_logic",
    "reasoning: spanish_logic",
    "Si todos los gatos son mamíferos y Feli es un gato, ¿es Feli un mamífero? Responde exactamente sí o no.",
    preset="template_default",
    no_tools=True,
    finish=("stop",),
    # The model answers "Sí." / "Sí" — capitalization and one trailing mark
    # are its deterministic style. Accept them; reject anything else.
    content_pattern=r"(?i)sí[.!]?",
    expect_reasoning=True,
)
case(
    "qx_reasoning_missing_open_think_tag",
    "template pre-seeds <think>, so reasoning_content must populate from a close-tag-only stream",
    "Think step by step about why 2+2=4, then return only the integer 4.",
    preset="template_default",
    no_tools=True,
    finish=("stop",),
    expected_content="4",
    expect_reasoning=True,
)
case(
    "qx_reasoning_structured_json",
    "reasoning then a strict JSON body",
    'Return only this JSON object with no prose: {"answer": 42, "unit": "cm"}',
    preset="template_default",
    no_tools=True,
    finish=("stop",),
    expected_json={"answer": 42, "unit": "cm"},
    expect_reasoning=True,
)
case(
    "qx_reasoning_system_instruction",
    "a system message must not break the reasoning split",
    "What is 12 times 12?",
    preset="template_default",
    system="You are terse. Answer with only the integer, no words.",
    no_tools=True,
    finish=("stop",),
    expected_content="144",
    expect_reasoning=True,
)
case(
    "qx_reasoning_long_context_retrieval",
    "retrieve one fact from a long preamble",
    "Here are records. "
    + " ".join(f"record {i} value {i * 3}." for i in range(1, 61))
    + " What is the value of record 47? Return only the integer.",
    preset="template_default",
    no_tools=True,
    finish=("stop",),
    expected_content="141",
    expect_reasoning=True,
)
case(
    "qx_reasoning_truncate_at_16_tokens",
    "a hard max_tokens cut mid-reasoning must still yield a clean length finish",
    "Explain in detail how a KV cache works in transformer inference.",
    preset="template_default",
    no_tools=True,
    finish=("length",),
    max_tokens=16,
)
case(
    "qx_reasoning_truncate_at_32_tokens",
    "a slightly later cut, still inside the reasoning block",
    "Explain in detail how paged attention allocates blocks.",
    preset="template_default",
    no_tools=True,
    finish=("length",),
    max_tokens=32,
)

case(
    "qx_reasoning_simple_ack",
    "trivial instruction still reasons and answers cleanly",
    "Remember the codeword ORCHID. Reply with exactly ACK.",
    preset="template_default",
    no_tools=True,
    finish=("stop",),
    expected_content="ACK",
    expect_reasoning=True,
)

# ---------------------------------------------------------------------------
# F7  Reasoning interacting with tool calls.
# ---------------------------------------------------------------------------
case(
    "qx_reasoning_tool_required",
    "reasoning must populate alongside a forced call",
    "Use get_weather exactly once for Paris in celsius.",
    preset="template_default",
    tools=["get_weather"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[tc("get_weather", {"city": "Paris", "unit": "celsius"})],
    expect_reasoning=True,
)
case(
    "qx_reasoning_tool_auto",
    "reasoning plus an auto-selected call",
    "I need today's weather for Berlin in celsius.",
    preset="template_default",
    tools=["get_weather"],
    tool_choice="auto",
    exact_tool_calls=1,
    expected_tool_calls=[tc("get_weather", {"city": "Berlin", "unit": "celsius"})],
    expect_reasoning=True,
)
case(
    "qx_reasoning_tool_named",
    "a named tool_choice with reasoning enabled",
    "Look up inventory using sku=XY9 and warehouse=west.",
    preset="template_default",
    tools=["lookup_inventory", "get_weather"],
    tool_choice={"type": "function", "function": {"name": "lookup_inventory"}},
    exact_tool_calls=1,
    expected_tool_calls=[tc("lookup_inventory", {"sku": "XY9", "warehouse": "west"})],
    expect_reasoning=True,
)
case(
    "qx_reasoning_tool_none_arithmetic",
    "tool_choice=none with reasoning on and tools declared",
    "Tools are available but do not use them. Compute 9 times 6 and return only the integer.",
    preset="template_default",
    tools=["multiply_numbers"],
    tool_choice="none",
    no_tools=True,
    finish=("stop",),
    expected_content="54",
    expect_reasoning=True,
)
case(
    "qx_reasoning_named_tool_time",
    "a second named tool_choice shape with a distractor tool declared",
    "Get the time in Asia/Tokyo.",
    preset="template_default",
    tools=["get_time", "get_weather"],
    tool_choice={"type": "function", "function": {"name": "get_time"}},
    exact_tool_calls=1,
    expected_tool_calls=[tc("get_time", {"timezone": "Asia/Tokyo"})],
    expect_reasoning=True,
)

# ---------------------------------------------------------------------------
# F8  Marker containment. The defaults block already forbids every Qwen3.6
#     control token in content and reasoning; these prompts actively bait the
#     model into emitting them.
# ---------------------------------------------------------------------------
case(
    "qx_marker_no_leak_on_plain_answer",
    "tool markup must not appear when answering without a call",
    "Reply with exactly the word GREEN. Do not call any function.",
    tools=["get_weather", "add_numbers"],
    tool_choice="none",
    no_tools=True,
    finish=("stop",),
    expected_content="GREEN",
)
case(
    "qx_marker_no_leak_with_unicode_arg",
    "unicode arguments must not drag markup into content",
    "Call translate_text with text 안녕하세요 and target_language English.",
    tools=["translate_text"],
    tool_choice="required",
    exact_tool_calls=1,
    expected_tool_calls=[
        tc("translate_text", {"text": "안녕하세요", "target_language": "English"})
    ],
)
case(
    "qx_marker_reasoning_stays_out_of_content",
    "reasoning text must not bleed into the visible answer",
    "Think about the capital of France, then reply with exactly Paris and nothing else.",
    preset="template_default",
    no_tools=True,
    finish=("stop",),
    expected_content="Paris",
    expect_reasoning=True,
)

# ---------------------------------------------------------------------------
# F9  reasoning_content ingress. The template re-renders assistant turns as
#     "<think>reasoning_content</think>content" unconditionally, so a client
#     that sends reasoning_content on an assistant history message changes the
#     rendered prompt IF the serving path forwards the field to the template.
#     Both cases below must PASS on their own; the discriminator is the
#     usage.prompt_tokens delta between them. Per path:
#       delta > 0   -> the path forwards reasoning_content into the render
#       delta == 0  -> the path drops it before rendering
#     A cross-path disagreement on the delta is a parity divergence even when
#     every record passes. compare_runs reads the deltas from usage.
#     (The live round-trip variant is already covered: the probe's
#     scripted_followup turns echo the real first-response reasoning_content
#     back on the second request for every lifecycle case.)
# ---------------------------------------------------------------------------
_INGRESS_HISTORY_BASE = [
    {"role": "user", "content": "What is the capital of France? Answer in one word."},
]
case(
    "qx_rt_ingress_with_reasoning",
    "assistant history message CARRIES reasoning_content; prompt must render and answer cleanly",
    "Repeat your previous answer exactly, one word only.",
    preset="template_default",
    history=_INGRESS_HISTORY_BASE
    + [
        {
            "role": "assistant",
            "content": "Paris",
            "reasoning_content": (
                "The user asks for the capital of France. That is Paris. "
                "This sentence exists to make the rendered prompt measurably "
                "longer when the field is forwarded to the chat template."
            ),
        }
    ],
    no_tools=True,
    finish=("stop",),
    expected_content="Paris",
    expect_reasoning=True,
)
case(
    "qx_rt_ingress_without_reasoning",
    "identical history minus reasoning_content; usage.prompt_tokens baseline for the delta check",
    "Repeat your previous answer exactly, one word only.",
    preset="template_default",
    history=_INGRESS_HISTORY_BASE + [{"role": "assistant", "content": "Paris"}],
    no_tools=True,
    finish=("stop",),
    expected_content="Paris",
    expect_reasoning=True,
)


def main() -> None:
    payload = {
        "schema_version": 1,
        "profile": "qwen3_coder_xml",
        "description": "qwen3_coder-format hybrid reasoning models tool-calling and reasoning qualification matrix.",
        "default_modes": ["nonstream", "stream"],
        "logical_cases": len(CASES),
        "expected_records": len(CASES) * 2,
        "defaults": {"forbidden_output_fragments": FORBIDDEN},
        "request_presets": PRESETS,
        "tools": TOOLS,
        "cases": CASES,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    ids = [c["case_id"] for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    print(f"wrote {OUT} with {len(CASES)} cases")


if __name__ == "__main__":
    main()
