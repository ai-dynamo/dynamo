# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay cases for tool-argument schema conformance.

The cases are reconstructions, not copies of the nightly OpenRouter formats
suite: they follow the three failure patterns reported for GPT-OSS with
``strict: false`` tools (missing required nullable field, wrong scalar type,
wrong nested structure) and the case name
``toolcall_formats-bash_wrong_type_requested-omitted``.

Suite ``A`` declares every tool ``strict: false``. Suite ``B`` is the control:
the same tools with ``strict`` omitted or ``true``, schemas that a grammar
backend may fail to compile, and requests that must not produce a tool call.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

BASH = {
    "name": "bash",
    "description": "Run a shell command and return its output.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run."},
            "timeout": {
                "type": ["number", "null"],
                "description": "Timeout in seconds, or null for no timeout.",
            },
        },
        "required": ["command", "timeout"],
        "additionalProperties": False,
    },
}

SET_TIMER = {
    "name": "set_timer",
    "description": "Start a countdown timer.",
    "parameters": {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "minutes": {"type": "integer", "description": "Duration in minutes."},
        },
        "required": ["label", "minutes"],
        "additionalProperties": False,
    },
}

RESIZE_IMAGE = {
    "name": "resize_image",
    "description": "Resize an image file.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
        },
        "required": ["path", "width", "height"],
        "additionalProperties": False,
    },
}

_PERSON = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
    "required": ["name", "email"],
    "additionalProperties": False,
}

CREATE_EVENT = {
    "name": "create_event",
    "description": "Create a calendar event and invite attendees.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "start": {"type": "string", "description": "ISO 8601 start time."},
            "attendees": {"type": "array", "items": _PERSON},
        },
        "required": ["title", "start", "attendees"],
        "additionalProperties": False,
    },
}

CREATE_ORDER = {
    "name": "create_order",
    "description": "Place an order for a customer.",
    "parameters": {
        "type": "object",
        "properties": {
            "customer": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                            "zip": {"type": "string"},
                        },
                        "required": ["street", "city", "zip"],
                        "additionalProperties": False,
                    },
                },
                "required": ["name", "address"],
                "additionalProperties": False,
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sku": {"type": "string"},
                        "quantity": {"type": "integer"},
                    },
                    "required": ["sku", "quantity"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["customer", "items"],
        "additionalProperties": False,
    },
}

GET_WEATHER = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
}

# Compile-rejection probes: valid JSON Schema that a grammar backend may not
# compile once the schema is enforced (regex lookahead, `$ref`).
REGISTER_HOST = {
    "name": "register_host",
    "description": "Register a host name in the inventory.",
    "parameters": {
        "type": "object",
        "properties": {
            "hostname": {"type": "string", "pattern": "^(?!-)[a-z0-9-]{1,63}$"},
        },
        "required": ["hostname"],
        "additionalProperties": False,
    },
}

SHIP_PARCEL = {
    "name": "ship_parcel",
    "description": "Ship a parcel between two addresses.",
    "parameters": {
        "type": "object",
        "$defs": {
            "address": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "zip": {"type": "string"}},
                "required": ["city", "zip"],
                "additionalProperties": False,
            }
        },
        "properties": {
            "sender": {"$ref": "#/$defs/address"},
            "recipient": {"$ref": "#/$defs/address"},
        },
        "required": ["sender", "recipient"],
        "additionalProperties": False,
    },
}

_UNSET = object()


def tool(definition: dict[str, Any], strict: Any = False) -> dict[str, Any]:
    """Chat Completions tool entry; ``strict=_UNSET`` omits the field."""
    function = copy.deepcopy(definition)
    if strict is not _UNSET:
        function["strict"] = strict
    return {"type": "function", "function": function}


@dataclass(frozen=True)
class Case:
    name: str
    suite: str
    prompt: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    # Name of the tool the answer must call; None means a text answer.
    expect_tool: str | None = None
    # "chat" (/v1/chat/completions) or "responses" (/v1/responses).
    api: str = "chat"


SYSTEM_PROMPT = "You are a helpful assistant. Use the provided tools when they help."

BASH_OMITTED = "List the files in /var/log with `ls -la /var/log`."
BASH_WRONG_TYPE = 'Run `make test` and set the timeout to "30".'
SCALAR_WORDS = "Start a timer called tea for forty-five minutes."
SCALAR_QUOTED = 'Resize photo.png to width "800" and height "600".'
NESTED_ATTENDEES = (
    "Create an event titled Design review on 2026-12-01T15:00:00Z with "
    "Alice (alice@example.com) and Bob (bob@example.com)."
)
NESTED_ORDER = (
    "Order 2 of SKU A-100 and 1 of SKU B-200 for Carol, 12 Elm St, Springfield, "
    "zip 01101."
)

_ALL = [BASH, SET_TIMER, CREATE_EVENT, GET_WEATHER]


def _suite_a() -> list[Case]:
    s = False
    return [
        # Pattern 1: required nullable field / wrong scalar type on bash.timeout.
        Case(
            "bash_timeout_omitted",
            "A",
            BASH_OMITTED,
            [tool(BASH, s)],
            "auto",
            expect_tool="bash",
        ),
        Case(
            "bash_wrong_type_requested",
            "A",
            BASH_WRONG_TYPE,
            [tool(BASH, s)],
            "auto",
            expect_tool="bash",
        ),
        Case(
            "bash_multi_tool",
            "A",
            BASH_OMITTED,
            [tool(t, s) for t in _ALL],
            "auto",
            expect_tool="bash",
        ),
        Case(
            "bash_parallel",
            "A",
            "Run `uname -a` and `df -h` as two separate commands.",
            [tool(BASH, s)],
            "auto",
            parallel_tool_calls=True,
            expect_tool="bash",
        ),
        Case(
            "bash_required",
            "A",
            BASH_OMITTED,
            [tool(BASH, s)],
            "required",
            expect_tool="bash",
        ),
        # Pattern 2: integer stated in words or as a quoted string.
        Case(
            "scalar_words",
            "A",
            SCALAR_WORDS,
            [tool(SET_TIMER, s)],
            "auto",
            expect_tool="set_timer",
        ),
        Case(
            "scalar_quoted",
            "A",
            SCALAR_QUOTED,
            [tool(RESIZE_IMAGE, s)],
            "auto",
            expect_tool="resize_image",
        ),
        Case(
            "scalar_multi_tool",
            "A",
            SCALAR_WORDS,
            [tool(t, s) for t in _ALL],
            "auto",
            expect_tool="set_timer",
        ),
        Case(
            "scalar_parallel",
            "A",
            'Start two timers: pasta for twelve minutes and eggs for "9" minutes.',
            [tool(SET_TIMER, s)],
            "auto",
            parallel_tool_calls=True,
            expect_tool="set_timer",
        ),
        Case(
            "scalar_required",
            "A",
            SCALAR_QUOTED,
            [tool(RESIZE_IMAGE, s)],
            "required",
            expect_tool="resize_image",
        ),
        # Pattern 3: array of objects / deeper nested object.
        Case(
            "nested_attendees",
            "A",
            NESTED_ATTENDEES,
            [tool(CREATE_EVENT, s)],
            "auto",
            expect_tool="create_event",
        ),
        Case(
            "nested_order",
            "A",
            NESTED_ORDER,
            [tool(CREATE_ORDER, s)],
            "auto",
            expect_tool="create_order",
        ),
        Case(
            "nested_multi_tool",
            "A",
            NESTED_ATTENDEES,
            [tool(t, s) for t in _ALL],
            "auto",
            expect_tool="create_event",
        ),
        Case(
            "nested_parallel",
            "A",
            "Create two events: Standup on 2026-12-02T09:00:00Z with Dan (dan@example.com), "
            "and Retro on 2026-12-05T16:00:00Z with Eve (eve@example.com).",
            [tool(CREATE_EVENT, s)],
            "auto",
            parallel_tool_calls=True,
            expect_tool="create_event",
        ),
        Case(
            "nested_required",
            "A",
            NESTED_ATTENDEES,
            [tool(CREATE_EVENT, s)],
            "required",
            expect_tool="create_event",
        ),
    ]


def _suite_b() -> list[Case]:
    return [
        Case(
            "bash_strict_omitted",
            "B",
            BASH_OMITTED,
            [tool(BASH, _UNSET)],
            "auto",
            expect_tool="bash",
        ),
        Case(
            "bash_strict_true",
            "B",
            BASH_WRONG_TYPE,
            [tool(BASH, True)],
            "auto",
            expect_tool="bash",
        ),
        Case(
            "scalar_strict_omitted",
            "B",
            SCALAR_QUOTED,
            [tool(RESIZE_IMAGE, _UNSET)],
            "auto",
            expect_tool="resize_image",
        ),
        Case(
            "nested_strict_true",
            "B",
            NESTED_ATTENDEES,
            [tool(CREATE_EVENT, True)],
            "auto",
            expect_tool="create_event",
        ),
        Case(
            "nested_order_strict_omitted",
            "B",
            NESTED_ORDER,
            [tool(CREATE_ORDER, _UNSET)],
            "auto",
            expect_tool="create_order",
        ),
        Case(
            "probe_pattern_lookahead",
            "B",
            "Register the host web-01 in the inventory.",
            [tool(REGISTER_HOST, False)],
            "auto",
            expect_tool="register_host",
        ),
        Case(
            "probe_ref",
            "B",
            "Ship a parcel from Austin 73301 to Boston 02108.",
            [tool(SHIP_PARCEL, False)],
            "auto",
            expect_tool="ship_parcel",
        ),
        Case("tool_choice_none", "B", BASH_OMITTED, [tool(BASH, False)], "none"),
        Case("no_tools", "B", "What is the capital of France? Answer in one word."),
        Case(
            "text_answer_with_tools",
            "B",
            "What is the capital of France? Answer in one word without using any tool.",
            [tool(t, False) for t in _ALL],
            "auto",
        ),
        Case(
            "responses_bash",
            "B",
            BASH_WRONG_TYPE,
            [tool(BASH, False)],
            "auto",
            expect_tool="bash",
            api="responses",
        ),
    ]


SUITES: dict[str, list[Case]] = {"A": _suite_a(), "B": _suite_b()}


def tool_schema(case: Case, name: str) -> dict[str, Any] | None:
    for entry in case.tools:
        if entry["function"]["name"] == name:
            return entry["function"]["parameters"]
    return None
