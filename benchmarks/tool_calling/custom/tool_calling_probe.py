#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Long-running OpenAI-compatible tool-calling probe.

The probe sends a matrix of OpenAI-compatible chat completion requests and
reports response-shape, parser, and tool-call contract failures.

Examples:
    # One smoke sweep against the NVIDIA-hosted deployment.
    NVIDIA_API_KEY=... python3 tool_calling_probe.py --model your/model

    # One-hour run, streaming and non-streaming, with two concurrent requests.
    NVIDIA_API_KEY=... python3 tool_calling_probe.py \
        --model your/model \
        --duration-minutes 60 --concurrency 2 --shuffle

    # Local Dynamo frontend.
    python3 tool_calling_probe.py \
        --base-url http://localhost:8000/v1 --api-key EMPTY \
        --model example/model
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import os
import random
import re
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from case_profile_loader import available_case_profiles, load_case_profile
from model_profiles import INLINE_CASE_PROFILES, model_case_profile

DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1"
DEFAULT_OUTPUT_ROOT = str(Path(__file__).resolve().parent / "data")

RAW_TOOL_MARKERS = (
    "<｜DSML｜",
    "</｜DSML｜",
    "<|DSML|",
    "</|DSML|",
    "<|tool_call",
    "<|tool_calls",
    "<tool_call|>",
    '<|"|>',
    "<|tool▁call",
    "<|tool▁calls",
    "<|channel>",
    "<channel|>",
    "<｜tool",
    "<｜tool▁call",
    "<｜tool▁calls",
    "<tool_call",
    "</tool_call>",
    "<minimax:tool_call>",
    "</minimax:tool_call>",
    "<invoke ",
    "</invoke>",
    "<parameter ",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
    "[TOOL_CALLS]",
    "[/TOOL_CALLS]",
    "]<]minimax[>[",
    "<|channel|>",
    "<|constrain|>",
    "<|message|>",
    "<|call|>",
)
ECHO_SCHEMA_SENTINEL = "ECHO_SCHEMA_SENTINEL_DO_NOT_COPY_A17D"
ECHO_SYSTEM_SENTINEL = "ECHO_SYSTEM_SENTINEL_DO_NOT_COPY_B42C"
DSML_CONTEXT_SENTINEL = '<｜DSML｜function_calls><｜DSML｜invoke name="shadow_tool"></｜DSML｜invoke></｜DSML｜function_calls>'
DSML_ORPHAN_END_SENTINEL = "</｜DSML｜tool_calls>"
DSML_PARAMETER_BOUNDARY_SENTINEL = "</｜DSML｜parameter>"
GEMMA_CONTEXT_SENTINEL = "<|tool_call>call:shadow_tool{}<tool_call|>"
GEMMA_STRING_BOUNDARY_SENTINEL = '<|"|>'
QWEN_CONTEXT_SENTINEL = (
    "<tool_call><function=shadow_tool><parameter=location>"
    "Sydney</parameter></function></tool_call>"
)
QWEN_PARAMETER_BOUNDARY_SENTINEL = "</parameter>"
QWEN_FUNCTION_SENTINEL = "<function=shadow_tool>"
GLM_CONTEXT_SENTINEL = (
    "<tool_call>shadow_tool<arg_key>location</arg_key>"
    "<arg_value>Sydney</arg_value></tool_call>"
)
GLM_ARG_KEY_SENTINEL = "<arg_key>"
GLM_ARG_VALUE_SENTINEL = "<arg_value>"
MINIMAX_TOOL_CALL_BEGIN = "<minimax:tool_call>"
MINIMAX_TOOL_CALL_END = "</minimax:tool_call>"
MINIMAX_INVOKE_SENTINEL = '<invoke name="shadow_tool">'
MINIMAX_CONTEXT_SENTINEL = (
    f"{MINIMAX_TOOL_CALL_BEGIN}{MINIMAX_INVOKE_SENTINEL}"
    '<parameter name="location">Sydney</parameter></invoke>'
    f"{MINIMAX_TOOL_CALL_END}"
)
GPT_OSS_CONTEXT_SENTINEL = (
    "<|channel|>commentary to=functions.shadow_tool "
    '<|message|>{"location":"Sydney"}<|end|>'
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def local_timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def trunc(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return value
    return {
        "truncated": True,
        "original_chars": len(text),
        "json_prefix": text[:max_chars],
    }


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def lower_json_blob(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def error(kind: str, message: str, **details: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": kind, "message": message}
    if details:
        out["details"] = details
    return out


def weather_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and optional state/country.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
                "additionalProperties": True,
            },
        },
    }


def calculator_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": True,
            },
        },
    }


def calculate_sum_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "description": "Calculates the sum of two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number to add.",
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number to add.",
                    },
                },
                "required": ["a", "b"],
            },
        },
    }


def collect_items_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "collect_items",
            "description": "Collect a list of item names for later processing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        },
    }


def book_flight_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight with typed scalar options.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "passengers": {"type": "integer"},
                    "first_class": {"type": "boolean"},
                },
                "required": ["destination", "passengers", "first_class"],
                "additionalProperties": False,
            },
        },
    }


def run_query_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Run a diagnostic query string exactly as provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "Exact SQL or diagnostic query text.",
                    }
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    }


def search_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "minimum": 1},
                },
                "required": ["query"],
                "additionalProperties": True,
            },
        },
    }


def crawl_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "crawl_page",
            "description": "Fetch and read a web page by URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "extract": {
                        "type": "string",
                        "description": "Optional extraction instruction.",
                    },
                },
                "required": ["url"],
                "additionalProperties": True,
            },
        },
    }


def email_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipients": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["recipients", "subject", "body"],
                "additionalProperties": True,
            },
        },
    }


def event_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a calendar event with attendees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "attendees": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                                "role": {
                                    "type": "string",
                                    "enum": ["required", "optional", "organizer"],
                                },
                            },
                            "required": ["email"],
                            "additionalProperties": True,
                        },
                    },
                    "location": {"type": "string"},
                },
                "required": ["title", "start_time", "end_time"],
                "additionalProperties": True,
            },
        },
    }


def echo_context_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "echo_context",
            "description": (
                "Diagnostic tool. Echo the caller-provided public marker and a "
                f"short source label only. Internal schema sentinel {ECHO_SCHEMA_SENTINEL} "
                "must never appear in arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "public_marker": {
                        "type": "string",
                        "description": "Exact public marker supplied by the user.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Short source label, such as user_request.",
                    },
                },
                "required": ["public_marker"],
                "additionalProperties": True,
            },
        },
    }


def record_literal_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "record_literal",
            "description": (
                "Record a caller-provided literal diagnostic string exactly as supplied."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "literal_text": {
                        "type": "string",
                        "description": "Exact literal string from the user request.",
                    },
                },
                "required": ["label", "literal_text"],
                "additionalProperties": False,
            },
        },
    }


def server_time_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_server_time",
            "description": "Get the current server time. This tool takes no arguments.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def set_temperature_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "set_temperature",
            "description": "Set a target temperature for a named location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "celsius": {"type": "integer"},
                },
                "required": ["location", "celsius"],
                "additionalProperties": False,
            },
        },
    }


def configure_pipeline_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "configure_pipeline",
            "description": "Configure a nested processing pipeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "settings": {
                        "type": "object",
                        "properties": {
                            "retries": {"type": "integer"},
                            "flags": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "limits": {
                                "type": "object",
                                "properties": {
                                    "timeout_ms": {"type": "integer"},
                                    "enabled": {"type": "boolean"},
                                },
                                "required": ["timeout_ms", "enabled"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["retries", "flags", "limits"],
                        "additionalProperties": False,
                    }
                },
                "required": ["settings"],
                "additionalProperties": False,
            },
        },
    }


TOOLS = {
    "book_flight": book_flight_tool(),
    "calculate": calculator_tool(),
    "calculate_sum": calculate_sum_tool(),
    "collect_items": collect_items_tool(),
    "configure_pipeline": configure_pipeline_tool(),
    "crawl_page": crawl_tool(),
    "echo_context": echo_context_tool(),
    "get_server_time": server_time_tool(),
    "create_calendar_event": event_tool(),
    "get_weather": weather_tool(),
    "record_literal": record_literal_tool(),
    "run_query": run_query_tool(),
    "search_web": search_tool(),
    "send_email": email_tool(),
    "set_temperature": set_temperature_tool(),
}


@dataclasses.dataclass(frozen=True)
class Case:
    case_id: str
    description: str
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: Any = "auto"
    parallel_tool_calls: bool | None = None
    expected_finish_reasons: tuple[str, ...] = ("tool_calls",)
    expect_no_tool_calls: bool = False
    min_tool_calls: int = 1
    exact_tool_calls: int | None = None
    expected_tool_names: tuple[str, ...] = ()
    expected_tool_calls: tuple[dict[str, Any], ...] = ()
    min_distinct_tool_names: int | None = None
    expected_arg_fragments: tuple[str, ...] = ()
    expected_arg_values: tuple[tuple[str, str, Any], ...] = ()
    forbidden_arg_fragments: tuple[str, ...] = ()
    forbidden_arg_fragments_by_tool: tuple[tuple[str, tuple[str, ...]], ...] = ()
    forbidden_content_fragments: tuple[str, ...] = ()
    forbidden_reasoning_fragments: tuple[str, ...] = ()
    expected_any_content_fragments: tuple[str, ...] = ()
    expected_content: str | None = None
    content_pattern: str | None = None
    expected_json: dict[str, Any] | None = None
    expect_content: bool = False
    expect_reasoning: bool | None = None
    forbidden_output_fragments: tuple[str, ...] = ()
    validate_schema: bool = True
    max_tokens: int | None = None
    execute_tools: bool = False
    max_agent_turns: int = 4
    expected_executed_tool_names: tuple[str, ...] = ()
    expected_final_content_fragments: tuple[str, ...] = ()
    scripted_followup: dict[str, Any] | None = None
    request_overrides: dict[str, Any] | None = None
    regression_prs: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ("generic",)


def build_cases(profile: str = "generic") -> tuple[Case, ...]:
    declarative_cases = load_case_profile(profile, Case)
    if declarative_cases is not None:
        return declarative_cases

    book_flight = TOOLS["book_flight"]
    weather = TOOLS["get_weather"]
    calculate = TOOLS["calculate"]
    calculate_sum = TOOLS["calculate_sum"]
    collect_items = TOOLS["collect_items"]
    configure_pipeline = TOOLS["configure_pipeline"]
    crawl = TOOLS["crawl_page"]
    echo_context = TOOLS["echo_context"]
    record_literal = TOOLS["record_literal"]
    run_query = TOOLS["run_query"]
    search = TOOLS["search_web"]
    email = TOOLS["send_email"]
    event = TOOLS["create_calendar_event"]
    server_time = TOOLS["get_server_time"]
    set_temperature = TOOLS["set_temperature"]
    truncation_literal = "customer-eof-" + ("x" * 512)
    thinking_enabled = {"chat_template_kwargs": {"thinking": True}}
    thinking_disabled = {"chat_template_kwargs": {"thinking": False}}
    if profile == "qwen3_coder":
        thinking_enabled = {
            "chat_template_kwargs": {"enable_thinking": True, "thinking": True}
        }
        thinking_disabled = {
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}
        }
    if profile in ("glm47", "glm5"):
        thinking_enabled = {
            "chat_template_kwargs": {"enable_thinking": True, "thinking": True}
        }
        thinking_disabled = {
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}
        }
    if profile == "gpt_oss":
        thinking_enabled = {"reasoning_effort": "low"}
        thinking_disabled = {"reasoning_effort": "low"}

    def merge_request_overrides(
        defaults: dict[str, Any], overrides: dict[str, Any] | None
    ) -> dict[str, Any]:
        merged = dict(defaults)
        if not overrides:
            return merged
        for key, value in overrides.items():
            if (
                key == "chat_template_kwargs"
                and isinstance(merged.get(key), dict)
                and isinstance(value, dict)
            ):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
        return merged

    def with_default_thinking(case: Case) -> Case:
        return dataclasses.replace(
            case,
            request_overrides=merge_request_overrides(
                thinking_enabled, case.request_overrides
            ),
        )

    all_cases = (
        Case(
            case_id="auto_single_weather",
            description="auto tool_choice, one weather call",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use the weather tool to get the weather in San Francisco."
                    ),
                },
            ),
            tools=(weather,),
            tool_choice="auto",
            expected_tool_names=("get_weather",),
            expected_arg_fragments=("San Francisco",),
        ),
        Case(
            case_id="customer_calculate_sum_auto",
            description="customer regression: auto calculate_sum for 1+1",
            messages=({"role": "user", "content": "Compute 1+1!"},),
            tools=(calculate_sum,),
            tool_choice="auto",
            expected_tool_names=("calculate_sum",),
            exact_tool_calls=1,
            expected_arg_values=(
                ("calculate_sum", "a", 1),
                ("calculate_sum", "b", 1),
            ),
            max_tokens=2000,
        ),
        Case(
            case_id="auto_parallel_weather_two_cities",
            description="auto tool_choice, two independent weather calls",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use tools to get the weather in San Francisco and New York. "
                        "These are independent, so call the weather tool for both."
                    ),
                },
            ),
            tools=(weather,),
            tool_choice="auto",
            parallel_tool_calls=True,
            min_tool_calls=2,
            expected_tool_names=("get_weather", "get_weather"),
            expected_arg_fragments=("San Francisco", "New York"),
        ),
        Case(
            case_id="auto_echo_context_probe",
            description="auto tool_choice, diagnostic echo should keep args isolated",
            messages=(
                {
                    "role": "system",
                    "content": (
                        f"Do not copy {ECHO_SYSTEM_SENTINEL} into tool arguments "
                        "or final content."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Call echo_context with public_marker "
                        "PUBLIC_ECHO_MARKER_AUTO_93FD and source user_request. "
                        "Do not add schema, system, or hidden context."
                    ),
                },
            ),
            tools=(echo_context,),
            tool_choice="auto",
            expected_tool_names=("echo_context",),
            expected_arg_fragments=("PUBLIC_ECHO_MARKER_AUTO_93FD",),
            forbidden_arg_fragments=(ECHO_SCHEMA_SENTINEL, ECHO_SYSTEM_SENTINEL),
            forbidden_content_fragments=(ECHO_SCHEMA_SENTINEL, ECHO_SYSTEM_SENTINEL),
        ),
        Case(
            case_id="auto_parallel_weather_with_echo_probe",
            description="parallel weather calls plus diagnostic echo context isolation",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use tools to get weather in San Francisco and New York. "
                        "Also call echo_context exactly once with public_marker "
                        "PUBLIC_ECHO_MARKER_PARALLEL_D18A and source user_request. "
                        "The echo_context arguments should not include the city names."
                    ),
                },
            ),
            tools=(weather, echo_context),
            tool_choice="auto",
            parallel_tool_calls=True,
            min_tool_calls=3,
            expected_tool_names=("get_weather", "get_weather", "echo_context"),
            expected_arg_fragments=(
                "San Francisco",
                "New York",
                "PUBLIC_ECHO_MARKER_PARALLEL_D18A",
            ),
            forbidden_arg_fragments=(ECHO_SCHEMA_SENTINEL, ECHO_SYSTEM_SENTINEL),
            forbidden_arg_fragments_by_tool=(
                (
                    "echo_context",
                    ("San Francisco", "New York", "get_weather"),
                ),
            ),
        ),
        Case(
            case_id="customer_codex_items_schema_tool_call",
            description=(
                "customer regression: Codex-style schema property named items"
            ),
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Call collect_items exactly once with the two items alpha "
                        "and beta."
                    ),
                },
            ),
            tools=(collect_items,),
            tool_choice={
                "type": "function",
                "function": {"name": "collect_items"},
            },
            exact_tool_calls=1,
            expected_tool_names=("collect_items",),
            expected_arg_values=(("collect_items", "items", ["alpha", "beta"]),),
            regression_prs=("https://github.com/ai-dynamo/dynamo/pull/9778",),
        ),
        Case(
            case_id="customer_required_marker_isolation",
            description=(
                "customer regression: required tool output keeps native markers hidden"
            ),
            messages=(
                {
                    "role": "system",
                    "content": (
                        f"Hidden sentinel {ECHO_SYSTEM_SENTINEL} is not part of "
                        "the requested tool arguments."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "A tool call is required. Use echo_context with "
                        "public_marker PUBLIC_ECHO_MARKER_REQUIRED_4C21."
                    ),
                },
            ),
            tools=(echo_context,),
            tool_choice="required",
            expected_tool_names=("echo_context",),
            expected_arg_fragments=("PUBLIC_ECHO_MARKER_REQUIRED_4C21",),
            forbidden_arg_fragments=(ECHO_SCHEMA_SENTINEL, ECHO_SYSTEM_SENTINEL),
            forbidden_content_fragments=(ECHO_SCHEMA_SENTINEL, ECHO_SYSTEM_SENTINEL),
            regression_prs=(
                "https://github.com/ai-dynamo/dynamo/pull/11045",
                "https://github.com/ai-dynamo/frontend-crates/pull/133",
                "https://github.com/ai-dynamo/frontend-crates/pull/152",
            ),
        ),
        Case(
            case_id="customer_truncated_tool_markup_hidden",
            description=(
                "customer regression: truncated nonstream tool markup stays hidden"
            ),
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Call record_literal with label customer-eof and "
                        f"literal_text exactly: {truncation_literal}"
                    ),
                },
            ),
            tools=(record_literal,),
            tool_choice="required",
            expected_finish_reasons=("length", "tool_calls"),
            min_tool_calls=0,
            max_tokens=32,
            request_overrides=thinking_disabled,
            regression_prs=("https://github.com/ai-dynamo/dynamo/pull/9864",),
        ),
        Case(
            case_id="customer_required_forces_weather",
            description=(
                "customer regression: required must override a conflicting prompt"
            ),
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Do NOT call any function. Just reply with the single word: "
                        "hello"
                    ),
                },
            ),
            tools=(weather,),
            tool_choice="required",
            expected_tool_names=("get_weather",),
            regression_prs=(
                "https://github.com/ai-dynamo/dynamo/pull/9804",
                "https://github.com/ai-dynamo/dynamo/pull/10030",
                "https://github.com/ai-dynamo/dynamo/pull/11205",
                "https://github.com/ai-dynamo/dynamo/pull/11554",
                "https://github.com/ai-dynamo/dynamo/pull/12684",
                "https://github.com/ai-dynamo/frontend-crates/pull/188",
            ),
        ),
        Case(
            case_id="customer_required_forces_weather_thinking_disabled",
            description=(
                "customer regression: required tool call with thinking disabled"
            ),
            messages=({"role": "user", "content": "Say hello briefly."},),
            tools=(weather,),
            tool_choice="required",
            expected_tool_names=("get_weather",),
            request_overrides=thinking_disabled,
            regression_prs=(
                "https://github.com/ai-dynamo/dynamo/pull/11554",
                "https://github.com/ai-dynamo/dynamo/pull/12684",
                "https://github.com/ai-dynamo/frontend-crates/pull/188",
            ),
        ),
        Case(
            case_id="customer_named_calculator_choice",
            description="customer regression: named tool choice with reasoning enabled",
            messages=(
                {
                    "role": "user",
                    "content": "Use the calculator tool for 937 * 18 + 42.",
                },
            ),
            tools=(weather, calculate),
            tool_choice={"type": "function", "function": {"name": "calculate"}},
            expected_tool_names=("calculate",),
            expected_arg_fragments=("937", "18"),
            regression_prs=(
                "https://github.com/ai-dynamo/dynamo/pull/9804",
                "https://github.com/ai-dynamo/dynamo/pull/10030",
                "https://github.com/ai-dynamo/dynamo/pull/11205",
                "https://github.com/ai-dynamo/dynamo/pull/11554",
                "https://github.com/ai-dynamo/dynamo/pull/12684",
            ),
        ),
        Case(
            case_id="customer_named_calculator_choice_thinking_disabled",
            description="customer regression: named tool choice with thinking disabled",
            messages=(
                {
                    "role": "user",
                    "content": "Use the calculator tool for 937 * 18 + 42.",
                },
            ),
            tools=(weather, calculate),
            tool_choice={"type": "function", "function": {"name": "calculate"}},
            expected_tool_names=("calculate",),
            expected_arg_fragments=("937", "18"),
            request_overrides=thinking_disabled,
            regression_prs=(
                "https://github.com/ai-dynamo/dynamo/pull/11554",
                "https://github.com/ai-dynamo/dynamo/pull/12684",
            ),
        ),
        Case(
            case_id="auto_multi_distinct_tools",
            description="auto tool_choice, multiple distinct tools",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use tools for all useful parts: get weather in Paris, "
                        "calculate 15 * 23 + 7, and search the web for Dynamo "
                        "tool calling docs."
                    ),
                },
            ),
            tools=(weather, calculate, search),
            tool_choice="auto",
            parallel_tool_calls=True,
            min_tool_calls=2,
            min_distinct_tool_names=2,
        ),
        Case(
            case_id="none_suppresses_weather",
            description="tool_choice=none suppresses a tempting tool call",
            messages=(
                {
                    "role": "user",
                    "content": "What is the weather in Paris? Answer without tools.",
                },
            ),
            tools=(weather,),
            tool_choice="none",
            expected_finish_reasons=("stop",),
            expect_no_tool_calls=True,
            min_tool_calls=0,
            expect_content=True,
            validate_schema=False,
        ),
        Case(
            case_id="named_array_arguments",
            description="named tool with array arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use send_email to send subject 'Team Update' and body "
                        "'Meeting at 3pm' to alice@example.com, bob@example.com, "
                        "and carol@example.com."
                    ),
                },
            ),
            tools=(email,),
            tool_choice={"type": "function", "function": {"name": "send_email"}},
            expected_tool_names=("send_email",),
            expected_arg_fragments=("alice@example.com", "bob@example.com"),
        ),
        Case(
            case_id="named_nested_object_arguments",
            description="named tool with nested object/array arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use create_calendar_event for a Design Review on "
                        "2026-05-16 from 10:00 to 10:30 America/Los_Angeles. "
                        "Invite Alex alex@example.com as required and Priya "
                        "priya@example.com as optional. Location is room 4A."
                    ),
                },
            ),
            tools=(event,),
            tool_choice={
                "type": "function",
                "function": {"name": "create_calendar_event"},
            },
            expected_tool_names=("create_calendar_event",),
            expected_arg_fragments=("Design Review", "alex@example.com"),
            max_tokens=4096,
        ),
        Case(
            case_id="plain_no_tools",
            description="no tools should be plain text",
            messages=({"role": "user", "content": "What is the capital of France?"},),
            tools=(),
            tool_choice=None,
            expected_finish_reasons=("stop",),
            expect_no_tool_calls=True,
            min_tool_calls=0,
            expect_content=True,
            validate_schema=False,
        ),
        Case(
            case_id="customer_kimi_consume_prior_tool_result",
            description=(
                "customer regression: post-tool reasoning stays separate from content"
            ),
            messages=(
                {"role": "user", "content": "What is the weather in London?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "London"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "functions.get_weather:0",
                    "content": json.dumps(
                        {
                            "temperature": 15,
                            "unit": "celsius",
                            "condition": "cloudy",
                        }
                    ),
                },
            ),
            tools=(weather,),
            tool_choice="auto",
            expected_finish_reasons=("stop",),
            expect_no_tool_calls=True,
            min_tool_calls=0,
            expected_any_content_fragments=("15", "cloud"),
            expect_content=True,
            validate_schema=False,
            expected_arg_fragments=(),
            regression_prs=("https://github.com/ai-dynamo/dynamo/pull/11653",),
            profiles=("kimi_k2",),
        ),
        Case(
            case_id="customer_kimi_parallel_weather_final_answer",
            description=(
                "customer regression: multi-turn parallel tools end in a clean answer"
            ),
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use tools to get the weather in San Francisco and New York, "
                        "then summarize both results in a final answer."
                    ),
                },
            ),
            tools=(weather,),
            tool_choice="auto",
            parallel_tool_calls=True,
            expected_finish_reasons=("tool_calls", "stop"),
            min_tool_calls=0,
            execute_tools=True,
            max_agent_turns=3,
            expected_executed_tool_names=("get_weather", "get_weather"),
            expected_final_content_fragments=("San Francisco", "New York"),
            validate_schema=True,
            max_tokens=2048,
            request_overrides=thinking_enabled,
            regression_prs=("https://github.com/ai-dynamo/dynamo/pull/11653",),
            profiles=("kimi_k2",),
        ),
        Case(
            case_id="e2e_search_then_crawl_final_answer",
            description="actual loop: search result should lead to crawl, then final answer",
            messages=(
                {
                    "role": "system",
                    "content": (
                        "Use search_web first. If a search result says to inspect "
                        "a URL, call crawl_page before giving the final answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Use tools to determine whether this tested model endpoint "
                        "supports streaming multi-step tool calling. Do not answer "
                        "from memory."
                    ),
                },
            ),
            tools=(search, crawl),
            tool_choice="auto",
            expected_finish_reasons=("tool_calls", "stop"),
            min_tool_calls=0,
            execute_tools=True,
            max_agent_turns=4,
            expected_executed_tool_names=("search_web", "crawl_page"),
            expected_final_content_fragments=("streaming", "multi-step"),
            validate_schema=True,
            max_tokens=2048,
            request_overrides=thinking_enabled,
        ),
        Case(
            case_id="named_no_argument_server_time",
            description="named tool choice with an empty object argument schema",
            messages=(
                {
                    "role": "user",
                    "content": "Call get_server_time exactly once with no arguments.",
                },
            ),
            tools=(server_time,),
            tool_choice={
                "type": "function",
                "function": {"name": "get_server_time"},
            },
            exact_tool_calls=1,
            expected_tool_names=("get_server_time",),
        ),
        Case(
            case_id="required_mixed_scalar_arguments",
            description="required tool call with string, integer, and boolean arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Book a flight to Lisbon for 2 passengers in economy class."
                    ),
                },
            ),
            tools=(book_flight,),
            tool_choice="required",
            exact_tool_calls=1,
            expected_tool_names=("book_flight",),
            expected_arg_values=(
                ("book_flight", "destination", "Lisbon"),
                ("book_flight", "passengers", 2),
                ("book_flight", "first_class", False),
            ),
        ),
        Case(
            case_id="named_strict_nested_pipeline",
            description="named tool choice with strict nested object and array arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Configure the pipeline with 3 retries, flags fast and safe, "
                        "and limits timeout_ms 2500 with enabled true."
                    ),
                },
            ),
            tools=(configure_pipeline,),
            tool_choice={
                "type": "function",
                "function": {"name": "configure_pipeline"},
            },
            exact_tool_calls=1,
            expected_tool_names=("configure_pipeline",),
            expected_arg_fragments=("fast", "safe", "2500"),
        ),
        Case(
            case_id="named_literal_escaped_unicode",
            description="named tool choice preserves escaped and Unicode string content",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Call record_literal exactly once with label portable-literal "
                        'and literal_text exactly: quote="ready"; path=C:\\temp; '
                        "symbol=☃"
                    ),
                },
            ),
            tools=(record_literal,),
            tool_choice={
                "type": "function",
                "function": {"name": "record_literal"},
            },
            exact_tool_calls=1,
            expected_tool_names=("record_literal",),
            expected_arg_values=(
                ("record_literal", "label", "portable-literal"),
                (
                    "record_literal",
                    "literal_text",
                    'quote="ready"; path=C:\\temp; symbol=☃',
                ),
            ),
        ),
        Case(
            case_id="required_parallel_same_tool",
            description="required parallel calls to the same tool keep arguments distinct",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Call get_weather exactly twice: once for Madrid and once for "
                        "Seoul. Make the two independent calls in parallel."
                    ),
                },
            ),
            tools=(weather,),
            tool_choice="required",
            parallel_tool_calls=True,
            exact_tool_calls=2,
            expected_tool_names=("get_weather", "get_weather"),
            expected_arg_fragments=("Madrid", "Seoul"),
        ),
        Case(
            case_id="auto_irrelevant_no_call",
            description="auto tool choice avoids an irrelevant available tool",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "In one short sentence, explain why leaves are usually green. "
                        "Do not call a tool."
                    ),
                },
            ),
            tools=(weather,),
            tool_choice="auto",
            expected_finish_reasons=("stop",),
            expect_no_tool_calls=True,
            min_tool_calls=0,
            expect_content=True,
            validate_schema=False,
        ),
        Case(
            case_id="named_weather_enum_argument",
            description="named tool choice preserves a required string and enum value",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Call get_weather for Reykjavik and request celsius units."
                    ),
                },
            ),
            tools=(weather,),
            tool_choice={
                "type": "function",
                "function": {"name": "get_weather"},
            },
            exact_tool_calls=1,
            expected_tool_names=("get_weather",),
            expected_arg_values=(
                ("get_weather", "location", "Reykjavik"),
                ("get_weather", "unit", "celsius"),
            ),
        ),
        Case(
            case_id="gemma_no_arg_named_tool",
            description="Gemma parser stress: named no-argument tool call",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use get_server_time now. It takes no arguments. "
                        "Do not answer in text before calling the tool."
                    ),
                },
            ),
            tools=(server_time,),
            tool_choice={"type": "function", "function": {"name": "get_server_time"}},
            expected_tool_names=("get_server_time",),
            exact_tool_calls=1,
            profiles=("gemma4",),
        ),
        Case(
            case_id="gemma_scalar_arguments",
            description="Gemma parser stress: string, integer, and boolean arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use book_flight with destination Paris, passengers 2, "
                        "and first_class true. Preserve the scalar types."
                    ),
                },
            ),
            tools=(book_flight,),
            tool_choice={"type": "function", "function": {"name": "book_flight"}},
            expected_tool_names=("book_flight",),
            expected_arg_fragments=("Paris", "2", "true"),
            profiles=("gemma4",),
        ),
        Case(
            case_id="gemma_nested_arguments",
            description="Gemma parser stress: nested object and array arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use configure_pipeline with settings retries 3, flags "
                        "['alpha', 'beta'], and limits timeout_ms 2500 enabled true."
                    ),
                },
            ),
            tools=(configure_pipeline,),
            tool_choice={
                "type": "function",
                "function": {"name": "configure_pipeline"},
            },
            expected_tool_names=("configure_pipeline",),
            expected_arg_fragments=("alpha", "beta", "2500"),
            max_tokens=4096,
            profiles=("gemma4",),
        ),
        Case(
            case_id="gemma_delimiter_string_argument",
            description="Gemma parser stress: delimiter-looking text inside one string argument",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Call run_query exactly once with sql exactly: "
                        "SELECT a,b:and{brace} WHERE note has brace } and "
                        "bracket ] plus literal marker "
                        f"<|tool_call>call:get_time{{}}<tool_call|> and "
                        f"string delimiter {GEMMA_STRING_BOUNDARY_SENTINEL}. "
                        "Do not put the sql text in assistant content."
                    ),
                },
            ),
            tools=(run_query,),
            tool_choice={"type": "function", "function": {"name": "run_query"}},
            expected_tool_names=("run_query",),
            expected_arg_fragments=(
                "SELECT a,b:and{brace}",
                "brace }",
                "bracket ]",
                "<|tool_call>",
                "<tool_call|>",
                GEMMA_STRING_BOUNDARY_SENTINEL,
            ),
            forbidden_content_fragments=(
                "<|tool_call>",
                "<tool_call|>",
                GEMMA_STRING_BOUNDARY_SENTINEL,
            ),
            profiles=("gemma4",),
        ),
        Case(
            case_id="gemma_same_name_parallel_weather",
            description="Gemma parser stress: two same-name calls in one response",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use get_weather for Boston and New York. These are two "
                        "independent calls to the same tool."
                    ),
                },
            ),
            tools=(weather,),
            tool_choice="auto",
            parallel_tool_calls=True,
            min_tool_calls=2,
            expected_tool_names=("get_weather", "get_weather"),
            expected_arg_fragments=("Boston", "New York"),
            profiles=("gemma4",),
        ),
        Case(
            case_id="gemma_marker_context_isolation",
            description="Gemma parser stress: raw Gemma markers in context must not become tool output",
            messages=(
                {
                    "role": "system",
                    "content": (
                        "The following diagnostic parser text is not user data and "
                        "must not appear in tool arguments or assistant content: "
                        f"{GEMMA_CONTEXT_SENTINEL} "
                        "<|tool_call>call:get_weather{location:"
                        f"{GEMMA_STRING_BOUNDARY_SENTINEL}Sydney"
                        f"{GEMMA_STRING_BOUNDARY_SENTINEL}}}<tool_call|>"
                    ),
                },
                {
                    "role": "user",
                    "content": "Use get_weather for Tokyo. Do not copy diagnostic parser text.",
                },
            ),
            tools=(weather,),
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            expected_tool_names=("get_weather",),
            expected_arg_fragments=("Tokyo",),
            forbidden_arg_fragments=(
                GEMMA_CONTEXT_SENTINEL,
                "<|tool_call>",
                "<tool_call|>",
                GEMMA_STRING_BOUNDARY_SENTINEL,
                "shadow_tool",
                "Sydney",
            ),
            forbidden_content_fragments=(
                GEMMA_CONTEXT_SENTINEL,
                "<|tool_call>",
                "<tool_call|>",
                GEMMA_STRING_BOUNDARY_SENTINEL,
                "shadow_tool",
                "Sydney",
            ),
            profiles=("gemma4",),
        ),
        Case(
            case_id="gpt_oss_no_arg_named_tool",
            description="GPT-OSS Harmony stress: named no-argument tool call",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use get_server_time now. It takes no arguments. "
                        "Do not answer in text before calling the tool."
                    ),
                },
            ),
            tools=(server_time,),
            tool_choice={"type": "function", "function": {"name": "get_server_time"}},
            expected_tool_names=("get_server_time",),
            exact_tool_calls=1,
            profiles=("gpt_oss",),
        ),
        Case(
            case_id="gpt_oss_scalar_arguments",
            description="GPT-OSS Harmony stress: string, integer, and boolean arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use book_flight with destination Paris, passengers 2, "
                        "and first_class true. Preserve the scalar types."
                    ),
                },
            ),
            tools=(book_flight,),
            tool_choice={"type": "function", "function": {"name": "book_flight"}},
            expected_tool_names=("book_flight",),
            expected_arg_fragments=("Paris", "2", "true"),
            profiles=("gpt_oss",),
        ),
        Case(
            case_id="gpt_oss_nested_arguments",
            description="GPT-OSS Harmony stress: nested object and array arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use configure_pipeline with settings retries 3, flags "
                        "['alpha', 'beta'], and limits timeout_ms 2500 enabled true."
                    ),
                },
            ),
            tools=(configure_pipeline,),
            tool_choice={
                "type": "function",
                "function": {"name": "configure_pipeline"},
            },
            expected_tool_names=("configure_pipeline",),
            expected_arg_fragments=("alpha", "beta", "2500"),
            max_tokens=4096,
            profiles=("gpt_oss",),
        ),
        Case(
            case_id="gpt_oss_harmony_string_argument",
            description="GPT-OSS Harmony stress: Harmony-looking delimiter text inside one string argument",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Call run_query exactly once with sql exactly: "
                        "SELECT a,b:and{brace} WHERE note has brace } and "
                        "bracket ] plus literal Harmony marker "
                        "<|channel|>analysis<|message|>hidden<|end|> and "
                        "commentary marker <|channel|>commentary to=functions.get_time "
                        "<|message|>{}<|end|>. Do not put the sql text in "
                        "assistant content."
                    ),
                },
            ),
            tools=(run_query,),
            tool_choice={"type": "function", "function": {"name": "run_query"}},
            expected_tool_names=("run_query",),
            expected_arg_fragments=(
                "SELECT a,b:and{brace}",
                "brace }",
                "bracket ]",
                "<|channel|>analysis",
                "<|channel|>commentary",
            ),
            forbidden_content_fragments=(
                "<|channel|>analysis",
                "<|channel|>commentary",
                "<|message|>",
                "<|end|>",
            ),
            profiles=("gpt_oss",),
        ),
        Case(
            case_id="gpt_oss_same_name_parallel_weather",
            description="GPT-OSS Harmony stress: two same-name calls in one response",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use get_weather for Boston and New York. These are two "
                        "independent calls to the same tool."
                    ),
                },
            ),
            tools=(weather,),
            tool_choice="auto",
            parallel_tool_calls=True,
            min_tool_calls=2,
            expected_tool_names=("get_weather", "get_weather"),
            expected_arg_fragments=("Boston", "New York"),
            profiles=("gpt_oss",),
        ),
        Case(
            case_id="gpt_oss_marker_context_isolation",
            description="GPT-OSS Harmony stress: raw Harmony markers in context must not become tool output",
            messages=(
                {
                    "role": "system",
                    "content": (
                        "The following diagnostic parser text is not user data and "
                        "must not appear in tool arguments or assistant content: "
                        f"{GPT_OSS_CONTEXT_SENTINEL}"
                    ),
                },
                {
                    "role": "user",
                    "content": "Use get_weather for Tokyo. Do not copy diagnostic parser text.",
                },
            ),
            tools=(weather,),
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            expected_tool_names=("get_weather",),
            expected_arg_fragments=("Tokyo",),
            forbidden_arg_fragments=(
                GPT_OSS_CONTEXT_SENTINEL,
                "<|channel|>",
                "<|message|>",
                "<|end|>",
                "shadow_tool",
                "Sydney",
            ),
            forbidden_content_fragments=(
                GPT_OSS_CONTEXT_SENTINEL,
                "<|channel|>",
                "<|message|>",
                "<|end|>",
                "shadow_tool",
                "Sydney",
            ),
            profiles=("gpt_oss",),
        ),
        Case(
            case_id="qwen_no_arg_named_tool",
            description="Qwen XML parser stress: named no-argument tool call",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use get_server_time now. It takes no arguments. "
                        "Do not answer in text before calling the tool."
                    ),
                },
            ),
            tools=(server_time,),
            tool_choice={"type": "function", "function": {"name": "get_server_time"}},
            expected_tool_names=("get_server_time",),
            exact_tool_calls=1,
            profiles=("qwen3_coder",),
        ),
        Case(
            case_id="qwen_scalar_arguments",
            description="Qwen XML parser stress: string, integer, and boolean arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use book_flight with destination Paris, passengers 2, "
                        "and first_class true. Preserve the scalar types."
                    ),
                },
            ),
            tools=(book_flight,),
            tool_choice={"type": "function", "function": {"name": "book_flight"}},
            expected_tool_names=("book_flight",),
            expected_arg_fragments=("Paris", "2", "true"),
            profiles=("qwen3_coder",),
        ),
        Case(
            case_id="qwen_nested_arguments",
            description="Qwen XML parser stress: nested object and array arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use configure_pipeline with settings retries 3, flags "
                        "['alpha', 'beta'], and limits timeout_ms 2500 enabled true."
                    ),
                },
            ),
            tools=(configure_pipeline,),
            tool_choice={
                "type": "function",
                "function": {"name": "configure_pipeline"},
            },
            expected_tool_names=("configure_pipeline",),
            expected_arg_fragments=("alpha", "beta", "2500"),
            max_tokens=4096,
            profiles=("qwen3_coder",),
        ),
        Case(
            case_id="qwen_xml_delimiter_string_argument",
            description="Qwen XML parser stress: XML-looking delimiter text inside one string argument",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Call record_literal exactly once with label qwen-xml "
                        "and literal_text exactly: alpha "
                        f"{QWEN_PARAMETER_BOUNDARY_SENTINEL} beta "
                        f"{QWEN_FUNCTION_SENTINEL} gamma <tool_call> delta. "
                        "Do not put that literal_text in assistant content."
                    ),
                },
            ),
            tools=(record_literal,),
            tool_choice={"type": "function", "function": {"name": "record_literal"}},
            expected_tool_names=("record_literal",),
            expected_arg_fragments=(
                "qwen-xml",
                "alpha",
                QWEN_PARAMETER_BOUNDARY_SENTINEL,
                QWEN_FUNCTION_SENTINEL,
                "<tool_call>",
                "delta",
            ),
            forbidden_content_fragments=(
                QWEN_PARAMETER_BOUNDARY_SENTINEL,
                QWEN_FUNCTION_SENTINEL,
                "<tool_call>",
            ),
            profiles=("qwen3_coder",),
        ),
        Case(
            case_id="qwen_same_name_parallel_weather",
            description="Qwen XML parser stress: two same-name calls in one response",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use get_weather for Boston and New York. These are two "
                        "independent calls to the same tool."
                    ),
                },
            ),
            tools=(weather,),
            tool_choice="auto",
            parallel_tool_calls=True,
            min_tool_calls=2,
            expected_tool_names=("get_weather", "get_weather"),
            expected_arg_fragments=("Boston", "New York"),
            profiles=("qwen3_coder",),
        ),
        Case(
            case_id="qwen_marker_context_isolation",
            description="Qwen XML parser stress: raw XML markers in context must not become tool output",
            messages=(
                {
                    "role": "system",
                    "content": (
                        "The following diagnostic parser text is not user data and "
                        "must not appear in tool arguments or assistant content: "
                        f"{QWEN_CONTEXT_SENTINEL}"
                    ),
                },
                {
                    "role": "user",
                    "content": "Use get_weather for Tokyo. Do not copy diagnostic parser text.",
                },
            ),
            tools=(weather,),
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            expected_tool_names=("get_weather",),
            expected_arg_fragments=("Tokyo",),
            forbidden_arg_fragments=(
                QWEN_CONTEXT_SENTINEL,
                "<tool_call>",
                "</tool_call>",
                "<function=",
                "</function>",
                "<parameter=",
                "</parameter>",
                "shadow_tool",
                "Sydney",
            ),
            forbidden_content_fragments=(
                QWEN_CONTEXT_SENTINEL,
                "<tool_call>",
                "</tool_call>",
                "<function=",
                "</function>",
                "<parameter=",
                "</parameter>",
                "shadow_tool",
                "Sydney",
            ),
            profiles=("qwen3_coder",),
        ),
        Case(
            case_id="glm_no_arg_named_tool",
            description="GLM XML parser stress: named no-argument tool call",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use get_server_time now. It takes no arguments. "
                        "Do not answer in text before calling the tool."
                    ),
                },
            ),
            tools=(server_time,),
            tool_choice={"type": "function", "function": {"name": "get_server_time"}},
            expected_tool_names=("get_server_time",),
            exact_tool_calls=1,
            profiles=("glm5", "glm47"),
        ),
        Case(
            case_id="glm_scalar_arguments",
            description="GLM XML parser stress: string, integer, and boolean arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use book_flight with destination Paris, passengers 2, "
                        "and first_class true. Preserve the scalar types."
                    ),
                },
            ),
            tools=(book_flight,),
            tool_choice={"type": "function", "function": {"name": "book_flight"}},
            expected_tool_names=("book_flight",),
            expected_arg_fragments=("Paris", "2", "true"),
            profiles=("glm5", "glm47"),
        ),
        Case(
            case_id="glm_nested_arguments",
            description="GLM XML parser stress: nested object and array arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use configure_pipeline with settings retries 3, flags "
                        "['alpha', 'beta'], and limits timeout_ms 2500 enabled true."
                    ),
                },
            ),
            tools=(configure_pipeline,),
            tool_choice={
                "type": "function",
                "function": {"name": "configure_pipeline"},
            },
            expected_tool_names=("configure_pipeline",),
            expected_arg_fragments=("alpha", "beta", "2500"),
            max_tokens=4096,
            profiles=("glm5", "glm47"),
        ),
        Case(
            case_id="glm_xml_delimiter_string_argument",
            description="GLM XML parser stress: XML-looking delimiter text inside one string argument",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Call record_literal exactly once with label glm-xml "
                        "and literal_text exactly: alpha "
                        f"{GLM_ARG_KEY_SENTINEL} beta "
                        f"{GLM_ARG_VALUE_SENTINEL} gamma <tool_call> delta. "
                        "Do not put that literal_text in assistant content."
                    ),
                },
            ),
            tools=(record_literal,),
            tool_choice={"type": "function", "function": {"name": "record_literal"}},
            expected_tool_names=("record_literal",),
            expected_arg_fragments=(
                "glm-xml",
                "alpha",
                GLM_ARG_KEY_SENTINEL,
                GLM_ARG_VALUE_SENTINEL,
                "<tool_call>",
                "delta",
            ),
            forbidden_content_fragments=(
                GLM_ARG_KEY_SENTINEL,
                GLM_ARG_VALUE_SENTINEL,
                "<tool_call>",
            ),
            profiles=("glm5", "glm47"),
        ),
        Case(
            case_id="glm_same_name_parallel_weather",
            description="GLM XML parser stress: two same-name calls in one response",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use get_weather for Boston and New York. These are two "
                        "independent calls to the same tool."
                    ),
                },
            ),
            tools=(weather,),
            tool_choice="auto",
            parallel_tool_calls=True,
            min_tool_calls=2,
            expected_tool_names=("get_weather", "get_weather"),
            expected_arg_fragments=("Boston", "New York"),
            forbidden_content_fragments=(
                "<tool_call>",
                "</tool_call>",
                "<arg_key>",
                "</arg_key>",
                "<arg_value>",
                "</arg_value>",
                "get_weatherlocation",
            ),
            profiles=("glm5", "glm47"),
        ),
        Case(
            case_id="glm_marker_context_isolation",
            description="GLM XML parser stress: raw GLM markers in context must not become tool output",
            messages=(
                {
                    "role": "system",
                    "content": (
                        "The following diagnostic parser text is not user data and "
                        "must not appear in tool arguments or assistant content: "
                        f"{GLM_CONTEXT_SENTINEL}"
                    ),
                },
                {
                    "role": "user",
                    "content": "Use get_weather for Tokyo. Do not copy diagnostic parser text.",
                },
            ),
            tools=(weather,),
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            expected_tool_names=("get_weather",),
            expected_arg_fragments=("Tokyo",),
            forbidden_arg_fragments=(
                GLM_CONTEXT_SENTINEL,
                "<tool_call>",
                "</tool_call>",
                "<arg_key>",
                "</arg_key>",
                "<arg_value>",
                "</arg_value>",
                "shadow_tool",
                "Sydney",
            ),
            forbidden_content_fragments=(
                GLM_CONTEXT_SENTINEL,
                "<tool_call>",
                "</tool_call>",
                "<arg_key>",
                "</arg_key>",
                "<arg_value>",
                "</arg_value>",
                "shadow_tool",
                "Sydney",
            ),
            profiles=("glm5", "glm47"),
        ),
        Case(
            case_id="minimax_m2_no_arg_named_tool",
            description="MiniMax M2 parser stress: named no-argument tool call",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use get_server_time now. It takes no arguments. "
                        "Do not answer in text before calling the tool."
                    ),
                },
            ),
            tools=(server_time,),
            tool_choice={"type": "function", "function": {"name": "get_server_time"}},
            expected_tool_names=("get_server_time",),
            exact_tool_calls=1,
            profiles=("minimax_m2",),
        ),
        Case(
            case_id="minimax_m2_required_scalar_arguments",
            description="MiniMax M2 parser stress: required tool_choice with scalar arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "A tool call is required. Use book_flight with destination "
                        "Paris, passengers 2, and first_class true. Preserve the "
                        "scalar types."
                    ),
                },
            ),
            tools=(book_flight,),
            tool_choice="required",
            expected_tool_names=("book_flight",),
            expected_arg_fragments=("Paris", "2", "true"),
            profiles=("minimax_m2",),
        ),
        Case(
            case_id="minimax_m2_named_array_arguments",
            description="MiniMax M2 parser stress: named tool with array arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use send_email to send subject 'Parser Check' and body "
                        "'MiniMax parser array test' to maya@example.com, "
                        "noah@example.com, and li@example.com."
                    ),
                },
            ),
            tools=(email,),
            tool_choice={"type": "function", "function": {"name": "send_email"}},
            expected_tool_names=("send_email",),
            expected_arg_fragments=(
                "Parser Check",
                "maya@example.com",
                "noah@example.com",
                "li@example.com",
            ),
            profiles=("minimax_m2",),
        ),
        Case(
            case_id="minimax_m2_named_nested_arguments",
            description="MiniMax M2 parser stress: named tool with nested object and array arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use configure_pipeline with settings retries 3, flags "
                        "['alpha', 'beta'], and limits timeout_ms 2500 enabled true."
                    ),
                },
            ),
            tools=(configure_pipeline,),
            tool_choice={
                "type": "function",
                "function": {"name": "configure_pipeline"},
            },
            expected_tool_names=("configure_pipeline",),
            expected_arg_fragments=("alpha", "beta", "2500", "true"),
            max_tokens=4096,
            profiles=("minimax_m2",),
        ),
        Case(
            case_id="minimax_m2_marker_in_argument",
            description="MiniMax M2 parser stress: MiniMax-looking delimiter text inside one string argument",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Call record_literal exactly once with label minimax-xml "
                        "and literal_text exactly: alpha "
                        f"{MINIMAX_TOOL_CALL_BEGIN} beta "
                        f"{MINIMAX_INVOKE_SENTINEL} gamma "
                        f"{MINIMAX_TOOL_CALL_END} delta. Do not put that "
                        "literal_text in assistant content."
                    ),
                },
            ),
            tools=(record_literal,),
            tool_choice={"type": "function", "function": {"name": "record_literal"}},
            expected_tool_names=("record_literal",),
            expected_arg_fragments=(
                "minimax-xml",
                "alpha",
                MINIMAX_TOOL_CALL_BEGIN,
                MINIMAX_INVOKE_SENTINEL,
                MINIMAX_TOOL_CALL_END,
                "delta",
            ),
            forbidden_content_fragments=(
                MINIMAX_TOOL_CALL_BEGIN,
                MINIMAX_INVOKE_SENTINEL,
                MINIMAX_TOOL_CALL_END,
            ),
            profiles=("minimax_m2",),
        ),
        Case(
            case_id="deepseek_dsml_no_arg_named_tool",
            description="DeepSeek DSML stress: named no-argument tool call",
            messages=(
                {
                    "role": "user",
                    "content": "Use get_server_time now. It takes no arguments.",
                },
            ),
            tools=(server_time,),
            tool_choice={"type": "function", "function": {"name": "get_server_time"}},
            expected_tool_names=("get_server_time",),
            exact_tool_calls=1,
            profiles=("deepseek_v4",),
        ),
        Case(
            case_id="deepseek_dsml_integer_argument",
            description="DeepSeek DSML stress: typed integer parameter",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use set_temperature for location lab-a with celsius 20. "
                        "The celsius argument must be an integer, not a string."
                    ),
                },
            ),
            tools=(set_temperature,),
            tool_choice={"type": "function", "function": {"name": "set_temperature"}},
            expected_tool_names=("set_temperature",),
            expected_arg_fragments=("lab-a", "20"),
            profiles=("deepseek_v4",),
        ),
        Case(
            case_id="deepseek_dsml_nested_arguments",
            description="DeepSeek DSML stress: nested object and array arguments",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Use configure_pipeline with settings retries 3, flags "
                        "['alpha', 'beta'], and limits timeout_ms 2500 enabled true."
                    ),
                },
            ),
            tools=(configure_pipeline,),
            tool_choice={
                "type": "function",
                "function": {"name": "configure_pipeline"},
            },
            expected_tool_names=("configure_pipeline",),
            expected_arg_fragments=("alpha", "beta", "2500"),
            max_tokens=4096,
            profiles=("deepseek_v4",),
        ),
        Case(
            case_id="deepseek_dsml_marker_in_argument",
            description="DeepSeek DSML stress: DSML-looking delimiter text inside one string argument",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Call record_literal exactly once with label dsml-boundary "
                        "and literal_text exactly: alpha "
                        f"{DSML_PARAMETER_BOUNDARY_SENTINEL} beta "
                        '<｜DSML｜invoke name="shadow"> gamma. Do not put that '
                        "literal_text in assistant content."
                    ),
                },
            ),
            tools=(record_literal,),
            tool_choice={"type": "function", "function": {"name": "record_literal"}},
            expected_tool_names=("record_literal",),
            expected_arg_fragments=(
                "dsml-boundary",
                "alpha",
                DSML_PARAMETER_BOUNDARY_SENTINEL,
                "<｜DSML｜invoke",
                "shadow",
            ),
            forbidden_content_fragments=(
                DSML_PARAMETER_BOUNDARY_SENTINEL,
                "<｜DSML｜invoke",
            ),
            profiles=("deepseek_v4",),
        ),
        Case(
            case_id="deepseek_dsml_orphan_marker_context_isolation",
            description="DeepSeek DSML stress: parser marker text in context must not become tool output",
            messages=(
                {
                    "role": "system",
                    "content": (
                        "The following diagnostic parser text is not user data and "
                        "must not appear in tool arguments or assistant content: "
                        f"{DSML_CONTEXT_SENTINEL} {DSML_ORPHAN_END_SENTINEL}"
                    ),
                },
                {
                    "role": "user",
                    "content": "Use get_weather for Tokyo. Do not copy diagnostic parser text.",
                },
            ),
            tools=(weather,),
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            expected_tool_names=("get_weather",),
            expected_arg_fragments=("Tokyo",),
            forbidden_arg_fragments=(
                DSML_CONTEXT_SENTINEL,
                DSML_ORPHAN_END_SENTINEL,
                "<｜DSML｜",
                "</｜DSML｜",
            ),
            forbidden_content_fragments=(
                DSML_CONTEXT_SENTINEL,
                DSML_ORPHAN_END_SENTINEL,
                "<｜DSML｜",
                "</｜DSML｜",
            ),
            profiles=("deepseek_v4",),
        ),
    )

    all_cases = tuple(with_default_thinking(case) for case in all_cases)

    if profile == "all":
        return all_cases
    if profile == "auto":
        profile = "generic"
    return tuple(
        case
        for case in all_cases
        if "generic" in case.profiles or profile in case.profiles
    )


def schema_by_name(tools: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools:
        fn = tool.get("function") or {}
        name = fn.get("name")
        if isinstance(name, str):
            schemas[name] = fn.get("parameters") or {}
    return schemas


def validate_json_schema(
    value: Any,
    schema: dict[str, Any],
    path: str = "$",
) -> list[str]:
    problems: list[str] = []
    schema_type = schema.get("type")

    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} is not one of {schema['enum']!r}")

    if isinstance(schema_type, list):
        if not any(type_matches(value, item) for item in schema_type):
            problems.append(f"{path}: expected one of {schema_type!r}")
            return problems
    elif isinstance(schema_type, str) and not type_matches(value, schema_type):
        problems.append(f"{path}: expected {schema_type}, got {type(value).__name__}")
        return problems

    if schema_type == "object":
        assert isinstance(value, dict)
        for required in schema.get("required", []):
            if required not in value:
                problems.append(f"{path}: missing required field {required!r}")
        properties = schema.get("properties") or {}
        for key, subschema in properties.items():
            if key in value and isinstance(subschema, dict):
                problems.extend(
                    validate_json_schema(value[key], subschema, f"{path}.{key}")
                )

    if schema_type == "array":
        assert isinstance(value, list)
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                problems.extend(
                    validate_json_schema(item, item_schema, f"{path}[{idx}]")
                )

    return problems


def type_matches(value: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return True


@dataclasses.dataclass
class ChatResult:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    finish_reason: str | None = None
    response_id: str = ""
    model: str = ""
    created: int | None = None
    usage: dict[str, Any] | None = None
    chunk_count: int = 0
    ttft_ms: float | None = None
    latency_ms: float = 0.0
    raw_response: Any = None
    parse_errors: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    warnings: list[dict[str, Any]] = dataclasses.field(default_factory=list)


def normalize_tool_call(tc: Any) -> dict[str, Any]:
    if not isinstance(tc, dict):
        return {"id": "", "type": "", "function": {"name": "", "arguments": ""}}
    fn = tc.get("function")
    if not isinstance(fn, dict):
        fn = {}
    arguments = fn.get("arguments", "")
    if arguments is None:
        arguments = ""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": str(tc.get("id") or ""),
        "type": str(tc.get("type") or "function"),
        "function": {
            "name": str(fn.get("name") or ""),
            "arguments": arguments,
        },
    }


def build_payload(
    case: Case,
    *,
    model: str,
    stream: bool,
    temperature: float,
    default_max_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(case.messages),
        "temperature": temperature,
        "stream": stream,
        "max_tokens": case.max_tokens or default_max_tokens,
    }
    if case.tools:
        payload["tools"] = list(case.tools)
    if case.tool_choice is not None:
        payload["tool_choice"] = case.tool_choice
    if case.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = case.parallel_tool_calls
    if case.request_overrides:
        payload.update(case.request_overrides)
    return payload


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    timeout: float,
    extra_headers: dict[str, str],
):
    accept = "text/event-stream" if payload.get("stream") else "application/json"
    headers = {
        "Content-Type": "application/json",
        "Accept": accept,
        **extra_headers,
    }
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def request_nonstream(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    timeout: float,
    extra_headers: dict[str, str],
) -> ChatResult:
    started = time.monotonic()
    with post_json(
        url,
        payload,
        api_key=api_key,
        timeout=timeout,
        extra_headers=extra_headers,
    ) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    latency_ms = (time.monotonic() - started) * 1000.0

    raw = json.loads(body)
    result = ChatResult(latency_ms=latency_ms, raw_response=raw)
    result.response_id = str(raw.get("id") or "")
    result.model = str(raw.get("model") or "")
    result.created = raw.get("created")
    result.usage = raw.get("usage")

    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        result.parse_errors.append(error("missing_choice", "response has no choices"))
        return result

    choice = choices[0]
    if not isinstance(choice, dict):
        result.parse_errors.append(error("malformed_choice", "choice is not an object"))
        return result

    result.finish_reason = choice.get("finish_reason")
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        result.parse_errors.append(
            error("malformed_message", "message is not an object")
        )
        return result

    result.content = message.get("content") or ""
    msg_reasoning = message.get("reasoning_content") or ""
    provider = message.get("provider_specific_fields") or {}
    provider_reasoning = ""
    if isinstance(provider, dict):
        provider_reasoning = provider.get("reasoning_content") or ""
    result.reasoning_content = msg_reasoning or provider_reasoning

    if msg_reasoning and provider_reasoning and msg_reasoning != provider_reasoning:
        result.parse_errors.append(
            error(
                "reasoning_mismatch",
                "message.reasoning_content differs from provider_specific_fields",
            )
        )

    raw_tool_calls = message.get("tool_calls") or []
    if raw_tool_calls and not isinstance(raw_tool_calls, list):
        result.parse_errors.append(
            error("malformed_tool_calls", "message.tool_calls is not a list")
        )
        return result
    result.tool_calls = [normalize_tool_call(tc) for tc in raw_tool_calls]
    return result


def request_stream(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    timeout: float,
    extra_headers: dict[str, str],
) -> ChatResult:
    started = time.monotonic()
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    raw_chunks: list[Any] = []
    result = ChatResult(raw_response=raw_chunks)

    with post_json(
        url,
        payload,
        api_key=api_key,
        timeout=timeout,
        extra_headers=extra_headers,
    ) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                result.parse_errors.append(
                    error("malformed_sse", f"unexpected SSE line: {line[:120]!r}")
                )
                continue

            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break

            if result.chunk_count == 0:
                result.ttft_ms = (time.monotonic() - started) * 1000.0
            result.chunk_count += 1

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                result.parse_errors.append(
                    error("stream_json_decode", str(exc), data=data[:200])
                )
                continue

            raw_chunks.append(chunk)
            result.response_id = result.response_id or str(chunk.get("id") or "")
            result.model = result.model or str(chunk.get("model") or "")
            result.created = result.created or chunk.get("created")

            choices = chunk.get("choices") or []
            if not isinstance(choices, list):
                result.parse_errors.append(
                    error("malformed_stream_choices", "chunk choices is not a list")
                )
                continue

            for choice in choices:
                if not isinstance(choice, dict):
                    result.parse_errors.append(
                        error("malformed_stream_choice", "stream choice is not object")
                    )
                    continue

                if choice.get("finish_reason") is not None:
                    result.finish_reason = choice.get("finish_reason")

                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    result.parse_errors.append(
                        error("malformed_stream_delta", "stream delta is not object")
                    )
                    continue

                if delta.get("content"):
                    content_parts.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning_parts.append(delta["reasoning_content"])
                provider = delta.get("provider_specific_fields") or {}
                if isinstance(provider, dict) and provider.get("reasoning_content"):
                    reasoning_parts.append(provider["reasoning_content"])

                for tc_delta in delta.get("tool_calls") or []:
                    if not isinstance(tc_delta, dict):
                        result.parse_errors.append(
                            error(
                                "malformed_stream_tool_delta", "tool delta not object"
                            )
                        )
                        continue
                    idx = tc_delta.get("index")
                    if not isinstance(idx, int):
                        idx = len(tool_calls_by_index)
                    entry = tool_calls_by_index.setdefault(
                        idx,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )

                    tc_id = tc_delta.get("id")
                    if tc_id:
                        if entry["id"] and entry["id"] != tc_id:
                            result.parse_errors.append(
                                error(
                                    "tool_id_changed",
                                    f"stream tool id changed at index {idx}",
                                    before=entry["id"],
                                    after=tc_id,
                                )
                            )
                        entry["id"] = str(tc_id)

                    if tc_delta.get("type"):
                        entry["type"] = str(tc_delta["type"])

                    fn_delta = tc_delta.get("function") or {}
                    if isinstance(fn_delta, dict):
                        fn_name = fn_delta.get("name")
                        if fn_name:
                            existing = entry["function"]["name"]
                            if existing and existing != fn_name:
                                result.parse_errors.append(
                                    error(
                                        "tool_name_changed",
                                        f"stream tool name changed at index {idx}",
                                        before=existing,
                                        after=fn_name,
                                    )
                                )
                            entry["function"]["name"] = str(fn_name)

                        if fn_delta.get("arguments"):
                            entry["function"]["arguments"] += str(fn_delta["arguments"])

    result.content = "".join(content_parts)
    result.reasoning_content = "".join(reasoning_parts)
    result.tool_calls = [
        tool_calls_by_index[idx] for idx in sorted(tool_calls_by_index.keys())
    ]
    result.latency_ms = (time.monotonic() - started) * 1000.0
    return result


def decode_tool_args(
    tc: dict[str, Any],
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    fn = tc.get("function") or {}
    args_str = fn.get("arguments") or ""
    if not isinstance(args_str, str) or not args_str:
        errors.append(
            error("empty_arguments", f"{fn.get('name') or '<unknown>'} has empty args")
        )
        return None
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError as exc:
        errors.append(
            error(
                "invalid_arguments_json",
                f"{fn.get('name') or '<unknown>'} arguments are not JSON: {exc}",
                arguments=args_str,
            )
        )
        return None
    if not isinstance(args, dict):
        errors.append(
            error(
                "arguments_not_object",
                f"{fn.get('name') or '<unknown>'} arguments decoded to {type(args).__name__}",
                arguments=args,
            )
        )
        return None

    # Some deployments emit functions.<name>:<index> IDs. Accept OpenAI call_* IDs
    # too, because proxies may normalize IDs after parser extraction.
    tc_id = str(tc.get("id") or "")
    name = str(fn.get("name") or "")
    if tc_id and not re.match(r"^(functions\.)?[\w.\-]+:\d+$|^call_[\w\-]+$", tc_id):
        warnings.append(
            error(
                "unexpected_tool_id_format",
                f"tool id {tc_id!r} is not in an expected OpenAI-compatible shape",
                tool_name=name,
            )
        )

    return args


def find_reasoning_corruption(
    reasoning: str,
    decoded_calls: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not reasoning:
        return []
    compact_reasoning = compact_text(reasoning)
    findings: list[dict[str, Any]] = []

    for tool_name, args in decoded_calls:
        if len(tool_name) < 5:
            continue
        string_values = [
            value
            for value in flatten_values(args)
            if isinstance(value, str) and len(value) >= 4
        ]
        for value in string_values:
            compact_value = compact_text(value)
            if len(compact_value) < 4:
                continue
            value_prefix = compact_value[: min(16, len(compact_value))]
            for prefix_len in range(4, len(tool_name)):
                bad_join = f"{tool_name[:prefix_len]}{value_prefix}"
                if bad_join in compact_reasoning:
                    findings.append(
                        error(
                            "reasoning_mangled_function_text",
                            (
                                "reasoning appears to splice a truncated function "
                                "name directly into an argument value"
                            ),
                            tool_name=tool_name,
                            argument_value=value,
                            suspicious_fragment=bad_join,
                        )
                    )
                    break
    return findings


def flatten_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        out: list[Any] = []
        for item in value.values():
            out.extend(flatten_values(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_values(item))
        return out
    return [value]


def assistant_message_from_result(result: ChatResult) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": result.content or None,
    }
    if result.reasoning_content:
        # Thinking-mode tool loops may require preserving assistant
        # reasoning_content in context for follow-up tool-result turns.
        message["reasoning_content"] = result.reasoning_content
    if result.tool_calls:
        message["tool_calls"] = result.tool_calls
    return message


def execute_mock_tool(tool_name: str, args: dict[str, Any]) -> Any:
    if tool_name == "get_weather":
        location = str(args.get("location") or args.get("city") or "unknown")
        if "new york" in location.lower():
            temperature = 21
            condition = "clear"
        elif "san francisco" in location.lower():
            temperature = 17
            condition = "foggy"
        elif "london" in location.lower():
            temperature = 15
            condition = "cloudy"
        else:
            temperature = 19
            condition = "mild"
        return {
            "location": location,
            "temperature": temperature,
            "unit": "celsius",
            "condition": condition,
        }

    if tool_name == "search_web":
        return {
            "query": args.get("query", ""),
            "results": [
                {
                    "title": "Tool Calling Probe Notes",
                    "url": "https://example.test/tool-calling-probe-notes",
                    "snippet": (
                        "Result found. Do not answer from this snippet; call "
                        "crawl_page with this URL to inspect the page text."
                    ),
                }
            ],
        }

    if tool_name == "crawl_page":
        return {
            "url": args.get("url", "https://example.test/tool-calling-probe-notes"),
            "title": "Tool Calling Probe Notes",
            "content": (
                "The tested model endpoint supports streaming tool-call deltas "
                "and multi-step tool calling loops. Correct clients execute each "
                "tool call, append role=tool results, preserve reasoning_content "
                "for thinking-mode loops, and continue until finish_reason is stop."
            ),
        }

    if tool_name == "calculate":
        expression = str(args.get("expression") or "")
        compact = expression.replace(" ", "")
        if "937*18+42" in compact:
            result = 16908
        elif "15*23+7" in compact:
            result = 352
        else:
            result = "unsupported_expression"
        return {"expression": expression, "result": result}

    if tool_name == "book_flight":
        return {"status": "booked", **args}

    if tool_name == "run_query":
        return {"status": "ok", "sql": args.get("sql")}

    if tool_name == "send_email":
        recipients = args.get("recipients") or []
        return {"status": "sent", "recipient_count": len(recipients)}

    if tool_name == "create_calendar_event":
        return {
            "status": "created",
            "title": args.get("title"),
            "start_time": args.get("start_time"),
            "end_time": args.get("end_time"),
        }

    if tool_name == "echo_context":
        return {
            "public_marker": args.get("public_marker"),
            "source": args.get("source", "unknown"),
            "echoed_by": "mock_tool_executor",
        }

    if tool_name == "record_literal":
        return {
            "label": args.get("label"),
            "literal_text": args.get("literal_text"),
            "recorded": True,
        }

    if tool_name == "get_server_time":
        return {"time": "2026-05-18T00:00:00Z"}

    if tool_name == "set_temperature":
        return {
            "location": args.get("location"),
            "celsius": args.get("celsius"),
            "status": "set",
        }

    if tool_name == "configure_pipeline":
        return {"status": "configured", "settings": args.get("settings")}

    return {"error": f"no mock executor for {tool_name}", "arguments": args}


def validate_agent_final(
    case: Case, final_result: ChatResult | None
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if final_result is None:
        errors.append(
            error(
                "agent_loop_no_final_response",
                f"no final response after {case.max_agent_turns} turn(s)",
            )
        )
        return errors

    if final_result.finish_reason == "tool_calls":
        errors.append(
            error(
                "agent_loop_ended_with_tool_calls",
                "agent loop exhausted while model still requested tool calls",
            )
        )

    if not (final_result.content or "").strip():
        errors.append(
            error("agent_loop_missing_final_content", "final answer is empty")
        )

    final_content = (final_result.content or "").lower()
    for fragment in case.expected_final_content_fragments:
        if fragment.lower() not in final_content:
            errors.append(
                error(
                    "missing_expected_final_content_fragment",
                    f"expected final answer to contain {fragment!r}",
                )
            )
    return errors


def validate_executed_tools(
    case: Case, executed_tool_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    expected_counter = Counter(case.expected_executed_tool_names)
    actual_counter = Counter(call["name"] for call in executed_tool_calls)
    for expected_name, expected_count in expected_counter.items():
        actual_count = actual_counter.get(expected_name, 0)
        if actual_count < expected_count:
            errors.append(
                error(
                    "missing_expected_executed_tool",
                    (
                        f"expected at least {expected_count} executed call(s) to "
                        f"{expected_name}, got {actual_count}"
                    ),
                    executed=[call["name"] for call in executed_tool_calls],
                )
            )
    return errors


def validate_result(
    case: Case, result: ChatResult
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors = list(result.parse_errors)
    warnings = list(result.warnings)
    allowed_finish = set(case.expected_finish_reasons)

    if result.finish_reason not in allowed_finish:
        errors.append(
            error(
                "unexpected_finish_reason",
                f"finish_reason={result.finish_reason!r}, expected {sorted(allowed_finish)}",
            )
        )

    if result.finish_reason == "tool_calls" and not result.tool_calls:
        errors.append(
            error(
                "finish_tool_calls_without_calls",
                "finish_reason=tool_calls but no calls",
            )
        )

    if result.finish_reason != "tool_calls" and result.tool_calls:
        warnings.append(
            error(
                "tool_calls_with_non_tool_finish",
                f"tool calls present with finish_reason={result.finish_reason!r}",
            )
        )

    if case.expect_no_tool_calls:
        if result.tool_calls:
            errors.append(
                error(
                    "unexpected_tool_calls",
                    f"expected no tool calls, got {len(result.tool_calls)}",
                )
            )
    else:
        if len(result.tool_calls) < case.min_tool_calls:
            errors.append(
                error(
                    "too_few_tool_calls",
                    f"expected at least {case.min_tool_calls}, got {len(result.tool_calls)}",
                )
            )

    if (
        case.exact_tool_calls is not None
        and len(result.tool_calls) != case.exact_tool_calls
    ):
        errors.append(
            error(
                "wrong_tool_call_count",
                f"expected exactly {case.exact_tool_calls}, got {len(result.tool_calls)}",
            )
        )

    names = []
    decoded_calls: list[tuple[str, dict[str, Any]]] = []
    schemas = schema_by_name(case.tools)
    ids = []

    for idx, tc in enumerate(result.tool_calls):
        if tc.get("type") != "function":
            errors.append(
                error("wrong_tool_type", f"tool call {idx} type is {tc.get('type')!r}")
            )

        tc_id = tc.get("id") or ""
        if not tc_id:
            errors.append(error("missing_tool_id", f"tool call {idx} is missing id"))
        ids.append(tc_id)

        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        if not name:
            errors.append(
                error("missing_function_name", f"tool call {idx} missing name")
            )
            continue
        names.append(name)

        if schemas and name not in schemas:
            errors.append(
                error(
                    "unknown_tool_name",
                    f"tool call {idx} used unknown function {name!r}",
                    known=sorted(schemas),
                )
            )

        args = decode_tool_args(tc, errors, warnings)
        if args is None:
            continue
        decoded_calls.append((name, args))

        if case.validate_schema and name in schemas:
            schema_errors = validate_json_schema(args, schemas[name])
            for schema_error in schema_errors:
                errors.append(error("schema_validation", schema_error, tool_name=name))

    nonempty_ids = [tc_id for tc_id in ids if tc_id]
    if len(nonempty_ids) != len(set(nonempty_ids)):
        errors.append(
            error("duplicate_tool_ids", "tool call IDs are not unique", ids=ids)
        )

    expected_counter = Counter(case.expected_tool_names)
    actual_counter = Counter(names)
    for expected_name, expected_count in expected_counter.items():
        actual_count = actual_counter.get(expected_name, 0)
        if actual_count < expected_count:
            errors.append(
                error(
                    "missing_expected_tool",
                    (
                        f"expected at least {expected_count} call(s) to {expected_name}, "
                        f"got {actual_count}"
                    ),
                    actual_names=names,
                )
            )

    if case.expected_tool_calls:
        expected_calls = Counter(
            (
                str(call.get("name") or ""),
                json_dumps(call.get("arguments") or {}),
            )
            for call in case.expected_tool_calls
        )
        actual_calls = Counter(
            (name, json_dumps(arguments)) for name, arguments in decoded_calls
        )
        if actual_calls != expected_calls:
            errors.append(
                error(
                    "unexpected_tool_calls",
                    "decoded tool calls did not exactly match the expected calls",
                    expected=[
                        {
                            "name": name,
                            "arguments": json.loads(arguments),
                            "count": count,
                        }
                        for (name, arguments), count in expected_calls.items()
                    ],
                    actual=[
                        {
                            "name": name,
                            "arguments": json.loads(arguments),
                            "count": count,
                        }
                        for (name, arguments), count in actual_calls.items()
                    ],
                )
            )

    if case.min_distinct_tool_names is not None:
        distinct = set(names)
        if len(distinct) < case.min_distinct_tool_names:
            errors.append(
                error(
                    "too_few_distinct_tools",
                    (
                        f"expected at least {case.min_distinct_tool_names} distinct "
                        f"tools, got {sorted(distinct)}"
                    ),
                )
            )

    for fragment in case.forbidden_arg_fragments:
        lower_fragment = fragment.lower()
        for tool_name, args in decoded_calls:
            if lower_fragment in lower_json_blob(args):
                errors.append(
                    error(
                        "context_leak_to_tool_arguments",
                        f"forbidden context fragment {fragment!r} appeared in {tool_name} arguments",
                        tool_name=tool_name,
                        forbidden_fragment=fragment,
                    )
                )

    for tool_name, fragments in case.forbidden_arg_fragments_by_tool:
        for actual_name, args in decoded_calls:
            if actual_name != tool_name:
                continue
            args_blob = lower_json_blob(args)
            for fragment in fragments:
                if fragment.lower() in args_blob:
                    errors.append(
                        error(
                            "context_leak_to_tool_arguments",
                            (
                                f"forbidden context fragment {fragment!r} appeared "
                                f"in {tool_name} arguments"
                            ),
                            tool_name=tool_name,
                            forbidden_fragment=fragment,
                        )
                    )

    arg_blob = lower_json_blob([args for _, args in decoded_calls])
    content_blob = (result.content or "").lower()
    for fragment in case.expected_arg_fragments:
        lower_fragment = fragment.lower()
        if lower_fragment not in arg_blob and lower_fragment not in content_blob:
            errors.append(
                error(
                    "missing_expected_argument_fragment",
                    f"expected fragment {fragment!r} in decoded args or content",
                )
            )

    for tool_name, arg_name, expected_value in case.expected_arg_values:
        matched = False
        for actual_name, args in decoded_calls:
            if actual_name != tool_name:
                continue
            if args.get(arg_name) == expected_value:
                matched = True
                break
        if not matched:
            errors.append(
                error(
                    "missing_expected_argument_value",
                    (
                        f"expected {tool_name}.{arg_name}={expected_value!r} "
                        "in decoded tool arguments"
                    ),
                )
            )

    if case.expect_content and not (result.content or "").strip():
        errors.append(error("missing_content", "expected non-empty assistant content"))

    stripped_content = (result.content or "").strip()
    if case.expected_content is not None and stripped_content != case.expected_content:
        errors.append(
            error(
                "unexpected_content",
                (f"content={stripped_content!r}, expected {case.expected_content!r}"),
            )
        )

    if (
        case.content_pattern is not None
        and re.fullmatch(case.content_pattern, stripped_content) is None
    ):
        errors.append(
            error(
                "content_pattern_mismatch",
                (
                    f"content={stripped_content!r} did not match "
                    f"/{case.content_pattern}/"
                ),
            )
        )

    if case.expected_json is not None:
        try:
            decoded_content = json.loads(stripped_content)
        except json.JSONDecodeError as exc:
            errors.append(
                error(
                    "content_json_decode",
                    f"assistant content is not valid JSON: {exc}",
                )
            )
        else:
            if decoded_content != case.expected_json:
                errors.append(
                    error(
                        "unexpected_content_json",
                        "assistant JSON did not match the expected value",
                        expected=case.expected_json,
                        actual=decoded_content,
                    )
                )

    if case.expect_reasoning is True and not (result.reasoning_content or "").strip():
        errors.append(
            error(
                "missing_reasoning_content",
                "thinking is enabled but reasoning_content is empty",
            )
        )
    if case.expect_reasoning is False and (result.reasoning_content or "").strip():
        errors.append(
            error(
                "unexpected_reasoning_content",
                "thinking is disabled but reasoning_content is non-empty",
            )
        )

    if case.expected_any_content_fragments:
        lower_content = (result.content or "").lower()
        if not any(
            fragment.lower() in lower_content
            for fragment in case.expected_any_content_fragments
        ):
            errors.append(
                error(
                    "missing_expected_content_fragment",
                    (
                        "expected at least one response content fragment from "
                        f"{case.expected_any_content_fragments!r}"
                    ),
                )
            )

    for fragment in case.forbidden_content_fragments:
        if fragment.lower() in (result.content or "").lower():
            errors.append(
                error(
                    "context_leak_to_content",
                    f"forbidden context fragment {fragment!r} appeared in assistant content",
                    forbidden_fragment=fragment,
                )
            )

    for fragment in case.forbidden_reasoning_fragments:
        if fragment.lower() in (result.reasoning_content or "").lower():
            errors.append(
                error(
                    "context_leak_to_reasoning",
                    f"forbidden context fragment {fragment!r} appeared in reasoning_content",
                    forbidden_fragment=fragment,
                )
            )

    for fragment in case.forbidden_output_fragments:
        if fragment in (result.content or ""):
            errors.append(
                error(
                    "reserved_marker_leaked_to_content",
                    f"content contains reserved marker {fragment!r}",
                )
            )
        if fragment in (result.reasoning_content or ""):
            errors.append(
                error(
                    "reserved_marker_leaked_to_reasoning",
                    f"reasoning contains reserved marker {fragment!r}",
                )
            )

    for marker in RAW_TOOL_MARKERS:
        if marker in (result.content or ""):
            errors.append(
                error("tool_marker_leaked_to_content", f"content contains {marker!r}")
            )
        if marker in (result.reasoning_content or ""):
            errors.append(
                error(
                    "tool_marker_leaked_to_reasoning", f"reasoning contains {marker!r}"
                )
            )

    errors.extend(find_reasoning_corruption(result.reasoning_content, decoded_calls))
    return errors, warnings


def request_chat(
    case: Case,
    mode: str,
    *,
    messages: list[dict[str, Any]],
    url: str,
    api_key: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    extra_headers: dict[str, str],
) -> tuple[ChatResult, dict[str, Any]]:
    stream = mode == "stream"
    payload = build_payload(
        case,
        model=model,
        stream=stream,
        temperature=temperature,
        default_max_tokens=max_tokens,
    )
    payload["messages"] = messages
    if stream:
        result = request_stream(
            url,
            payload,
            api_key=api_key,
            timeout=timeout,
            extra_headers=extra_headers,
        )
    else:
        result = request_nonstream(
            url,
            payload,
            api_key=api_key,
            timeout=timeout,
            extra_headers=extra_headers,
        )
    return result, payload


def _step_findings(findings: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for finding in findings:
        details = dict(finding.get("details") or {})
        details["step"] = step
        output.append(
            error(
                finding["kind"],
                f"step {step}: {finding['message']}",
                **details,
            )
        )
    return output


def run_scripted_followup_case(
    case: Case,
    mode: str,
    *,
    iteration: int,
    url: str,
    api_key: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    extra_headers: dict[str, str],
    raw_chars: int,
    record_success_raw: bool,
) -> dict[str, Any]:
    """Run a deterministic two-step K3 tool-result lifecycle case."""

    started = utc_now()
    first_case = dataclasses.replace(case, scripted_followup=None)
    requests: list[dict[str, Any]] = []
    results: list[ChatResult] = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    try:
        first_result, first_payload = request_chat(
            first_case,
            mode,
            messages=list(first_case.messages),
            url=url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        requests.append(first_payload)
        results.append(first_result)
        step_errors, step_warnings = validate_result(first_case, first_result)
        errors.extend(_step_findings(step_errors, 1))
        warnings.extend(_step_findings(step_warnings, 1))

        followup = dict(case.scripted_followup or {})
        if not errors:
            messages = list(case.messages)
            messages.append(assistant_message_from_result(first_result))
            tool_results = list(followup.get("tool_results") or ())
            pairs = list(zip(first_result.tool_calls, tool_results))
            if followup.get("reverse_results"):
                pairs.reverse()
            for tool_call, tool_result in pairs:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "",
                        "content": str(tool_result),
                    }
                )

            followup_case = dataclasses.replace(
                first_case,
                messages=tuple(messages),
                tool_choice=followup.get("tool_choice"),
                expected_finish_reasons=tuple(
                    followup.get("expected_finish_reasons") or ("stop",)
                ),
                expect_no_tool_calls=bool(followup.get("expect_no_tool_calls", False)),
                min_tool_calls=int(followup.get("min_tool_calls") or 0),
                exact_tool_calls=followup.get("exact_tool_calls"),
                expected_tool_names=tuple(followup.get("expected_tool_names") or ()),
                expected_tool_calls=tuple(followup.get("expected_tool_calls") or ()),
                expected_content=followup.get("expected_content"),
                content_pattern=followup.get("content_pattern"),
                expected_json=followup.get("expected_json"),
                expect_reasoning=followup.get("expect_reasoning"),
            )
            second_result, second_payload = request_chat(
                followup_case,
                mode,
                messages=messages,
                url=url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_headers=extra_headers,
            )
            requests.append(second_payload)
            results.append(second_result)
            step_errors, step_warnings = validate_result(followup_case, second_result)
            errors.extend(_step_findings(step_errors, 2))
            warnings.extend(_step_findings(step_warnings, 2))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        results.append(
            ChatResult(
                latency_ms=0.0,
                raw_response={"status": exc.code, "body": body},
            )
        )
        errors.append(
            error(
                "http_error",
                f"HTTP {exc.code}: {exc.reason}",
                body=body[:raw_chars],
            )
        )
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        results.append(ChatResult(raw_response={"error": repr(exc)}))
        errors.append(error("request_error", repr(exc)))
    except Exception as exc:  # noqa: BLE001 - probe should report and continue.
        results.append(ChatResult(raw_response={"exception": repr(exc)}))
        errors.append(
            error(
                "probe_exception",
                repr(exc),
                traceback=traceback.format_exc(limit=8),
            )
        )

    final_result = results[-1] if results else ChatResult()
    passed = not errors
    total_latency_ms = round(sum(result.latency_ms for result in results), 3)
    record: dict[str, Any] = {
        "timestamp": started,
        "iteration": iteration,
        "case_id": case.case_id,
        "description": case.description,
        "mode": mode,
        "pass": passed,
        "errors": errors,
        "warnings": warnings,
        "scripted_followup": True,
        "steps": [
            {
                "step": index,
                "finish_reason": result.finish_reason,
                "content_chars": len(result.content or ""),
                "reasoning_chars": len(result.reasoning_content or ""),
                "tool_calls": [
                    {
                        "id": tool_call.get("id"),
                        "name": (tool_call.get("function") or {}).get("name"),
                        "arguments": (tool_call.get("function") or {}).get("arguments"),
                    }
                    for tool_call in result.tool_calls
                ],
            }
            for index, result in enumerate(results, start=1)
        ],
        "response": {
            "id": final_result.response_id,
            "model": final_result.model,
            "created": final_result.created,
            "finish_reason": final_result.finish_reason,
            "latency_ms": total_latency_ms,
            "ttft_ms": final_result.ttft_ms,
            "chunk_count": final_result.chunk_count,
            "content_chars": len(final_result.content or ""),
            "reasoning_chars": len(final_result.reasoning_content or ""),
            "tool_calls": [
                {
                    "id": tool_call.get("id"),
                    "name": (tool_call.get("function") or {}).get("name"),
                    "arguments": (tool_call.get("function") or {}).get("arguments"),
                }
                for tool_call in final_result.tool_calls
            ],
            "usage": final_result.usage,
        },
    }
    if errors or record_success_raw:
        record["requests"] = trunc(requests, raw_chars)
        record["raw_responses"] = trunc(
            [result.raw_response for result in results], raw_chars
        )
        if final_result.content:
            record["content"] = final_result.content[:raw_chars]
        if final_result.reasoning_content:
            record["reasoning_content"] = final_result.reasoning_content[:raw_chars]
    return record


def run_agent_case(
    case: Case,
    mode: str,
    *,
    iteration: int,
    url: str,
    api_key: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    extra_headers: dict[str, str],
    raw_chars: int,
    record_success_raw: bool,
) -> dict[str, Any]:
    started = utc_now()
    messages = list(case.messages)
    turn_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    executed_tool_calls: list[dict[str, Any]] = []
    final_result: ChatResult | None = None
    last_payload: dict[str, Any] | None = None

    try:
        for turn in range(1, case.max_agent_turns + 1):
            result, payload = request_chat(
                case,
                mode,
                messages=messages,
                url=url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_headers=extra_headers,
            )
            last_payload = payload

            turn_case = dataclasses.replace(
                case,
                expected_finish_reasons=("tool_calls", "stop"),
                expect_no_tool_calls=False,
                min_tool_calls=0,
                exact_tool_calls=None,
                expected_tool_names=(),
                min_distinct_tool_names=None,
                expected_arg_fragments=(),
                expected_any_content_fragments=(),
                expect_content=False,
                execute_tools=False,
                expected_executed_tool_names=(),
                expected_final_content_fragments=(),
            )
            turn_errors, turn_warnings = validate_result(turn_case, result)
            errors.extend(
                error(
                    turn_error["kind"],
                    turn_error["message"],
                    turn=turn,
                    **turn_error.get("details", {}),
                )
                for turn_error in turn_errors
            )
            warnings.extend(
                error(
                    turn_warning["kind"],
                    turn_warning["message"],
                    turn=turn,
                    **turn_warning.get("details", {}),
                )
                for turn_warning in turn_warnings
            )

            turn_records.append(
                {
                    "turn": turn,
                    "id": result.response_id,
                    "finish_reason": result.finish_reason,
                    "latency_ms": round(result.latency_ms, 3),
                    "ttft_ms": None
                    if result.ttft_ms is None
                    else round(result.ttft_ms, 3),
                    "chunk_count": result.chunk_count,
                    "content_chars": len(result.content or ""),
                    "reasoning_chars": len(result.reasoning_content or ""),
                    "tool_calls": [
                        {
                            "id": tc.get("id"),
                            "name": (tc.get("function") or {}).get("name"),
                            "arguments": (tc.get("function") or {}).get("arguments"),
                        }
                        for tc in result.tool_calls
                    ],
                }
            )

            if errors:
                final_result = result
                break

            if result.finish_reason != "tool_calls" or not result.tool_calls:
                final_result = result
                break

            messages.append(assistant_message_from_result(result))
            for tc in result.tool_calls:
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "")
                args = decode_tool_args(tc, errors, warnings)
                if args is None:
                    continue
                tool_result = execute_mock_tool(name, args)
                executed_tool_calls.append(
                    {
                        "turn": turn,
                        "id": tc.get("id") or "",
                        "name": name,
                        "arguments": args,
                        "result": tool_result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id") or "",
                        "name": name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        errors.extend(validate_agent_final(case, final_result))
        errors.extend(validate_executed_tools(case, executed_tool_calls))

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        final_result = ChatResult(
            latency_ms=0.0, raw_response={"status": exc.code, "body": body}
        )
        errors.append(
            error(
                "http_error",
                f"HTTP {exc.code}: {exc.reason}",
                body=body[:raw_chars],
            )
        )
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        final_result = ChatResult(latency_ms=0.0, raw_response={"error": repr(exc)})
        errors.append(error("request_error", repr(exc)))
    except Exception as exc:  # noqa: BLE001 - probe should report and continue.
        final_result = ChatResult(latency_ms=0.0, raw_response={"exception": repr(exc)})
        errors.append(
            error(
                "probe_exception",
                repr(exc),
                traceback=traceback.format_exc(limit=8),
            )
        )

    final_result = final_result or ChatResult()
    passed = not errors
    total_latency_ms = round(
        sum(turn.get("latency_ms") or 0.0 for turn in turn_records), 3
    )
    record: dict[str, Any] = {
        "timestamp": started,
        "iteration": iteration,
        "case_id": case.case_id,
        "description": case.description,
        "mode": mode,
        "pass": passed,
        "errors": errors,
        "warnings": warnings,
        "agent_loop": True,
        "turns": turn_records,
        "executed_tool_calls": [
            {
                "turn": call["turn"],
                "id": call["id"],
                "name": call["name"],
                "arguments": call["arguments"],
                "result": call["result"],
            }
            for call in executed_tool_calls
        ],
        "response": {
            "id": final_result.response_id,
            "model": final_result.model,
            "created": final_result.created,
            "finish_reason": final_result.finish_reason,
            "latency_ms": total_latency_ms,
            "ttft_ms": None
            if final_result.ttft_ms is None
            else round(final_result.ttft_ms, 3),
            "chunk_count": final_result.chunk_count,
            "content_chars": len(final_result.content or ""),
            "reasoning_chars": len(final_result.reasoning_content or ""),
            "tool_calls": [
                {
                    "id": tc.get("id"),
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": (tc.get("function") or {}).get("arguments"),
                }
                for tc in final_result.tool_calls
            ],
            "usage": final_result.usage,
        },
    }

    if errors or record_success_raw:
        record["request"] = trunc(
            last_payload
            or build_payload(
                case,
                model=model,
                stream=mode == "stream",
                temperature=temperature,
                default_max_tokens=max_tokens,
            ),
            raw_chars,
        )
        record["final_messages"] = trunc(messages, raw_chars)
        record["raw_response"] = trunc(final_result.raw_response, raw_chars)
        if final_result.content:
            record["content"] = final_result.content[:raw_chars]
        if final_result.reasoning_content:
            record["reasoning_content"] = final_result.reasoning_content[:raw_chars]

    return record


def run_case(
    case: Case,
    mode: str,
    *,
    iteration: int,
    url: str,
    api_key: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    extra_headers: dict[str, str],
    raw_chars: int,
    record_success_raw: bool,
) -> dict[str, Any]:
    if case.scripted_followup:
        return run_scripted_followup_case(
            case,
            mode,
            iteration=iteration,
            url=url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_headers=extra_headers,
            raw_chars=raw_chars,
            record_success_raw=record_success_raw,
        )
    if case.execute_tools:
        return run_agent_case(
            case,
            mode,
            iteration=iteration,
            url=url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_headers=extra_headers,
            raw_chars=raw_chars,
            record_success_raw=record_success_raw,
        )

    stream = mode == "stream"
    payload = build_payload(
        case,
        model=model,
        stream=stream,
        temperature=temperature,
        default_max_tokens=max_tokens,
    )
    started = utc_now()

    try:
        if stream:
            result = request_stream(
                url,
                payload,
                api_key=api_key,
                timeout=timeout,
                extra_headers=extra_headers,
            )
        else:
            result = request_nonstream(
                url,
                payload,
                api_key=api_key,
                timeout=timeout,
                extra_headers=extra_headers,
            )
        errors, warnings = validate_result(case, result)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        result = ChatResult(
            latency_ms=0.0, raw_response={"status": exc.code, "body": body}
        )
        errors = [
            error(
                "http_error",
                f"HTTP {exc.code}: {exc.reason}",
                body=body[:raw_chars],
            )
        ]
        warnings = []
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        result = ChatResult(latency_ms=0.0, raw_response={"error": repr(exc)})
        errors = [error("request_error", repr(exc))]
        warnings = []
    except Exception as exc:  # noqa: BLE001 - probe should report and continue.
        result = ChatResult(latency_ms=0.0, raw_response={"exception": repr(exc)})
        errors = [
            error(
                "probe_exception",
                repr(exc),
                traceback=traceback.format_exc(limit=8),
            )
        ]
        warnings = []

    passed = not errors
    record: dict[str, Any] = {
        "timestamp": started,
        "iteration": iteration,
        "case_id": case.case_id,
        "description": case.description,
        "mode": mode,
        "pass": passed,
        "errors": errors,
        "warnings": warnings,
        "response": {
            "id": result.response_id,
            "model": result.model,
            "created": result.created,
            "finish_reason": result.finish_reason,
            "latency_ms": round(result.latency_ms, 3),
            "ttft_ms": None if result.ttft_ms is None else round(result.ttft_ms, 3),
            "chunk_count": result.chunk_count,
            "content_chars": len(result.content or ""),
            "reasoning_chars": len(result.reasoning_content or ""),
            "tool_calls": [
                {
                    "id": tc.get("id"),
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": (tc.get("function") or {}).get("arguments"),
                }
                for tc in result.tool_calls
            ],
            "usage": result.usage,
        },
    }

    if errors or record_success_raw:
        record["request"] = trunc(payload, raw_chars)
        record["raw_response"] = trunc(result.raw_response, raw_chars)
        if result.content:
            record["content"] = result.content[:raw_chars]
        if result.reasoning_content:
            record["reasoning_content"] = result.reasoning_content[:raw_chars]

    return record


class ReportWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        config: dict[str, Any],
        cases: tuple[Case, ...],
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = output_dir / "results.jsonl"
        self.failures_path = output_dir / "failures.jsonl"
        self.summary_path = output_dir / "summary.md"
        self.config_path = output_dir / "run_config.json"
        self.cases_path = output_dir / "cases.json"

        self.results_file = self.results_path.open("a", encoding="utf-8")
        self.failures_file = self.failures_path.open("a", encoding="utf-8")
        self.records: list[dict[str, Any]] = []
        self.started = utc_now()

        self.config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        self.cases_path.write_text(
            json.dumps(
                [dataclasses.asdict(case) for case in cases],
                indent=2,
                ensure_ascii=False,
                default=list,
            ),
            encoding="utf-8",
        )

    def close(self) -> None:
        self.results_file.close()
        self.failures_file.close()

    def record(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        print(json_dumps(record), file=self.results_file, flush=True)
        if not record["pass"]:
            print(json_dumps(record), file=self.failures_file, flush=True)

    def write_summary(self) -> None:
        total = len(self.records)
        failed = sum(1 for record in self.records if not record["pass"])
        passed = total - failed
        by_case = Counter(record["case_id"] for record in self.records)
        failed_by_case = Counter(
            record["case_id"] for record in self.records if not record["pass"]
        )
        by_mode = Counter(record["mode"] for record in self.records)
        failed_by_mode = Counter(
            record["mode"] for record in self.records if not record["pass"]
        )
        failure_kinds = Counter(
            err["kind"] for record in self.records for err in record.get("errors", [])
        )

        lines = [
            "# Tool Calling Probe Report",
            "",
            f"- Started: `{self.started}`",
            f"- Updated: `{utc_now()}`",
            f"- Total requests: `{total}`",
            f"- Passed: `{passed}`",
            f"- Failed: `{failed}`",
            f"- Results JSONL: `{self.results_path}`",
            f"- Failures JSONL: `{self.failures_path}`",
            "",
            "## By Mode",
            "",
            "| Mode | Total | Failed |",
            "|---|---:|---:|",
        ]
        for mode, count in sorted(by_mode.items()):
            lines.append(f"| `{mode}` | {count} | {failed_by_mode.get(mode, 0)} |")

        lines.extend(
            [
                "",
                "## By Case",
                "",
                "| Case | Total | Failed |",
                "|---|---:|---:|",
            ]
        )
        for case_id, count in sorted(by_case.items()):
            lines.append(
                f"| `{case_id}` | {count} | {failed_by_case.get(case_id, 0)} |"
            )

        lines.extend(
            [
                "",
                "## Failure Kinds",
                "",
                "| Kind | Count |",
                "|---|---:|",
            ]
        )
        if failure_kinds:
            for kind, count in failure_kinds.most_common():
                lines.append(f"| `{kind}` | {count} |")
        else:
            lines.append("| none | 0 |")

        failed_records = [record for record in self.records if not record["pass"]]
        lines.extend(
            [
                "",
                "## Failure Examples",
                "",
                "| Iteration | Case | Mode | Finish | Error Kinds | Response ID |",
                "|---:|---|---|---|---|---|",
            ]
        )
        for record in failed_records[:50]:
            kinds = ", ".join(err["kind"] for err in record.get("errors", []))
            response = record.get("response", {})
            lines.append(
                "| {iteration} | `{case}` | `{mode}` | `{finish}` | {kinds} | `{rid}` |".format(
                    iteration=record["iteration"],
                    case=record["case_id"],
                    mode=record["mode"],
                    finish=response.get("finish_reason"),
                    kinds=kinds,
                    rid=response.get("id") or "",
                )
            )

        self.summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"header {value!r} must be formatted as Name: value")
        name, header_value = value.split(":", 1)
        headers[name.strip()] = header_value.strip()
    return headers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe OpenAI-compatible chat-completions tool calling."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Do not send an Authorization header.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Extra request header, formatted as 'Name: value'. May be repeated.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument(
        "--modes",
        default="nonstream,stream",
        help="Comma-separated modes: nonstream,stream.",
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="Comma-separated case IDs, or 'all'. Use --list-cases to inspect.",
    )
    parser.add_argument(
        "--exclude-cases",
        default="",
        help=(
            "Comma-separated case ID glob patterns to omit after --cases is "
            "applied. Useful for keeping adversarial diagnostics out of a "
            "qualification score."
        ),
    )
    parser.add_argument(
        "--case-profile",
        default="auto",
        choices=(
            "auto",
            *INLINE_CASE_PROFILES,
            *available_case_profiles(),
            "all",
        ),
        help="Case profile to run. auto infers from --model.",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of full sweeps. Use 0 with --duration-minutes for duration-only.",
    )
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=0.0,
        help="Keep sweeping until this duration expires. 0 disables duration limit.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--case-delay-seconds", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--raw-chars", type=int, default=20000)
    parser.add_argument("--record-success-raw", action="store_true")
    parser.add_argument(
        "--summary-every",
        type=int,
        default=10,
        help="Rewrite summary after this many records.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected cases and exit without sending requests.",
    )
    return parser


def select_cases(
    all_cases: tuple[Case, ...], selector: str, exclude_selector: str = ""
) -> tuple[Case, ...]:
    if selector == "all":
        selected = all_cases
    else:
        requested = {item.strip() for item in selector.split(",") if item.strip()}
        by_id = {case.case_id: case for case in all_cases}
        unknown = sorted(requested - set(by_id))
        if unknown:
            raise ValueError(f"unknown case ID(s): {', '.join(unknown)}")
        selected = tuple(by_id[case_id] for case_id in sorted(requested))

    exclude_patterns = tuple(
        item.strip() for item in exclude_selector.split(",") if item.strip()
    )
    if exclude_patterns:
        selected = tuple(
            case
            for case in selected
            if not any(
                fnmatchcase(case.case_id, pattern) for pattern in exclude_patterns
            )
        )
    if not selected:
        raise ValueError("case selection is empty after exclusions")
    return selected


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {"nonstream", "stream"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"unknown mode(s): {', '.join(unknown)}")
    if not modes:
        raise ValueError("at least one mode is required")
    return modes


def endpoint_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def print_progress(record: dict[str, Any]) -> None:
    status = "PASS" if record["pass"] else "FAIL"
    response = record.get("response", {})
    latency = response.get("latency_ms")
    kinds = ",".join(err["kind"] for err in record.get("errors", []))
    suffix = f" {kinds}" if kinds else ""
    print(
        (
            f"[{status}] iter={record['iteration']} case={record['case_id']} "
            f"mode={record['mode']} finish={response.get('finish_reason')} "
            f"latency_ms={latency}{suffix}"
        ),
        flush=True,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    random.seed(args.seed)

    case_profile = (
        model_case_profile(args.model)
        if args.case_profile == "auto"
        else args.case_profile
    )
    all_cases = build_cases(case_profile)
    if args.list_cases:
        for case in all_cases:
            print(f"{case.case_id}\t{case.description}")
        return 0

    try:
        cases = select_cases(all_cases, args.cases, args.exclude_cases)
        modes = parse_modes(args.modes)
        extra_headers = parse_headers(args.header)
    except ValueError as exc:
        parser.error(str(exc))

    api_key: str | None = None
    if not args.no_auth:
        api_key = args.api_key or os.environ.get(args.api_key_env)
        if not api_key:
            parser.error(
                f"missing API key: set ${args.api_key_env}, pass --api-key, "
                "or use --no-auth"
            )

    url = endpoint_url(args.base_url)
    output_dir = Path(
        args.output_dir
        or Path(DEFAULT_OUTPUT_ROOT)
        / f"{local_timestamp()}-{args.model.split('/')[-1]}"
    )

    config = {
        "base_url": args.base_url,
        "url": url,
        "model": args.model,
        "api_key_env": args.api_key_env if not args.no_auth else None,
        "auth": "disabled" if args.no_auth else "bearer",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "modes": modes,
        "case_ids": [case.case_id for case in cases],
        "exclude_cases": args.exclude_cases,
        "iterations": args.iterations,
        "duration_minutes": args.duration_minutes,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "output_dir": str(output_dir),
    }

    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0

    writer = ReportWriter(output_dir, config=config, cases=cases)
    deadline = (
        time.monotonic() + args.duration_minutes * 60.0
        if args.duration_minutes > 0
        else None
    )

    print(f"Writing probe report under {output_dir}", flush=True)
    print(f"Endpoint: {url}", flush=True)
    print(f"Model: {args.model}", flush=True)

    iteration = 0
    stop = False
    try:
        while not stop:
            if args.iterations > 0 and iteration >= args.iterations:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break

            iteration += 1
            work = [(case, mode) for case in cases for mode in modes]
            if args.shuffle:
                random.shuffle(work)

            def submit_one(item: tuple[Case, str]) -> dict[str, Any]:
                case, mode = item
                if args.case_delay_seconds > 0:
                    time.sleep(args.case_delay_seconds)
                return run_case(
                    case,
                    mode,
                    iteration=iteration,
                    url=url,
                    api_key=api_key,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout_seconds,
                    extra_headers=extra_headers,
                    raw_chars=args.raw_chars,
                    record_success_raw=args.record_success_raw,
                )

            if args.concurrency <= 1:
                for item in work:
                    record = submit_one(item)
                    writer.record(record)
                    print_progress(record)
                    if len(writer.records) % max(1, args.summary_every) == 0:
                        writer.write_summary()
                    if args.stop_on_failure and not record["pass"]:
                        stop = True
                        break
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.concurrency
                ) as executor:
                    future_to_item = {
                        executor.submit(submit_one, item): item for item in work
                    }
                    for future in concurrent.futures.as_completed(future_to_item):
                        record = future.result()
                        writer.record(record)
                        print_progress(record)
                        if len(writer.records) % max(1, args.summary_every) == 0:
                            writer.write_summary()
                        if args.stop_on_failure and not record["pass"]:
                            stop = True

            writer.write_summary()

            if stop:
                break
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

            if args.iterations == 0 and deadline is None:
                break

    except KeyboardInterrupt:
        print("Interrupted; writing final summary.", file=sys.stderr)
    finally:
        writer.write_summary()
        writer.close()

    failed = sum(1 for record in writer.records if not record["pass"])
    print(f"Summary: {writer.summary_path}", flush=True)
    print(f"Failures: {writer.failures_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
