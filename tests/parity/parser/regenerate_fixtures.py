# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixture (re-)generator for the parity (parser) harness.

Walks (family × case) combinations, runs each input through Dynamo's
PyO3 parser, and writes the result as a fixture JSON. Run from inside
a container with `dynamo._core` installed:

    python3 tests/parity/parser/regenerate_fixtures.py

Re-running overwrites existing fixtures under `fixtures/`. Edit
INPUTS to add or change templates; the generator captures Dynamo's
output verbatim, so fixtures always reflect the current Rust
contract. Cases per family follow PARSER_CASES.md.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

# `dynamo._core` is a PyO3 native extension. The `parse_tool_call`
# symbol is added by a sibling Rust binding change that may not be
# present in every local build, so the import is type-ignored — this
# is a manual-run utility, not part of the test contract.
from dynamo._core import parse_tool_call  # type: ignore[attr-defined]

FIXTURES_ROOT = Path(__file__).parent / "fixtures"

# Tool definitions reused across cases. Each family picks the subset
# of tools relevant to its case inputs.
_GET_WEATHER_LOC = {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
}
_GET_WEATHER_LOC_UNIT = {
    "name": "get_weather",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string"},
        },
    },
}
_GET_TIME_TZ = {
    "name": "get_time",
    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
}
_GET_TIME_NOARG = {
    "name": "get_time",
    "parameters": {"type": "object", "properties": {}},
}
_PROCESS_DATA_NESTED = {
    "name": "process_data",
    "parameters": {
        "type": "object",
        "properties": {
            "items": {"type": "array"},
            "config": {"type": "object"},
        },
    },
}

# (family, case_id) -> {"text": str, "tools": list[dict] | None, "description": str}
# Cases marked with text=None are intentionally skipped (N/A or not yet
# defined for that family).
INPUTS: dict[tuple[str, str], dict[str, Any] | None] = {
    # ----- kimi_k2 -----
    ("kimi_k2", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.2"): {
        "description": "Multiple tool calls",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{"timezone":"EST"}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("kimi_k2", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.4"): {
        "description": "Malformed JSON args (missing close brace)",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.5"): {
        "description": "Missing section_end (max_tokens truncation, PR #8208)",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_time:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>",
        "tools": [_GET_TIME_NOARG],
    },
    ("kimi_k2", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array)",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.process_data:0<|tool_call_argument_begin|>{"items":[1,2,3],"config":{"nested":true}}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("kimi_k2", "PARSER.batch.8"): {
        "description": "Interleaved normal text",
        "text": 'I\'ll help you with that. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"Dallas"}<|tool_call_end|><|tool_calls_section_end|> Let me check.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{"location":"LA"}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- qwen3_coder -----
    ("qwen3_coder", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.2"): {
        "description": "Multiple tool calls",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_time>\n<parameter=timezone>\nEST\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("qwen3_coder", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.4"): {
        "description": "Malformed (missing </parameter> closing tag)",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.5"): {
        "description": "Missing </tool_call> end marker",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": "<tool_call>\n<function=get_time>\n</function>\n</tool_call>",
        "tools": [_GET_TIME_NOARG],
    },
    ("qwen3_coder", "PARSER.batch.7"): {
        "description": "Complex args (multi-parameter)",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC_UNIT],
    },
    ("qwen3_coder", "PARSER.batch.8"): {
        "description": "Interleaved normal text",
        "text": "I'll help you check the weather. <tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call> Let me get that information for you.",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=location>\nLA\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- glm47 -----
    ("glm47", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.2"): {
        "description": "Multiple tool calls",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_time<arg_key>timezone</arg_key><arg_value>EST</arg_value></tool_call>",
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("glm47", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.4"): {
        "description": "Malformed (missing arg_value end tag)",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.5"): {
        "description": "Missing </tool_call> end marker",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": "<tool_call>get_time</tool_call>",
        "tools": [_GET_TIME_NOARG],
    },
    ("glm47", "PARSER.batch.7"): {
        "description": "Complex args (multi-parameter)",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value><arg_key>unit</arg_key><arg_value>fahrenheit</arg_value></tool_call>",
        "tools": [_GET_WEATHER_LOC_UNIT],
    },
    ("glm47", "PARSER.batch.8"): {
        "description": "Interleaved normal text",
        "text": "I'll check the weather. <tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>LA</arg_value></tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- deepseek_v3_1 -----
    ("deepseek_v3_1", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.2"): {
        "description": "Multiple tool calls",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone":"EST"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v3_1", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.4"): {
        "description": "Malformed JSON inside call",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.5"): {
        "description": "Missing tool_calls_end (truncation)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v3_1", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>process_data<｜tool▁sep｜>{"items":[1,2,3],"config":{"nested":true}}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("deepseek_v3_1", "PARSER.batch.8"): {
        "description": "Interleaved normal text",
        "text": 'The following tool call retrieves weather information: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"LA"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- harmony -----
    ("harmony", "PARSER.batch.1"): {
        "description": "Single tool call (basic complete envelope)",
        "text": '<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.2"): {
        "description": "Multiple tool calls (back-to-back commentary blocks)",
        "text": '<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_time <|constrain|>json<|message|>{"timezone":"EST"}<|call|>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("harmony", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.4"): {
        "description": "Malformed (truncated JSON args)",
        "text": '<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC<|call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.5"): {
        "description": "Missing <|call|> end marker (bare envelope)",
        "text": '<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": "<|channel|>commentary to=functions.get_time <|constrain|>json<|message|>{}",
        "tools": [_GET_TIME_NOARG],
    },
    ("harmony", "PARSER.batch.7"): {
        "description": "Complex args (multi-parameter)",
        "text": '<|channel|>analysis<|message|>Need to use function get_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC","unit":"fahrenheit"}<|call|>',
        "tools": [_GET_WEATHER_LOC_UNIT],
    },
    ("harmony", "PARSER.batch.8"): {
        "description": "Interleaved analysis-channel text + tool call",
        "text": '<|channel|>analysis<|message|>Need to use function get_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"LA"}<|call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- minimax_m2 -----
    ("minimax_m2", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.2"): {
        "description": "Multiple tool calls",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>\n<invoke name="get_time">\n<parameter name="timezone">EST</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("minimax_m2", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.4"): {
        "description": "Malformed (missing closing invoke tag)",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.5"): {
        "description": "Missing </minimax:tool_call> end marker",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": '<minimax:tool_call>\n<invoke name="get_time">\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_TIME_NOARG],
    },
    ("minimax_m2", "PARSER.batch.7"): {
        "description": "Complex args (multi-parameter)",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n<parameter name="unit">fahrenheit</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC_UNIT],
    },
    ("minimax_m2", "PARSER.batch.8"): {
        "description": "Interleaved normal text",
        "text": 'I\'ll help you check the weather. <minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">Tokyo</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>\n<invoke name="get_weather">\n<parameter name="location">LA</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- nemotron_deci -----
    ("nemotron_deci", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.2"): {
        "description": "Multiple tool calls",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}]</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("nemotron_deci", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.4"): {
        "description": "Malformed (truncated JSON inside TOOLCALL)",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.5"): {
        "description": "Missing </TOOLCALL> end marker",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": '<TOOLCALL>[{"name": "get_time", "arguments": {}}]</TOOLCALL>',
        "tools": [_GET_TIME_NOARG],
    },
    ("nemotron_deci", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array)",
        "text": '<TOOLCALL>[{"name": "process_data", "arguments": {"items": [1,2,3], "config": {"nested": true}}}]</TOOLCALL>',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("nemotron_deci", "PARSER.batch.8"): {
        "description": "Interleaved normal text",
        "text": 'Hey How are you? <TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_weather", "arguments": {"location": "LA"}}]</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC],
    },
}


async def _run_one(family: str, text: str, tools: list[dict] | None) -> dict[str, Any]:
    tools_json = json.dumps(tools) if tools else None
    result_json = await parse_tool_call(family, text, tools_json)
    raw = json.loads(result_json)
    calls = []
    for c in raw.get("calls") or []:
        args_str = c["function"]["arguments"]
        try:
            args = json.loads(args_str) if args_str else {}
        except (json.JSONDecodeError, TypeError):
            args = args_str
        calls.append({"name": c["function"]["name"], "arguments": args})
    return {"calls": calls, "normal_text": raw.get("normal_text") or ""}


def _write_family_fixtures(
    family: str, mode: str, cases: dict[str, dict[str, Any]]
) -> None:
    """Write one file per (family, mode) holding all cases for that mode."""
    family_dir = FIXTURES_ROOT / family
    family_dir.mkdir(parents=True, exist_ok=True)
    out = {"family": family, "mode": mode, "cases": cases}
    (family_dir / f"PARSER.{mode}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


async def main() -> None:
    # Group inputs by (family, mode) so we can write one file per pair.
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for (family, case_id), entry in INPUTS.items():
        if entry is None:
            continue
        # case_id is e.g. "PARSER.batch.5" — split into mode and number.
        _, mode, num = case_id.split(".", 2)
        expected = await _run_one(family, entry["text"], entry["tools"])
        grouped.setdefault((family, mode), {})[num] = {
            "description": entry["description"],
            "model_text": entry["text"],
            "tools": entry["tools"],
            "expected": expected,
        }

    for (family, mode), cases in grouped.items():
        # Sort cases numerically so output is stable across runs.
        ordered = dict(sorted(cases.items(), key=lambda kv: int(kv[0])))
        _write_family_fixtures(family, mode, ordered)
        print(f"  wrote {family}/PARSER.{mode}.json with {len(ordered)} cases")
    total = sum(1 for v in INPUTS.values() if v is not None)
    print(f"\nGenerated {total} cases across {len(grouped)} files.")


if __name__ == "__main__":
    asyncio.run(main())
