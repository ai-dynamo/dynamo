# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixture (re-)generator for the parity (parser) harness.

Walks (family × case) combinations, runs each input through Dynamo's
PyO3 parser, and writes the result as a fixture YAML. Run from the
repo root inside a container with `dynamo._core` installed:

    python3 -m tests.parity.parser.regenerate_fixtures

(Run as a module — the local `dynamo.py` wrapper would shadow the
real `dynamo` package if invoked as a script directly.)

Default behavior is **non-destructive**: cases that already exist on
disk are left alone. To refresh after an intentional Dynamo
parser-behavior change, pass `--overwrite-if-exists`. Cases on disk
but not in INPUTS today are always preserved (so editing INPUTS
can't accidentally delete other contributors' cases).

Cases per family follow PARSER_CASES.md. N/A combinations
(empty INPUTS entry) are skipped.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from dynamo._core import parse_tool_call

FIXTURES_ROOT = Path(__file__).parent / "fixtures"


def _yaml_str_presenter(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    """Use a literal block scalar (`|-`) for multi-line strings so
    fixture `model_text` reads as wire-format text rather than a
    `\\n`-escaped one-liner. Single-line strings keep the default style."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _yaml_str_presenter)

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
    # ----- pythonic -----
    # Format: [name(arg=val, ...), name2(...)] — Python-call-style. Also
    # accepts <|python_start|>...<|python_end|> wrapping (e.g. Llama 4).
    ("pythonic", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '[get_weather(location="NYC")]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.2"): {
        "description": "Multiple tool calls (parallel)",
        "text": '[get_weather(location="NYC"), get_time(timezone="EST")]',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("pythonic", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.4"): {
        "description": "Malformed (missing closing bracket)",
        "text": '[get_weather(location="NYC"',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.5"): {
        "description": "Missing closing `]` end marker",
        "text": '[get_weather(location="NYC")',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": "[get_time()]",
        "tools": [_GET_TIME_NOARG],
    },
    ("pythonic", "PARSER.batch.7"): {
        "description": "Complex args (nested dict + array)",
        "text": '[process_data(items=[1, 2, 3], config={"nested": True})]',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("pythonic", "PARSER.batch.8"): {
        "description": "Interleaved normal text",
        "text": 'Hey yo ! [get_weather(location="NYC")] Hey yo',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '[get_weather(location="NYC"), get_weather(location="LA")]',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- gemma4 -----
    # Format: <|tool_call>call:NAME{key:val,...}<tool_call|>
    # String values are wrapped with `<|"|>` literal markers (not standard JSON quotes).
    ("gemma4", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.2"): {
        "description": "Multiple tool calls (parallel)",
        "text": '<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|><|tool_call>call:get_time{timezone:<|"|>EST<|"|>}<tool_call|>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("gemma4", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.4"): {
        "description": "Malformed (missing close brace)",
        "text": '<|tool_call>call:get_weather{location:<|"|>NYC<|"|><tool_call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.5"): {
        "description": "Missing <tool_call|> end marker",
        "text": '<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": "<|tool_call>call:get_time{}<tool_call|>",
        "tools": [_GET_TIME_NOARG],
    },
    ("gemma4", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array)",
        "text": "<|tool_call>call:process_data{items:[1,2,3],config:{nested:true}}<tool_call|>",
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("gemma4", "PARSER.batch.8"): {
        "description": "Interleaved normal text",
        "text": 'I will check that. <|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|> Done.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|><|tool_call>call:get_weather{location:<|"|>LA<|"|>}<tool_call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- deepseek_v3 (legacy) -----
    # Format: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
    # ```json\n{args}\n```<｜tool▁call▁end｜>...<｜tool▁calls▁end｜>
    # Note: distinct from `deepseek_v3_1` (no markdown fence).
    ("deepseek_v3", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.2"): {
        "description": "Multiple tool calls (parallel)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n```json\n{"timezone": "EST"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v3", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.4"): {
        "description": "Malformed JSON args (missing close brace)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.5"): {
        "description": "Missing calls_end / call_end markers",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v3", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>process_data\n```json\n{"items": [1, 2, 3], "config": {"nested": true}}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("deepseek_v3", "PARSER.batch.8"): {
        "description": "Interleaved normal text",
        "text": 'The following tool call retrieves weather information: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "New York"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "LA"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
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


def _load_existing(family: str, mode: str) -> dict[str, dict[str, Any]]:
    """Read the on-disk cases dict for `(family, mode)`, or empty if absent.

    Keys on disk are full case IDs (`PARSER.batch.5`); strip the prefix
    so internal bookkeeping stays keyed by the case number (`"5"`)."""
    fp = FIXTURES_ROOT / family / f"PARSER.{mode}.yaml"
    if not fp.exists():
        return {}
    raw = yaml.safe_load(fp.read_text(encoding="utf-8")).get("cases", {}) or {}
    return {k.rsplit(".", 1)[1]: v for k, v in raw.items()}


def _write_family_fixtures(
    family: str, mode: str, cases: dict[str, dict[str, Any]]
) -> None:
    """Write one file per (family, mode) holding all cases for that mode.

    On-disk keys are the full case ID (e.g. `PARSER.batch.5`) so they
    match the IDs used in PARSER_CASES.md and `KNOWN_DIVERGENCES`. A
    single `grep PARSER.batch.5` then finds the case across docs,
    fixtures, and Rust source comments."""
    family_dir = FIXTURES_ROOT / family
    family_dir.mkdir(parents=True, exist_ok=True)
    ordered = {f"PARSER.{mode}.{n}": cases[n] for n in sorted(cases, key=int)}
    out = {"family": family, "mode": mode, "cases": ordered}
    header = (
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )
    (family_dir / f"PARSER.{mode}.yaml").write_text(
        header + yaml.dump(out, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


async def main(overwrite_if_exists: bool = False) -> None:
    # Group inputs by (family, mode) so we can merge with existing per file.
    inputs_by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for (family, case_id), entry in INPUTS.items():
        if entry is None:
            continue
        # case_id is e.g. "PARSER.batch.5" — split into mode and number.
        _, mode, num = case_id.split(".", 2)
        inputs_by_pair.setdefault((family, mode), {})[num] = entry

    n_written = n_skipped = n_orphan_kept = 0
    for (family, mode), entries in inputs_by_pair.items():
        existing = _load_existing(family, mode)
        merged: dict[str, dict[str, Any]] = {}

        # 1. Process every case the user listed in INPUTS.
        for num, entry in entries.items():
            if num in existing and not overwrite_if_exists:
                merged[num] = existing[num]
                n_skipped += 1
                continue
            expected = await _run_one(family, entry["text"], entry["tools"])
            merged[num] = {
                "description": entry["description"],
                "model_text": entry["text"],
                "tools": entry["tools"],
                "expected": expected,
            }
            n_written += 1

        # 2. Preserve any on-disk cases that aren't in INPUTS today, so a
        #    contributor's INPUTS edit can't accidentally delete other
        #    contributors' fixture cases.
        for num, case in existing.items():
            if num not in merged:
                merged[num] = case
                n_orphan_kept += 1

        _write_family_fixtures(family, mode, merged)
        print(f"  wrote {family}/PARSER.{mode}.yaml with {len(merged)} cases")

    print(
        f"\n{n_written} written, {n_skipped} skipped (already on disk), "
        f"{n_orphan_kept} preserved (on disk but not in INPUTS).\n"
        f"Pass --overwrite-if-exists to refresh the {n_skipped} skipped case(s)."
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--overwrite-if-exists",
        action="store_true",
        help=(
            "Re-run Dynamo for cases that already exist on disk and overwrite "
            "the recorded `expected` output. Default: skip existing cases "
            "(adds new ones only). Use this when intentionally refreshing a "
            "fixture after a Dynamo parser-behavior change."
        ),
    )
    args = p.parse_args()
    asyncio.run(main(overwrite_if_exists=args.overwrite_if_exists))
