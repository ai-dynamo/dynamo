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
    # PARSER.batch.8 sub-case pilot. Once any sub-case is introduced, the bare
    # `PARSER.batch.8` is retired — the four positional shapes below replace
    # it. The flat-file `PARSER.batch.8` entry should be removed from
    # kimi_k2/PARSER.batch.yaml after the regenerator runs.
    ("kimi_k2", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "ref": "inspired-by https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L272",
        "text": 'I\'ll check the weather. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"Dallas"}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "ref": "inspired-by https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L435",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"Dallas"}<|tool_call_end|><|tool_calls_section_end|> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": 'I\'ll check the weather. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"Dallas"}<|tool_call_end|><|tool_calls_section_end|> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'First check Dallas. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"Dallas"}<|tool_call_end|><|tool_calls_section_end|> Then check NYC. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_calls_section_end|>',
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
    ("qwen3_coder", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": "I will check the weather. <tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call> Let me know if you need more.",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": "I will check the weather. <tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call> Let me know if you need more.",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": "I will check the weather. <tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call> Then check LA weather. <tool_call>\n<function=get_weather>\n<parameter=location>\nLA\n</parameter>\n</function>\n</tool_call>",
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
    ("glm47", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "ref": "inspired-by https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L94",
        "text": "I will check the weather. <tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call> Let me know if you need more.",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": "I will check the weather. <tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call> Let me know if you need more.",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": "I will check the weather. <tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call> Then check LA weather. <tool_call>get_weather<arg_key>location</arg_key><arg_value>LA</arg_value></tool_call>",
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
    ("deepseek_v3_1", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": 'I will check the weather. <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": 'I will check the weather. <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜> Then check LA weather. <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"LA"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
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
    ("harmony", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": 'I will check the weather. <|channel|>analysis<|message|>Need to use function get_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '<|channel|>analysis<|message|>Need to use function get_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "ref": "inspired-by https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_openai_tool_parser.py#L220",
        "text": 'I will check the weather. <|channel|>analysis<|message|>Need to use function get_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. <|channel|>analysis<|message|>Need to use function get_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|> Then check LA weather. <|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"LA"}<|call|>',
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
    ("minimax_m2", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "ref": "inspired-by https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L126",
        "text": 'I will check the weather. <minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>\n</minimax:tool_call> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": 'I will check the weather. <minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>\n</minimax:tool_call> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. <minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>\n</minimax:tool_call> Then check LA weather. <minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">LA</parameter>\n</invoke>\n</minimax:tool_call>',
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
    ("nemotron_deci", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": 'I will check the weather. <TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</TOOLCALL> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": 'I will check the weather. <TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</TOOLCALL> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. <TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</TOOLCALL> Then check LA weather. <TOOLCALL>[{"name": "get_weather", "arguments": {"location": "LA"}}]</TOOLCALL>',
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
    ("pythonic", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": 'I will check the weather. [get_weather(location="NYC")]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '[get_weather(location="NYC")] Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": 'I will check the weather. [get_weather(location="NYC")] Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. [get_weather(location="NYC")] Then check LA weather. [get_weather(location="LA")]',
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
    ("gemma4", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "ref": "inspired-by https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L194",
        "text": 'I will check the weather. <|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": 'I will check the weather. <|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. <|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|> Then check LA weather. <|tool_call>call:get_weather{location:<|"|>LA<|"|>}<tool_call|>',
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
    ("deepseek_v3", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": 'I will check the weather. <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "ref": "inspired-by https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282",
        "text": 'I will check the weather. <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜> Then check LA weather. <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "LA"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
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
    # ----- deepseek_v4 (DSML) -----
    # Format: <｜DSML｜tool_calls>
    #          <｜DSML｜invoke name="NAME">
    #          <｜DSML｜parameter name="K" string="true|false">V</｜DSML｜parameter>
    #          ...
    #          </｜DSML｜invoke>
    #          </｜DSML｜tool_calls>
    # `string="true"` means the parameter value is a literal string;
    # `string="false"` means the value is a JSON literal (bool/int/array/etc).
    ("deepseek_v4", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.2"): {
        "description": "Multiple tool calls (parallel)",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_time">\n<｜DSML｜parameter name="timezone" string="true">EST</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v4", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.4"): {
        "description": "Malformed (missing </｜DSML｜parameter> end tag)",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.5"): {
        "description": "Missing </｜DSML｜tool_calls> end marker",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_time">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v4", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array, JSON-typed)",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="process_data">\n<｜DSML｜parameter name="items" string="false">[1, 2, 3]</｜DSML｜parameter>\n<｜DSML｜parameter name="config" string="false">{"nested": true}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("deepseek_v4", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": 'I will check the weather. <｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": 'I will check the weather. <｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. <｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> Then check LA weather. <｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">LA</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">LA</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- hermes -----
    ("hermes", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.2"): {
        "description": "Multiple tool calls (parallel)",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call><tool_call>{"name": "get_time", "arguments": {"timezone": "EST"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("hermes", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.4"): {
        "description": "Malformed JSON args (missing close brace)",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.5"): {
        "description": "Missing </tool_call> end marker",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>',
        "tools": [_GET_TIME_NOARG],
    },
    ("hermes", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array)",
        "text": '<tool_call>{"name": "process_data", "arguments": {"items": [1, 2, 3], "config": {"nested": true}}}</tool_call>',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("hermes", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "ref": "inspired-by https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L221",
        "text": 'I will check the weather. <tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": 'I will check the weather. <tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. <tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call> Then check LA weather. <tool_call>{"name": "get_weather", "arguments": {"location": "LA"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call><tool_call>{"name": "get_weather", "arguments": {"location": "LA"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- qwen25 -----
    ("qwen25", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.2"): {
        "description": "Multiple tool calls (parallel)",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call><tool_call>{"name": "get_time", "arguments": {"timezone": "EST"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("qwen25", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.4"): {
        "description": "Malformed JSON args (missing close brace)",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.5"): {
        "description": "Missing </tool_call> end marker",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>',
        "tools": [_GET_TIME_NOARG],
    },
    ("qwen25", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array)",
        "text": '<tool_call>{"name": "process_data", "arguments": {"items": [1, 2, 3], "config": {"nested": true}}}</tool_call>',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("qwen25", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": 'I will check the weather. <tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": 'I will check the weather. <tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. <tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call> Then check LA weather. <tool_call>{"name": "get_weather", "arguments": {"location": "LA"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call><tool_call>{"name": "get_weather", "arguments": {"location": "LA"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- mistral -----
    ("mistral", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}][/TOOL_CALLS]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.2"): {
        "description": "Multiple tool calls (parallel)",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}][/TOOL_CALLS]',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("mistral", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.4"): {
        "description": "Malformed JSON args (missing close brace)",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"][/TOOL_CALLS]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.5"): {
        "description": "Missing [/TOOL_CALLS] end marker",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": '[TOOL_CALLS][{"name": "get_time", "arguments": {}}][/TOOL_CALLS]',
        "tools": [_GET_TIME_NOARG],
    },
    ("mistral", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array)",
        "text": '[TOOL_CALLS][{"name": "process_data", "arguments": {"items": [1, 2, 3], "config": {"nested": true}}}][/TOOL_CALLS]',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("mistral", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": 'I will check the weather. [TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}][/TOOL_CALLS]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}][/TOOL_CALLS] Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "ref": "inspired-by https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1858",
        "text": 'I will check the weather. [TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}][/TOOL_CALLS] Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. [TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}][/TOOL_CALLS] Then check LA weather. [TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "LA"}}][/TOOL_CALLS]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_weather", "arguments": {"location": "LA"}}][/TOOL_CALLS]',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- jamba -----
    ("jamba", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.2"): {
        "description": "Multiple tool calls (parallel)",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}]</tool_calls>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("jamba", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.4"): {
        "description": "Malformed JSON args (missing close brace)",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"]</tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.5"): {
        "description": "Missing </tool_calls> end marker",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": '<tool_calls>[{"name": "get_time", "arguments": {}}]</tool_calls>',
        "tools": [_GET_TIME_NOARG],
    },
    ("jamba", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array)",
        "text": '<tool_calls>[{"name": "process_data", "arguments": {"items": [1, 2, 3], "config": {"nested": true}}}]</tool_calls>',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("jamba", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": 'I will check the weather. <tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</tool_calls> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": 'I will check the weather. <tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</tool_calls> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. <tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</tool_calls> Then check LA weather. <tool_calls>[{"name": "get_weather", "arguments": {"location": "LA"}}]</tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_weather", "arguments": {"location": "LA"}}]</tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- llama3_json -----
    ("llama3_json", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '<|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.2"): {
        "description": "Multiple tool calls (parallel)",
        "text": '<|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}};{"name": "get_time", "arguments": {"timezone": "EST"}}',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("llama3_json", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.4"): {
        "description": "Malformed JSON args (missing close brace)",
        "text": '<|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"',
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.5"): {
        "description": "No explicit end (truncation)",
        "text": '<|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": '<|python_tag|>{"name": "get_time", "arguments": {}}',
        "tools": [_GET_TIME_NOARG],
    },
    ("llama3_json", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array)",
        "text": '<|python_tag|>{"name": "process_data", "arguments": {"items": [1, 2, 3], "config": {"nested": true}}}',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("llama3_json", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": 'I will check the weather. <|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '<|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}} Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "ref": "inspired-by https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L128",
        "text": 'I will check the weather. <|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}} Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. <|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}} Then check LA weather. <|python_tag|>{"name": "get_weather", "arguments": {"location": "LA"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}};{"name": "get_weather", "arguments": {"location": "LA"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- phi4 -----
    ("phi4", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": 'functools[{"name": "get_weather", "arguments": {"location": "NYC"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.2"): {
        "description": "Multiple tool calls (parallel)",
        "text": 'functools[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}]',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("phi4", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.4"): {
        "description": "Malformed JSON args (missing close brace)",
        "text": 'functools[{"name": "get_weather", "arguments": {"location": "NYC"]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.5"): {
        "description": "No explicit end (truncation)",
        "text": 'functools[{"name": "get_weather", "arguments": {"location": "NYC"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": 'functools[{"name": "get_time", "arguments": {}}]',
        "tools": [_GET_TIME_NOARG],
    },
    ("phi4", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array)",
        "text": 'functools[{"name": "process_data", "arguments": {"items": [1, 2, 3], "config": {"nested": true}}}]',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("phi4", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": 'I will check the weather. functools[{"name": "get_weather", "arguments": {"location": "NYC"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": 'functools[{"name": "get_weather", "arguments": {"location": "NYC"}}] Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "ref": "inspired-by https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282",
        "text": 'I will check the weather. functools[{"name": "get_weather", "arguments": {"location": "NYC"}}] Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. functools[{"name": "get_weather", "arguments": {"location": "NYC"}}] Then check LA weather. functools[{"name": "get_weather", "arguments": {"location": "LA"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": 'functools[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_weather", "arguments": {"location": "LA"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- nemotron_nano -----
    ("nemotron_nano", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.2"): {
        "description": "Multiple tool calls (parallel)",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_time>\n<parameter=timezone>\nEST\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("nemotron_nano", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.4"): {
        "description": "Malformed (missing </parameter> closing tag)",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.5"): {
        "description": "Missing </tool_call> end marker",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": "<tool_call>\n<function=get_time>\n</function>\n</tool_call>",
        "tools": [_GET_TIME_NOARG],
    },
    ("nemotron_nano", "PARSER.batch.7"): {
        "description": "Complex args (multi-parameter)",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC_UNIT],
    },
    ("nemotron_nano", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "text": "I will check the weather. <tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call> Let me know if you need more.",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": "I will check the weather. <tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call> Let me know if you need more.",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": "I will check the weather. <tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call> Then check LA weather. <tool_call>\n<function=get_weather>\n<parameter=location>\nLA\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=location>\nLA\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    # ----- deepseek_v3_2 -----
    ("deepseek_v3_2", "PARSER.batch.1"): {
        "description": "Single tool call (happy path)",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.2"): {
        "description": "Multiple tool calls (parallel)",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_time">\n<｜DSML｜parameter name="timezone" string="true">EST</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v3_2", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.4"): {
        "description": "Malformed (missing </｜DSML｜parameter> end tag)",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.5"): {
        "description": "Missing </｜DSML｜function_calls> end marker",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.6"): {
        "description": "Empty args (no-arg call)",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_time">\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v3_2", "PARSER.batch.7"): {
        "description": "Complex args (nested object + array, JSON-typed)",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="process_data">\n<｜DSML｜parameter name="items" string="false">[1, 2, 3]</｜DSML｜parameter>\n<｜DSML｜parameter name="config" string="false">{"nested": true}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("deepseek_v3_2", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "ref": "inspired-by https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L158",
        "text": 'I will check the weather. <｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.8.c"): {
        "description": "Narration both before and after (sandwich)",
        "text": 'I will check the weather. <｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls> Let me know if you need more.',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.8.d"): {
        "description": "Narration between multiple tool calls",
        "text": 'I will check the weather. <｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls> Then check LA weather. <｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">LA</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.9"): {
        "description": "Empty input",
        "text": "",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.10"): {
        "description": "Duplicate calls (same name twice)",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">LA</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
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


def _file_stem_for(case_id: str) -> str:
    """Map a case ID to the YAML file stem it belongs in.

    `PARSER.batch.5`   → `PARSER.batch`        (legacy flat file: cases 1..10)
    `PARSER.batch.8.a` → `PARSER.batch.8`      (per-top-level-case file: 8.a, 8.b, ...)

    Once any sub-case `PARSER.<mode>.<n>.<sub>` is introduced, the bare
    `PARSER.<mode>.<n>` key is retired — its sub-cases live in the per-case
    file together. The loader's two-layout merge keeps it conflict-free.
    """
    parts = case_id.split(".")
    if len(parts) >= 4:  # has sub-case → per-case file
        return ".".join(parts[:3])  # e.g. "PARSER.batch.8"
    return ".".join(parts[:2])  # e.g. "PARSER.batch"


def _case_sort_key(case_id: str) -> tuple[int, str]:
    """Sort key for case IDs that may carry a sub-letter."""
    parts = case_id.split(".")
    return (int(parts[2]), parts[3] if len(parts) > 3 else "")


def _load_existing(family: str, file_stem: str) -> dict[str, dict[str, Any]]:
    """Read the on-disk cases dict for `<family>/<file_stem>.yaml`, or {} if absent.

    Keyed by the full case ID (`PARSER.batch.5`, `PARSER.batch.8.a`) so callers
    don't have to re-stitch the prefix.
    """
    fp = FIXTURES_ROOT / family / f"{file_stem}.yaml"
    if not fp.exists():
        return {}
    raw = yaml.safe_load(fp.read_text(encoding="utf-8")).get("cases", {}) or {}
    return dict(raw)


def _write_family_fixtures(
    family: str, file_stem: str, mode: str, cases: dict[str, dict[str, Any]]
) -> None:
    """Write `<family>/<file_stem>.yaml` holding `cases` (full-case-ID-keyed).

    Sort respects sub-case suffixes (`PARSER.batch.8.a` < `8.b`). The `mode`
    field in the YAML header is the parser mode (`batch`/`stream`), not the
    file stem — so `PARSER.batch.8.yaml` has `mode: batch`.
    """
    family_dir = FIXTURES_ROOT / family
    family_dir.mkdir(parents=True, exist_ok=True)
    ordered = {cid: cases[cid] for cid in sorted(cases, key=_case_sort_key)}
    out = {"family": family, "mode": mode, "cases": ordered}
    header = (
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )
    (family_dir / f"{file_stem}.yaml").write_text(
        header + yaml.dump(out, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


async def main(overwrite_if_exists: bool = False) -> None:
    # Group inputs by destination (family, file_stem) so we merge per file.
    # mode is recoverable from case_id; we track it for the YAML header.
    inputs_by_file: dict[tuple[str, str], tuple[str, dict[str, dict[str, Any]]]] = {}
    # Track which (family, "PARSER.<mode>.<n>") top-level case IDs have been
    # split into sub-cases via INPUTS. The bare top-level case is retired
    # whenever any sub-case exists for the same (family, mode, n).
    retired_bare_ids: set[tuple[str, str]] = set()
    for (family, case_id), entry in INPUTS.items():
        if entry is None:
            continue
        mode = case_id.split(".")[1]  # PARSER.batch.8.a → "batch"
        file_stem = _file_stem_for(case_id)  #                  → "PARSER.batch.8"
        slot = inputs_by_file.setdefault((family, file_stem), (mode, {}))
        slot[1][case_id] = entry
        # If this is a sub-case, mark the corresponding bare top-level for retirement.
        parts = case_id.split(".")
        if len(parts) >= 4:
            retired_bare_ids.add((family, ".".join(parts[:3])))

    n_written = n_skipped = n_orphan_kept = n_retired = 0
    for (family, file_stem), (mode, entries) in inputs_by_file.items():
        existing = _load_existing(family, file_stem)
        merged: dict[str, dict[str, Any]] = {}

        # 1. Process every case the user listed in INPUTS.
        for case_id, entry in entries.items():
            if case_id in existing and not overwrite_if_exists:
                merged[case_id] = existing[case_id]
                n_skipped += 1
                continue
            expected = await _run_one(family, entry["text"], entry["tools"])
            merged_case: dict[str, Any] = {"description": entry["description"]}
            # `ref` is required on per-sub-case files only (PARSER.batch.<n>.yaml).
            # URL pointing at the upstream test the fixture was sourced from —
            # the URL itself names the impl (`vllm-project/vllm`,
            # `sgl-project/sglang`, ...). For sub-cases authored fresh in this
            # repo, the literal `"dynamo"` records that we made it up rather
            # than mirrored an upstream test. The legacy flat `PARSER.batch.yaml`
            # (cases without sub-cases) does NOT carry `ref` — those entries
            # predate the convention.
            if len(case_id.split(".")) >= 4:
                merged_case["ref"] = entry.get("ref", "dynamo")
            merged_case["model_text"] = entry["text"]
            merged_case["tools"] = entry["tools"]
            merged_case["expected"] = expected
            merged[case_id] = merged_case
            n_written += 1

        # 2. Preserve any on-disk cases that aren't in INPUTS today, so a
        #    contributor's INPUTS edit can't accidentally delete other
        #    contributors' fixture cases — EXCEPT a bare `PARSER.<mode>.<n>`
        #    that's been superseded by sub-cases (`<n>.<sub>`) elsewhere in
        #    INPUTS. That bare ID gets dropped from the flat file so the
        #    retired top-level doesn't end up running alongside its
        #    replacement sub-cases after regeneration.
        for case_id, case in existing.items():
            if case_id in merged:
                continue
            if (family, case_id) in retired_bare_ids:
                n_retired += 1
                continue
            merged[case_id] = case
            n_orphan_kept += 1

        _write_family_fixtures(family, file_stem, mode, merged)
        print(f"  wrote {family}/{file_stem}.yaml with {len(merged)} cases")

    print(
        f"\n{n_written} written, {n_skipped} skipped (already on disk), "
        f"{n_orphan_kept} preserved (on disk but not in INPUTS), "
        f"{n_retired} bare-IDs retired (replaced by sub-cases).\n"
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
