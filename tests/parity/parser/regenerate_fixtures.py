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
# Multi-type tool for batch.7.a (standard scalar/container types).
_BOOK_FLIGHT_MIXED = {
    "name": "book_flight",
    "parameters": {
        "type": "object",
        "properties": {
            "destination": {"type": "string"},
            "passengers": {"type": "integer"},
            "first_class": {"type": "boolean"},
        },
    },
}
# Type-coercion tool for batch.7.c — numeric param schema.
_SET_TEMP_NUMERIC = {
    "name": "set_temperature",
    "parameters": {
        "type": "object",
        "properties": {
            "celsius": {"type": "integer"},
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
    ("kimi_k2", "PARSER.batch.2.a"): {
        "description": "Parallel calls in single section",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{"timezone":"EST"}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("kimi_k2", "PARSER.batch.2.b"): {
        "description": "Two back-to-back sections (each with one call)",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_calls_section_end|><|tool_calls_section_begin|><|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{"timezone":"EST"}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("kimi_k2", "PARSER.batch.2.c"): {
        "description": "Parallel calls with surrounding narration",
        "text": 'I will check both. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{"timezone":"EST"}<|tool_call_end|><|tool_calls_section_end|> Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("kimi_k2", "PARSER.batch.2.d"): {
        "description": "Same-name twice (id distinctness)",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{"location":"LA"}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.4.a"): {
        "description": "Garbage between section fences (no call_begin)",
        "text": "<|tool_calls_section_begin|>nonsense<|tool_calls_section_end|>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.4.b"): {
        "description": "Invalid JSON args (missing close brace)",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L151",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.4.c"): {
        "description": "Missing function name (call_begin without functions.X:N)",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|><|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.4.d"): {
        "description": "Mismatched fences (call_end before call_begin closes)",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L165",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.5.a"): {
        "description": "Missing section_end (max_tokens truncation, PR #8208)",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.5.b"): {
        "description": "Closing section_end without matching section_begin",
        "text": '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.5.c"): {
        "description": "Truncation mid-arguments (incomplete JSON body)",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L413",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"loc',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.6.a"): {
        "description": "Canonical empty {} arguments",
        "text": "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_time:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>",
        "tools": [_GET_TIME_NOARG],
    },
    ("kimi_k2", "PARSER.batch.6.b"): {
        "description": "Whitespace inside empty {}",
        "text": "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_time:0<|tool_call_argument_begin|>{ }<|tool_call_end|><|tool_calls_section_end|>",
        "tools": [_GET_TIME_NOARG],
    },
    ("kimi_k2", "PARSER.batch.6.c"): {
        "description": "No argument body (skip argument_begin → end)",
        "text": "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_time:0<|tool_call_end|><|tool_calls_section_end|>",
        "tools": [_GET_TIME_NOARG],
    },
    ("kimi_k2", "PARSER.batch.7.a"): {
        "description": "Standard scalar types (string + int + bool)",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.book_flight:0<|tool_call_argument_begin|>{"destination":"Paris","passengers":2,"first_class":true}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("kimi_k2", "PARSER.batch.7.b"): {
        "description": "Unicode + escaped chars in string value",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"Tōkyō \\"central\\""}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.set_temperature:0<|tool_call_argument_begin|>{"celsius":"20"}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("kimi_k2", "PARSER.batch.7.d"): {
        "description": "Nested object + array (existing shape)",
        "text": '<|tool_calls_section_begin|><|tool_call_begin|>functions.process_data:0<|tool_call_argument_begin|>{"items":[1,2,3],"config":{"nested":true}}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_PROCESS_DATA_NESTED],
    },
    # PARSER.batch.8 sub-case pilot. Once any sub-case is introduced, the bare
    # `PARSER.batch.8` is retired — the four positional shapes below replace
    # it. The flat-file `PARSER.batch.8` entry should be removed from
    # kimi_k2/PARSER.batch.yaml after the regenerator runs.
    ("kimi_k2", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L272",
        "text": 'I\'ll check the weather. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"Dallas"}<|tool_call_end|><|tool_calls_section_end|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("kimi_k2", "PARSER.batch.8.b"): {
        "description": "Narration after tool call only",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_kimi_k2_tool_parser.py#L435",
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
    ("qwen3_coder", "PARSER.batch.2.a"): {
        "description": "Parallel calls (different names)",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L185",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_time>\n<parameter=timezone>\nEST\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("qwen3_coder", "PARSER.batch.2.c"): {
        "description": "Parallel calls with surrounding narration",
        "text": "Both queries: <tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_time>\n<parameter=timezone>\nEST\n</parameter>\n</function>\n</tool_call> Results above.",
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("qwen3_coder", "PARSER.batch.2.d"): {
        "description": "Same-name twice (id distinctness)",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=location>\nLA\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.4.a"): {
        "description": "Tool_call wrapper with no <function> body",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L322",
        "text": "<tool_call>\nrandom text without function tag\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.4.d"): {
        "description": "Missing </parameter> closing tag",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L729",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.5.a"): {
        "description": "Missing </tool_call> end marker",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L401",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.5.b"): {
        "description": "Closing </tool_call> without matching <tool_call> open",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_qwen3coder_tool_parser.py#L943",
        "text": "<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.5.c"): {
        "description": "Truncation mid-parameter value",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNY",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen3_coder", "PARSER.batch.6.a"): {
        "description": "Function block with no <parameter>",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L260",
        "text": "<tool_call>\n<function=get_time>\n</function>\n</tool_call>",
        "tools": [_GET_TIME_NOARG],
    },
    ("qwen3_coder", "PARSER.batch.6.b"): {
        "description": "Function block with extra whitespace inside",
        "text": "<tool_call>\n<function=get_time>\n\n</function>\n</tool_call>",
        "tools": [_GET_TIME_NOARG],
    },
    ("qwen3_coder", "PARSER.batch.7.a"): {
        "description": "Multiple parameters (existing multi-arg shape)",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L220",
        "text": "<tool_call>\n<function=book_flight>\n<parameter=destination>\nParis\n</parameter>\n<parameter=passengers>\n2\n</parameter>\n<parameter=first_class>\ntrue\n</parameter>\n</function>\n</tool_call>",
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("qwen3_coder", "PARSER.batch.7.b"): {
        "description": "Unicode in parameter value",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L302",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nTōkyō central\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
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
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282",
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
    ("glm47", "PARSER.batch.2.a"): {
        "description": "Parallel calls (different names)",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_time<arg_key>timezone</arg_key><arg_value>EST</arg_value></tool_call>",
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("glm47", "PARSER.batch.2.c"): {
        "description": "Parallel calls with surrounding narration",
        "text": "Checking: <tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_time<arg_key>timezone</arg_key><arg_value>EST</arg_value></tool_call> Done.",
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("glm47", "PARSER.batch.2.d"): {
        "description": "Same-name twice (id distinctness)",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>LA</arg_value></tool_call>",
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.4.a"): {
        "description": "Tool_call wrapper with no function name",
        "text": "<tool_call><arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.4.d"): {
        "description": "Missing </arg_value> end tag",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.5.a"): {
        "description": "Missing </tool_call> end marker",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.5.b"): {
        "description": "Closing </tool_call> without matching <tool_call> open",
        "text": "get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.5.c"): {
        "description": "Truncation mid-arg_value",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>N",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.6.a"): {
        "description": "Function-only call (no arg keys)",
        "text": "<tool_call>get_time</tool_call>",
        "tools": [_GET_TIME_NOARG],
    },
    ("glm47", "PARSER.batch.6.b"): {
        "description": "Function-only with whitespace",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L68",
        "text": "<tool_call>get_time </tool_call>",
        "tools": [_GET_TIME_NOARG],
    },
    ("glm47", "PARSER.batch.7.a"): {
        "description": "Multiple parameters (existing multi-arg shape)",
        "text": "<tool_call>book_flight<arg_key>destination</arg_key><arg_value>Paris</arg_value><arg_key>passengers</arg_key><arg_value>2</arg_value><arg_key>first_class</arg_key><arg_value>true</arg_value></tool_call>",
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("glm47", "PARSER.batch.7.b"): {
        "description": "Unicode in arg value",
        "text": "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Tōkyō central</arg_value></tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("glm47", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_glm47_moe_tool_parser.py#L94",
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
    ("deepseek_v3_1", "PARSER.batch.2.a"): {
        "description": "Parallel calls in single fence pair",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv31_tool_parser.py#L39",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone":"EST"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v3_1", "PARSER.batch.2.b"): {
        "description": "Two back-to-back fence pairs",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone":"EST"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v3_1", "PARSER.batch.2.c"): {
        "description": "Parallel with surrounding narration",
        "text": 'Both: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone":"EST"}<｜tool▁call▁end｜><｜tool▁calls▁end｜> Results.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v3_1", "PARSER.batch.2.d"): {
        "description": "Same-name twice (id distinctness)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"LA"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.4.a"): {
        "description": "Garbage between calls fences (no tool_sep / args)",
        "text": "<｜tool▁calls▁begin｜>nonsense<｜tool▁calls▁end｜>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.4.b"): {
        "description": "Unterminated JSON string in args",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.4.c"): {
        "description": "Missing function name (no token before tool_sep)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜><｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.4.d"): {
        "description": "Mismatched fences (calls_end without call_end)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.5.a"): {
        "description": "Missing tool_calls_end (truncation)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.5.b"): {
        "description": "Closing tool_calls_end without matching tool_calls_begin",
        "text": '<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.5.c"): {
        "description": "Truncation mid-arguments JSON",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"loc',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.6.a"): {
        "description": "Canonical empty {} after tool_sep",
        "text": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v3_1", "PARSER.batch.6.b"): {
        "description": "Whitespace inside empty { }",
        "text": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{ }<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v3_1", "PARSER.batch.6.c"): {
        "description": "No tool_sep / args section",
        "text": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v3_1", "PARSER.batch.7.a"): {
        "description": "Standard scalar types",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>book_flight<｜tool▁sep｜>{"destination":"Paris","passengers":2,"first_class":true}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("deepseek_v3_1", "PARSER.batch.7.b"): {
        "description": "Unicode + escaped chars",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Tōkyō \\"central\\""}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_1", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>set_temperature<｜tool▁sep｜>{"celsius":"20"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("deepseek_v3_1", "PARSER.batch.7.d"): {
        "description": "Nested object + array (existing)",
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
    ("harmony", "PARSER.batch.2.a"): {
        "description": "Two back-to-back commentary envelopes",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_openai_tool_parser.py#L130",
        "text": '<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_time <|constrain|>json<|message|>{"timezone":"EST"}<|call|>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("harmony", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": 'I need both. <|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_time <|constrain|>json<|message|>{"timezone":"EST"}<|call|> Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("harmony", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "text": '<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"LA"}<|call|>',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.4.a"): {
        "description": "Channel envelope with garbage body",
        "text": "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>not json at all<|call|>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.4.b"): {
        "description": "Unterminated JSON in message body",
        "text": '<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC<|call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.4.c"): {
        "description": "No to=functions.X recipient",
        "text": '<|start|>assistant<|channel|>commentary <|constrain|>json<|message|>{"location":"NYC"}<|call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.5.a"): {
        "description": "Missing <|call|> end marker (bare envelope)",
        "text": '<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.5.b"): {
        "description": "Bare <|call|> without preceding channel envelope",
        "text": "<|call|>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.5.c"): {
        "description": "Truncation mid-message JSON",
        "text": '<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"loc',
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.6.a"): {
        "description": "Canonical empty {} message body",
        "text": "<|channel|>commentary to=functions.get_time <|constrain|>json<|message|>{}",
        "tools": [_GET_TIME_NOARG],
    },
    ("harmony", "PARSER.batch.6.b"): {
        "description": "Whitespace inside empty {}",
        "text": "<|channel|>commentary to=functions.get_time <|constrain|>json<|message|>{ }",
        "tools": [_GET_TIME_NOARG],
    },
    ("harmony", "PARSER.batch.6.c"): {
        "description": "No <|message|> body",
        "text": "<|channel|>commentary to=functions.get_time <|constrain|>json",
        "tools": [_GET_TIME_NOARG],
    },
    ("harmony", "PARSER.batch.7.a"): {
        "description": "Standard scalar types",
        "text": '<|channel|>commentary to=functions.book_flight <|constrain|>json<|message|>{"destination":"Paris","passengers":2,"first_class":true}<|call|>',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("harmony", "PARSER.batch.7.b"): {
        "description": "Unicode + escaped chars",
        "text": '<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"Tōkyō \\"central\\""}<|call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("harmony", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": '<|channel|>commentary to=functions.set_temperature <|constrain|>json<|message|>{"celsius":"20"}<|call|>',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("harmony", "PARSER.batch.7.d"): {
        "description": "Nested object + array",
        "text": '<|channel|>commentary to=functions.process_data <|constrain|>json<|message|>{"items":[1,2,3],"config":{"nested":true}}<|call|>',
        "tools": [_PROCESS_DATA_NESTED],
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
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_openai_tool_parser.py#L220",
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
    ("minimax_m2", "PARSER.batch.2.a"): {
        "description": "Two invokes inside one tool_call wrapper",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>\n<invoke name="get_time">\n<parameter name="timezone">EST</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("minimax_m2", "PARSER.batch.2.b"): {
        "description": "Two back-to-back tool_call wrappers",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L207",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>\n</minimax:tool_call>\n<minimax:tool_call>\n<invoke name="get_time">\n<parameter name="timezone">EST</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("minimax_m2", "PARSER.batch.2.c"): {
        "description": "Parallel with surrounding narration",
        "text": 'Both: <minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>\n<invoke name="get_time">\n<parameter name="timezone">EST</parameter>\n</invoke>\n</minimax:tool_call> Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("minimax_m2", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L331",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>\n<invoke name="get_weather">\n<parameter name="location">LA</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.4.a"): {
        "description": "Tool_call wrapper with garbage body",
        "text": "<minimax:tool_call>\nnot a valid invoke\n</minimax:tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.4.c"): {
        "description": "Invoke without name attribute",
        "text": '<minimax:tool_call>\n<invoke>\n<parameter name="location">NYC</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.4.d"): {
        "description": "Missing closing </invoke>",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.5.a"): {
        "description": "Missing </minimax:tool_call> end marker",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.5.b"): {
        "description": "Closing </minimax:tool_call> without matching open",
        "text": '<invoke name="get_weather">\n<parameter name="location">NYC</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.5.c"): {
        "description": "Truncation mid-parameter value",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">NY',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.6.a"): {
        "description": "Invoke with no <parameter>",
        "text": '<minimax:tool_call>\n<invoke name="get_time">\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_TIME_NOARG],
    },
    ("minimax_m2", "PARSER.batch.6.b"): {
        "description": "Invoke with extra newline",
        "text": '<minimax:tool_call>\n<invoke name="get_time">\n\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_TIME_NOARG],
    },
    ("minimax_m2", "PARSER.batch.7.a"): {
        "description": "Multiple parameters",
        "text": '<minimax:tool_call>\n<invoke name="book_flight">\n<parameter name="destination">Paris</parameter>\n<parameter name="passengers">2</parameter>\n<parameter name="first_class">true</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("minimax_m2", "PARSER.batch.7.b"): {
        "description": "Unicode in parameter",
        "text": '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">Tōkyō central</parameter>\n</invoke>\n</minimax:tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("minimax_m2", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_minimax_m2_tool_parser.py#L126",
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
    ("nemotron_deci", "PARSER.batch.2.a"): {
        "description": "Parallel calls in JSON array",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}]</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("nemotron_deci", "PARSER.batch.2.b"): {
        "description": "Two back-to-back TOOLCALL wrappers",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</TOOLCALL><TOOLCALL>[{"name": "get_time", "arguments": {"timezone": "EST"}}]</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("nemotron_deci", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": 'Both: <TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}]</TOOLCALL> Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("nemotron_deci", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_weather", "arguments": {"location": "LA"}}]</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.4.a"): {
        "description": "TOOLCALL wrapper with garbage",
        "text": "<TOOLCALL>not a json array</TOOLCALL>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.4.b"): {
        "description": "Unterminated JSON string in args",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.4.c"): {
        "description": "Missing name key in JSON",
        "text": '<TOOLCALL>[{"arguments": {"location": "NYC"}}]</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.4.d"): {
        "description": "Mismatched JSON brackets",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.5.a"): {
        "description": "Missing </TOOLCALL> end marker",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.5.b"): {
        "description": "Closing </TOOLCALL> without matching <TOOLCALL> open",
        "text": '[{"name": "get_weather", "arguments": {"location": "NYC"}}]</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.5.c"): {
        "description": "Truncation mid-arguments JSON",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"loc',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.6.a"): {
        "description": 'Canonical "arguments": {}',
        "text": '<TOOLCALL>[{"name": "get_time", "arguments": {}}]</TOOLCALL>',
        "tools": [_GET_TIME_NOARG],
    },
    ("nemotron_deci", "PARSER.batch.6.b"): {
        "description": "Whitespace inside arguments {}",
        "text": '<TOOLCALL>[{"name": "get_time", "arguments": { }}]</TOOLCALL>',
        "tools": [_GET_TIME_NOARG],
    },
    ("nemotron_deci", "PARSER.batch.6.c"): {
        "description": "No arguments key",
        "text": '<TOOLCALL>[{"name": "get_time"}]</TOOLCALL>',
        "tools": [_GET_TIME_NOARG],
    },
    ("nemotron_deci", "PARSER.batch.7.a"): {
        "description": "Standard scalar types",
        "text": '<TOOLCALL>[{"name": "book_flight", "arguments": {"destination": "Paris", "passengers": 2, "first_class": true}}]</TOOLCALL>',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("nemotron_deci", "PARSER.batch.7.b"): {
        "description": "Unicode + escaped chars",
        "text": '<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "Tōkyō \\"central\\""}}]</TOOLCALL>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_deci", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": '<TOOLCALL>[{"name": "set_temperature", "arguments": {"celsius": "20"}}]</TOOLCALL>',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("nemotron_deci", "PARSER.batch.7.d"): {
        "description": "Nested object + array (existing)",
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
    ("pythonic", "PARSER.batch.2.a"): {
        "description": "Parallel calls in single bracket",
        "text": '[get_weather(location="NYC"), get_time(timezone="EST")]',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("pythonic", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": 'Both: [get_weather(location="NYC"), get_time(timezone="EST")] Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("pythonic", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "text": '[get_weather(location="NYC"), get_weather(location="LA")]',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.4.a"): {
        "description": "Bracket form with garbage call",
        "text": "[not a valid python call]",
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.4.b"): {
        "description": "Missing closing bracket",
        "text": '[get_weather(location="NYC"',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.4.d"): {
        "description": "Mismatched paren / bracket",
        "text": '[get_weather(location="NYC"]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.5.a"): {
        "description": "Missing closing `]` end marker",
        "text": '[get_weather(location="NYC")',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.5.b"): {
        "description": "Closing `]` without matching `[` open",
        "text": 'get_weather(location="NYC")]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.5.c"): {
        "description": "Truncation mid-argument value",
        "text": '[get_weather(location="N',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.6.a"): {
        "description": "Empty parens (no args)",
        "text": "[get_time()]",
        "tools": [_GET_TIME_NOARG],
    },
    ("pythonic", "PARSER.batch.6.b"): {
        "description": "Whitespace inside parens",
        "text": "[get_time( )]",
        "tools": [_GET_TIME_NOARG],
    },
    ("pythonic", "PARSER.batch.7.a"): {
        "description": "Standard scalar types",
        "text": '[book_flight(destination="Paris", passengers=2, first_class=True)]',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("pythonic", "PARSER.batch.7.b"): {
        "description": "Unicode + escaped chars",
        "text": '[get_weather(location="Tōkyō \\"central\\"")]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("pythonic", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": '[set_temperature(celsius="20")]',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("pythonic", "PARSER.batch.7.d"): {
        "description": "Nested dict + array (existing)",
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
    ("gemma4", "PARSER.batch.2.a"): {
        "description": "Parallel calls (back-to-back sentinels)",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L207",
        "text": '<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|><|tool_call>call:get_time{timezone:<|"|>EST<|"|>}<tool_call|>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("gemma4", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": 'Both: <|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|><|tool_call>call:get_time{timezone:<|"|>EST<|"|>}<tool_call|> Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("gemma4", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "text": '<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|><|tool_call>call:get_weather{location:<|"|>LA<|"|>}<tool_call|>',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.4.a"): {
        "description": "Sentinel pair with garbage",
        "text": "<|tool_call>nonsense<tool_call|>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.4.b"): {
        "description": "Missing close brace in body",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L572",
        "text": '<|tool_call>call:get_weather{location:<|"|>NYC<|"|><tool_call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.4.c"): {
        "description": "No call:name prefix",
        "text": '<|tool_call>{location:<|"|>NYC<|"|>}<tool_call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.4.d"): {
        "description": "Mismatched sentinels",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L252",
        "text": '<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.5.a"): {
        "description": "Missing <tool_call|> end marker",
        "text": '<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.5.b"): {
        "description": "Closing <tool_call|> without matching <|tool_call> open",
        "text": 'call:get_weather{location:<|"|>NYC<|"|>}<tool_call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.5.c"): {
        "description": "Truncation mid-argument value",
        "text": '<|tool_call>call:get_weather{location:<|"|>N',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.6.a"): {
        "description": "Canonical empty {} body",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L560",
        "text": "<|tool_call>call:get_time{}<tool_call|>",
        "tools": [_GET_TIME_NOARG],
    },
    ("gemma4", "PARSER.batch.6.b"): {
        "description": "Whitespace inside empty {}",
        "text": "<|tool_call>call:get_time{ }<tool_call|>",
        "tools": [_GET_TIME_NOARG],
    },
    ("gemma4", "PARSER.batch.6.c"): {
        "description": "No body (just call:name)",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L280",
        "text": "<|tool_call>call:get_time<tool_call|>",
        "tools": [_GET_TIME_NOARG],
    },
    ("gemma4", "PARSER.batch.7.a"): {
        "description": "Standard scalar types",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L236",
        "text": '<|tool_call>call:book_flight{destination:<|"|>Paris<|"|>,passengers:2,first_class:true}<tool_call|>',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("gemma4", "PARSER.batch.7.b"): {
        "description": "Unicode in value",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L630",
        "text": '<|tool_call>call:get_weather{location:<|"|>Tōkyō central<|"|>}<tool_call|>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("gemma4", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": '<|tool_call>call:set_temperature{celsius:<|"|>20<|"|>}<tool_call|>',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("gemma4", "PARSER.batch.7.d"): {
        "description": "Nested object + array (existing)",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L179",
        "text": "<|tool_call>call:process_data{items:[1,2,3],config:{nested:true}}<tool_call|>",
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("gemma4", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_gemma4_tool_parser.py#L194",
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
    ("deepseek_v3", "PARSER.batch.2.a"): {
        "description": "Parallel calls in single fence pair",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L185",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n```json\n{"timezone": "EST"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v3", "PARSER.batch.2.b"): {
        "description": "Two back-to-back fence pairs",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n```json\n{"timezone": "EST"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v3", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": 'Both: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n```json\n{"timezone": "EST"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜> Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v3", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "LA"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.4.a"): {
        "description": "Garbage between calls fences",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L322",
        "text": "<｜tool▁calls▁begin｜>nonsense<｜tool▁calls▁end｜>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.4.b"): {
        "description": "Missing close brace inside fenced JSON",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.4.c"): {
        "description": "Missing function name token",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.4.d"): {
        "description": "Missing fenced JSON block (no ```json)",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n{"location": "NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.5.a"): {
        "description": "Missing calls_end / call_end markers",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.5.b"): {
        "description": "Closing tool_calls_end without matching tool_calls_begin",
        "text": '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.5.c"): {
        "description": "Truncation mid-arguments JSON inside fenced block",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"loc',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.6.a"): {
        "description": "Canonical empty {} in fenced code block",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L260",
        "text": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v3", "PARSER.batch.6.b"): {
        "description": "Whitespace inside empty { } (newlines)",
        "text": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n```json\n{\n}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v3", "PARSER.batch.6.c"): {
        "description": "No fenced JSON block (function name only)",
        "text": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v3", "PARSER.batch.7.a"): {
        "description": "Standard scalar types",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L220",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>book_flight\n```json\n{"destination": "Paris", "passengers": 2, "first_class": true}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("deepseek_v3", "PARSER.batch.7.b"): {
        "description": "Unicode + escaped chars",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L302",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"location": "Tōkyō \\"central\\""}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>set_temperature\n```json\n{"celsius": "20"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("deepseek_v3", "PARSER.batch.7.d"): {
        "description": "Nested object + array (existing)",
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
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282",
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
    ("deepseek_v4", "PARSER.batch.2.a"): {
        "description": "Two invokes inside one tool_calls wrapper",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_time">\n<｜DSML｜parameter name="timezone" string="true">EST</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v4", "PARSER.batch.2.b"): {
        "description": "Two back-to-back tool_calls wrappers",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_time">\n<｜DSML｜parameter name="timezone" string="true">EST</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v4", "PARSER.batch.2.c"): {
        "description": "Parallel with surrounding narration",
        "text": 'Both: <｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_time">\n<｜DSML｜parameter name="timezone" string="true">EST</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v4", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">LA</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.4.a"): {
        "description": "Tool_calls wrapper with garbage",
        "text": "<｜DSML｜tool_calls>\nnot a valid invoke\n</｜DSML｜tool_calls>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.4.c"): {
        "description": "Invoke without name attribute",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke>\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.4.d"): {
        "description": "Missing </｜DSML｜parameter> end tag",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.5.a"): {
        "description": "Missing </｜DSML｜tool_calls> end marker",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.5.b"): {
        "description": "Closing </｜DSML｜tool_calls> without matching open",
        "text": '<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.5.c"): {
        "description": "Truncation mid-parameter value",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NY',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v4", "PARSER.batch.6.a"): {
        "description": "Invoke with no <parameter>",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_time">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v4", "PARSER.batch.6.b"): {
        "description": "Invoke with extra newline",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_time">\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v4", "PARSER.batch.7.a"): {
        "description": "Multiple parameters",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="book_flight">\n<｜DSML｜parameter name="destination" string="true">Paris</｜DSML｜parameter>\n<｜DSML｜parameter name="passengers" string="false">2</｜DSML｜parameter>\n<｜DSML｜parameter name="first_class" string="false">true</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("deepseek_v4", "PARSER.batch.7.b"): {
        "description": "Unicode in parameter",
        "text": '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">Tōkyō central</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        "tools": [_GET_WEATHER_LOC],
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
    ("hermes", "PARSER.batch.2.a"): {
        "description": "Parallel calls (back-to-back tool_call wrappers)",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call><tool_call>{"name": "get_time", "arguments": {"timezone": "EST"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("hermes", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": 'Both: <tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call><tool_call>{"name": "get_time", "arguments": {"timezone": "EST"}}</tool_call> Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("hermes", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call><tool_call>{"name": "get_weather", "arguments": {"location": "LA"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.4.a"): {
        "description": "Tool_call wrapper with non-JSON garbage",
        "text": "<tool_call>not even json</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.4.b"): {
        "description": "Missing close brace in JSON body",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L376",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.4.c"): {
        "description": "Missing name key",
        "text": '<tool_call>{"arguments": {"location": "NYC"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.4.d"): {
        "description": "Missing </tool_call> close",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.5.a"): {
        "description": "Missing </tool_call> end marker",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.5.b"): {
        "description": "Closing </tool_call> without matching <tool_call> open",
        "text": '{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.5.c"): {
        "description": "Truncation mid-arguments JSON",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L356",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"loc',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.6.a"): {
        "description": 'Canonical "arguments": {}',
        "text": '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>',
        "tools": [_GET_TIME_NOARG],
    },
    ("hermes", "PARSER.batch.6.b"): {
        "description": "Whitespace inside arguments {}",
        "text": '<tool_call>{"name": "get_time", "arguments": { }}</tool_call>',
        "tools": [_GET_TIME_NOARG],
    },
    ("hermes", "PARSER.batch.6.c"): {
        "description": "No arguments key",
        "text": '<tool_call>{"name": "get_time"}</tool_call>',
        "tools": [_GET_TIME_NOARG],
    },
    ("hermes", "PARSER.batch.7.a"): {
        "description": "Standard scalar types",
        "text": '<tool_call>{"name": "book_flight", "arguments": {"destination": "Paris", "passengers": 2, "first_class": true}}</tool_call>',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("hermes", "PARSER.batch.7.b"): {
        "description": "Unicode + escaped chars",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "Tōkyō \\"central\\""}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("hermes", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": '<tool_call>{"name": "set_temperature", "arguments": {"celsius": "20"}}</tool_call>',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("hermes", "PARSER.batch.7.d"): {
        "description": "Nested object + array",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L284",
        "text": '<tool_call>{"name": "process_data", "arguments": {"items": [1,2,3], "config": {"nested": true}}}</tool_call>',
        "tools": [_PROCESS_DATA_NESTED],
    },
    ("hermes", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_hermes_tool_parser.py#L221",
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
    ("qwen25", "PARSER.batch.2.a"): {
        "description": "Parallel calls (back-to-back tool_call wrappers)",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call><tool_call>{"name": "get_time", "arguments": {"timezone": "EST"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("qwen25", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": 'Both: <tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call><tool_call>{"name": "get_time", "arguments": {"timezone": "EST"}}</tool_call> Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("qwen25", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call><tool_call>{"name": "get_weather", "arguments": {"location": "LA"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.4.a"): {
        "description": "Tool_call wrapper with non-JSON garbage",
        "text": "<tool_call>not even json</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.4.b"): {
        "description": "Missing close brace in JSON body",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.4.c"): {
        "description": "Missing name key",
        "text": '<tool_call>{"arguments": {"location": "NYC"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.4.d"): {
        "description": "Missing </tool_call> close",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.5.a"): {
        "description": "Missing </tool_call> end marker",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.5.b"): {
        "description": "Closing </tool_call> without matching <tool_call> open",
        "text": '{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.5.c"): {
        "description": "Truncation mid-arguments JSON",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"loc',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.6.a"): {
        "description": 'Canonical "arguments": {}',
        "text": '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>',
        "tools": [_GET_TIME_NOARG],
    },
    ("qwen25", "PARSER.batch.6.b"): {
        "description": "Whitespace inside arguments {}",
        "text": '<tool_call>{"name": "get_time", "arguments": { }}</tool_call>',
        "tools": [_GET_TIME_NOARG],
    },
    ("qwen25", "PARSER.batch.6.c"): {
        "description": "No arguments key",
        "text": '<tool_call>{"name": "get_time"}</tool_call>',
        "tools": [_GET_TIME_NOARG],
    },
    ("qwen25", "PARSER.batch.7.a"): {
        "description": "Standard scalar types",
        "text": '<tool_call>{"name": "book_flight", "arguments": {"destination": "Paris", "passengers": 2, "first_class": true}}</tool_call>',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("qwen25", "PARSER.batch.7.b"): {
        "description": "Unicode + escaped chars",
        "text": '<tool_call>{"name": "get_weather", "arguments": {"location": "Tōkyō \\"central\\""}}</tool_call>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("qwen25", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": '<tool_call>{"name": "set_temperature", "arguments": {"celsius": "20"}}</tool_call>',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("qwen25", "PARSER.batch.7.d"): {
        "description": "Nested object + array",
        "text": '<tool_call>{"name": "process_data", "arguments": {"items": [1,2,3], "config": {"nested": true}}}</tool_call>',
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
    ("mistral", "PARSER.batch.2.a"): {
        "description": "Parallel calls in JSON array",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}][/TOOL_CALLS]',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("mistral", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": 'Both: [TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}][/TOOL_CALLS] Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("mistral", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_weather", "arguments": {"location": "LA"}}][/TOOL_CALLS]',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.4.a"): {
        "description": "TOOL_CALLS wrapper with garbage",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L405",
        "text": "[TOOL_CALLS]not a json array[/TOOL_CALLS]",
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.4.b"): {
        "description": "Missing close brace in JSON",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"][/TOOL_CALLS]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.4.c"): {
        "description": "Missing name key",
        "text": '[TOOL_CALLS][{"arguments": {"location": "NYC"}}][/TOOL_CALLS]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.4.d"): {
        "description": "Missing [/TOOL_CALLS] close",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.5.a"): {
        "description": "Missing [/TOOL_CALLS] end marker",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "NYC"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.5.b"): {
        "description": "Closing [/TOOL_CALLS] without matching [TOOL_CALLS] open",
        "text": '[{"name": "get_weather", "arguments": {"location": "NYC"}}][/TOOL_CALLS]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.5.c"): {
        "description": "Truncation mid-arguments JSON",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"loc',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.6.a"): {
        "description": 'Canonical "arguments": {}',
        "text": '[TOOL_CALLS][{"name": "get_time", "arguments": {}}][/TOOL_CALLS]',
        "tools": [_GET_TIME_NOARG],
    },
    ("mistral", "PARSER.batch.6.b"): {
        "description": "Whitespace inside arguments {}",
        "text": '[TOOL_CALLS][{"name": "get_time", "arguments": { }}][/TOOL_CALLS]',
        "tools": [_GET_TIME_NOARG],
    },
    ("mistral", "PARSER.batch.6.c"): {
        "description": "No arguments key",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L514",
        "text": '[TOOL_CALLS][{"name": "get_time"}][/TOOL_CALLS]',
        "tools": [_GET_TIME_NOARG],
    },
    ("mistral", "PARSER.batch.7.a"): {
        "description": "Standard scalar types",
        "text": '[TOOL_CALLS][{"name": "book_flight", "arguments": {"destination": "Paris", "passengers": 2, "first_class": true}}][/TOOL_CALLS]',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("mistral", "PARSER.batch.7.b"): {
        "description": "Unicode + escaped chars",
        "text": '[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "Tōkyō \\"central\\""}}][/TOOL_CALLS]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("mistral", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": '[TOOL_CALLS][{"name": "set_temperature", "arguments": {"celsius": "20"}}][/TOOL_CALLS]',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("mistral", "PARSER.batch.7.d"): {
        "description": "Nested object + array",
        "text": '[TOOL_CALLS][{"name": "process_data", "arguments": {"items": [1,2,3], "config": {"nested": true}}}][/TOOL_CALLS]',
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
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_mistral_tool_parser.py#L1858",
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
    ("jamba", "PARSER.batch.2.a"): {
        "description": "Parallel calls in JSON array",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}]</tool_calls>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("jamba", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": 'Both: <tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}]</tool_calls> Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("jamba", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_weather", "arguments": {"location": "LA"}}]</tool_calls>',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.4.a"): {
        "description": "Tool_calls wrapper with garbage",
        "text": "<tool_calls>not a json array</tool_calls>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.4.b"): {
        "description": "Missing close brace in JSON",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"]</tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.4.c"): {
        "description": "Missing name key",
        "text": '<tool_calls>[{"arguments": {"location": "NYC"}}]</tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.4.d"): {
        "description": "Missing </tool_calls> close",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.5.a"): {
        "description": "Missing </tool_calls> end marker",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"location": "NYC"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.5.b"): {
        "description": "Closing </tool_calls> without matching <tool_calls> open",
        "text": '[{"name": "get_weather", "arguments": {"location": "NYC"}}]</tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.5.c"): {
        "description": "Truncation mid-arguments JSON",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"loc',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.6.a"): {
        "description": 'Canonical "arguments": {}',
        "text": '<tool_calls>[{"name": "get_time", "arguments": {}}]</tool_calls>',
        "tools": [_GET_TIME_NOARG],
    },
    ("jamba", "PARSER.batch.6.b"): {
        "description": "Whitespace inside arguments {}",
        "text": '<tool_calls>[{"name": "get_time", "arguments": { }}]</tool_calls>',
        "tools": [_GET_TIME_NOARG],
    },
    ("jamba", "PARSER.batch.6.c"): {
        "description": "No arguments key",
        "text": '<tool_calls>[{"name": "get_time"}]</tool_calls>',
        "tools": [_GET_TIME_NOARG],
    },
    ("jamba", "PARSER.batch.7.a"): {
        "description": "Standard scalar types",
        "text": '<tool_calls>[{"name": "book_flight", "arguments": {"destination": "Paris", "passengers": 2, "first_class": true}}]</tool_calls>',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("jamba", "PARSER.batch.7.b"): {
        "description": "Unicode + escaped chars",
        "text": '<tool_calls>[{"name": "get_weather", "arguments": {"location": "Tōkyō \\"central\\""}}]</tool_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("jamba", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": '<tool_calls>[{"name": "set_temperature", "arguments": {"celsius": "20"}}]</tool_calls>',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("jamba", "PARSER.batch.7.d"): {
        "description": "Nested object + array",
        "text": '<tool_calls>[{"name": "process_data", "arguments": {"items": [1,2,3], "config": {"nested": true}}}]</tool_calls>',
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
    ("llama3_json", "PARSER.batch.2.a"): {
        "description": "Parallel calls (semicolon-separated)",
        "text": '<|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}};{"name": "get_time", "arguments": {"timezone": "EST"}}',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("llama3_json", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": 'Both: <|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}};{"name": "get_time", "arguments": {"timezone": "EST"}} Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("llama3_json", "PARSER.batch.2.d"): {
        "description": "Same-name twice (semicolon-separated)",
        "text": '<|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}};{"name": "get_weather", "arguments": {"location": "LA"}}',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.4.a"): {
        "description": "Python_tag prefix with non-JSON garbage",
        "text": "<|python_tag|>not even json",
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.4.b"): {
        "description": "Missing close brace",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L66",
        "text": '<|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"',
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.4.c"): {
        "description": "Missing name key",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L234",
        "text": '<|python_tag|>{"arguments": {"location": "NYC"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.5.a"): {
        "description": "No explicit end (truncation)",
        "text": '<|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.5.c"): {
        "description": "Truncation deeper inside arguments JSON (vs .a which truncates at the closing brace)",
        "text": '<|python_tag|>{"name": "get_weather", "arguments": {"loc',
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.6.a"): {
        "description": 'Canonical "arguments": {}',
        "text": '<|python_tag|>{"name": "get_time", "arguments": {}}',
        "tools": [_GET_TIME_NOARG],
    },
    ("llama3_json", "PARSER.batch.6.b"): {
        "description": "Whitespace inside arguments {}",
        "text": '<|python_tag|>{"name": "get_time", "arguments": { }}',
        "tools": [_GET_TIME_NOARG],
    },
    ("llama3_json", "PARSER.batch.6.c"): {
        "description": "No arguments key",
        "text": '<|python_tag|>{"name": "get_time"}',
        "tools": [_GET_TIME_NOARG],
    },
    ("llama3_json", "PARSER.batch.7.a"): {
        "description": "Standard scalar types",
        "text": '<|python_tag|>{"name": "book_flight", "arguments": {"destination": "Paris", "passengers": 2, "first_class": true}}',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("llama3_json", "PARSER.batch.7.b"): {
        "description": "Unicode + escaped chars",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L217",
        "text": '<|python_tag|>{"name": "get_weather", "arguments": {"location": "Tōkyō \\"central\\""}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("llama3_json", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": '<|python_tag|>{"name": "set_temperature", "arguments": {"celsius": "20"}}',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("llama3_json", "PARSER.batch.7.d"): {
        "description": "Nested object + array",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L146",
        "text": '<|python_tag|>{"name": "process_data", "arguments": {"items": [1,2,3], "config": {"nested": true}}}',
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
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_llama3_json_tool_parser.py#L128",
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
    ("phi4", "PARSER.batch.2.a"): {
        "description": "Parallel calls in JSON array",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L185",
        "text": 'functools[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}]',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("phi4", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": 'Both: functools[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}] Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("phi4", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "text": 'functools[{"name": "get_weather", "arguments": {"location": "NYC"}}, {"name": "get_weather", "arguments": {"location": "LA"}}]',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.4.a"): {
        "description": "functools[ wrapper with garbage",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L322",
        "text": "functools[not a json array]",
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.4.b"): {
        "description": "Missing close brace",
        "text": 'functools[{"name": "get_weather", "arguments": {"location": "NYC"]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.4.c"): {
        "description": "Missing name key",
        "text": 'functools[{"arguments": {"location": "NYC"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.4.d"): {
        "description": "Missing closing ]",
        "text": 'functools[{"name": "get_weather", "arguments": {"location": "NYC"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.5.a"): {
        "description": "No explicit end (truncation)",
        "text": 'functools[{"name": "get_weather", "arguments": {"location": "NYC"}}',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.5.b"): {
        "description": "Closing `]` without matching `functools[` open",
        "text": 'functools{"name": "get_weather", "arguments": {"location": "NYC"}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.5.c"): {
        "description": "Truncation mid-arguments JSON",
        "text": 'functools[{"name": "get_weather", "arguments": {"loc',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.6.a"): {
        "description": 'Canonical "arguments": {}',
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L260",
        "text": 'functools[{"name": "get_time", "arguments": {}}]',
        "tools": [_GET_TIME_NOARG],
    },
    ("phi4", "PARSER.batch.6.b"): {
        "description": "Whitespace inside arguments {}",
        "text": 'functools[{"name": "get_time", "arguments": { }}]',
        "tools": [_GET_TIME_NOARG],
    },
    ("phi4", "PARSER.batch.6.c"): {
        "description": "No arguments key",
        "text": 'functools[{"name": "get_time"}]',
        "tools": [_GET_TIME_NOARG],
    },
    ("phi4", "PARSER.batch.7.a"): {
        "description": "Standard scalar types",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L220",
        "text": 'functools[{"name": "book_flight", "arguments": {"destination": "Paris", "passengers": 2, "first_class": true}}]',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("phi4", "PARSER.batch.7.b"): {
        "description": "Unicode + escaped chars",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L302",
        "text": 'functools[{"name": "get_weather", "arguments": {"location": "Tōkyō \\"central\\""}}]',
        "tools": [_GET_WEATHER_LOC],
    },
    ("phi4", "PARSER.batch.7.c"): {
        "description": "Schema mismatch — string value where schema declares integer",
        "text": 'functools[{"name": "set_temperature", "arguments": {"celsius": "20"}}]',
        "tools": [_SET_TEMP_NUMERIC],
    },
    ("phi4", "PARSER.batch.7.d"): {
        "description": "Nested object + array",
        "text": 'functools[{"name": "process_data", "arguments": {"items": [1,2,3], "config": {"nested": true}}}]',
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
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/common_tests.py#L282",
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
    ("nemotron_nano", "PARSER.batch.2.a"): {
        "description": "Parallel calls (back-to-back tool_call wrappers)",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_time>\n<parameter=timezone>\nEST\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("nemotron_nano", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": "Both: <tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_time>\n<parameter=timezone>\nEST\n</parameter>\n</function>\n</tool_call> Done.",
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("nemotron_nano", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=location>\nLA\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.4.a"): {
        "description": "Tool_call wrapper with no <function> body",
        "text": "<tool_call>\nrandom text\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.4.d"): {
        "description": "Missing </parameter> closing tag",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.5.a"): {
        "description": "Missing </tool_call> end marker",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.5.b"): {
        "description": "Closing </tool_call> without matching <tool_call> open",
        "text": "<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.5.c"): {
        "description": "Truncation mid-parameter value",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nNY",
        "tools": [_GET_WEATHER_LOC],
    },
    ("nemotron_nano", "PARSER.batch.6.a"): {
        "description": "Function block with no <parameter>",
        "text": "<tool_call>\n<function=get_time>\n</function>\n</tool_call>",
        "tools": [_GET_TIME_NOARG],
    },
    ("nemotron_nano", "PARSER.batch.6.b"): {
        "description": "Function block with extra whitespace inside",
        "text": "<tool_call>\n<function=get_time>\n\n</function>\n</tool_call>",
        "tools": [_GET_TIME_NOARG],
    },
    ("nemotron_nano", "PARSER.batch.7.a"): {
        "description": "Multiple parameters",
        "text": "<tool_call>\n<function=book_flight>\n<parameter=destination>\nParis\n</parameter>\n<parameter=passengers>\n2\n</parameter>\n<parameter=first_class>\ntrue\n</parameter>\n</function>\n</tool_call>",
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("nemotron_nano", "PARSER.batch.7.b"): {
        "description": "Unicode in parameter",
        "text": "<tool_call>\n<function=get_weather>\n<parameter=location>\nTōkyō central\n</parameter>\n</function>\n</tool_call>",
        "tools": [_GET_WEATHER_LOC],
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
    ("deepseek_v3_2", "PARSER.batch.2.a"): {
        "description": "Two invokes inside one function_calls wrapper",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L172",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_time">\n<｜DSML｜parameter name="timezone" string="true">EST</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v3_2", "PARSER.batch.2.b"): {
        "description": "Two back-to-back function_calls wrappers",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L327",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_time">\n<｜DSML｜parameter name="timezone" string="true">EST</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v3_2", "PARSER.batch.2.c"): {
        "description": "With surrounding narration",
        "text": 'Both: <｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_time">\n<｜DSML｜parameter name="timezone" string="true">EST</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls> Done.',
        "tools": [_GET_WEATHER_LOC, _GET_TIME_TZ],
    },
    ("deepseek_v3_2", "PARSER.batch.2.d"): {
        "description": "Same-name twice",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L370",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">LA</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_WEATHER_LOC, _GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.3"): {
        "description": "No tool call (plain text)",
        "text": "Hello, how can I help you today?",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.4.a"): {
        "description": "Function_calls wrapper with garbage",
        "text": "<｜DSML｜function_calls>\nnot a valid invoke\n</｜DSML｜function_calls>",
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.4.c"): {
        "description": "Invoke without name attribute",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke>\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.4.d"): {
        "description": "Missing </｜DSML｜parameter> end tag",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L499",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.5.a"): {
        "description": "Missing </｜DSML｜function_calls> end marker",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.5.b"): {
        "description": "Closing </｜DSML｜function_calls> without matching open",
        "text": '<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.5.c"): {
        "description": "Truncation mid-parameter value",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">NY',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.6.a"): {
        "description": "Invoke with no <parameter>",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L363",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_time">\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v3_2", "PARSER.batch.6.b"): {
        "description": "Invoke with extra newline",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_time">\n\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_TIME_NOARG],
    },
    ("deepseek_v3_2", "PARSER.batch.7.a"): {
        "description": "Multiple parameters",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="book_flight">\n<｜DSML｜parameter name="destination" string="true">Paris</｜DSML｜parameter>\n<｜DSML｜parameter name="passengers" string="false">2</｜DSML｜parameter>\n<｜DSML｜parameter name="first_class" string="false">true</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_BOOK_FLIGHT_MIXED],
    },
    ("deepseek_v3_2", "PARSER.batch.7.b"): {
        "description": "Unicode in parameter",
        "text": '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_weather">\n<｜DSML｜parameter name="location" string="true">Tōkyō central</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>',
        "tools": [_GET_WEATHER_LOC],
    },
    ("deepseek_v3_2", "PARSER.batch.8.a"): {
        "description": "Narration before tool call only",
        "ref": "originated from https://github.com/vllm-project/vllm/blob/b53c507bc91f87e28b03e9b54bbff7c76e97d58b/tests/tool_parsers/test_deepseekv32_tool_parser.py#L158",
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
    body = yaml.dump(out, sort_keys=False, allow_unicode=True, width=120)
    header = (
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
    )
    if "|2+" in body:
        header += (
            "#\n"
            '# `|2+` simply means a single newline ("\\n"); '
            "see tests/parity/README.md for the full decode.\n"
        )
    (family_dir / f"{file_stem}.yaml").write_text(
        header + "\n" + body, encoding="utf-8"
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
