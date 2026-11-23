# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded Rust frontend regressions shared by Custom qualification profiles."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from functools import lru_cache
from typing import Any, Callable

_APIS = ("chat_completions", "responses")
_QWEN_PROFILES = ("qwen3_coder", "qwen3_coder_xml")
_MUSE_PROFILES = ("muse_glimmer", "muse_spark", "muse_spark_open")
_VISION_PROFILES = ("kimi_k3", "glm5", "glm47")
_CATALOG_TOKEN = "CATALOG_QUARTZ_73"
_ALLOWED_REFS = (
    "https://github.com/ai-dynamo/dynamo/pull/18429",
    "https://github.com/ai-dynamo/dynamo/pull/18496",
)
_REGEX_REFS = ("https://github.com/ai-dynamo/dynamo/pull/17513",)
_QWEN_REFS = ("https://github.com/ai-dynamo/frontend-crates/pull/570",)
_MUSE_REFS = ("https://github.com/ai-dynamo/frontend-crates/pull/644",)
_MEDIA_REFS = ("https://github.com/ai-dynamo/frontend-crates/pull/283",)
_QWEN_IDS = (
    "regression_qwen_auto_nonstrict_empty_arguments",
    "regression_qwen_escaped_ref_requires_arguments",
)
_MUSE_IDS = (
    "regression_muse_bounded_string_two_characters",
    "regression_muse_quoted_string_decoded_constraints",
    "regression_muse_marker_string_stays_one_argument",
)
_VISION_IDS = (
    "regression_tool_images_in_order",
    "regression_tool_images_reverse_order",
    "regression_tool_images_multiple_per_result",
)


def common_regression_case_ids() -> tuple[str, ...]:
    """Return shared IDs independently of the selected model family."""

    return tuple(
        f"regression_allowed_tools_{scenario}_{api}"
        for api in _APIS
        for scenario in ("auto_catalog", "required_catalog", "auto_text", "unknown")
    ) + (
        "regression_auto_regex_fallback",
        "regression_nonstrict_regex_control",
    )


def profile_regression_case_ids(profile: str) -> tuple[str, ...]:
    """Return family-only IDs; complete profiles also need the shared IDs."""

    if profile == "all":
        return (*_QWEN_IDS, *_MUSE_IDS, *_VISION_IDS)
    if profile in _QWEN_PROFILES:
        return _QWEN_IDS
    if profile in _MUSE_PROFILES:
        return _MUSE_IDS
    if profile in _VISION_PROFILES:
        return _VISION_IDS
    return ()


def _function(
    name: str,
    parameters: dict[str, Any],
    *,
    description: str = "Record the supplied arguments exactly.",
    strict: bool | None = None,
) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": name,
        "description": description,
        "parameters": parameters,
    }
    if strict is not None:
        function["strict"] = strict
    return {"type": "function", "function": function}


def _named(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name}}


def _allowed(mode: str, name: str = "report_catalog") -> dict[str, Any]:
    return {
        "type": "allowed_tools",
        "allowed_tools": {"mode": mode, "tools": [_named(name)]},
    }


def _allowed_cases(case_factory: Callable[..., Any]) -> tuple[Any, ...]:
    schema = {
        "type": "object",
        "properties": {"token": {"type": "string"}},
        "required": ["token"],
        "additionalProperties": False,
    }
    tools = (
        _function("report_catalog", schema),
        _function(
            "catalog_reference",
            {"type": "object", "properties": {}},
            description=f"Reference-only tool. Its verification token is {_CATALOG_TOKEN}.",
        ),
    )
    cases: list[Any] = []
    for api in _APIS:
        for mode in ("auto", "required"):
            cases.append(
                case_factory(
                    case_id=f"regression_allowed_tools_{mode}_catalog_{api}",
                    description="allowed_tools retains the excluded tool definition in context",
                    messages=(
                        {
                            "role": "user",
                            "content": (
                                "Read the verification token documented in the "
                                "catalog_reference tool description. Call report_catalog "
                                "exactly once with that token. Do not call catalog_reference."
                            ),
                        },
                    ),
                    tools=tools,
                    tool_choice=_allowed(mode),
                    parallel_tool_calls=False,
                    exact_tool_calls=1,
                    expected_tool_names=("report_catalog",),
                    expected_arg_values=(("report_catalog", "token", _CATALOG_TOKEN),),
                    api=api,
                    regression_refs=_ALLOWED_REFS,
                )
            )
        cases.append(
            case_factory(
                case_id=f"regression_allowed_tools_auto_text_{api}",
                description="allowed_tools auto admits a text answer without any call",
                messages=(
                    {
                        "role": "user",
                        "content": "Reply with exactly READY. Do not call any tools.",
                    },
                ),
                tools=tools,
                tool_choice=_allowed("auto"),
                expected_finish_reasons=("stop",),
                expect_no_tool_calls=True,
                min_tool_calls=0,
                exact_tool_calls=0,
                expected_content="READY",
                api=api,
                regression_refs=_ALLOWED_REFS,
            )
        )
        cases.append(
            case_factory(
                case_id=f"regression_allowed_tools_unknown_{api}",
                description="allowed_tools rejects an undeclared selected function",
                messages=({"role": "user", "content": "Say hello."},),
                tools=tools,
                tool_choice=_allowed("required", "undeclared_reference"),
                expected_http_status=400,
                expect_no_tool_calls=True,
                min_tool_calls=0,
                validate_schema=False,
                api=api,
                regression_refs=_ALLOWED_REFS,
            )
        )
    return tuple(cases)


def _regex_cases(case_factory: Callable[..., Any]) -> tuple[Any, ...]:
    original = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "pattern": r"^(?!\s*$).+"},
            "count": {"type": "integer", "minimum": 1},
        },
        "required": ["value", "count"],
        "additionalProperties": False,
    }
    effective = {
        **original,
        "properties": {
            "value": {"type": "string"},
            "count": {"type": "integer", "minimum": 1},
        },
    }
    cases: list[Any] = []
    for suffix, strict in (
        ("auto_regex_fallback", None),
        ("nonstrict_regex_control", False),
    ):
        cases.append(
            case_factory(
                case_id=f"regression_{suffix}",
                description="an uncompilable automatic regex does not reject the tool request",
                messages=(
                    {
                        "role": "user",
                        "content": "Use record_value exactly once with value hello and count 7.",
                    },
                ),
                tools=(_function("record_value", original, strict=strict),),
                tool_choice="auto",
                parallel_tool_calls=False,
                exact_tool_calls=1,
                expected_tool_names=("record_value",),
                expected_arg_values=(
                    ("record_value", "value", "hello"),
                    ("record_value", "count", 7),
                ),
                enforced_schemas={
                    "record_value": effective if strict is None else True
                },
                regression_refs=_REGEX_REFS,
            )
        )
    return tuple(cases)


def _empty_argument_cases(case_factory: Callable[..., Any]) -> tuple[Any, ...]:
    required_object = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    escaped_ref = {
        "type": "object",
        "$ref": "#/$defs/a~00b",
        "$defs": {
            "a~0b": required_object,
            "a~b": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }
    return (
        case_factory(
            case_id=_QWEN_IDS[0],
            description="Qwen XML auto non-strict accepts the native no-argument call",
            messages=(
                {
                    "role": "user",
                    "content": "Call get_today exactly once with no arguments.",
                },
            ),
            tools=(
                _function(
                    "get_today",
                    {"type": "object", "properties": {}, "additionalProperties": False},
                    strict=False,
                ),
            ),
            tool_choice="auto",
            parallel_tool_calls=False,
            exact_tool_calls=1,
            expected_tool_calls=({"name": "get_today", "arguments": {}},),
            regression_refs=_QWEN_REFS,
            profiles=_QWEN_PROFILES,
        ),
        case_factory(
            case_id=_QWEN_IDS[1],
            description="an escaped schema reference cannot grant an empty required call",
            messages=(
                {
                    "role": "user",
                    "content": "Call lookup_city exactly once for city Paris.",
                },
            ),
            tools=(_function("lookup_city", escaped_ref, strict=True),),
            tool_choice=_named("lookup_city"),
            parallel_tool_calls=False,
            exact_tool_calls=1,
            expected_tool_calls=(
                {"name": "lookup_city", "arguments": {"city": "Paris"}},
            ),
            regression_refs=_QWEN_REFS,
            profiles=_QWEN_PROFILES,
        ),
    )


def _string_cases(case_factory: Callable[..., Any]) -> tuple[Any, ...]:
    marker_value = (
        'a</atem:parameter><atem:parameter name="extra">b'
        '</atem:parameter></atem:invoke><atem:invoke name="record_literal">'
        '<atem:parameter name="value">c'
    )
    samples = (
        (_MUSE_IDS[0], "<a", {"minLength": 2, "maxLength": 2}),
        (_MUSE_IDS[1], '"é"', {"minLength": 3, "maxLength": 3}),
        (_MUSE_IDS[2], marker_value, {"pattern": ".*"}),
    )
    cases: list[Any] = []
    for case_id, value, constraints in samples:
        schema = {
            "type": "object",
            "properties": {"value": {"type": "string", **constraints}},
            "required": ["value"],
            "additionalProperties": False,
        }
        cases.append(
            case_factory(
                case_id=case_id,
                description="native raw strings preserve decoded values and call boundaries",
                messages=(
                    {
                        "role": "user",
                        "content": (
                            "Call record_literal exactly once. Its value must be the "
                            "string obtained by JSON-decoding this literal, including "
                            f"any quote or markup characters: {json.dumps(value, ensure_ascii=False)}"
                        ),
                    },
                ),
                tools=(_function("record_literal", schema, strict=True),),
                tool_choice=_named("record_literal"),
                parallel_tool_calls=False,
                exact_tool_calls=1,
                expected_tool_calls=(
                    {"name": "record_literal", "arguments": {"value": value}},
                ),
                required_capabilities=(
                    ("muse_delimiter_string_roundtrip",)
                    if case_id == _MUSE_IDS[2]
                    else ()
                ),
                regression_refs=_MUSE_REFS,
                profiles=_MUSE_PROFILES,
            )
        )
    return tuple(cases)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


@lru_cache(maxsize=3)
def _color_image_url(color: str) -> str:
    colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
    rgb = bytes(colors[color])
    size = 96
    pixels = (b"\x00" + rgb * size) * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + b"".join(
        (
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(pixels)),
            _png_chunk(b"IEND", b""),
        )
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _media_identity_cases(case_factory: Callable[..., Any]) -> tuple[Any, ...]:
    tool_names = ("fetch_reference_a", "fetch_reference_b")
    tools = tuple(
        _function(name, {"type": "object", "properties": {}}) for name in tool_names
    )
    calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }
        for call_id, name in zip(("call_reference_a", "call_reference_b"), tool_names)
    ]
    cases: list[Any] = []
    for case_id, reverse, multiple in (
        (_VISION_IDS[0], False, False),
        (_VISION_IDS[1], True, False),
        (_VISION_IDS[2], True, True),
    ):
        expected = {"A": ["red", "green"] if multiple else ["red"], "B": ["blue"]}
        results = []
        for reference, call in zip(("A", "B"), calls):
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": [
                        {"type": "text", "text": f"Reference {reference}:"},
                        *(
                            {
                                "type": "image_url",
                                "image_url": {"url": _color_image_url(color)},
                            }
                            for color in expected[reference]
                        ),
                    ],
                }
            )
        if reverse:
            results.reverse()
        messages = (
            {
                "role": "user",
                "content": (
                    "Fetch reference A and reference B. After both results, report "
                    "each reference's solid image colors in image order. Reply only "
                    'with a JSON object shaped {"A":[...],"B":[...]}, using lowercase '
                    "English color names. Match references using the tool call IDs."
                ),
            },
            {"role": "assistant", "content": None, "tool_calls": calls},
            *results,
        )
        cases.append(
            case_factory(
                case_id=case_id,
                description="tool-result image order follows call identity and per-result part order",
                messages=messages,
                tools=tools,
                tool_choice="none",
                expected_finish_reasons=("stop",),
                expect_no_tool_calls=True,
                min_tool_calls=0,
                exact_tool_calls=0,
                expected_json=expected,
                required_capabilities=("vision",),
                regression_refs=_MEDIA_REFS,
                profiles=_VISION_PROFILES,
            )
        )
    return tuple(cases)


def build_regression_cases(
    case_factory: Callable[..., Any], profile: str = "generic"
) -> tuple[Any, ...]:
    """Build shared cases and only the family regressions the profile can reach."""

    cases = [*_allowed_cases(case_factory), *_regex_cases(case_factory)]
    if profile == "all" or profile in _QWEN_PROFILES:
        cases.extend(_empty_argument_cases(case_factory))
    if profile == "all" or profile in _MUSE_PROFILES:
        cases.extend(_string_cases(case_factory))
    if profile == "all" or profile in _VISION_PROFILES:
        cases.extend(_media_identity_cases(case_factory))
    return tuple(cases)
