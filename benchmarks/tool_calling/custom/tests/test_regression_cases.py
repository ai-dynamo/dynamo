# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import copy
import dataclasses
import json
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tool_calling_probe as probe  # noqa: E402 -- Custom also supports direct script entrypoints.
from regression_cases import (  # noqa: E402 -- Custom also supports direct script entrypoints.
    build_regression_cases,
    common_regression_case_ids,
    profile_regression_case_ids,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def _cases(profile: str) -> dict[str, SimpleNamespace]:
    return {
        case.case_id: case for case in build_regression_cases(SimpleNamespace, profile)
    }


def _png_pixels(url: str) -> tuple[int, int, bytes]:
    png = base64.b64decode(url.removeprefix("data:image/png;base64,"), validate=True)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    compressed = []
    width = height = 0
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        kind = png[offset + 4 : offset + 8]
        payload = png[offset + 8 : offset + 8 + length]
        crc = struct.unpack(">I", png[offset + 8 + length : offset + 12 + length])[0]
        assert crc == zlib.crc32(kind + payload) & 0xFFFFFFFF
        if kind == b"IHDR":
            (
                width,
                height,
                depth,
                color,
                compression,
                filtering,
                interlace,
            ) = struct.unpack(">IIBBBBB", payload)
            assert (depth, color, compression, filtering, interlace) == (8, 2, 0, 0, 0)
        elif kind == b"IDAT":
            compressed.append(payload)
        offset += length + 12
    return width, height, zlib.decompress(b"".join(compressed))


@pytest.mark.parametrize(
    "profile",
    [
        "generic",
        "qwen3_coder_xml",
        "qwen3_coder",
        "muse_glimmer",
        "muse_spark_open",
        "kimi_k3",
        "glm5",
        "all",
    ],
)
def test_regressions_are_reachable_from_their_selection_manifests(profile):
    cases = build_regression_cases(SimpleNamespace, profile)
    actual = [case.case_id for case in cases]
    selected = (*common_regression_case_ids(), *profile_regression_case_ids(profile))

    assert len(actual) == len(set(actual))
    assert set(actual) == set(selected)
    assert all(case.regression_refs for case in cases)


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_allowed_catalog_probe_has_no_answer_in_the_messages(api):
    cases = _cases("generic")
    for mode in ("auto", "required"):
        case = cases[f"regression_allowed_tools_{mode}_catalog_{api}"]
        report, reference = case.tools
        token = case.expected_arg_values[0][2]

        assert token not in json.dumps(case.messages)
        assert token not in json.dumps(report)
        assert token in reference["function"]["description"]
        assert case.tool_choice["allowed_tools"]["tools"] == [
            {"type": "function", "function": {"name": "report_catalog"}}
        ]
        assert case.tool_choice["allowed_tools"]["mode"] == mode
        assert case.api == api

    text = cases[f"regression_allowed_tools_auto_text_{api}"]
    assert text.tool_choice["allowed_tools"]["mode"] == "auto"
    assert text.min_tool_calls == 0
    assert text.expected_finish_reasons == ("stop",)

    invalid = cases[f"regression_allowed_tools_unknown_{api}"]
    selected = invalid.tool_choice["allowed_tools"]["tools"][0]["function"]["name"]
    assert selected not in {tool["function"]["name"] for tool in invalid.tools}
    assert invalid.expected_http_status == 400


def test_regex_controls_preserve_declared_schema_and_only_relax_effective_constraints():
    cases = _cases("generic")
    auto = cases["regression_auto_regex_fallback"]
    relaxed = cases["regression_nonstrict_regex_control"]
    declared = auto.tools[0]["function"]["parameters"]

    assert "strict" not in auto.tools[0]["function"]
    assert relaxed.tools[0]["function"]["strict"] is False
    assert declared == relaxed.tools[0]["function"]["parameters"]
    assert declared["properties"]["value"]["pattern"] == r"^(?!\s*$).+"

    original_validator = Draft202012Validator(declared)
    effective_validator = Draft202012Validator(auto.enforced_schemas["record_value"])
    assert not original_validator.is_valid({"value": "", "count": 7})
    assert effective_validator.is_valid({"value": "", "count": 7})
    assert not effective_validator.is_valid({"value": "hello", "count": "7"})
    assert not effective_validator.is_valid(
        {"value": "hello", "count": 7, "extra": True}
    )
    assert relaxed.enforced_schemas["record_value"] is True


def test_escaped_reference_fixture_requires_city_under_standard_pointer_resolution():
    cases = _cases("qwen3_coder_xml")
    case = cases["regression_qwen_escaped_ref_requires_arguments"]
    schema = case.tools[0]["function"]["parameters"]
    validator = Draft202012Validator(schema)

    assert validator.is_valid({"city": "Paris"})
    assert not validator.is_valid({})
    assert case.expected_tool_calls == (
        {"name": "lookup_city", "arguments": {"city": "Paris"}},
    )
    alternate = dict(schema, **{"$ref": "#/$defs/a~0b"})
    assert Draft202012Validator(alternate).is_valid({})

    empty = cases["regression_qwen_auto_nonstrict_empty_arguments"]
    assert empty.tool_choice == "auto"
    assert empty.tools[0]["function"]["strict"] is False
    assert empty.expected_tool_calls == ({"name": "get_today", "arguments": {}},)


def test_raw_string_fixtures_distinguish_decoded_values_from_atom_and_wire_lengths():
    cases = _cases("muse_glimmer")
    for case in cases.values():
        if not case.case_id.startswith("regression_muse_"):
            continue
        call = case.expected_tool_calls[0]
        validator = Draft202012Validator(case.tools[0]["function"]["parameters"])
        assert validator.is_valid(call["arguments"])
        assert case.parallel_tool_calls is False
        assert case.exact_tool_calls == 1

    bounded = cases["regression_muse_bounded_string_two_characters"]
    assert bounded.expected_tool_calls[0]["arguments"]["value"] == "<a"

    quoted = cases["regression_muse_quoted_string_decoded_constraints"]
    value = quoted.expected_tool_calls[0]["arguments"]["value"]
    assert len(value) == 3 and len(value.encode("utf-8")) == 4
    validator = Draft202012Validator(quoted.tools[0]["function"]["parameters"])
    assert not validator.is_valid({"value": json.loads(value)})

    markers = cases["regression_muse_marker_string_stays_one_argument"]
    validator = Draft202012Validator(markers.tools[0]["function"]["parameters"])
    assert not validator.is_valid({"value": "a", "extra": "b"})
    assert (
        '<atem:invoke name="record_literal">'
        in markers.expected_tool_calls[0]["arguments"]["value"]
    )


def test_marker_roundtrip_needs_explicit_support_and_never_silently_passes(monkeypatch):
    cases = {
        case.case_id: case
        for case in build_regression_cases(probe.Case, "muse_glimmer")
    }
    target = cases["regression_muse_marker_string_stays_one_argument"]
    assert target.required_capabilities == ("muse_delimiter_string_roundtrip",)
    assert (
        cases["regression_muse_bounded_string_two_characters"].required_capabilities
        == ()
    )
    assert (
        cases["regression_muse_quoted_string_decoded_constraints"].required_capabilities
        == ()
    )

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("an unqualified delimiter-string contract must not dispatch")

    monkeypatch.setattr(probe, "post_json", unexpected_dispatch)
    record = probe.run_case(
        target,
        "nonstream",
        iteration=1,
        url="unused",
        api_key=None,
        model="test-model",
        temperature=0,
        max_tokens=64,
        timeout=1,
        extra_headers={},
        raw_chars=2000,
        record_success_raw=True,
    )

    assert record["pass"] is False
    assert record["failure_category"] == "unclassified"
    assert record["failure_kinds"] == ["capability_unavailable"]
    assert record["errors"][0]["details"]["missing"] == [
        "muse_delimiter_string_roundtrip"
    ]


@pytest.mark.parametrize("content_media_type", [None, "text/plain"])
def test_quoted_argument_fixture_preserves_plain_and_freeform_values(
    content_media_type,
):
    original = next(
        case
        for case in build_regression_cases(probe.Case, "muse_glimmer")
        if case.case_id == "regression_muse_quoted_string_decoded_constraints"
    )
    tools = copy.deepcopy(original.tools)
    if content_media_type:
        tools[0]["function"]["parameters"]["properties"]["value"][
            "contentMediaType"
        ] = content_media_type
    target = dataclasses.replace(original, tools=tools)
    literal = original.expected_tool_calls[0]["arguments"]["value"]

    def result(value):
        return probe.ChatResult(
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_literal",
                    "type": "function",
                    "function": {
                        "name": "record_literal",
                        "arguments": json.dumps({"value": value}, ensure_ascii=False),
                    },
                }
            ],
        )

    errors, _ = probe.validate_result(target, result(literal))
    assert errors == []
    errors, _ = probe.validate_result(target, result(json.loads(literal)))
    assert "enforced_schema_violation" in {finding["kind"] for finding in errors}


def test_media_histories_keep_call_identity_and_real_image_pixels_without_text_answers():
    cases = _cases("kimi_k3")
    colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
    for suffix, expected_order in (
        ("in_order", ["call_reference_a", "call_reference_b"]),
        ("reverse_order", ["call_reference_b", "call_reference_a"]),
        ("multiple_per_result", ["call_reference_b", "call_reference_a"]),
    ):
        case = cases[f"regression_tool_images_{suffix}"]
        assert case.required_capabilities == ("vision",)
        assert case.tool_choice == "none"
        assistant = case.messages[1]
        assert [call["id"] for call in assistant["tool_calls"]] == [
            "call_reference_a",
            "call_reference_b",
        ]
        results = case.messages[2:]
        assert [result["tool_call_id"] for result in results] == expected_order
        for result in results:
            reference = "A" if result["tool_call_id"] == "call_reference_a" else "B"
            parts = result["content"]
            assert parts[0] == {"type": "text", "text": f"Reference {reference}:"}
            urls = [part["image_url"]["url"] for part in parts[1:]]
            assert len(urls) == len(case.expected_json[reference])
            for url, color in zip(urls, case.expected_json[reference]):
                width, height, pixels = _png_pixels(url)
                assert (width, height) == (96, 96)
                assert pixels == (b"\x00" + bytes(colors[color]) * width) * height
