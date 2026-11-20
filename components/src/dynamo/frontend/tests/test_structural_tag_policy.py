# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from dynamo.frontend.structural_tag_policy import (
    effective_tool_strict,
    guidance_tool_schema,
    runtime_backend_defers_grammar,
    runtime_structural_tag_excludes_reasoning,
    runtime_structural_tag_options,
    should_attempt_structural_tag,
    unsupported_regex_construct,
    unsupported_schema_construct,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

requires_xgrammar = pytest.mark.skipif(
    importlib.util.find_spec("xgrammar") is None,
    reason="the regex check asks XGrammar, which vLLM and SGLang install",
)


@pytest.mark.parametrize(
    ("advertised", "boundary", "expected"),
    [
        (None, "backend", False),
        (False, "backend", False),
        (True, "auto", True),
        (True, "structural_tag", True),
        ("true", "auto", False),
        (1, "auto", False),
    ],
)
def test_backend_grammar_capability_is_not_a_frontend_policy(
    advertised, boundary, expected
):
    """A boundary override must not claim a scheduling capability the worker lacks."""
    assert (
        runtime_backend_defers_grammar(
            {
                "runtime_data": {
                    "tool_call_structural_tag_excludes_reasoning": advertised
                },
                "structural_tag": {"reasoning_boundary": boundary},
            }
        )
        is expected
    )


@pytest.mark.parametrize(
    ("mode", "scope", "choice", "strict", "single", "expected"),
    [
        ("off", "always", "auto", False, False, False),
        ("on", "always", "auto", False, False, True),
        ("on", "auto", "auto", True, False, True),
        ("on", "auto", "auto", False, True, True),
        ("on", "auto", "auto", False, False, False),
        ("on", "auto", "required", False, False, True),
        ("on", "auto", "named", False, False, True),
        ("on", "always", "none", True, True, False),
    ],
)
def test_structural_tag_activation_policy(
    mode, scope, choice, strict, single, expected
):
    assert (
        should_attempt_structural_tag(
            mode=mode,
            scope=scope,
            tool_choice_kind=choice,
            has_tools=True,
            any_explicit_strict=strict,
            parallel_tool_calls_explicitly_false=single,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("request_strict", "schema_mode", "expected"),
    [
        (None, "auto", True),
        (True, "auto", True),
        (False, "auto", False),
        (False, "strict", True),
    ],
)
def test_effective_tool_strict(request_strict, schema_mode, expected):
    assert effective_tool_strict(request_strict, schema_mode) is expected


# The verdicts are shared with the Rust preprocessor test, which checks its
# hand-written list against them. This test checks them against XGrammar itself,
# so that list cannot drift from the compiler unnoticed.
# components/src/dynamo/frontend/tests -> the repository root.
_REGEX_CONSTRUCT_CASES = [
    tuple(case)
    for case in json.loads(
        (
            Path(__file__).resolve().parents[5]
            / "lib/llm/tests/data/structural_tag/unsupported_regex_constructs.json"
        ).read_text(encoding="utf-8")
    )["cases"]
]


@requires_xgrammar
@pytest.mark.parametrize(("pattern", "expected"), _REGEX_CONSTRUCT_CASES)
def test_unsupported_regex_constructs_match_the_grammar_compiler(pattern, expected):
    assert (unsupported_regex_construct(pattern) is None) == (expected is None)


_LOOKAHEAD = r"^(?!\s*$).+"


@requires_xgrammar
@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"properties": {"a": {"items": {"pattern": _LOOKAHEAD}}}}, "lookahead"),
        ({"anyOf": [{"type": "integer"}, {"pattern": _LOOKAHEAD}]}, "lookahead"),
        ({"patternProperties": {_LOOKAHEAD: {"type": "string"}}}, "lookahead"),
        ({"$defs": {"name": {"pattern": _LOOKAHEAD}}}, "lookahead"),
        ({"properties": {"default": {"pattern": _LOOKAHEAD}}}, "lookahead"),
        ({"dependencies": {"enum": {"pattern": _LOOKAHEAD}}}, "lookahead"),
        ({"properties": {"a": {"type": "string", "pattern": "^[a-z]+$"}}}, None),
        ({"properties": {"pattern": {"type": "string"}}}, None),
        ({"const": {"pattern": _LOOKAHEAD}}, None),
        ({"default": {"pattern": _LOOKAHEAD}}, None),
        ({"enum": [{"pattern": _LOOKAHEAD}]}, None),
        ({"examples": [{"pattern": _LOOKAHEAD}]}, None),
        (None, None),
    ],
)
def test_unsupported_schema_constructs_are_found_in_subschemas_only(schema, expected):
    assert (unsupported_schema_construct(schema) is None) == (expected is None)


def _value_schema(pattern):
    return {
        "type": "object",
        "properties": {"value": {"type": "string", "pattern": pattern}},
        "required": ["value"],
    }


_UNSUPPORTED = _value_schema(_LOOKAHEAD)
_SUPPORTED = _value_schema("^[a-z]+$")
_STRIPPED = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
}
_UNSUPPORTED_NAMES = {
    "type": "object",
    "patternProperties": {_LOOKAHEAD: {"type": "string"}},
    "additionalProperties": False,
}


@requires_xgrammar
@pytest.mark.parametrize(
    ("request_strict", "schema_mode", "parameters", "expected"),
    [
        (None, "auto", _UNSUPPORTED, (_STRIPPED, True)),
        (True, "auto", _UNSUPPORTED, (_UNSUPPORTED, True)),
        (False, "auto", _UNSUPPORTED, (_UNSUPPORTED, False)),
        (None, "strict", _UNSUPPORTED, (_UNSUPPORTED, True)),
        (False, "strict", _UNSUPPORTED, (_UNSUPPORTED, True)),
        (None, "auto", _SUPPORTED, (_SUPPORTED, True)),
        (None, "auto", _UNSUPPORTED_NAMES, (_UNSUPPORTED_NAMES, False)),
        (None, "auto", None, (None, True)),
    ],
)
def test_guidance_tool_schema_adapts_only_automatic_enforcement(
    request_strict, schema_mode, parameters, expected
):
    client_schema = copy.deepcopy(parameters)
    assert (
        guidance_tool_schema(
            request_strict, schema_mode, parameters, tool_name="record_value"
        )
        == expected
    )
    assert parameters == client_schema


def test_runtime_structural_tag_options_are_conservative_when_absent():
    assert runtime_structural_tag_options(None) == ("off", "auto", "auto")
    assert runtime_structural_tag_options({}) == ("off", "auto", "auto")


def test_runtime_structural_tag_options_read_model_card_policy():
    assert runtime_structural_tag_options(
        {
            "structural_tag_mode": "on",
            "structural_tag_scope": "always",
            "structural_tag_schema": "strict",
        }
    ) == ("on", "always", "strict")


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {"structural_tag": {"scope": "always", "schema": "strict"}},
            ("on", "always", "strict"),
        ),
        (
            {"structural_tag": None, "structural_tag_mode": "on"},
            ("off", "auto", "auto"),
        ),
        (
            {"structural_tag": {"scope": "always"}, "structural_tag_mode": "off"},
            ("on", "always", "auto"),
        ),
    ],
)
def test_canonical_structural_tag_policy_takes_precedence(config, expected):
    assert runtime_structural_tag_options(config) == expected


@pytest.mark.parametrize(
    ("runtime_config", "expected"),
    [
        (None, False),
        ({}, False),
        ({"runtime_data": {}}, False),
        ({"runtime_data": {"tool_call_structural_tag_excludes_reasoning": True}}, True),
        (
            {"runtime_data": {"tool_call_structural_tag_excludes_reasoning": False}},
            False,
        ),
        (
            {"runtime_data": {"tool_call_structural_tag_excludes_reasoning": "true"}},
            False,
        ),
    ],
)
def test_runtime_structural_tag_excludes_reasoning(runtime_config, expected):
    assert runtime_structural_tag_excludes_reasoning(runtime_config) is expected


_EXCLUDES = {"runtime_data": {"tool_call_structural_tag_excludes_reasoning": True}}


@pytest.mark.parametrize(
    ("boundary", "backend_excludes", "expected"),
    [
        ("auto", False, False),
        ("auto", True, True),
        ("backend", False, True),
        ("backend", True, True),
        ("structural_tag", False, False),
        # Conflicting pairing: Rust rejects it; follow the backend.
        ("structural_tag", True, True),
    ],
)
def test_reasoning_boundary_resolution(boundary, backend_excludes, expected):
    config = {"structural_tag": {"reasoning_boundary": boundary}}
    if backend_excludes:
        config.update(_EXCLUDES)
    assert runtime_structural_tag_excludes_reasoning(config) is expected
