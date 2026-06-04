#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenAI ``response_format`` guided-decoding mapping.

Covers the patch that wires ``response_format`` (json_object / json_schema)
into the existing guided-decoding mechanism (the same path tool_choice
required/named already uses).  Tests are GPU/tokenizer-free: they exercise
only the pure functions

  * ``build_response_format_guided_decoding`` -- response_format -> guided dict
  * ``create_parsers``                        -- parser-gating precedence
  * ``_build_dynamo_preproc``                 -- guided_decoding passthrough

Parallels test_sglang_processor_unit.py (same imports/markers/class style).
The end-to-end precedence (tool_choice required > response_format > auto
structural tag) lives in preprocess_chat_request, which needs a real
tokenizer, and is covered by the e2e suite instead.
"""


import json

import pytest
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.parser.reasoning_parser import ReasoningParser

from dynamo.frontend.sglang_prepost import (
    _response_format_constraint_requested,
    build_response_format_guided_decoding,
    create_parsers,
)
from dynamo.frontend.sglang_processor import _build_dynamo_preproc
from dynamo.frontend.utils import PreprocessError

# Same gate as test_sglang_processor_unit.py: needs the sglang packages
# (parsers) but no GPU kernels or model weights -- gpu_1 container is fine.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


CITY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "country": {"type": "string"},
        "population": {"type": "number"},
        "landmark": {"type": "string"},
    },
    "required": ["name", "country", "population", "landmark"],
    "additionalProperties": False,
}


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def _schema_rf(schema, name="city_info"):
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


# ---------------------------------------------------------------------------
# build_response_format_guided_decoding
# ---------------------------------------------------------------------------


class TestBuildResponseFormatGuidedDecoding:  # FRONTEND.3 — response_format → guided decoding
    """response_format -> {'json': <schema dict>} mapping (and rejections)."""

    def test_json_object_maps_to_generic_object_grammar(self):
        guided = build_response_format_guided_decoding(
            {"response_format": {"type": "json_object"}}
        )
        assert guided == {"json": {"type": "object"}}

    def test_json_schema_dict_maps_to_schema(self):
        """json_schema -> {'json': <schema>}; schema stays a dict (no double-encode)."""
        guided = build_response_format_guided_decoding(
            {"response_format": _schema_rf(CITY_SCHEMA)}
        )
        assert guided == {"json": CITY_SCHEMA}
        # Must remain a dict so the backend's json.dumps(value) does not
        # double-encode a pre-serialized string.
        assert isinstance(guided["json"], dict)

    def test_json_schema_as_string_is_parsed(self):
        """SDKs that serialize the schema to a JSON string are tolerated."""
        guided = build_response_format_guided_decoding(
            {"response_format": _schema_rf(json.dumps(CITY_SCHEMA))}
        )
        assert guided == {"json": CITY_SCHEMA}
        assert isinstance(guided["json"], dict)

    def test_text_type_returns_none(self):
        assert (
            build_response_format_guided_decoding({"response_format": {"type": "text"}})
            is None
        )

    def test_type_none_returns_none(self):
        """response_format dict with explicit type=None is a no-op."""
        assert (
            build_response_format_guided_decoding({"response_format": {"type": None}})
            is None
        )

    def test_response_format_absent_returns_none(self):
        assert build_response_format_guided_decoding({}) is None

    def test_response_format_explicit_none_returns_none(self):
        assert build_response_format_guided_decoding({"response_format": None}) is None

    def test_non_dict_response_format_raises(self):
        with pytest.raises(PreprocessError, match="must be an object"):
            build_response_format_guided_decoding({"response_format": "json_object"})

    def test_unsupported_type_raises(self):
        with pytest.raises(PreprocessError, match="Unsupported response_format type"):
            build_response_format_guided_decoding(
                {"response_format": {"type": "banana"}}
            )

    def test_json_schema_top_level_schema_shorthand(self):
        """Native-SGLang shorthand: schema at the response_format top level."""
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        result = build_response_format_guided_decoding(
            {"response_format": {"type": "json_schema", "schema": schema}}
        )
        assert result == {"json": schema}

    def test_json_schema_missing_json_schema_field_raises(self):
        with pytest.raises(PreprocessError, match="json_schema"):
            build_response_format_guided_decoding(
                {"response_format": {"type": "json_schema"}}
            )

    def test_json_schema_field_not_a_dict_raises(self):
        with pytest.raises(PreprocessError, match="json_schema"):
            build_response_format_guided_decoding(
                {"response_format": {"type": "json_schema", "json_schema": "nope"}}
            )

    def test_json_schema_missing_schema_raises(self):
        """json_schema present but without a 'schema' key is rejected."""
        with pytest.raises(PreprocessError, match="schema"):
            build_response_format_guided_decoding(
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "x", "strict": True},
                    }
                }
            )

    def test_json_schema_schema_not_dict_raises(self):
        with pytest.raises(PreprocessError, match="schema"):
            build_response_format_guided_decoding({"response_format": _schema_rf(123)})

    def test_json_schema_schema_invalid_json_string_raises(self):
        with pytest.raises(PreprocessError, match="not valid JSON"):
            build_response_format_guided_decoding(
                {"response_format": _schema_rf("{not valid json")}
            )


# ---------------------------------------------------------------------------
# _response_format_constraint_requested helper
# ---------------------------------------------------------------------------


class TestResponseFormatConstraintRequested:  # FRONTEND.3 — response_format detection
    def test_json_object_true(self):
        assert _response_format_constraint_requested(
            {"response_format": {"type": "json_object"}}
        )

    def test_json_schema_true(self):
        assert _response_format_constraint_requested(
            {"response_format": {"type": "json_schema", "json_schema": {}}}
        )

    def test_text_false(self):
        assert not _response_format_constraint_requested(
            {"response_format": {"type": "text"}}
        )

    def test_absent_false(self):
        assert not _response_format_constraint_requested({})

    def test_non_dict_false(self):
        """A non-dict response_format is not a *requested* constraint here.

        (Validation/rejection happens later in
        build_response_format_guided_decoding.)
        """
        assert not _response_format_constraint_requested(
            {"response_format": "json_object"}
        )


# ---------------------------------------------------------------------------
# create_parsers: response_format parser gating
# ---------------------------------------------------------------------------


class TestCreateParsersResponseFormat:  # FRONTEND.2 — response_format parser gating
    """response_format suppresses BOTH tool-call and reasoning parsers when
    it is the active grammar; tool_choice required/named still take precedence.
    """

    def test_json_schema_with_tools_auto_suppresses_both_parsers(self):
        """response_format active (auto tool_choice) -> (None, None)."""
        tcp, rp = create_parsers(
            {
                "tools": _tools(),
                "tool_choice": "auto",
                "response_format": _schema_rf(CITY_SCHEMA),
            },
            tool_call_parser_name="qwen25",
            reasoning_parser_name="qwen3",
        )
        assert tcp is None
        assert rp is None

    def test_json_object_with_tools_auto_suppresses_both_parsers(self):
        tcp, rp = create_parsers(
            {
                "tools": _tools(),
                "tool_choice": "auto",
                "response_format": {"type": "json_object"},
            },
            tool_call_parser_name="qwen25",
            reasoning_parser_name="qwen3",
        )
        assert tcp is None
        assert rp is None

    def test_json_schema_without_tools_suppresses_reasoning(self):
        """response_format with no tools at all still suppresses reasoning."""
        tcp, rp = create_parsers(
            {
                "response_format": _schema_rf(CITY_SCHEMA),
            },
            tool_call_parser_name="qwen25",
            reasoning_parser_name="qwen3",
        )
        assert tcp is None
        assert rp is None

    def test_response_format_plus_tool_choice_required_tool_wins(self):
        """tool_choice=required outranks response_format -> JsonArrayParser, no reasoning."""
        tcp, rp = create_parsers(
            {
                "tools": _tools(),
                "tool_choice": "required",
                "response_format": _schema_rf(CITY_SCHEMA),
            },
            tool_call_parser_name="qwen25",
            reasoning_parser_name="qwen3",
        )
        assert isinstance(tcp, JsonArrayParser)
        assert rp is None

    def test_response_format_plus_tool_choice_named_tool_wins(self):
        """Named tool_choice outranks response_format -> JsonArrayParser, no reasoning."""
        tcp, rp = create_parsers(
            {
                "tools": _tools(),
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
                "response_format": _schema_rf(CITY_SCHEMA),
            },
            tool_call_parser_name="qwen25",
            reasoning_parser_name="qwen3",
        )
        assert isinstance(tcp, JsonArrayParser)
        assert rp is None

    def test_response_format_text_leaves_behavior_unchanged_with_tools(self):
        """response_format=text is a no-op: tools auto -> FunctionCallParser + reasoning."""
        tcp, rp = create_parsers(
            {
                "tools": _tools(),
                "tool_choice": "auto",
                "response_format": {"type": "text"},
            },
            tool_call_parser_name="qwen25",
            reasoning_parser_name="qwen3",
        )
        assert isinstance(tcp, FunctionCallParser)
        assert isinstance(rp, ReasoningParser)

    def test_response_format_text_reasoning_only(self):
        """response_format=text with no tools: reasoning parser still created."""
        tcp, rp = create_parsers(
            {"response_format": {"type": "text"}},
            tool_call_parser_name="qwen25",
            reasoning_parser_name="qwen3",
        )
        assert tcp is None
        assert isinstance(rp, ReasoningParser)

    def test_no_response_format_unchanged_with_tools(self):
        """Absent response_format: tools auto -> FunctionCallParser + reasoning."""
        tcp, rp = create_parsers(
            {"tools": _tools(), "tool_choice": "auto"},
            tool_call_parser_name="qwen25",
            reasoning_parser_name="qwen3",
        )
        assert isinstance(tcp, FunctionCallParser)
        assert isinstance(rp, ReasoningParser)

    def test_no_response_format_no_parser_names_unchanged(self):
        """Absent response_format and no parser names: (None, None)."""
        tcp, rp = create_parsers(
            {"tools": _tools(), "tool_choice": "auto"},
            tool_call_parser_name=None,
            reasoning_parser_name=None,
        )
        assert tcp is None
        assert rp is None

    def test_response_format_active_without_parser_names_returns_none(self):
        """response_format active but no parser names configured -> (None, None)."""
        tcp, rp = create_parsers(
            {
                "tools": _tools(),
                "tool_choice": "auto",
                "response_format": _schema_rf(CITY_SCHEMA),
            },
            tool_call_parser_name=None,
            reasoning_parser_name=None,
        )
        assert tcp is None
        assert rp is None


# ---------------------------------------------------------------------------
# _build_dynamo_preproc: response_format guided_decoding passthrough
# ---------------------------------------------------------------------------


class TestBuildDynamoPreprocResponseFormat:  # FRONTEND.7 — guided_decoding passthrough
    """The mapped response_format guided dict survives into sampling_options."""

    def test_json_object_guided_decoding_passthrough(self):
        guided = build_response_format_guided_decoding(
            {"response_format": {"type": "json_object"}}
        )
        result = _build_dynamo_preproc(
            {"model": "test"},
            prompt_token_ids=[1, 2, 3],
            model_name="test",
            eos_token_id=None,
            guided_decoding=guided,
        )
        assert result["sampling_options"]["guided_decoding"] == {
            "json": {"type": "object"}
        }

    def test_json_schema_guided_decoding_passthrough(self):
        guided = build_response_format_guided_decoding(
            {"response_format": _schema_rf(CITY_SCHEMA)}
        )
        result = _build_dynamo_preproc(
            {"model": "test"},
            prompt_token_ids=[1],
            model_name="test",
            eos_token_id=None,
            guided_decoding=guided,
        )
        assert result["sampling_options"]["guided_decoding"] == {"json": CITY_SCHEMA}

    def test_no_guided_decoding_is_none(self):
        """No response_format -> guided_decoding stays None in sampling_options."""
        result = _build_dynamo_preproc(
            {"model": "test"},
            prompt_token_ids=[1],
            model_name="test",
            eos_token_id=None,
            guided_decoding=None,
        )
        assert result["sampling_options"]["guided_decoding"] is None

    def test_response_format_grammar_keeps_skip_special_tokens_true(self):
        """No tool_call_parser under response_format -> skip_special_tokens=True.

        Design decision 3: the response_format grammar suppresses the
        tool-call parser, so special tokens must NOT leak into the
        constrained JSON content.
        """
        result = _build_dynamo_preproc(
            {"model": "test"},
            prompt_token_ids=[1],
            model_name="test",
            eos_token_id=None,
            guided_decoding={"json": {"type": "object"}},
            tool_call_parser=None,
        )
        assert result["output_options"]["skip_special_tokens"] is True
