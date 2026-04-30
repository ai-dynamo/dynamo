#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang processor components.

Tests for preprocessing, sampling parameter projection, finish reason mapping,
incremental detokenization, error handling, and deprecation warnings.

Parallels test_vllm_unit.py for the vLLM backend.
"""


import json
import sys
import types

import pytest
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

import dynamo.frontend.sglang_prepost as sglang_prepost_module
import dynamo.frontend.sglang_processor as sglang_processor_module
from dynamo.frontend.sglang_prepost import (
    SglangPreprocessResult,
    SglangStreamingPostProcessor,
    _normalize_assistant_tool_call_arguments,
    _normalize_prompt_token_ids,
    _parse_json_array_buffer,
    build_tool_call_guided_decoding,
    convert_tools,
    create_parsers,
    preprocess_chat_request,
)
from dynamo.frontend.sglang_processor import (
    SglangPreprocessWorkerResult,
    _build_dynamo_preproc,
    _init_worker,
    _map_finish_reason,
    _runtime_config_parser_name,
)
from dynamo.frontend.utils import PreprocessError, random_call_id, random_uuid

# Needs sglang packages (gpu_1 container).  No need for parallel marker.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL)


# ---------------------------------------------------------------------------
# _build_dynamo_preproc: sampling parameter projection
# ---------------------------------------------------------------------------


class TestBuildDynamoPreproc:  # FE.7 — worker subprocess preproc construction
    """Test sampling parameter projection from request to Dynamo format."""

    def test_defaults(self):
        """Default sampling options when request has minimal fields."""
        result = _build_dynamo_preproc(
            {"model": "test", "messages": []},
            prompt_token_ids=[1, 2, 3],
            model_name="test",
            eos_token_id=2,
        )
        sampling = result["sampling_options"]
        assert sampling["n"] == 1
        assert sampling["temperature"] == 1.0
        assert sampling["top_p"] == 1.0
        assert sampling["top_k"] == -1  # 0 -> -1 for SGLang
        assert sampling["min_p"] == 0.0
        assert sampling["presence_penalty"] == 0.0
        assert sampling["frequency_penalty"] == 0.0
        assert sampling["repetition_penalty"] == 1.0
        assert sampling["seed"] is None

    def test_top_k_zero_maps_to_negative_one(self):
        """SGLang uses -1 for disabled top_k, OpenAI uses 0."""
        result = _build_dynamo_preproc(
            {"model": "test", "top_k": 0},
            prompt_token_ids=[1],
            model_name="test",
            eos_token_id=None,
        )
        assert result["sampling_options"]["top_k"] == -1

    def test_top_k_positive_preserved(self):
        """Positive top_k values pass through unchanged."""
        result = _build_dynamo_preproc(
            {"model": "test", "top_k": 50},
            prompt_token_ids=[1],
            model_name="test",
            eos_token_id=None,
        )
        assert result["sampling_options"]["top_k"] == 50

    def test_sampling_options_from_request(self):
        """All sampling fields are projected from request."""
        request = {
            "model": "test",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "repetition_penalty": 1.1,
            "seed": 42,
            "n": 1,
        }
        result = _build_dynamo_preproc(request, [1], "test", None)
        sampling = result["sampling_options"]
        assert sampling["temperature"] == 0.7
        assert sampling["top_p"] == 0.9
        assert sampling["top_k"] == 40
        assert sampling["min_p"] == 0.05
        assert sampling["presence_penalty"] == 0.1
        assert sampling["frequency_penalty"] == 0.2
        assert sampling["repetition_penalty"] == 1.1
        assert sampling["seed"] == 42

    def test_guided_decoding_passthrough(self):
        result = _build_dynamo_preproc(
            {"model": "test"},
            prompt_token_ids=[1, 2, 3],
            model_name="test",
            eos_token_id=None,
            guided_decoding={"json": {"type": "object"}},
        )
        assert result["sampling_options"]["guided_decoding"] == {
            "json": {"type": "object"}
        }

    def test_stop_conditions_string(self):
        """Single stop string is wrapped in a list."""
        result = _build_dynamo_preproc(
            {"model": "test", "stop": "END"},
            [1],
            "test",
            None,
        )
        assert result["stop_conditions"]["stop"] == ["END"]

    def test_stop_conditions_list(self):
        """Stop list passes through."""
        result = _build_dynamo_preproc(
            {"model": "test", "stop": ["END", "STOP"]},
            [1],
            "test",
            None,
        )
        assert result["stop_conditions"]["stop"] == ["END", "STOP"]

    def test_stop_conditions_none(self):
        """None stop becomes empty list."""
        result = _build_dynamo_preproc(
            {"model": "test"},
            [1],
            "test",
            None,
        )
        assert result["stop_conditions"]["stop"] == []

    def test_max_tokens_from_max_completion_tokens(self):
        """max_completion_tokens takes precedence over max_tokens."""
        result = _build_dynamo_preproc(
            {"model": "test", "max_completion_tokens": 200, "max_tokens": 100},
            [1],
            "test",
            None,
        )
        assert result["stop_conditions"]["max_tokens"] == 200

    def test_max_tokens_fallback(self):
        """max_tokens used when max_completion_tokens not set."""
        result = _build_dynamo_preproc(
            {"model": "test", "max_tokens": 100},
            [1],
            "test",
            None,
        )
        assert result["stop_conditions"]["max_tokens"] == 100

    def test_eos_token_id_present(self):
        """eos_token_id is wrapped in a list."""
        result = _build_dynamo_preproc({"model": "test"}, [1], "test", 151643)
        assert result["eos_token_ids"] == [151643]

    def test_eos_token_id_none(self):
        """None eos_token_id becomes empty list."""
        result = _build_dynamo_preproc({"model": "test"}, [1], "test", None)
        assert result["eos_token_ids"] == []

    def test_logprobs_true_with_top_logprobs(self):
        """logprobs=True with top_logprobs=5 yields 5."""
        result = _build_dynamo_preproc(
            {"model": "test", "logprobs": True, "top_logprobs": 5},
            [1],
            "test",
            None,
        )
        assert result["output_options"]["logprobs"] == 5

    def test_logprobs_true_without_top_logprobs(self):
        """logprobs=True without top_logprobs yields 1."""
        result = _build_dynamo_preproc(
            {"model": "test", "logprobs": True},
            [1],
            "test",
            None,
        )
        assert result["output_options"]["logprobs"] == 1

    def test_logprobs_integer(self):
        """Integer logprobs pass through."""
        result = _build_dynamo_preproc(
            {"model": "test", "logprobs": 3},
            [1],
            "test",
            None,
        )
        assert result["output_options"]["logprobs"] == 3

    def test_logprobs_disabled(self):
        """No logprobs yields None."""
        result = _build_dynamo_preproc(
            {"model": "test"},
            [1],
            "test",
            None,
        )
        assert result["output_options"]["logprobs"] is None

    def test_model_name_and_token_ids(self):
        """Model name and token_ids are set correctly."""
        result = _build_dynamo_preproc(
            {"model": "test"},
            [10, 20, 30],
            "my-model",
            None,
        )
        assert result["model"] == "my-model"
        assert result["token_ids"] == [10, 20, 30]


# ---------------------------------------------------------------------------
# _map_finish_reason
# ---------------------------------------------------------------------------


class TestMapFinishReason:  # FE.5 — finish_reason remap (frontend layer)
    """Test Dynamo-to-OpenAI finish reason mapping."""

    def test_none_passthrough(self):
        assert _map_finish_reason(None) is None

    def test_eos_to_stop(self):
        assert _map_finish_reason("eos") == "stop"

    def test_stop_to_stop(self):
        assert _map_finish_reason("stop") == "stop"

    def test_length(self):
        assert _map_finish_reason("length") == "length"

    def test_error(self):
        assert _map_finish_reason("error") == "error"

    def test_error_prefix(self):
        """error:* strings all map to 'error'."""
        assert _map_finish_reason("error:timeout") == "error"

    def test_abort_exact(self):
        assert _map_finish_reason("abort") == "stop"

    def test_abort_prefix(self):
        """abort:* strings all map to 'stop'."""
        assert _map_finish_reason("abort:cancelled") == "stop"

    def test_cancelled(self):
        assert _map_finish_reason("cancelled") == "stop"

    def test_content_filter(self):
        assert _map_finish_reason("content_filter") == "stop"

    def test_unknown_passthrough(self):
        """Unknown reasons pass through unchanged."""
        assert _map_finish_reason("tool_calls") == "tool_calls"


# ---------------------------------------------------------------------------
# convert_tools
# ---------------------------------------------------------------------------


class TestConvertTools:  # FE.3 — OpenAI tool schema → SGLang Tool/Function
    """Test OpenAI tool dict to SGLang Tool conversion."""

    def test_none_returns_none(self):
        assert convert_tools(None) is None

    def test_empty_list_returns_none(self):
        assert convert_tools([]) is None

    def test_single_tool(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        result = convert_tools(tools)
        assert len(result) == 1
        assert result[0].function.name == "get_weather"
        assert result[0].type == "function"

    def test_multiple_tools(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "f1", "description": "d1", "parameters": {}},
            },
            {
                "type": "function",
                "function": {"name": "f2", "description": "d2", "parameters": {}},
            },
        ]
        result = convert_tools(tools)
        assert len(result) == 2
        assert result[0].function.name == "f1"
        assert result[1].function.name == "f2"

    def test_model_dump_roundtrip(self):
        """Converted tools can be model_dump()'d for chat templates."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ]
        result = convert_tools(tools)
        dumped = result[0].model_dump()
        assert dumped["function"]["name"] == "search"
        assert "properties" in dumped["function"]["parameters"]


# ---------------------------------------------------------------------------
# create_parsers
# ---------------------------------------------------------------------------


class TestCreateParsers:  # FE.2 — tool/reasoning parser dispatch
    """Test parser creation logic."""

    def test_no_parsers(self):
        tcp, rp = create_parsers(
            {}, tool_call_parser_name=None, reasoning_parser_name=None
        )
        assert tcp is None
        assert rp is None

    def test_reasoning_only(self):
        tcp, rp = create_parsers(
            {}, tool_call_parser_name=None, reasoning_parser_name="qwen3"
        )
        assert tcp is None
        assert rp is not None


class TestBuildToolCallGuidedDecoding:  # FE.3 — guided-decoding setup for tool_choice
    def test_none_when_no_tools(self):
        assert (
            build_tool_call_guided_decoding(
                {"tool_choice": "auto"},
                tool_call_parser_name="hermes",
                sglang_tools=None,
            )
            is None
        )

    def test_none_when_tool_choice_none(self):
        tools = convert_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )
        assert (
            build_tool_call_guided_decoding(
                {"tool_choice": "none"},
                tool_call_parser_name="hermes",
                sglang_tools=tools,
            )
            is None
        )

    def test_required_tool_choice_builds_json_schema_guidance(self):
        tools = convert_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
        )

        guided = build_tool_call_guided_decoding(
            {"tool_choice": "required"},
            tool_call_parser_name="hermes",
            sglang_tools=tools,
        )

        assert isinstance(guided, dict)
        assert "json" in guided

    def test_required_tool_choice_supports_older_sglang_constraint_signature(
        self, monkeypatch
    ):
        tools = convert_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ]
        )

        def old_get_json_schema_constraint(sglang_tools, tool_choice):
            assert sglang_tools == tools
            assert tool_choice == "required"
            return {"type": "array", "items": {"type": "object"}}

        monkeypatch.setattr(
            sglang_prepost_module,
            "get_json_schema_constraint",
            old_get_json_schema_constraint,
        )

        guided = build_tool_call_guided_decoding(
            {"tool_choice": "required", "parallel_tool_calls": False},
            tool_call_parser_name=None,
            sglang_tools=tools,
        )

        assert guided == {"json": {"type": "array", "items": {"type": "object"}}}

    def test_auto_tool_choice_supports_older_structure_constraint_signature(
        self, monkeypatch
    ):
        tools = convert_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ]
        )

        class OldFunctionCallParser:
            def __init__(self, *, tools, tool_call_parser):
                self.tools = tools
                self.tool_call_parser = tool_call_parser

            def get_structure_constraint(self, tool_choice):
                assert tool_choice == "auto"
                return "structural_tag", {"type": "object"}

        monkeypatch.setattr(
            sglang_prepost_module,
            "FunctionCallParser",
            OldFunctionCallParser,
        )

        guided = build_tool_call_guided_decoding(
            {"tool_choice": "auto", "parallel_tool_calls": False},
            tool_call_parser_name="kimi_k2",
            sglang_tools=tools,
        )

        assert guided == {"structural_tag": {"type": "object"}}

    def test_auto_strict_tools_can_build_structural_tag_guidance(self):
        tools = convert_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
        )

        guided = build_tool_call_guided_decoding(
            {"tool_choice": "auto"},
            tool_call_parser_name="kimi_k2",
            sglang_tools=tools,
        )

        assert isinstance(guided, dict)
        assert "structural_tag" in guided

    def test_tool_parser_requires_tools(self):
        """Tool parser is not created if no tools in request."""
        tcp, rp = create_parsers(
            {}, tool_call_parser_name="hermes", reasoning_parser_name=None
        )
        assert tcp is None

    def test_tool_parser_with_tools(self):
        """Tool parser is created when tools are present."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": "d",
                        "parameters": {},
                    },
                }
            ]
        }
        tcp, rp = create_parsers(
            request, tool_call_parser_name="hermes", reasoning_parser_name=None
        )
        assert tcp is not None
        assert rp is None

    def test_tool_choice_none_skips_parser(self):
        """tool_choice='none' should skip tool parser creation."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": "d",
                        "parameters": {},
                    },
                }
            ],
            "tool_choice": "none",
        }
        tcp, rp = create_parsers(
            request, tool_call_parser_name="hermes", reasoning_parser_name=None
        )
        assert tcp is None

    def test_both_parsers(self):
        """Both parsers created when tools and reasoning requested."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": "d",
                        "parameters": {},
                    },
                }
            ]
        }
        tcp, rp = create_parsers(
            request,
            tool_call_parser_name="hermes",
            reasoning_parser_name="qwen3",
        )
        assert tcp is not None
        assert rp is not None

    def test_required_creates_json_array_parser(self):
        """tool_choice='required' creates JsonArrayParser, not FunctionCallParser."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": "d",
                        "parameters": {},
                    },
                }
            ],
            "tool_choice": "required",
        }
        tcp, _ = create_parsers(
            request, tool_call_parser_name="hermes", reasoning_parser_name=None
        )
        assert isinstance(tcp, JsonArrayParser)

    def test_named_tool_choice_creates_json_array_parser(self):
        """Named tool_choice creates JsonArrayParser."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {},
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
        }
        tcp, _ = create_parsers(
            request, tool_call_parser_name="hermes", reasoning_parser_name=None
        )
        assert isinstance(tcp, JsonArrayParser)

    def test_auto_creates_function_call_parser(self):
        """tool_choice='auto' creates FunctionCallParser."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": "d",
                        "parameters": {},
                    },
                }
            ],
            "tool_choice": "auto",
        }
        tcp, _ = create_parsers(
            request, tool_call_parser_name="hermes", reasoning_parser_name=None
        )
        assert isinstance(tcp, FunctionCallParser)

    def test_required_without_parser_name_still_creates_json_array_parser(self):
        """tool_choice='required' doesn't need tool_call_parser_name."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": "d",
                        "parameters": {},
                    },
                }
            ],
            "tool_choice": "required",
        }
        tcp, _ = create_parsers(
            request, tool_call_parser_name=None, reasoning_parser_name=None
        )
        assert isinstance(tcp, JsonArrayParser)


# ---------------------------------------------------------------------------
# _parse_json_array_buffer
# ---------------------------------------------------------------------------


class TestParseJsonArrayBuffer:  # FE.6 — incremental JSON-array buffer parsing
    """Test JSON array fallback parser for constrained decoding output."""

    def test_single_tool_call(self):
        buffer = json.dumps([{"name": "get_weather", "parameters": {"city": "NYC"}}])
        calls = _parse_json_array_buffer(buffer)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].tool_index == 0
        assert json.loads(calls[0].parameters) == {"city": "NYC"}

    def test_multiple_tool_calls(self):
        buffer = json.dumps(
            [
                {"name": "get_weather", "parameters": {"city": "NYC"}},
                {"name": "search", "parameters": {"q": "hello"}},
            ]
        )
        calls = _parse_json_array_buffer(buffer)
        assert len(calls) == 2
        assert calls[0].name == "get_weather"
        assert calls[0].tool_index == 0
        assert calls[1].name == "search"
        assert calls[1].tool_index == 1

    def test_arguments_key_also_accepted(self):
        """Some formats use 'arguments' instead of 'parameters'."""
        buffer = json.dumps([{"name": "f", "arguments": {"x": 1}}])
        calls = _parse_json_array_buffer(buffer)
        assert len(calls) == 1
        assert json.loads(calls[0].parameters) == {"x": 1}

    def test_string_parameters_preserved(self):
        buffer = json.dumps([{"name": "f", "parameters": "already_a_string"}])
        calls = _parse_json_array_buffer(buffer)
        assert calls[0].parameters == "already_a_string"

    def test_invalid_json_returns_empty(self):
        assert _parse_json_array_buffer("not json") == []

    def test_non_array_returns_empty(self):
        assert _parse_json_array_buffer('{"name": "f"}') == []

    def test_empty_buffer_returns_empty(self):
        assert _parse_json_array_buffer("") == []

    def test_non_dict_items_skipped(self):
        buffer = json.dumps(["not_a_dict", {"name": "f", "parameters": {}}])
        calls = _parse_json_array_buffer(buffer)
        assert len(calls) == 1
        assert calls[0].name == "f"
        assert calls[0].tool_index == 1

    def test_trailing_special_token(self):
        """Trailing EOS/special tokens should not break parsing."""
        buffer = '[{"name": "f", "parameters": {"x": 1}}]<|endoftext|>'
        calls = _parse_json_array_buffer(buffer)
        assert len(calls) == 1
        assert calls[0].name == "f"
        assert json.loads(calls[0].parameters) == {"x": 1}

    def test_leading_text_with_array(self):
        """Leading non-JSON text before the array should be tolerated."""
        buffer = 'some preamble [{"name": "f", "parameters": {"x": 1}}]'
        calls = _parse_json_array_buffer(buffer)
        assert len(calls) == 1
        assert calls[0].name == "f"

    def test_trailing_and_leading_noise(self):
        """Both leading and trailing noise."""
        buffer = 'text [{"name": "g", "parameters": {"y": 2}}] <|end|>'
        calls = _parse_json_array_buffer(buffer)
        assert len(calls) == 1
        assert calls[0].name == "g"


class TestNormalizePromptTokenIds:  # FE.6 — prompt-token-id normalization
    def test_batch_encoding_like_object_uses_input_ids(self):
        class FakeBatchEncoding:
            def __init__(self):
                self.input_ids = [11, 22, 33]

            def __iter__(self):
                yield from ("input_ids", "attention_mask")

        assert _normalize_prompt_token_ids(FakeBatchEncoding()) == [11, 22, 33]

    def test_mapping_uses_input_ids(self):
        assert _normalize_prompt_token_ids(
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        ) == [1, 2, 3]


class TestRuntimeConfigParserName:  # FE.2 — parser name resolution from runtime config
    def test_missing_runtime_config_returns_none(self):
        class FakeMdc:
            def runtime_config(self):
                return None

        assert _runtime_config_parser_name(FakeMdc(), "tool_call_parser") is None

    def test_missing_key_returns_none(self):
        class FakeMdc:
            def runtime_config(self):
                return {"reasoning_parser": "qwen3"}

        assert _runtime_config_parser_name(FakeMdc(), "tool_call_parser") is None

    def test_reads_non_empty_string_value(self):
        class FakeMdc:
            def runtime_config(self):
                return {"tool_call_parser": "hermes"}

        assert _runtime_config_parser_name(FakeMdc(), "tool_call_parser") == "hermes"


# ---------------------------------------------------------------------------
# preprocess_chat_request
# ---------------------------------------------------------------------------


class TestPreprocessChatRequest:  # FE.1 — chat-template input preprocessing (multi-turn assistant tool_calls, role handling)
    """Test end-to-end preprocessing with a real tokenizer."""

    def test_basic_chat(self, tokenizer):
        """Simple user message preprocesses to non-empty token IDs."""
        request = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = preprocess_chat_request(
            request,
            tokenizer=tokenizer,
            tool_call_parser_name=None,
            reasoning_parser_name=None,
        )
        assert isinstance(result, SglangPreprocessResult)
        assert len(result.prompt_token_ids) > 0
        assert result.tool_call_parser is None
        assert result.reasoning_parser is None

    def test_multi_turn(self, tokenizer):
        """Multi-turn conversation produces more tokens than single turn."""
        single = preprocess_chat_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            tokenizer=tokenizer,
            tool_call_parser_name=None,
            reasoning_parser_name=None,
        )
        multi = preprocess_chat_request(
            {
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ],
            },
            tokenizer=tokenizer,
            tool_call_parser_name=None,
            reasoning_parser_name=None,
        )
        assert len(multi.prompt_token_ids) > len(single.prompt_token_ids)

    def test_with_tools(self, tokenizer):
        """Tools are passed through to chat template, producing more tokens."""
        without_tools = preprocess_chat_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            tokenizer=tokenizer,
            tool_call_parser_name=None,
            reasoning_parser_name=None,
        )
        with_tools = preprocess_chat_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
            tokenizer=tokenizer,
            tool_call_parser_name="hermes",
            reasoning_parser_name=None,
        )
        assert len(with_tools.prompt_token_ids) > len(without_tools.prompt_token_ids)
        assert with_tools.tool_call_parser is not None

    def test_assistant_tool_calls_with_string_arguments(self, tokenizer):
        """Multi-turn with prior assistant tool_calls renders without raising.

        Regression: the qwen3-coder Jinja template calls ``arguments | items``
        on assistant tool_calls and required ``arguments`` to be a mapping.
        Dynamo carries arguments as a JSON string per the OpenAI wire
        contract; the prepost must parse them to dict before templating.
        """
        request = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "Weather in Tokyo?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Tokyo"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_001",
                    "content": json.dumps({"temperature": 22}),
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        }
        result = preprocess_chat_request(
            request,
            tokenizer=tokenizer,
            tool_call_parser_name="hermes",
            reasoning_parser_name=None,
        )
        assert len(result.prompt_token_ids) > 0
        # Original request dict must not be mutated by the normaliser
        # (callers reuse it for downstream processing).
        original_args = request["messages"][1]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(original_args, str)

    def test_normalize_assistant_tool_call_arguments_helper(self):
        """The string→dict normaliser parses valid JSON and skips bad input."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "f",
                            "arguments": json.dumps({"a": 1}),
                        }
                    },
                    {
                        "function": {
                            "name": "g",
                            "arguments": "not-json",
                        }
                    },
                    {
                        "function": {
                            "name": "h",
                            "arguments": {"already": "dict"},
                        }
                    },
                ],
            },
            # Tool messages are not assistant; arguments key shouldn't exist
            # but if it did we wouldn't touch it.
            {"role": "tool", "tool_call_id": "x", "content": "ok"},
        ]
        _normalize_assistant_tool_call_arguments(messages)
        tcs = messages[1]["tool_calls"]
        assert tcs[0]["function"]["arguments"] == {"a": 1}
        # Malformed JSON left as-is so the template error stays visible.
        assert tcs[1]["function"]["arguments"] == "not-json"
        # Already-dict values pass through untouched.
        assert tcs[2]["function"]["arguments"] == {"already": "dict"}

    def test_tool_choice_none_strips_tools_from_template(self, tokenizer):
        """When exclude flag is on and tool_choice=none, tools are excluded from template."""
        tool_request = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        }
        with_tools_auto = preprocess_chat_request(
            {**tool_request, "tool_choice": "auto"},
            tokenizer=tokenizer,
            tool_call_parser_name=None,
            reasoning_parser_name=None,
            exclude_tools_when_tool_choice_none=True,
        )
        with_tools_none = preprocess_chat_request(
            {**tool_request, "tool_choice": "none"},
            tokenizer=tokenizer,
            tool_call_parser_name=None,
            reasoning_parser_name=None,
            exclude_tools_when_tool_choice_none=True,
        )
        # tool_choice=none should produce fewer tokens (no tool defs in template)
        assert len(with_tools_none.prompt_token_ids) < len(
            with_tools_auto.prompt_token_ids
        ), "tool_choice=none with exclude flag should strip tools from template"

    def test_tool_choice_none_keeps_tools_when_flag_off(self, tokenizer):
        """When exclude flag is off, tool_choice=none still includes tools in template."""
        tool_request = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        }
        with_auto = preprocess_chat_request(
            {**tool_request, "tool_choice": "auto"},
            tokenizer=tokenizer,
            tool_call_parser_name=None,
            reasoning_parser_name=None,
            exclude_tools_when_tool_choice_none=False,
        )
        with_none = preprocess_chat_request(
            {**tool_request, "tool_choice": "none"},
            tokenizer=tokenizer,
            tool_call_parser_name=None,
            reasoning_parser_name=None,
            exclude_tools_when_tool_choice_none=False,
        )
        # With flag off, both should have similar token counts (tools in template)
        assert len(with_none.prompt_token_ids) == len(
            with_auto.prompt_token_ids
        ), "tool_choice=none with flag off should keep tools in template"

    def test_named_tool_choice_missing_function_raises(self, tokenizer):
        """Named tool_choice referencing a function absent from tools raises ValueError."""
        request = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "does_not_exist"},
            },
        }
        with pytest.raises(ValueError, match="does_not_exist"):
            preprocess_chat_request(
                request,
                tokenizer=tokenizer,
                tool_call_parser_name="hermes",
                reasoning_parser_name=None,
            )

    def test_init_worker_propagates_exclude_flag_true(self):
        """_init_worker sets the worker-global exclude_tools flag to True."""
        _init_worker(MODEL, None, None, exclude_tools_when_tool_choice_none=True)
        assert sglang_processor_module._w_exclude_tools_when_tool_choice_none is True

    def test_init_worker_propagates_exclude_flag_false(self):
        """_init_worker sets the worker-global exclude_tools flag to False."""
        _init_worker(MODEL, None, None, exclude_tools_when_tool_choice_none=False)
        assert sglang_processor_module._w_exclude_tools_when_tool_choice_none is False
        # Reset to default
        sglang_processor_module._w_exclude_tools_when_tool_choice_none = True

    def test_with_reasoning_parser(self, tokenizer):
        """Reasoning parser is attached to result."""
        result = preprocess_chat_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            tokenizer=tokenizer,
            tool_call_parser_name=None,
            reasoning_parser_name="qwen3",
        )
        assert result.reasoning_parser is not None

    def test_system_message(self, tokenizer):
        """System message is included in tokenization."""
        without_system = preprocess_chat_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            tokenizer=tokenizer,
            tool_call_parser_name=None,
            reasoning_parser_name=None,
        )
        with_system = preprocess_chat_request(
            {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ],
            },
            tokenizer=tokenizer,
            tool_call_parser_name=None,
            reasoning_parser_name=None,
        )
        assert len(with_system.prompt_token_ids) > len(without_system.prompt_token_ids)

    def test_deepseek_v4_uses_sglang_encoder_when_chat_template_missing(
        self, monkeypatch
    ):
        """DeepSeek-V4 uses SGLang's encoder instead of HF chat_template."""
        captured = {}
        fake_module = types.ModuleType("sglang.srt.entrypoints.openai.encoding_dsv4")

        def fake_encode_messages(messages, *, thinking_mode, reasoning_effort=None):
            captured["messages"] = messages
            captured["thinking_mode"] = thinking_mode
            captured["reasoning_effort"] = reasoning_effort
            return "<dsv4-prompt>"

        fake_module.encode_messages = fake_encode_messages
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.entrypoints.openai.encoding_dsv4",
            fake_module,
        )

        class NoTemplateTokenizer:
            chat_template = None

            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("apply_chat_template should not be called")

            def encode(self, prompt):
                assert prompt == "<dsv4-prompt>"
                return [1, 2, 3]

        request = {
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": "max",
            },
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        }

        result = preprocess_chat_request(
            request,
            tokenizer=NoTemplateTokenizer(),
            tool_call_parser_name=None,
            reasoning_parser_name="deepseek_v4",
        )

        assert result.prompt_token_ids == [1, 2, 3]
        assert captured["thinking_mode"] == "thinking"
        assert captured["reasoning_effort"] == "max"
        assert captured["messages"][0]["role"] == "system"
        assert captured["messages"][0]["tools"][0]["function"]["name"] == "get_weather"
        assert captured["messages"][1]["role"] == "user"

    def test_deepseek_v4_named_tool_choice_filters_encoder_tools(self, monkeypatch):
        captured = {}
        fake_module = types.ModuleType("sglang.srt.entrypoints.openai.encoding_dsv4")

        def fake_encode_messages(messages, *, thinking_mode, reasoning_effort=None):
            captured["messages"] = messages
            return "<dsv4-prompt>"

        fake_module.encode_messages = fake_encode_messages
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.entrypoints.openai.encoding_dsv4",
            fake_module,
        )

        class NoTemplateTokenizer:
            chat_template = None

            def encode(self, prompt):
                return [1]

        request = {
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                },
                {
                    "type": "function",
                    "function": {"name": "get_time", "parameters": {}},
                },
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_time"},
            },
        }

        preprocess_chat_request(
            request,
            tokenizer=NoTemplateTokenizer(),
            tool_call_parser_name=None,
            reasoning_parser_name="deepseek_v4",
        )

        tools = captured["messages"][0]["tools"]
        assert [tool["function"]["name"] for tool in tools] == ["get_time"]

    def test_deepseek_v4_respects_existing_chat_template(self, monkeypatch):
        fake_module = types.ModuleType("sglang.srt.entrypoints.openai.encoding_dsv4")

        def fake_encode_messages(messages, *, thinking_mode, reasoning_effort=None):
            raise AssertionError("encoding_dsv4 should not be called")

        fake_module.encode_messages = fake_encode_messages
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.entrypoints.openai.encoding_dsv4",
            fake_module,
        )

        class TemplateTokenizer:
            chat_template = (
                "{% for message in messages %}{{ message.content }}{% endfor %}"
            )

            def apply_chat_template(self, messages, **kwargs):
                assert kwargs["add_generation_prompt"] is True
                assert kwargs["tokenize"] is True
                return [4, 5, 6]

            def encode(self, prompt):
                raise AssertionError("encode should not be called")

        result = preprocess_chat_request(
            {
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            tokenizer=TemplateTokenizer(),
            tool_call_parser_name=None,
            reasoning_parser_name=None,
        )

        assert result.prompt_token_ids == [4, 5, 6]

    def test_deepseek_v4_normalizes_none_content_without_mutating_request(
        self, monkeypatch
    ):
        captured = {}
        fake_module = types.ModuleType("sglang.srt.entrypoints.openai.encoding_dsv4")

        def fake_encode_messages(messages, *, thinking_mode, reasoning_effort=None):
            captured["messages"] = messages
            return "<dsv4-prompt>"

        fake_module.encode_messages = fake_encode_messages
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.entrypoints.openai.encoding_dsv4",
            fake_module,
        )

        class NoTemplateTokenizer:
            chat_template = None

            def encode(self, prompt):
                return [7]

        request = {
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "messages": [{"role": "assistant", "content": None}],
        }

        result = preprocess_chat_request(
            request,
            tokenizer=NoTemplateTokenizer(),
            tool_call_parser_name=None,
            reasoning_parser_name=None,
        )

        assert result.prompt_token_ids == [7]
        assert captured["messages"] == [{"role": "assistant", "content": ""}]
        assert request["messages"] == [{"role": "assistant", "content": None}]

    def test_deepseek_v4_tool_choice_none_strips_encoder_tools(self, monkeypatch):
        captured = {}
        fake_module = types.ModuleType("sglang.srt.entrypoints.openai.encoding_dsv4")

        def fake_encode_messages(messages, *, thinking_mode, reasoning_effort=None):
            captured["messages"] = messages
            return "<dsv4-prompt>"

        fake_module.encode_messages = fake_encode_messages
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.entrypoints.openai.encoding_dsv4",
            fake_module,
        )

        class NoTemplateTokenizer:
            chat_template = None

            def encode(self, prompt):
                return [8]

        preprocess_chat_request(
            {
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "messages": [{"role": "system", "content": "Stay terse."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {}},
                    }
                ],
                "tool_choice": "none",
            },
            tokenizer=NoTemplateTokenizer(),
            tool_call_parser_name=None,
            reasoning_parser_name=None,
            exclude_tools_when_tool_choice_none=True,
        )

        assert "tools" not in captured["messages"][0]


# ---------------------------------------------------------------------------
# SglangStreamingPostProcessor: incremental detokenization
# ---------------------------------------------------------------------------


class TestIncrementalDetokenization:  # FE.6 — token-id stream → text
    """Test the sliding-window incremental detokenizer."""

    def test_basic_decode(self, tokenizer):
        """Tokens decode to expected text."""
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer, tool_call_parser=None, reasoning_parser=None
        )
        token_ids = tokenizer.encode("Hello world")
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        assert choice is not None
        assert "Hello world" in choice["delta"]["content"]

    def test_incremental_batches(self, tokenizer):
        """Batched tokens produce the full text when concatenated."""
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer, tool_call_parser=None, reasoning_parser=None
        )
        text = "The quick brown fox jumps over the lazy dog."
        token_ids = tokenizer.encode(text)

        content = ""
        batch_size = 3
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i : i + batch_size]
            is_last = i + batch_size >= len(token_ids)
            choice = post.process_output(
                {"token_ids": batch, "finish_reason": "stop" if is_last else None}
            )
            if choice and "content" in choice.get("delta", {}):
                content += choice["delta"]["content"]
        assert text in content

    def test_empty_token_ids(self, tokenizer):
        """Empty token_ids with no finish_reason returns None."""
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer, tool_call_parser=None, reasoning_parser=None
        )
        result = post.process_output({"token_ids": [], "finish_reason": None})
        assert result is None

    def test_finish_reason_only(self, tokenizer):
        """finish_reason without new tokens emits a finish chunk."""
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer, tool_call_parser=None, reasoning_parser=None
        )
        # First send some tokens
        token_ids = tokenizer.encode("Hello")
        post.process_output({"token_ids": token_ids, "finish_reason": None})
        # Then send finish with no new tokens
        choice = post.process_output({"token_ids": [], "finish_reason": "stop"})
        assert choice is not None
        assert choice["finish_reason"] == "stop"

    def test_lookback_trimming(self, tokenizer):
        """Verify _all_token_ids doesn't grow unbounded."""
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer, tool_call_parser=None, reasoning_parser=None
        )
        # Send enough tokens to trigger trimming (LOOKBACK * 16 = 96)
        for _ in range(200):
            post.process_output({"token_ids": [1], "finish_reason": None})
        # Should be trimmed, not 200 tokens
        assert len(post._all_token_ids) < 200


# ---------------------------------------------------------------------------
# SglangStreamingPostProcessor: fast plain text path
# ---------------------------------------------------------------------------


class TestFastPlainTextPath:  # FE.6 — fast path that skips parser when no markers
    """Test the fast path when no parsers are active."""

    def test_fast_path_active(self, tokenizer):
        """No parsers -> fast plain text path."""
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer, tool_call_parser=None, reasoning_parser=None
        )
        assert post._fast_plain_text is True

    def test_fast_path_inactive_with_reasoning(self, tokenizer):
        """Reasoning parser disables fast path."""
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        rp = ReasoningParser(model_type="qwen3", stream_reasoning=True)
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer, tool_call_parser=None, reasoning_parser=rp
        )
        assert post._fast_plain_text is False

    def test_fast_path_content_output(self, tokenizer):
        """Fast path produces role and content in delta."""
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer, tool_call_parser=None, reasoning_parser=None
        )
        token_ids = tokenizer.encode("Hello")
        choice = post.process_output({"token_ids": token_ids, "finish_reason": None})
        assert choice is not None
        assert choice["delta"]["role"] == "assistant"
        assert "content" in choice["delta"]
        assert choice["index"] == 0
        assert choice["logprobs"] is None


# ---------------------------------------------------------------------------
# SglangStreamingPostProcessor: reasoning parsing
# ---------------------------------------------------------------------------


class TestReasoningParsing:  # FE.9 — reasoning ↔ tool-call orchestration
    """Test reasoning content extraction via post-processor."""

    def test_reasoning_separated(self, tokenizer):
        """<think>...</think> content goes to reasoning_content field."""
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        rp = ReasoningParser(model_type="qwen3", stream_reasoning=True)
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer, tool_call_parser=None, reasoning_parser=rp
        )
        text = "<think>\nLet me think about this.\n</think>\n\nThe answer is 42."
        token_ids = tokenizer.encode(text)

        reasoning = ""
        content = ""
        for i in range(0, len(token_ids), 5):
            batch = token_ids[i : i + 5]
            is_last = i + 5 >= len(token_ids)
            choice = post.process_output(
                {"token_ids": batch, "finish_reason": "stop" if is_last else None}
            )
            if choice:
                delta = choice.get("delta", {})
                reasoning += delta.get("reasoning_content", "")
                content += delta.get("content", "")

        assert "think about this" in reasoning
        assert "42" in content


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestUtilities:  # (mixed — see per-test annotations)
    """Test shared utility functions."""

    def test_random_uuid_format(self):  # FE.4
        """random_uuid produces 16-char hex string."""
        uid = random_uuid()
        assert len(uid) == 16
        int(uid, 16)  # Should not raise

    def test_random_uuid_unique(self):  # FE.4
        """Two calls produce different UUIDs."""
        assert random_uuid() != random_uuid()

    def test_random_call_id_format(self):  # FE.4
        """random_call_id produces call_<16hex> format."""
        cid = random_call_id()
        assert cid.startswith("call_")
        assert len(cid) == 21  # "call_" + 16 hex chars
        int(cid[5:], 16)  # Should not raise

    def test_preprocess_error(self):  # FE.8
        """PreprocessError stores message and stringifies."""
        err = PreprocessError("n=2 unsupported")
        assert "n=2" in str(err)


# ---------------------------------------------------------------------------
# SglangPreprocessWorkerResult picklability
# ---------------------------------------------------------------------------


class TestWorkerResultPicklability:  # FE.7 — worker subprocess boundary picklability
    """Test that worker results survive ProcessPoolExecutor round-trip."""

    def test_full_result(self):
        """Full SglangPreprocessWorkerResult survives pickle round-trip."""
        import pickle

        result = SglangPreprocessWorkerResult(
            prompt_token_ids=[1, 2, 3],
            dynamo_preproc={
                "model": "test-model",
                "token_ids": [1, 2, 3],
                "stop_conditions": {
                    "max_tokens": 100,
                    "stop": [],
                    "stop_token_ids": [2],
                    "min_tokens": 0,
                    "ignore_eos": False,
                },
                "sampling_options": {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "seed": None,
                },
                "output_options": {
                    "logprobs": None,
                    "prompt_logprobs": None,
                    "skip_special_tokens": True,
                },
                "eos_token_ids": [2],
                "annotations": [],
            },
            request={"model": "test-model", "messages": [], "tools": None},
        )

        data = pickle.dumps(result)
        restored = pickle.loads(data)

        assert restored.prompt_token_ids == result.prompt_token_ids
        assert restored.dynamo_preproc == result.dynamo_preproc
        assert restored.request == result.request


# ---------------------------------------------------------------------------
# Deprecation warning for --use-sglang-tokenizer
# ---------------------------------------------------------------------------


class TestDeprecationWarning:  # FE.8 — legacy/deprecated field warnings
    """Test that --use-sglang-tokenizer deprecation warning is in place."""

    def test_deprecation_warning_in_source(self):
        """Verify parse_args contains FutureWarning for use_sglang_tokenizer.

        The warning is embedded in parse_args() which requires full ServerArgs
        initialization -- too heavy for a unit test.  Instead, verify the warning
        text exists in the source code so it isn't accidentally removed.
        """
        import inspect

        from dynamo.sglang import args as sglang_args

        source = inspect.getsource(sglang_args)
        assert "use_sglang_tokenizer" in source
        assert "FutureWarning" in source
        assert "--dyn-chat-processor sglang" in source
