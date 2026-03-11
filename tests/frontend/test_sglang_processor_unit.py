#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang processor components.

Tests for preprocessing, sampling parameter projection, finish reason mapping,
incremental detokenization, error handling, and deprecation warnings.

Parallels test_vllm_unit.py for the vLLM backend.
"""


import pytest
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

from dynamo.frontend.sglang_prepost import (
    SglangPreprocessResult,
    SglangStreamingPostProcessor,
    convert_tools,
    create_parsers,
    preprocess_chat_request,
)
from dynamo.frontend.sglang_processor import (
    SglangPreprocessWorkerResult,
    _build_dynamo_preproc,
    _map_finish_reason,
)
from dynamo.frontend.utils import PreprocessError, random_call_id, random_uuid

pytestmark = [
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL)


# ---------------------------------------------------------------------------
# _build_dynamo_preproc: sampling parameter projection
# ---------------------------------------------------------------------------


class TestBuildDynamoPreproc:
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


class TestMapFinishReason:
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


class TestConvertTools:
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


class TestCreateParsers:
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


# ---------------------------------------------------------------------------
# preprocess_chat_request
# ---------------------------------------------------------------------------


class TestPreprocessChatRequest:
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


# ---------------------------------------------------------------------------
# SglangStreamingPostProcessor: incremental detokenization
# ---------------------------------------------------------------------------


class TestIncrementalDetokenization:
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


class TestFastPlainTextPath:
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


class TestReasoningParsing:
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


class TestUtilities:
    """Test shared utility functions."""

    def test_random_uuid_format(self):
        """random_uuid produces 16-char hex string."""
        uid = random_uuid()
        assert len(uid) == 16
        int(uid, 16)  # Should not raise

    def test_random_uuid_unique(self):
        """Two calls produce different UUIDs."""
        assert random_uuid() != random_uuid()

    def test_random_call_id_format(self):
        """random_call_id produces call_<16hex> format."""
        cid = random_call_id()
        assert cid.startswith("call_")
        assert len(cid) == 21  # "call_" + 16 hex chars
        int(cid[5:], 16)  # Should not raise

    def test_preprocess_error(self):
        """PreprocessError stores error_dict and stringifies."""
        err = PreprocessError({"error": {"message": "n=2 unsupported"}})
        assert err.error_dict == {"error": {"message": "n=2 unsupported"}}
        assert "n=2" in str(err)


# ---------------------------------------------------------------------------
# SglangPreprocessWorkerResult picklability
# ---------------------------------------------------------------------------


class TestWorkerResultPicklability:
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


class TestDeprecationWarning:
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
