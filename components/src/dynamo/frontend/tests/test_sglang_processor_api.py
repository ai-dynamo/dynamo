#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Conformance tests for the SGLang API surface used by the sglang processor.

These tests lock down the SGLang interfaces we depend on so that SGLang
upgrades that break our integration surface are caught immediately.
"""

import inspect
import pickle

# ---------------------------------------------------------------------------
# Import tests -- verify all required modules and symbols exist
# ---------------------------------------------------------------------------


def test_get_tokenizer_importable():
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    assert callable(get_tokenizer)


def test_function_call_parser_importable():
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    assert callable(FunctionCallParser)


def test_tool_call_item_importable():
    from sglang.srt.function_call.core_types import ToolCallItem

    assert callable(ToolCallItem)


def test_reasoning_parser_importable():
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    assert callable(ReasoningParser)


def test_sglang_tool_importable():
    from sglang.srt.entrypoints.openai.protocol import Function, Tool

    assert callable(Tool)
    assert callable(Function)


# ---------------------------------------------------------------------------
# get_tokenizer signature
# ---------------------------------------------------------------------------


def test_get_tokenizer_accepts_tokenizer_mode():
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    sig = inspect.signature(get_tokenizer)
    params = sig.parameters
    assert "tokenizer_name" in params or list(params.keys())[0] != ""
    assert "tokenizer_mode" in params


# ---------------------------------------------------------------------------
# FunctionCallParser
# ---------------------------------------------------------------------------


def test_function_call_parser_init():
    """Verify FunctionCallParser constructor accepts tools and tool_call_parser."""
    from sglang.srt.entrypoints.openai.protocol import Function, Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    tools = [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather for a city",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        )
    ]
    parser = FunctionCallParser(tools=tools, tool_call_parser="hermes")
    assert parser is not None


def test_function_call_parser_enum_keys():
    """Verify commonly-used parser names are accepted."""
    from sglang.srt.entrypoints.openai.protocol import Function, Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    tools = [
        Tool(
            type="function",
            function=Function(
                name="f",
                description="d",
                parameters={"type": "object", "properties": {}},
            ),
        )
    ]
    # These parser names must remain available
    for name in ("hermes", "llama3", "qwen25"):
        parser = FunctionCallParser(tools=tools, tool_call_parser=name)
        assert parser is not None


def test_parse_stream_chunk_signature():
    """Verify parse_stream_chunk returns (str, list[ToolCallItem])."""
    from sglang.srt.entrypoints.openai.protocol import Function, Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    tools = [
        Tool(
            type="function",
            function=Function(
                name="f",
                description="d",
                parameters={"type": "object", "properties": {}},
            ),
        )
    ]
    parser = FunctionCallParser(tools=tools, tool_call_parser="hermes")
    result = parser.parse_stream_chunk("Hello world")
    assert isinstance(result, tuple)
    assert len(result) == 2
    normal_text, calls = result
    assert isinstance(normal_text, str)
    assert isinstance(calls, list)


def test_tool_call_item_fields():
    """Verify ToolCallItem has expected fields."""
    from sglang.srt.function_call.core_types import ToolCallItem

    item = ToolCallItem(tool_index=0, name="test", parameters='{"x": 1}')
    assert item.tool_index == 0
    assert item.name == "test"
    assert item.parameters == '{"x": 1}'


# ---------------------------------------------------------------------------
# ReasoningParser
# ---------------------------------------------------------------------------


def test_reasoning_parser_init():
    """Verify ReasoningParser constructor accepts model_type."""
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    parser = ReasoningParser(model_type="deepseek-r1", stream_reasoning=True)
    assert parser is not None


def test_reasoning_parser_detector_map():
    """Verify commonly-used detector names are accepted."""
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    for name in ("deepseek-r1", "qwen3"):
        parser = ReasoningParser(model_type=name, stream_reasoning=True)
        assert parser is not None


def test_reasoning_parser_parse_stream_chunk():
    """Verify parse_stream_chunk returns (reasoning_text, normal_text)."""
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    parser = ReasoningParser(model_type="deepseek-r1", stream_reasoning=True)
    result = parser.parse_stream_chunk("Hello")
    assert isinstance(result, tuple)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# StreamingParseResult (function call variant)
# ---------------------------------------------------------------------------


def test_streaming_parse_result_fields():
    """Verify function-call StreamingParseResult has expected fields."""
    from sglang.srt.function_call.core_types import StreamingParseResult

    r = StreamingParseResult(normal_text="hello", calls=[])
    assert r.normal_text == "hello"
    assert r.calls == []


# ---------------------------------------------------------------------------
# Tool / Function protocol models
# ---------------------------------------------------------------------------


def test_sglang_tool_model_dump():
    """Verify Tool.model_dump() produces a dict suitable for chat templates."""
    from sglang.srt.entrypoints.openai.protocol import Function, Tool

    tool = Tool(
        type="function",
        function=Function(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        ),
    )
    d = tool.model_dump()
    assert d["type"] == "function"
    assert d["function"]["name"] == "search"
    assert "properties" in d["function"]["parameters"]


# ---------------------------------------------------------------------------
# Picklability (required for ProcessPoolExecutor worker results)
# ---------------------------------------------------------------------------


def test_preprocess_result_picklability():
    """Verify SglangPreprocessWorkerResult survives pickle round-trip."""
    from dynamo.frontend.sglang_processor import SglangPreprocessWorkerResult

    result = SglangPreprocessWorkerResult(
        prompt_token_ids=[1, 2, 3],
        dynamo_preproc={
            "model": "test",
            "token_ids": [1, 2, 3],
            "stop_conditions": {},
            "sampling_options": {},
            "output_options": {},
            "eos_token_ids": [],
            "annotations": [],
        },
        request={"model": "test", "messages": []},
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored.prompt_token_ids == result.prompt_token_ids
    assert restored.dynamo_preproc == result.dynamo_preproc
    assert restored.request == result.request
