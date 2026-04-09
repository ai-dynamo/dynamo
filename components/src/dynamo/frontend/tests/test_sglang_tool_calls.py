#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for tool call parsing in SglangStreamingPostProcessor.

Covers the interaction between SGLang's FunctionCallParser, ReasoningParser,
and our post-processor's accumulate-and-emit-on-finish logic, including the
parse_non_stream fallback for the chunking-sensitivity issue in
BaseFormatDetector.parse_streaming_increment.
"""

import json

import pytest
from sglang.srt.entrypoints.openai.protocol import Function as SglangFunction
from sglang.srt.entrypoints.openai.protocol import Tool as SglangTool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

from dynamo.frontend.sglang_prepost import SglangStreamingPostProcessor

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


TOOLS = [
    SglangTool(
        type="function",
        function=SglangFunction(
            name="search_gutenberg_books",
            description="Search for books in the Project Gutenberg library",
            parameters={
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search terms to find books",
                    }
                },
                "required": ["search_terms"],
            },
        ),
    ),
    SglangTool(
        type="function",
        function=SglangFunction(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
    ),
]


def _run_postprocessor(tokenizer, full_text, batch_size, *, use_reasoning=True):
    """Tokenize text, feed through post-processor in batches, return all choices."""
    tcp = FunctionCallParser(tools=TOOLS, tool_call_parser="hermes")
    rp = (
        ReasoningParser(model_type="qwen3", stream_reasoning=True)
        if use_reasoning
        else None
    )

    post = SglangStreamingPostProcessor(
        tokenizer=tokenizer,
        tool_call_parser=tcp,
        reasoning_parser=rp,
    )

    token_ids = tokenizer.encode(full_text)
    results = []
    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i : i + batch_size]
        is_last = i + batch_size >= len(token_ids)
        choice = post.process_output(
            {"token_ids": batch, "finish_reason": "stop" if is_last else None}
        )
        if choice:
            results.append(choice)
    return results


def _extract_tool_calls(results):
    """Extract tool_calls from the list of choices."""
    for r in results:
        tc = r.get("delta", {}).get("tool_calls")
        if tc:
            return tc
    return []


# ---------------------------------------------------------------------------
# Single tool call
# ---------------------------------------------------------------------------


class TestSingleToolCall:
    """Single tool call with reasoning, various batch sizes."""

    TEXT = (
        "<think>\nLet me search for books.\n</think>\n\n"
        '<tool_call>\n{"name": "search_gutenberg_books", '
        '"arguments": {"search_terms": ["James Joyce"]}}\n</tool_call>'
    )

    def test_large_batches(self, tokenizer):
        """stream_interval=20 scenario -- complete JSON in one chunk."""
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 20))
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search_gutenberg_books"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"search_terms": ["James Joyce"]}

    def test_small_batches(self, tokenizer):
        """Token-by-token-ish scenario -- streaming deltas work directly."""
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 3))
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search_gutenberg_books"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"search_terms": ["James Joyce"]}

    def test_medium_batches(self, tokenizer):
        """Intermediate batch size."""
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 10))
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search_gutenberg_books"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"search_terms": ["James Joyce"]}

    def test_tool_call_has_id_and_type(self, tokenizer):
        """Each tool call must have id and type fields."""
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 20))
        assert tc[0]["id"].startswith("call_")
        assert tc[0]["type"] == "function"
        assert tc[0]["index"] == 0


# ---------------------------------------------------------------------------
# No reasoning parser
# ---------------------------------------------------------------------------


class TestNoReasoningParser:
    """Tool calls without reasoning parser active."""

    TEXT = (
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}\n</tool_call>'
    )

    def test_large_batches(self, tokenizer):
        tc = _extract_tool_calls(
            _run_postprocessor(tokenizer, self.TEXT, 15, use_reasoning=False)
        )
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"city": "Paris"}

    def test_small_batches(self, tokenizer):
        tc = _extract_tool_calls(
            _run_postprocessor(tokenizer, self.TEXT, 3, use_reasoning=False)
        )
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"city": "Paris"}


# ---------------------------------------------------------------------------
# Multiple tool calls
# ---------------------------------------------------------------------------


class TestMultipleToolCalls:
    """Two tool calls in a single response."""

    TEXT = (
        "<think>\nI'll search and check weather.\n</think>\n\n"
        '<tool_call>\n{"name": "search_gutenberg_books", '
        '"arguments": {"search_terms": ["Joyce"]}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "London"}}\n</tool_call>'
    )

    def test_both_tools_present(self, tokenizer):
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 10))
        assert len(tc) == 2
        names = {t["function"]["name"] for t in tc}
        assert names == {"search_gutenberg_books", "get_weather"}

    def test_arguments_correct(self, tokenizer):
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 10))
        by_name = {t["function"]["name"]: t for t in tc}
        assert json.loads(
            by_name["search_gutenberg_books"]["function"]["arguments"]
        ) == {"search_terms": ["Joyce"]}
        assert json.loads(by_name["get_weather"]["function"]["arguments"]) == {
            "city": "London"
        }

    def test_distinct_ids(self, tokenizer):
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 10))
        ids = [t["id"] for t in tc]
        assert len(set(ids)) == len(ids), "Tool call IDs must be unique"


# ---------------------------------------------------------------------------
# Content alongside tool calls
# ---------------------------------------------------------------------------


class TestContentWithToolCalls:
    """Reasoning content and regular content are preserved alongside tool calls."""

    TEXT = (
        "<think>\nThinking about it.\n</think>\n\n"
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "NYC"}}\n</tool_call>'
    )

    def test_reasoning_content_present(self, tokenizer):
        results = _run_postprocessor(tokenizer, self.TEXT, 20)
        reasoning = ""
        for r in results:
            rc = r.get("delta", {}).get("reasoning_content", "")
            reasoning += rc
        assert "Thinking about it" in reasoning

    def test_content_is_whitespace_only(self, tokenizer):
        """Content between </think> and <tool_call> should be whitespace only."""
        results = _run_postprocessor(tokenizer, self.TEXT, 20)
        content = ""
        for r in results:
            c = r.get("delta", {}).get("content", "")
            content += c
        assert content.strip() == ""


# ---------------------------------------------------------------------------
# No tool calls (plain text)
# ---------------------------------------------------------------------------


class TestNoToolCalls:
    """When no tool call markup is present, no tool_calls should appear."""

    TEXT = "<think>\nJust thinking.\n</think>\n\nHello, world!"

    def test_no_tool_calls_emitted(self, tokenizer):
        tc = _extract_tool_calls(_run_postprocessor(tokenizer, self.TEXT, 10))
        assert tc == []

    def test_content_preserved(self, tokenizer):
        results = _run_postprocessor(tokenizer, self.TEXT, 10)
        content = ""
        for r in results:
            c = r.get("delta", {}).get("content", "")
            content += c
        assert "Hello, world!" in content
