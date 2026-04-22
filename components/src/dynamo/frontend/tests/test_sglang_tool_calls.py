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
from sglang.srt.function_call.json_array_parser import JsonArrayParser
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


class TestKimiToolCallIds:
    def test_kimi_uses_history_adjusted_ids(self):
        class DummyTokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return "".join(chr(x) for x in token_ids)

        class DummyToolCall:
            def __init__(self, tool_index, name, parameters):
                self.tool_index = tool_index
                self.name = name
                self.parameters = parameters

        class DummyParser:
            tool_call_parser = "kimi_k2"
            detector = type("Detector", (), {"_buffer": ""})()

            def parse_stream_chunk(self, text):
                return "", [
                    DummyToolCall(0, "get_weather", '{"city":"Paris"}'),
                    DummyToolCall(
                        1, "search_gutenberg_books", '{"search_terms":["Joyce"]}'
                    ),
                ]

        post = SglangStreamingPostProcessor(
            tokenizer=DummyTokenizer(),
            tool_call_parser=DummyParser(),
            reasoning_parser=None,
            history_tool_calls_count=3,
            tool_call_parser_name="kimi_k2",
        )

        choice = post.process_output(
            {
                "token_ids": [ord("x")],
                "finish_reason": "stop",
            }
        )

        tc = choice["delta"]["tool_calls"]
        assert [item["id"] for item in tc] == [
            "functions.get_weather:3",
            "functions.search_gutenberg_books:4",
        ]

    def test_kimi_reparse_uses_sequential_index_not_tool_index(self):
        """kimi_k2 IDs after re-parse use the output position, not tool_index.

        ``FunctionCallParser.parse_non_stream`` can return
        ``ToolCallItem.tool_index`` values that reflect the tool-definition
        position rather than the call's sequential position.  IDs must
        align with the emitted ``index`` field, so they are built from
        the post-processor's ``seq_idx``.
        """

        class DummyTokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return "".join(chr(x) for x in token_ids)

        class DummyToolCall:
            def __init__(self, tool_index, name, parameters):
                self.tool_index = tool_index
                self.name = name
                self.parameters = parameters

        class DummyParser:
            tool_call_parser = "kimi_k2"
            detector = type("Detector", (), {"_buffer": ""})()

            def parse_stream_chunk(self, text):
                # Streaming misses both calls — forces the re-parse path.
                return "", []

            def has_tool_call(self, text):
                return True

            def parse_non_stream(self, text):
                # Non-sequential tool_index values, as parse_non_stream
                # sometimes returns tool-definition positions.
                return "", [
                    DummyToolCall(5, "get_weather", '{"city":"Paris"}'),
                    DummyToolCall(2, "search_gutenberg_books", '{"q":"Joyce"}'),
                ]

        post = SglangStreamingPostProcessor(
            tokenizer=DummyTokenizer(),
            tool_call_parser=DummyParser(),
            reasoning_parser=None,
            history_tool_calls_count=3,
            tool_call_parser_name="kimi_k2",
        )

        choice = post.process_output(
            {
                "token_ids": [ord("x")],
                "finish_reason": "stop",
            }
        )

        tc = choice["delta"]["tool_calls"]
        # IDs must use seq_idx (0, 1) + history (3), not tool_index (5, 2).
        assert [item["id"] for item in tc] == [
            "functions.get_weather:3",
            "functions.search_gutenberg_books:4",
        ]
        assert [item["index"] for item in tc] == [0, 1]


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


# ---------------------------------------------------------------------------
# Single-chunk tool calls (finish-time re-parse fallback)
# ---------------------------------------------------------------------------


class TestSingleChunkFallback:
    """When all tool call tokens + finish arrive in one batch, the streaming
    parser only processes one event.  The finish-time re-parse must recover
    arguments and any additional tool calls."""

    TEXT = (
        "<think>\nLet me search for books.\n</think>\n\n"
        '<tool_call>\n{"name": "search_gutenberg_books", '
        '"arguments": {"search_terms": ["James Joyce"]}}\n</tool_call>'
    )

    def test_all_tokens_plus_finish_in_one_batch(self, tokenizer):
        """Entire response + finish in a single process_output call."""
        tcp = FunctionCallParser(tools=TOOLS, tool_call_parser="hermes")
        rp = ReasoningParser(model_type="qwen3", stream_reasoning=True)
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=tcp,
            reasoning_parser=rp,
        )
        token_ids = tokenizer.encode(self.TEXT)
        # Feed ALL tokens at once with finish_reason
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        assert choice is not None
        tc = choice.get("delta", {}).get("tool_calls", [])
        assert len(tc) == 1, f"Expected 1 tool call, got {len(tc)}"
        assert tc[0]["function"]["name"] == "search_gutenberg_books"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"search_terms": ["James Joyce"]}

    def test_multiple_tools_single_chunk(self, tokenizer):
        """Multiple tool calls in one chunk -- re-parse must find all."""
        text = (
            "<think>\nI'll search and check weather.\n</think>\n\n"
            '<tool_call>\n{"name": "search_gutenberg_books", '
            '"arguments": {"search_terms": ["Joyce"]}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "London"}}\n</tool_call>'
        )
        tcp = FunctionCallParser(tools=TOOLS, tool_call_parser="hermes")
        rp = ReasoningParser(model_type="qwen3", stream_reasoning=True)
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=tcp,
            reasoning_parser=rp,
        )
        token_ids = tokenizer.encode(text)
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        assert choice is not None
        tc = choice.get("delta", {}).get("tool_calls", [])
        assert len(tc) == 2, f"Expected 2 tool calls, got {len(tc)}"
        names = {t["function"]["name"] for t in tc}
        assert names == {"search_gutenberg_books", "get_weather"}
        for t in tc:
            args = json.loads(t["function"]["arguments"])
            assert args, f"Arguments should not be empty for {t['function']['name']}"

    def test_finish_reason_rewritten_to_tool_calls(self, tokenizer):
        """finish_reason should be 'tool_calls' when re-parse finds calls."""
        tcp = FunctionCallParser(tools=TOOLS, tool_call_parser="hermes")
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=tcp,
            reasoning_parser=None,
        )
        text = (
            '<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "NYC"}}\n</tool_call>'
        )
        token_ids = tokenizer.encode(text)
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        assert choice is not None
        assert choice["finish_reason"] == "tool_calls"


class TestMalformedToolCalls:
    def test_incomplete_arguments_are_not_emitted(self):
        class DummyTokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return "".join(chr(x) for x in token_ids)

        class DummyToolCall:
            def __init__(self, tool_index, name, parameters):
                self.tool_index = tool_index
                self.name = name
                self.parameters = parameters

        class DummyParser:
            def parse_stream_chunk(self, text):
                return "", [DummyToolCall(0, "get_weather", '{"city": "Paris"')]

            def has_tool_call(self, text):
                return "<tool_call>" in text

            def parse_non_stream(self, text):
                return "", []

        post = SglangStreamingPostProcessor(
            tokenizer=DummyTokenizer(),
            tool_call_parser=DummyParser(),
            reasoning_parser=None,
        )

        malformed = (
            '<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "Paris"}\n</tool_call>'
        )
        choice = post.process_output(
            {
                "token_ids": [ord(c) for c in malformed],
                "finish_reason": "stop",
            }
        )

        assert choice is not None
        assert choice["finish_reason"] == "stop"
        assert choice.get("delta", {}).get("tool_calls", []) == []


# ---------------------------------------------------------------------------
# JsonArrayParser path (tool_choice="required" / named function)
# ---------------------------------------------------------------------------


class TestJsonArrayParserReparse:
    """Exercise the JsonArrayParser branch of the finish-time re-parse.

    Under ``tool_choice="required"`` or a named function, guided decoding
    constrains the model to emit a raw JSON array and
    SglangStreamingPostProcessor is constructed with a JsonArrayParser
    instead of a FunctionCallParser. The re-parse path uses
    ``has_tool_call`` on the parser as a cheap gate and
    ``_parse_json_array_buffer`` for recovery — this class locks in that
    API surface so a SGLang upgrade can't silently break it.
    """

    def test_single_call_reparse(self, tokenizer):
        """Full JSON array arriving in one chunk triggers the re-parse."""
        text = '[{"name": "get_weather", "parameters": {"city": "NYC"}}]'
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=JsonArrayParser(),
            reasoning_parser=None,
            sglang_tools=TOOLS,
        )
        token_ids = tokenizer.encode(text)
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        assert choice is not None
        tc = choice.get("delta", {}).get("tool_calls", [])
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"
        assert json.loads(tc[0]["function"]["arguments"]) == {"city": "NYC"}
        assert choice["finish_reason"] == "tool_calls"

    def test_multiple_calls_reparse(self, tokenizer):
        """Multiple calls in one chunk; re-parse must recover all."""
        text = (
            '[{"name": "search_gutenberg_books", '
            '"parameters": {"search_terms": ["Joyce"]}}, '
            '{"name": "get_weather", "parameters": {"city": "London"}}]'
        )
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=JsonArrayParser(),
            reasoning_parser=None,
            sglang_tools=TOOLS,
        )
        token_ids = tokenizer.encode(text)
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        assert choice is not None
        tc = choice.get("delta", {}).get("tool_calls", [])
        assert len(tc) == 2
        names = {t["function"]["name"] for t in tc}
        assert names == {"search_gutenberg_books", "get_weather"}

    def test_plain_text_skips_reparse(self, tokenizer):
        """Plain text with no JSON markers must not crash the re-parse path.

        Locks in that the ``has_tool_call`` gate on JsonArrayParser returns
        False for text without '[' or '{', so ``_parse_json_array_buffer``
        and the secondary FunctionCallParser fallback are never reached.
        """
        post = SglangStreamingPostProcessor(
            tokenizer=tokenizer,
            tool_call_parser=JsonArrayParser(),
            reasoning_parser=None,
            sglang_tools=TOOLS,
        )
        token_ids = tokenizer.encode("Hello, world!")
        choice = post.process_output({"token_ids": token_ids, "finish_reason": "stop"})
        # No tool calls, plain content preserved, no crash.
        tc = (choice or {}).get("delta", {}).get("tool_calls", [])
        assert tc == []
