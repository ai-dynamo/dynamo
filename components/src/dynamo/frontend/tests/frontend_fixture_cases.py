#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Shared cases + normalize + assert for the FE.process_output parity tests.

The cases are backend-agnostic and defined in code below, composed from shared
hermes tool-call / qwen3 ``<think>`` snippets so a block is written once and
reused. (They were YAML fixtures, but unlike the parser parity corpus these are
homogeneous -- one markup, one tool set, one shared ``expected`` -- so a data
file only added a parse step, a second reader in the dashboard, and verbatim
duplication YAML can't factor out.)

The per-backend adapters (``_vllm_frontend_adapter.py`` /
``_sglang_frontend_adapter.py``) build that backend's StreamingPostProcessor,
replay a case's ``model_text`` at each chunk granularity, and hand the assembled
OpenAI choice dicts here. Both backends emit the same choice shape
(``{delta: {content, reasoning_content, tool_calls}, finish_reason}``), so
normalize/assert is shared rather than duplicated.

Grouped into FE.process_output.4 (assembly, tool parser configured), .6 (detok,
fast plain-text path, no parser), and .9 (reasoning <-> tool orchestration).
This is a *coverage* matrix (write once, both engines must pass), NOT a
behavioral divergence grid -- vLLM/SGLang expose no callable frontend on the
same input, so there is no expected.{vllm,sglang}; per-engine gaps live in each
test's ``_KNOWN_GAPS``.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class FrontendCase:
    case_id: str
    model_text: str
    expected: dict[str, Any]
    batch_sizes: list[int] = field(default_factory=lambda: [20])
    single_chunk: bool = False
    description: str = ""


# --------------------------------------------------------------------------- #
# Shared markup snippets. Hermes tool calls and qwen3 <think> blocks are written
# once here; cases below compose them, so the same block is never repeated.
# --------------------------------------------------------------------------- #

_TC_WEATHER_NYC = (
    "<tool_call>\n"
    '{"name": "get_weather", "arguments": {"city": "NYC"}}\n'
    "</tool_call>"
)
_TC_WEATHER_LONDON = (
    "<tool_call>\n"
    '{"name": "get_weather", "arguments": {"city": "London"}}\n'
    "</tool_call>"
)
_TC_SEARCH_JOYCE = (
    "<tool_call>\n"
    '{"name": "search_books", "arguments": {"q": "Joyce"}}\n'
    "</tool_call>"
)
# Two parallel calls, one newline between them, exactly as a model streams them.
_TC_WEATHER_AND_SEARCH = _TC_WEATHER_LONDON + "\n" + _TC_SEARCH_JOYCE

_THINK_WEATHER = "<think>\nThe user wants the weather, I will call the tool.\n</think>"
_THINK_ANSWER = "<think>\nLet me think about the answer.\n</think>"
_THINK_MULTI = "<think>\nI'll check weather and search books.\n</think>"

# Expected parsed tool calls (the assert contract), reused across cases.
_EXP_WEATHER_NYC = {"name": "get_weather", "arguments": {"city": "NYC"}}
_EXP_WEATHER_LONDON = {"name": "get_weather", "arguments": {"city": "London"}}
_EXP_SEARCH_JOYCE = {"name": "search_books", "arguments": {"q": "Joyce"}}

# Canonical tool schemas; each adapter converts to its engine's tool type.
TOOLS = [
    {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
    {
        "name": "search_books",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
]


# --------------------------------------------------------------------------- #
# FE.process_output.4 — tool-call output assembly (hermes markup, reasoning-free)
# --------------------------------------------------------------------------- #
ASSEMBLY_CASES = [
    FrontendCase(
        "single_tool_call",
        _TC_WEATHER_NYC,
        {
            "tool_calls": [_EXP_WEATHER_NYC],
            "content": "",
            "finish_reason": "tool_calls",
        },
        batch_sizes=[20, 10, 3],
        description="one complete tool call, no narration",
    ),
    FrontendCase(
        "multiple_tool_calls",
        _TC_WEATHER_AND_SEARCH,
        {
            "tool_calls": [_EXP_WEATHER_LONDON, _EXP_SEARCH_JOYCE],
            "finish_reason": "tool_calls",
        },
        batch_sizes=[20, 10],
        description="two parallel tool calls, distinct names",
    ),
    FrontendCase(
        "content_then_tool_call",
        # Realistic token streaming: narration arrives in chunks BEFORE the
        # <tool_call> start. Known limitation (not asserted): when narration and
        # the <tool_call> start land in the SAME delta, the narration is dropped
        # -- prepost.py buffers post-</think> content but not plain pre-tool
        # narration.
        "Let me check that for you.\n" + _TC_WEATHER_NYC,
        {
            "tool_calls": [_EXP_WEATHER_NYC],
            "content_contains": "Let me check that for you.",
            "finish_reason": "tool_calls",
        },
        batch_sizes=[8, 5],
        description="narration before the tool call is preserved as content",
    ),
    FrontendCase(
        "no_tool_calls",
        "It is sunny in NYC today.",
        {
            "tool_calls": [],
            "content_contains": "It is sunny in NYC today.",
            "finish_reason": "stop",
        },
        batch_sizes=[20, 5],
        description="plain text answer, no tool markup -> no tool_calls, content kept",
    ),
    FrontendCase(
        "single_chunk_fallback",
        _TC_WEATHER_NYC,
        {"tool_calls": [_EXP_WEATHER_NYC], "finish_reason": "tool_calls"},
        single_chunk=True,
        description="whole response plus finish arrive in one chunk (non-streaming fallback)",
    ),
]


# --------------------------------------------------------------------------- #
# FE.process_output.6 — incremental detok / fast plain-text path (no parser)
# --------------------------------------------------------------------------- #
DETOK_CASES = [
    FrontendCase(
        "plain_text_single_chunk",
        "It is sunny in NYC today.",
        {
            "tool_calls": [],
            "content": "It is sunny in NYC today.",
            "finish_reason": "stop",
        },
        single_chunk=True,
        description="short plain answer, whole thing in one chunk",
    ),
    FrontendCase(
        "plain_text_streamed",
        "The quick brown fox jumps over the lazy dog.",
        {
            "tool_calls": [],
            "content": "The quick brown fox jumps over the lazy dog.",
            "finish_reason": "stop",
        },
        batch_sizes=[20, 5, 2],
        description="plain answer streamed in small chunks -> content reassembled intact",
    ),
    FrontendCase(
        "plain_text_with_markup_like_chars",
        'Use the format {"city": "NYC"} or a list like [1, 2, 3].',
        {
            "tool_calls": [],
            "content": 'Use the format {"city": "NYC"} or a list like [1, 2, 3].',
            "finish_reason": "stop",
        },
        batch_sizes=[20, 4],
        description="JSON/bracket chars without tool markup stay as content (no spurious parse)",
    ),
]


# --------------------------------------------------------------------------- #
# FE.process_output.9 — reasoning <-> tool-call orchestration
# --------------------------------------------------------------------------- #
REASONING_CASES = [
    FrontendCase(
        "reasoning_then_tool_call",
        _THINK_WEATHER + "\n\n" + _TC_WEATHER_NYC,
        {
            "reasoning_contains": "call the tool",
            "tool_calls": [_EXP_WEATHER_NYC],
            "content": "",
            "finish_reason": "tool_calls",
        },
        batch_sizes=[8, 5],
        description="reasoning block then a tool call -> reasoning_content + tool_calls, content empty",
    ),
    FrontendCase(
        "reasoning_then_text",
        _THINK_ANSWER + "\n\nThe answer is 42.",
        {
            "reasoning_contains": "think about the answer",
            "tool_calls": [],
            "content_contains": "The answer is 42.",
            "finish_reason": "stop",
        },
        batch_sizes=[8, 5],
        description="reasoning block then a plain answer -> reasoning_content + content, no tool_calls",
    ),
    FrontendCase(
        "reasoning_then_multiple_tool_calls",
        _THINK_MULTI + "\n\n" + _TC_WEATHER_AND_SEARCH,
        {
            "reasoning_contains": "check weather",
            "tool_calls": [_EXP_WEATHER_LONDON, _EXP_SEARCH_JOYCE],
            "finish_reason": "tool_calls",
        },
        batch_sizes=[10],
        description="reasoning block then two parallel tool calls",
    ),
]


_CASE_LISTS: dict[str, list[FrontendCase]] = {
    "frontend_assembly": ASSEMBLY_CASES,
    "frontend_detok": DETOK_CASES,
    "frontend_reasoning": REASONING_CASES,
}


def load_tools(name: str = "frontend_assembly") -> list[dict[str, Any]]:
    """Canonical tool schemas; each adapter converts to its engine's type."""
    return TOOLS


def load_cases(name: str = "frontend_assembly") -> list[FrontendCase]:
    """Return the shared, backend-agnostic cases for a fixture group."""
    return _CASE_LISTS[name]


def params(cases, known_gaps: dict | None = None):
    """Build pytest params (one per case x batch size). ``known_gaps`` maps
    (case_id, batch_size) -> reason and marks that combo a strict xfail, so a
    documented backend gap stays green while flipping to a failure the moment
    it is fixed."""
    known_gaps = known_gaps or {}
    out = []
    for case in cases:
        sizes = [None] if case.single_chunk else case.batch_sizes
        for bs in sizes:
            label = "all" if bs is None else bs
            marks = []
            reason = known_gaps.get((case.case_id, bs))
            if reason:
                marks.append(pytest.mark.xfail(reason=reason, strict=True))
            out.append(
                pytest.param(case, bs, id=f"{case.case_id}-{label}", marks=marks)
            )
    return out


def normalize(choices: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse a per-chunk choice stream into a single comparable result:
    ``{tool_calls: [{name, arguments}], content, reasoning_content,
    finish_reason}``. Streaming tool-call deltas sharing an ``index`` are
    merged (name from whichever delta carries it, arguments concatenated) the
    way an OpenAI client does; ``arguments`` is JSON-parsed when possible."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    by_index: dict[int, dict[str, Any]] = {}
    order: list[int] = []

    for choice in choices:
        delta = choice.get("delta", {}) or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
        for tc in delta.get("tool_calls", []) or []:
            idx = tc.get("index", 0)
            fn = tc.get("function", {}) or {}
            if idx not in by_index:
                by_index[idx] = {"name": None, "arguments": ""}
                order.append(idx)
            if fn.get("name"):
                by_index[idx]["name"] = fn["name"]
            if fn.get("arguments"):
                by_index[idx]["arguments"] += fn["arguments"]
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]

    tool_calls: list[dict[str, Any]] = []
    for idx in order:
        raw_args = by_index[idx]["arguments"]
        try:
            args: Any = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError):
            args = raw_args
        tool_calls.append({"name": by_index[idx]["name"], "arguments": args})

    return {
        "tool_calls": tool_calls,
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "finish_reason": finish_reason,
    }


def assert_case(
    result: dict[str, Any], expected: dict[str, Any], *, context: str
) -> None:
    """Assert a normalized result against a case's ``expected`` block."""
    exp_calls = expected.get("tool_calls")
    if exp_calls is not None:
        got = [
            {"name": c["name"], "arguments": c["arguments"]}
            for c in result["tool_calls"]
        ]
        assert (
            got == exp_calls
        ), f"{context}: tool_calls mismatch\n got={got}\n exp={exp_calls}"

    if "content" in expected:
        assert (
            result["content"].strip() == expected["content"].strip()
        ), f"{context}: content mismatch\n got={result['content']!r}\n exp={expected['content']!r}"

    if "content_contains" in expected:
        assert (
            expected["content_contains"] in result["content"]
        ), f"{context}: content missing {expected['content_contains']!r}\n got={result['content']!r}"

    if "reasoning_contains" in expected:
        assert (
            expected["reasoning_contains"] in result["reasoning_content"]
        ), f"{context}: reasoning missing {expected['reasoning_contains']!r}\n got={result['reasoning_content']!r}"

    if "finish_reason" in expected:
        assert (
            result["finish_reason"] == expected["finish_reason"]
        ), f"{context}: finish_reason {result['finish_reason']!r} != {expected['finish_reason']!r}"
