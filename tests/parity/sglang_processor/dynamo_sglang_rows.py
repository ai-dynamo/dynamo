# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay conformance text chunks through Dynamo's SGLang chat processor.

The rows match the per-chunk delta rows of the frontend-crates SGLang stream
harness (``conformance/utils/src/capture_sglang_unified.py``). Each row lists
``reasoning``, ``text`` and ``tool_call`` deltas.

The component under test is production code: ``create_parsers`` and
``SglangStreamingPostProcessor`` from ``dynamo.frontend.sglang_prepost``. Each
chunk enters ``SglangStreamingPostProcessor._process_text``, the step after
detokenization. This text-injection boundary bypasses incremental
detokenization, stop-token and stop-string filtering, token-id delimiter gating
(no tokenizer), logprobs, and the Rust HTTP/SSE projection.

An OpenAI delta has no order between its fields, so a row lists reasoning, then
text, then tool calls, which is the order the native harness uses.
"""

from __future__ import annotations

from typing import Any

from dynamo.frontend import sglang_prepost


class TextOnlyTokenizer:
    """Tokenizer stand-in without a vocabulary.

    No delimiter encodes to a single token, so ``tool_call_marker_token_ids``
    returns ``{}`` and the post-processor uses its text-only parser path, which
    is the boundary this replay exercises. Decoding is never reached from
    ``_process_text``.
    """

    def encode(self, *_args: Any, **_kwargs: Any) -> list[int]:
        return []

    def decode(self, *_args: Any, **_kwargs: Any) -> str:
        raise NotImplementedError("text-injection replay does not decode token ids")


def rows_from_output(out: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Convert one post-processor chunk into harness delta rows."""
    if not out:
        return []
    delta = out.get("delta") or {}
    rows: list[dict[str, Any]] = []
    if delta.get("reasoning_content"):
        rows.append({"kind": "reasoning", "text": delta["reasoning_content"]})
    if delta.get("content"):
        rows.append({"kind": "text", "text": delta["content"]})
    for call in delta.get("tool_calls") or []:
        function = call.get("function") or {}
        rows.append(
            {
                "kind": "tool_call",
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        )
    return rows


def stream_rows(
    *,
    reasoning_parser_name: str | None,
    tool_call_parser_name: str | None,
    chunks: list[str],
    tools: list[dict[str, Any]],
    finish_reason: str = "stop",
    terminal_step: bool = False,
    rows: list[list[dict[str, Any]]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Replay *chunks* and return one delta-row list per chunk.

    The final chunk carries *finish_reason*, as the last token batch does in
    production. With *terminal_step*, an empty finishing chunk follows the text,
    matching the harness's authored terminal row. Rows are appended to *rows*
    as they are produced, so a caller keeps partial output when a chunk raises.
    """
    request = {
        "tools": [{"type": "function", "function": tool} for tool in tools],
        "tool_choice": "auto",
    }
    sglang_tools = sglang_prepost.convert_tools(request["tools"])
    # A native/default request has no template forcing reasoning, matching the
    # native harness's force_reasoning=False.
    tool_parser, reasoning_parser = sglang_prepost.create_parsers(
        request,
        tool_call_parser_name=tool_call_parser_name,
        reasoning_parser_name=reasoning_parser_name,
        sglang_tools=sglang_tools,
        force_reasoning=False,
    )
    processor = sglang_prepost.SglangStreamingPostProcessor(
        tokenizer=TextOnlyTokenizer(),
        tool_call_parser=tool_parser,
        reasoning_parser=reasoning_parser,
        sglang_tools=sglang_tools,
        tool_call_parser_name=tool_call_parser_name,
    )
    rows = [] if rows is None else rows
    schedule = list(chunks) + ([""] if terminal_step else [])
    for index, text in enumerate(schedule):
        final = index == len(schedule) - 1
        out = processor._process_text(text, finish_reason if final else None, [])
        rows.append(rows_from_output(out))
    return rows
