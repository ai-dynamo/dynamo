# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-level postprocessor that uses SGLang's native parsers.

When --postprocessing sglang is set, this converts raw engine output
into GenerateChunk dicts with text, tool_calls, reasoning_content --
matching sglang.launch_server behavior.

Note: Dynamo forces incremental_streaming_output=True on SGLang, so
the ``text`` field in each engine chunk is an **incremental delta**
(not cumulative). We accumulate the full text ourselves for the
finish-time re-parse of tool calls.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from dynamo.common.backend.engine import GenerateChunk

logger = logging.getLogger(__name__)


class SglangPostProcessor:
    """Converts raw SGLang engine streaming output to OpenAI-compatible GenerateChunks.

    Mirrors the postprocessing logic in sglang.srt.entrypoints.openai.serving_chat
    but operates at the worker level, yielding GenerateChunk dicts instead of
    HTTP SSE frames.
    """

    def __init__(
        self,
        *,
        tool_call_parser_name: Optional[str] = None,
        reasoning_parser_name: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ):
        self._tool_call_parser = None
        self._reasoning_parser = None
        # Accumulated full text per index for finish-time re-parse
        self._accumulated_text: dict[int, str] = {}
        self._has_tool_calls: dict[int, bool] = {}

        if tool_call_parser_name and tools:
            from sglang.srt.entrypoints.openai.protocol import Tool as SglangTool
            from sglang.srt.function_call.function_call_parser import FunctionCallParser

            sglang_tools = [SglangTool(**t) for t in tools]
            self._tool_call_parser = FunctionCallParser(
                tools=sglang_tools,
                tool_call_parser=tool_call_parser_name,
            )

        if reasoning_parser_name:
            from sglang.srt.parser.reasoning_parser import ReasoningParser

            self._reasoning_parser = ReasoningParser(
                model_type=reasoning_parser_name,
                stream_reasoning=True,
            )

    def process_chunk(
        self, raw: dict[str, Any], is_final: bool
    ) -> Optional[GenerateChunk]:
        """Process a raw SGLang engine chunk into a GenerateChunk.

        Args:
            raw: Dict from engine.async_generate() with keys:
                 text (incremental delta), output_ids, meta_info, index.
                 Note: Dynamo sets incremental_streaming_output=True, so
                 ``text`` is a per-chunk delta, NOT cumulative.
            is_final: Whether this is the final chunk (finish_reason set)

        Returns:
            GenerateChunk or None if no content to emit this iteration.
        """
        index = raw.get("index") or 0
        meta_info = raw["meta_info"]

        # text is an incremental delta (not cumulative) because Dynamo
        # forces incremental_streaming_output=True on SGLang.
        delta = raw.get("text", "")

        # Accumulate for finish-time re-parse
        if delta:
            self._accumulated_text[index] = (
                self._accumulated_text.get(index, "") + delta
            )

        if not delta and not is_final:
            return None

        reasoning_text = None
        content_text = delta
        tool_call_deltas: list[dict[str, Any]] = []

        # Reasoning parsing
        if self._reasoning_parser and delta:
            r_text, n_text = self._reasoning_parser.parse_stream_chunk(delta)
            reasoning_text = r_text
            content_text = n_text if n_text else ""

        # Tool call parsing
        if self._tool_call_parser and content_text:
            normal_text, tool_items = self._tool_call_parser.parse_stream_chunk(
                content_text
            )
            content_text = normal_text

            for item in tool_items:
                tc: dict[str, Any] = {
                    "index": item.tool_index,
                    "type": "function",
                    "function": {"arguments": item.parameters},
                }
                if item.name is not None:
                    tc["id"] = f"call_{item.tool_index}"
                    tc["function"]["name"] = item.name
                tool_call_deltas.append(tc)
                self._has_tool_calls[index] = True

        # On finish: re-parse full accumulated text for missed tool calls
        if is_final and self._tool_call_parser:
            full_text = self._accumulated_text.get(index, "")
            if self._reasoning_parser:
                r, n = self._reasoning_parser.parse_non_stream(full_text)
                full_text = n if n else ""
            _, final_tools = self._tool_call_parser.parse_non_stream(full_text)
            if final_tools and not tool_call_deltas:
                for item in final_tools:
                    tc = {
                        "index": item.tool_index,
                        "type": "function",
                        "id": f"call_{item.tool_index}",
                        "function": {
                            "name": item.name,
                            "arguments": item.parameters,
                        },
                    }
                    tool_call_deltas.append(tc)
                    self._has_tool_calls[index] = True

        # Build chunk
        chunk: GenerateChunk = {"index": index}

        if content_text:
            chunk["text"] = content_text
        if reasoning_text:
            chunk["reasoning_content"] = reasoning_text
        if tool_call_deltas:
            chunk["tool_calls"] = tool_call_deltas

        if is_final:
            finish_reason = meta_info["finish_reason"]
            fr_type = (
                finish_reason["type"]
                if isinstance(finish_reason, dict)
                else finish_reason
            )
            if fr_type == "stop" and self._has_tool_calls.get(index, False):
                fr_type = "tool_calls"
            chunk["finish_reason"] = fr_type

            prompt_tokens = meta_info.get("prompt_tokens", 0)
            completion_tokens = meta_info.get("completion_tokens", 0)
            chunk["completion_usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        has_content = content_text or reasoning_text or tool_call_deltas
        if has_content or is_final:
            return chunk
        return None

    def reset(self) -> None:
        """Reset state for a new request."""
        self._accumulated_text.clear()
        self._has_tool_calls.clear()
        if self._reasoning_parser:
            from sglang.srt.parser.reasoning_parser import ReasoningParser

            self._reasoning_parser = ReasoningParser(
                model_type=self._reasoning_parser.model_type,
                stream_reasoning=True,
            )
        if self._tool_call_parser:
            from sglang.srt.function_call.function_call_parser import FunctionCallParser

            self._tool_call_parser = FunctionCallParser(
                tools=self._tool_call_parser.tools,
                tool_call_parser=self._tool_call_parser.tool_call_parser,
            )
