#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sglang.srt.entrypoints.openai.protocol import Function as SglangFunction
from sglang.srt.entrypoints.openai.protocol import Tool as SglangTool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser

_MASK_64_BITS = (1 << 64) - 1


@dataclass
class SglangPreprocessResult:
    """Result of SGLang preprocessing."""

    prompt_token_ids: list[int]
    tool_call_parser: FunctionCallParser | None
    reasoning_parser: ReasoningParser | None
    request: dict[str, Any]


def convert_tools(tools: list[dict[str, Any]] | None) -> list[SglangTool] | None:
    """Convert OpenAI tool dicts to SGLang Tool objects."""
    if not tools:
        return None
    sglang_tools = []
    for tool in tools:
        func = tool.get("function", {})
        sglang_tools.append(
            SglangTool(
                type=tool.get("type", "function"),
                function=SglangFunction(
                    name=func.get("name", ""),
                    description=func.get("description"),
                    parameters=func.get("parameters"),
                    strict=func.get("strict", False),
                ),
            )
        )
    return sglang_tools


def _materialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert message objects to plain dicts for apply_chat_template."""
    normalized = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            normalized.append(msg.model_dump(exclude_none=False))
        elif isinstance(msg, dict):
            normalized.append(msg)
        else:
            normalized.append(dict(msg))
    return normalized


def preprocess_chat_request(
    request: dict[str, Any],
    *,
    tokenizer,
    tool_call_parser_name: str | None,
    reasoning_parser_name: str | None,
) -> SglangPreprocessResult:
    """Preprocess a chat request using SGLang tokenizer and parser APIs.

    Synchronous -- suitable for both main-process and worker-process execution.
    """
    messages = _materialize_messages(request.get("messages", []))
    tools = request.get("tools")
    tool_choice = request.get("tool_choice", "auto")

    # Convert tools to SGLang format
    sglang_tools = convert_tools(tools)

    # Build template kwargs -- single call for rendering + tokenization
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
    }
    if sglang_tools:
        template_kwargs["tools"] = [t.model_dump() for t in sglang_tools]

    prompt_token_ids = tokenizer.apply_chat_template(messages, **template_kwargs)
    if not isinstance(prompt_token_ids, list):
        prompt_token_ids = list(prompt_token_ids)

    # Tool call parser
    tool_call_parser = None
    if tool_call_parser_name and sglang_tools and tool_choice != "none":
        tool_call_parser = FunctionCallParser(
            tools=sglang_tools,
            tool_call_parser=tool_call_parser_name,
        )

    # Reasoning parser
    reasoning_parser = None
    if reasoning_parser_name:
        reasoning_parser = ReasoningParser(
            model_type=reasoning_parser_name,
            stream_reasoning=True,
        )

    return SglangPreprocessResult(
        prompt_token_ids=prompt_token_ids,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        request=request,
    )


def _random_call_id() -> str:
    return f"call_{uuid.uuid4().int & _MASK_64_BITS:016x}"


class SglangStreamingPostProcessor:
    """Streaming post-processor using SGLang parsers and HF tokenizer detokenization.

    Handles:
    - Incremental detokenization via sliding-window decode (6-token lookback)
    - Reasoning content extraction via SGLang ReasoningParser
    - Tool call parsing via SGLang FunctionCallParser with parameter diffing
    """

    LOOKBACK = 6  # token lookback for multi-byte character boundary handling

    def __init__(
        self,
        *,
        tokenizer,
        tool_call_parser: FunctionCallParser | None,
        reasoning_parser: ReasoningParser | None,
    ) -> None:
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self._fast_plain_text = (
            tool_call_parser is None and reasoning_parser is None
        )

        self._all_token_ids: list[int] = []
        # Tool call state tracking
        self._tool_call_ids: dict[int, str] = {}  # tool_index -> call_id
        self._prev_tool_params: dict[int, str] = {}  # tool_index -> accumulated params

    def _incremental_decode(self, new_token_ids: list[int]) -> str:
        """Decode new tokens with lookback window for multi-byte char boundaries.

        Re-decodes a small window of previous tokens alongside new tokens so that
        multi-byte characters spanning token boundaries are correctly resolved.
        """
        prev_count = len(self._all_token_ids)
        self._all_token_ids.extend(new_token_ids)

        start = max(0, prev_count - self.LOOKBACK)

        # Decode lookback-only prefix (before new tokens)
        prefix_tokens = self._all_token_ids[start:prev_count]
        prefix_text = (
            self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)
            if prefix_tokens
            else ""
        )

        # Decode lookback + new tokens together
        window_tokens = self._all_token_ids[start:]
        window_text = self.tokenizer.decode(window_tokens, skip_special_tokens=True)

        return window_text[len(prefix_text):]

    def process_output(self, engine_response: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single engine response chunk into an OpenAI SSE choice dict.

        Args:
            engine_response: Dict with ``token_ids`` and optional ``finish_reason``.

        Returns:
            OpenAI choice dict or ``None`` if nothing to emit yet.
        """
        token_ids = list(engine_response.get("token_ids") or [])
        finish_reason = engine_response.get("finish_reason")

        delta_text = self._incremental_decode(token_ids) if token_ids else ""

        if self._fast_plain_text:
            if delta_text:
                return {
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta_text},
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            elif finish_reason:
                return {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            return None

        # -- Reasoning parsing --
        reasoning_text = None
        normal_text = delta_text

        if self.reasoning_parser and delta_text:
            r_text, n_text = self.reasoning_parser.parse_stream_chunk(delta_text)
            reasoning_text = r_text or None
            normal_text = n_text or ""

        # -- Tool call parsing --
        tool_call_deltas: list[dict[str, Any]] = []
        content_text = normal_text

        if self.tool_call_parser and normal_text:
            parsed_text, tool_calls = self.tool_call_parser.parse_stream_chunk(
                normal_text
            )
            content_text = parsed_text

            for tc in tool_calls:
                idx = tc.tool_index
                # Generate call_id for new tool calls
                if idx not in self._tool_call_ids:
                    self._tool_call_ids[idx] = _random_call_id()

                # Compute parameter delta (SGLang returns accumulated params)
                prev_params = self._prev_tool_params.get(idx, "")
                current_params = tc.parameters or ""
                param_delta = current_params[len(prev_params):]
                self._prev_tool_params[idx] = current_params

                tc_delta: dict[str, Any] = {
                    "index": idx,
                    "id": self._tool_call_ids[idx],
                    "type": "function",
                    "function": {},
                }
                if tc.name:
                    tc_delta["function"]["name"] = tc.name
                if param_delta:
                    tc_delta["function"]["arguments"] = param_delta

                tool_call_deltas.append(tc_delta)

        # -- Assemble delta --
        delta: dict[str, Any] = {"role": "assistant"}
        has_content = False

        if content_text:
            delta["content"] = content_text
            has_content = True
        if reasoning_text:
            delta["reasoning_content"] = reasoning_text
            has_content = True
        if tool_call_deltas:
            delta["tool_calls"] = tool_call_deltas
            has_content = True

        if has_content or finish_reason:
            return {
                "index": 0,
                "delta": delta if has_content else {},
                "finish_reason": finish_reason,
                "logprobs": None,
            }

        return None
