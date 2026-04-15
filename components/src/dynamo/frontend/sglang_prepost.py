#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang.srt.entrypoints.openai.protocol import Function as SglangFunction
from sglang.srt.entrypoints.openai.protocol import Tool as SglangTool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser

from .utils import random_call_id


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


def detect_force_reasoning(tokenizer, prompt_token_ids: list[int]) -> bool:
    """Check if the chat template's generation prompt ends with ``<think>``.

    When the template appends ``<think>`` to the prompt, the model output
    starts inside a reasoning block without an explicit opening tag.
    The reasoning parser must be told to begin in reasoning mode
    (``force_reasoning=True``) so that it correctly separates reasoning
    content from normal content.
    """
    if not prompt_token_ids:
        return False
    tail = tokenizer.decode(prompt_token_ids[-10:], skip_special_tokens=False)
    return tail.rstrip().endswith("<think>")


def create_parsers(
    request: dict[str, Any],
    *,
    tool_call_parser_name: str | None,
    reasoning_parser_name: str | None,
    sglang_tools: list[SglangTool] | None = None,
    force_reasoning: bool = False,
) -> tuple[FunctionCallParser | None, ReasoningParser | None]:
    """Create tool call and reasoning parsers for a request.

    Shared by both the single-process preprocessing path and the pool path
    (which must recreate non-picklable parsers in the main process).

    If ``sglang_tools`` is provided, reuses them; otherwise converts from
    the request's ``tools`` field.
    """
    if sglang_tools is None:
        sglang_tools = convert_tools(request.get("tools"))
    tool_choice = request.get("tool_choice", "auto")

    tool_call_parser = None
    if tool_call_parser_name and sglang_tools and tool_choice != "none":
        tool_call_parser = FunctionCallParser(
            tools=sglang_tools,
            tool_call_parser=tool_call_parser_name,
        )

    reasoning_parser = None
    if reasoning_parser_name:
        kwargs: dict[str, Any] = {
            "model_type": reasoning_parser_name,
            "stream_reasoning": True,
        }
        if force_reasoning:
            kwargs["force_reasoning"] = True
        reasoning_parser = ReasoningParser(**kwargs)

    return tool_call_parser, reasoning_parser


def _normalize_prompt_token_ids(prompt_token_ids: Any) -> list[int]:
    if isinstance(prompt_token_ids, list):
        return prompt_token_ids

    input_ids = getattr(prompt_token_ids, "input_ids", None)
    if input_ids is not None and not isinstance(input_ids, str):
        return list(input_ids)

    if isinstance(prompt_token_ids, dict):
        dict_input_ids = prompt_token_ids.get("input_ids")
        if dict_input_ids is not None and not isinstance(dict_input_ids, str):
            return list(dict_input_ids)

    return list(prompt_token_ids)


def preprocess_chat_request(
    request: dict[str, Any],
    *,
    tokenizer,
    tool_call_parser_name: str | None,
    reasoning_parser_name: str | None,
    exclude_tools_when_tool_choice_none: bool = True,
) -> SglangPreprocessResult:
    """Preprocess a chat request using SGLang tokenizer and parser APIs.

    Synchronous -- suitable for both main-process and worker-process execution.
    """
    messages = _materialize_messages(request.get("messages", []))

    # Convert tools to SGLang format (done once, shared with parser creation)
    sglang_tools = convert_tools(request.get("tools"))

    # Build template kwargs -- single call for rendering + tokenization
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": False,
    }
    # Strip tools from template when tool_choice=none so the model doesn't
    # see them and generate raw XML tool calls in its response.
    tool_choice = request.get("tool_choice", "auto")
    if sglang_tools and not (
        exclude_tools_when_tool_choice_none and tool_choice == "none"
    ):
        template_kwargs["tools"] = [t.model_dump() for t in sglang_tools]

    prompt_token_ids = _normalize_prompt_token_ids(
        tokenizer.apply_chat_template(messages, **template_kwargs)
    )

    force_reasoning = (
        detect_force_reasoning(tokenizer, prompt_token_ids)
        if reasoning_parser_name
        else False
    )

    tool_call_parser, reasoning_parser = create_parsers(
        request,
        tool_call_parser_name=tool_call_parser_name,
        reasoning_parser_name=reasoning_parser_name,
        sglang_tools=sglang_tools,
        force_reasoning=force_reasoning,
    )

    return SglangPreprocessResult(
        prompt_token_ids=prompt_token_ids,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        request=request,
    )


def _random_call_id() -> str:
    return random_call_id()


class SglangStreamingPostProcessor:
    """Streaming post-processor using SGLang parsers and HF tokenizer detokenization.

    Handles:
    - Incremental detokenization via sliding-window decode (6-token lookback)
    - Reasoning content extraction via SGLang ReasoningParser
    - Tool call parsing via SGLang FunctionCallParser (parameter deltas)
    """

    # Lookback window size for incremental detokenization.  UTF-8 characters
    # can span up to 4 bytes, each potentially its own token.  A lookback of
    # 6 covers the worst case (4-token char) plus margin for BPE merges that
    # cross the old/new boundary.
    LOOKBACK = 6

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
        self._fast_plain_text = tool_call_parser is None and reasoning_parser is None

        self._all_token_ids: list[int] = []
        # Tool call accumulation.  SGLang's streaming parser returns
        # deltas (name in one chunk, argument fragments across subsequent
        # chunks).  However, when the complete tool-call JSON arrives in a
        # single chunk the parser emits the name but never streams
        # arguments (a chunking-sensitivity issue in the base detector).
        # We accumulate names + arg fragments from streaming deltas and,
        # on finish, fall back to parse_non_stream on the detector buffer
        # for any tool call whose arguments are still missing.
        self._tool_call_ids: dict[int, str] = {}  # tool_index -> call_id
        self._tool_call_names: dict[int, str] = {}  # tool_index -> name
        self._tool_call_args: dict[int, list[str]] = {}  # tool_index -> arg chunks

    def _incremental_decode(self, new_token_ids: list[int]) -> str:
        """Decode new tokens with lookback window for multi-byte char boundaries.

        Re-decodes a small window of previous tokens alongside new tokens so that
        multi-byte characters spanning token boundaries are correctly resolved.
        Only retains the last LOOKBACK tokens to bound memory usage.
        """
        prev_count = len(self._all_token_ids)
        self._all_token_ids.extend(new_token_ids)

        start = max(0, prev_count - self.LOOKBACK)

        # Trim to avoid unbounded growth -- only the tail matters for decoding
        if len(self._all_token_ids) > self.LOOKBACK * 16:
            self._all_token_ids = self._all_token_ids[
                -(self.LOOKBACK + len(new_token_ids)) :
            ]
            prev_count = len(self._all_token_ids) - len(new_token_ids)
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

        return window_text[len(prefix_text) :]

    def process_output(self, engine_response: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single engine response chunk into an OpenAI SSE choice dict.

        Args:
            engine_response: Dict with ``token_ids`` and optional ``finish_reason``.

        Returns:
            OpenAI choice dict or ``None`` if nothing to emit yet.
        """
        raw_ids = engine_response.get("token_ids")
        token_ids = raw_ids if isinstance(raw_ids, list) else list(raw_ids or [])
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

        # -- Tool call parsing (accumulate deltas) --
        content_text = normal_text

        if self.tool_call_parser and normal_text:
            parsed_text, tool_calls = self.tool_call_parser.parse_stream_chunk(
                normal_text
            )
            content_text = parsed_text

            for tc in tool_calls:
                idx = tc.tool_index
                if idx not in self._tool_call_ids:
                    self._tool_call_ids[idx] = _random_call_id()
                if tc.name:
                    self._tool_call_names[idx] = tc.name
                if tc.parameters:
                    self._tool_call_args.setdefault(idx, []).append(tc.parameters)

        # -- Assemble delta --
        delta: dict[str, Any] = {"role": "assistant"}
        has_content = False

        if content_text:
            delta["content"] = content_text
            has_content = True
        if reasoning_text:
            delta["reasoning_content"] = reasoning_text
            has_content = True

        # Emit complete tool calls on finish.  For any tool call whose
        # arguments are still empty (chunking-sensitivity issue), fall
        # back to parse_non_stream on the detector's buffer.
        if finish_reason and self._tool_call_names:
            missing_args = any(
                idx not in self._tool_call_args for idx in self._tool_call_names
            )
            if missing_args and self.tool_call_parser is not None:
                buffer = getattr(self.tool_call_parser.detector, "_buffer", "")
                if buffer:
                    _, final_calls = self.tool_call_parser.parse_non_stream(buffer)
                    for tc in final_calls:
                        idx = tc.tool_index
                        if idx not in self._tool_call_ids:
                            self._tool_call_ids[idx] = _random_call_id()
                        if tc.name:
                            self._tool_call_names[idx] = tc.name
                        if tc.parameters:
                            self._tool_call_args[idx] = [tc.parameters]

            tool_calls_out: list[dict[str, Any]] = []
            for idx in sorted(self._tool_call_names):
                tool_calls_out.append(
                    {
                        "index": idx,
                        "id": self._tool_call_ids[idx],
                        "type": "function",
                        "function": {
                            "name": self._tool_call_names[idx],
                            "arguments": "".join(self._tool_call_args.get(idx, [])),
                        },
                    }
                )
            delta["tool_calls"] = tool_calls_out
            has_content = True

        if has_content or finish_reason:
            return {
                "index": 0,
                "delta": delta if has_content else {},
                "finish_reason": finish_reason,
                "logprobs": None,
            }

        return None
