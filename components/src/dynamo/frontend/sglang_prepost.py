#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, TypeAlias

from sglang.srt.entrypoints.openai.protocol import Function as SglangFunction
from sglang.srt.entrypoints.openai.protocol import Tool as SglangTool
from sglang.srt.entrypoints.openai.protocol import ToolChoice as SglangToolChoice
from sglang.srt.entrypoints.openai.protocol import (
    ToolChoiceFuncName as SglangToolChoiceFuncName,
)
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.utils import get_json_schema_constraint
from sglang.srt.parser.reasoning_parser import ReasoningParser

from .utils import random_call_id

logger = logging.getLogger(__name__)

# Union of parser types used for tool call detection.
# - FunctionCallParser: model-specific format detection (tool_choice="auto")
# - JsonArrayParser: direct JSON array parsing under constrained decoding
#   (tool_choice="required" or named function)
ToolCallParserType: TypeAlias = FunctionCallParser | JsonArrayParser


@dataclass
class SglangPreprocessResult:
    """Result of SGLang preprocessing."""

    prompt_token_ids: list[int]
    tool_call_parser: ToolCallParserType | None
    reasoning_parser: ReasoningParser | None
    guided_decoding: dict[str, Any] | None
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


def create_parsers(
    request: dict[str, Any],
    *,
    tool_call_parser_name: str | None,
    reasoning_parser_name: str | None,
    sglang_tools: list[SglangTool] | None = None,
) -> tuple[ToolCallParserType | None, ReasoningParser | None]:
    """Create tool call and reasoning parsers for a request.

    Shared by both the single-process preprocessing path and the pool path
    (which must recreate non-picklable parsers in the main process).

    If ``sglang_tools`` is provided, reuses them; otherwise converts from
    the request's ``tools`` field.

    For ``tool_choice="required"`` or a named function, uses
    :class:`JsonArrayParser` (matching native SGLang) since guided decoding
    constrains the output to a JSON array.  Otherwise uses the model-specific
    :class:`FunctionCallParser`.
    """
    if sglang_tools is None:
        sglang_tools = convert_tools(request.get("tools"))
    tool_choice = request.get("tool_choice", "auto")

    tool_call_parser: ToolCallParserType | None = None
    if sglang_tools and tool_choice != "none":
        if tool_choice == "required" or _is_named_tool_choice(tool_choice):
            tool_call_parser = JsonArrayParser()
        elif tool_call_parser_name:
            tool_call_parser = FunctionCallParser(
                tools=sglang_tools,
                tool_call_parser=tool_call_parser_name,
            )

    reasoning_parser = None
    if reasoning_parser_name:
        reasoning_parser = ReasoningParser(
            model_type=reasoning_parser_name,
            stream_reasoning=True,
        )

    return tool_call_parser, reasoning_parser


def _is_named_tool_choice(tool_choice: Any) -> bool:
    return (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and isinstance(tool_choice.get("function"), dict)
        and bool(tool_choice["function"].get("name"))
    )


def build_tool_call_guided_decoding(
    request: dict[str, Any],
    *,
    tool_call_parser_name: str | None,
    sglang_tools: list[SglangTool] | None,
) -> dict[str, Any] | None:
    """Build native-SGLang-like tool call constraints for guided decoding."""
    if not sglang_tools:
        return None

    tool_choice = request.get("tool_choice", "auto")
    if tool_choice == "none":
        return None

    parallel_tool_calls = request.get("parallel_tool_calls")
    constraint: Any = None

    if tool_choice == "required" or _is_named_tool_choice(tool_choice):
        # get_json_schema_constraint branches on isinstance(tool_choice,
        # ToolChoice) for the named-function case — passing our raw dict
        # would silently fall through and return None, disabling guided
        # decoding and letting the model omit required fields.
        sglang_tool_choice: Any = tool_choice
        if _is_named_tool_choice(tool_choice):
            sglang_tool_choice = SglangToolChoice(
                type="function",
                function=SglangToolChoiceFuncName(
                    name=tool_choice["function"]["name"],
                ),
            )
        constraint = (
            "json_schema",
            get_json_schema_constraint(
                sglang_tools,
                sglang_tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            ),
        )
    elif tool_call_parser_name:
        parser = FunctionCallParser(
            tools=sglang_tools,
            tool_call_parser=tool_call_parser_name,
        )
        constraint = parser.get_structure_constraint(
            tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

    if isinstance(constraint, tuple) and len(constraint) == 2:
        if constraint[0] == "json_schema":
            return {"json": constraint[1]}
        if constraint[0] == "structural_tag":
            tag_value = constraint[1]
            # SGLang returns a Pydantic model (LegacyStructuralTagResponseFormat)
            # here.  Convert to a plain dict before it hits the RPC layer —
            # msgpack/serde_json cannot serialize BaseModel instances.
            if hasattr(tag_value, "model_dump"):
                tag_value = tag_value.model_dump()
            return {"structural_tag": tag_value}

    return None


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

    # Reject a named tool_choice whose function is missing from tools —
    # otherwise the chat template would render with zero tools while
    # guided decoding still constrains the output to that function's
    # schema, producing confusing model behavior.
    tool_choice = request.get("tool_choice", "auto")
    if _is_named_tool_choice(tool_choice):
        chosen_name = tool_choice["function"]["name"]
        available_names = {t.function.name for t in (sglang_tools or [])}
        if chosen_name not in available_names:
            raise ValueError(
                f"tool_choice names function {chosen_name!r}, but it is not "
                f"present in tools (available: {sorted(available_names) or 'none'})"
            )

    # Build template kwargs -- single call for rendering + tokenization
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
    }
    # Strip tools from template when tool_choice=none so the model doesn't
    # see them and generate raw XML tool calls in its response.
    # When tool_choice names a specific function, only include that tool
    # in the template so the model doesn't see irrelevant definitions.
    if sglang_tools and not (
        exclude_tools_when_tool_choice_none and tool_choice == "none"
    ):
        if _is_named_tool_choice(tool_choice):
            chosen_name = tool_choice["function"]["name"]
            template_kwargs["tools"] = [
                t.model_dump() for t in sglang_tools if t.function.name == chosen_name
            ]
        else:
            template_kwargs["tools"] = [t.model_dump() for t in sglang_tools]

    prompt_token_ids = _normalize_prompt_token_ids(
        tokenizer.apply_chat_template(messages, **template_kwargs)
    )

    tool_call_parser, reasoning_parser = create_parsers(
        request,
        tool_call_parser_name=tool_call_parser_name,
        reasoning_parser_name=reasoning_parser_name,
        sglang_tools=sglang_tools,
    )
    guided_decoding = build_tool_call_guided_decoding(
        request,
        tool_call_parser_name=tool_call_parser_name,
        sglang_tools=sglang_tools,
    )

    return SglangPreprocessResult(
        prompt_token_ids=prompt_token_ids,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        guided_decoding=guided_decoding,
        request=request,
    )


def _random_call_id() -> str:
    return random_call_id()


def _get_history_tool_calls_count(messages: list[dict[str, Any]]) -> int:
    """Count prior assistant tool calls for parser-specific ID generation."""
    count = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            count += len(tool_calls)
    return count


def _tool_call_id_for_parser(
    parser_name: str | None,
    name: str,
    index: int,
    history_tool_calls_count: int,
) -> str:
    """Match native SGLang tool-call ID behavior for parser-specific formats.

    ``index`` is the sequential position of this call within the current
    response — callers must pass the same index they use as the dict key
    for the call, so the ID stays consistent with the emitted ``index``
    field.  For ``parse_non_stream`` output, ``ToolCallItem.tool_index``
    can instead reflect the tool-definition position, so it is not safe
    to read here directly.
    """
    if parser_name != "kimi_k2":
        return _random_call_id()
    return f"functions.{name or ''}:{history_tool_calls_count + index}"


def _parse_json_array_buffer(buffer: str) -> list[ToolCallItem]:
    """Parse a JSON array buffer from constrained decoding into ToolCallItems.

    Used as the fallback when JsonArrayParser's streaming parsing missed
    arguments (same chunking-sensitivity issue as FunctionCallParser).
    Mirrors SGLang native's ``orjson.loads`` path in ``_process_tool_calls``.

    The buffer may contain trailing special tokens (e.g. ``<|endoftext|>``)
    from incremental detokenization with ``skip_special_tokens=False``.
    If the full buffer is not valid JSON, we extract the substring between
    the first ``[`` and last ``]`` and retry.
    """
    data = _try_parse_json_array(buffer)
    if data is None:
        return []
    calls: list[ToolCallItem] = []
    for i, tool in enumerate(data):
        if not isinstance(tool, dict):
            continue
        name = tool.get("name", "")
        params = tool.get("parameters")
        if params is None:
            params = tool.get("arguments")
        if params is not None and not isinstance(params, str):
            params = json.dumps(params, ensure_ascii=False)
        calls.append(
            ToolCallItem(
                tool_index=i,
                name=name,
                parameters=params if params is not None else "",
            )
        )
    return calls


def _try_parse_json_array(text: str) -> list | None:
    """Try to parse a JSON array from *text*, tolerating surrounding noise."""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    # Retry: extract the outermost [...] substring (handles trailing
    # special tokens or leading content text).
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
    return None


class SglangStreamingPostProcessor:
    """Streaming post-processor using SGLang parsers and HF tokenizer detokenization.

    Handles:
    - Incremental detokenization via sliding-window decode (6-token lookback)
    - Reasoning content extraction via SGLang ReasoningParser
    - Tool call parsing via SGLang FunctionCallParser or JsonArrayParser
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
        tool_call_parser: ToolCallParserType | None,
        reasoning_parser: ReasoningParser | None,
        history_tool_calls_count: int = 0,
        sglang_tools: list[SglangTool] | None = None,
        tool_call_parser_name: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.history_tool_calls_count = history_tool_calls_count
        self._sglang_tools = sglang_tools or []
        self._tool_call_parser_name = tool_call_parser_name
        self._fast_plain_text = tool_call_parser is None and reasoning_parser is None
        # Preserve special tokens when a tool call parser is active so
        # delimiter tokens (e.g. <|tool_call|>) remain visible to the parser.
        self._skip_special_tokens = tool_call_parser is None
        self._is_json_array_parser = isinstance(tool_call_parser, JsonArrayParser)

        self._all_token_ids: list[int] = []
        # Tool call accumulation.  SGLang's streaming parser returns
        # deltas (name in one chunk, argument fragments across subsequent
        # chunks).  However, the base detector processes at most one event
        # per call (a name OR an argument diff), and the post-processor
        # calls it only once per token batch.  When multiple tool calls
        # arrive together, later calls may not be detected during streaming.
        # We accumulate all text fed to the parser and, on finish, re-parse
        # the full text to recover any missed tool calls or arguments.
        self._tool_call_ids: dict[int, str] = {}  # tool_index -> call_id
        self._tool_call_names: dict[int, str] = {}  # tool_index -> name
        self._tool_call_args: dict[int, list[str]] = {}  # tool_index -> arg chunks
        # Full text accumulator for robust finish-time re-parse.
        self._tool_text_parts: list[str] = []

    def _tool_call_id(self, name: str, index: int) -> str:
        return _tool_call_id_for_parser(
            self._tool_call_parser_name,
            name,
            index,
            self.history_tool_calls_count,
        )

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
            self.tokenizer.decode(
                prefix_tokens, skip_special_tokens=self._skip_special_tokens
            )
            if prefix_tokens
            else ""
        )

        # Decode lookback + new tokens together
        window_tokens = self._all_token_ids[start:]
        window_text = self.tokenizer.decode(
            window_tokens, skip_special_tokens=self._skip_special_tokens
        )

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
            # Accumulate raw text for finish-time re-parse.
            self._tool_text_parts.append(normal_text)

            if self._is_json_array_parser:
                result = self.tool_call_parser.parse_streaming_increment(
                    normal_text, self._sglang_tools
                )
                parsed_text, tool_calls = result.normal_text, result.calls
            else:
                parsed_text, tool_calls = self.tool_call_parser.parse_stream_chunk(
                    normal_text
                )
            content_text = parsed_text

            for tc in tool_calls:
                idx = tc.tool_index
                if idx not in self._tool_call_ids:
                    self._tool_call_ids[idx] = self._tool_call_id(tc.name or "", idx)
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

        # On finish, re-parse the full accumulated text to recover tool
        # calls or arguments that the streaming parser missed.
        #
        # The streaming parser (BaseFormatDetector.parse_streaming_increment)
        # processes at most one event per invocation — a tool name OR an
        # argument diff — and the post-processor calls it once per token
        # batch.  When multiple tool calls arrive together or the complete
        # JSON lands in a single chunk, later calls (or arguments) may
        # never be detected during streaming.
        #
        # The re-parse uses the accumulated text (not the parser's internal
        # _buffer, which is consumed during streaming) and assigns
        # sequential indices to match the OpenAI API convention.
        if (
            finish_reason
            and self.tool_call_parser is not None
            and self._tool_text_parts
        ):
            # Purge streaming results that don't match any known tool.
            # When guided decoding is not enforced the streaming parser
            # can misidentify words in the prompt (e.g. a person's name)
            # as function names.
            known_names = (
                {t.function.name for t in self._sglang_tools}
                if self._sglang_tools
                else set()
            )
            if known_names:
                for idx in list(self._tool_call_names):
                    if self._tool_call_names[idx] not in known_names:
                        del self._tool_call_names[idx]
                        self._tool_call_ids.pop(idx, None)
                        self._tool_call_args.pop(idx, None)

            # Discard malformed (non-JSON) argument fragments that the
            # streaming parser accumulated from mixed content.
            for idx in list(self._tool_call_args):
                combined = "".join(self._tool_call_args[idx])
                if combined:
                    try:
                        json.loads(combined)
                    except (json.JSONDecodeError, ValueError):
                        del self._tool_call_args[idx]

            missing_names = not self._tool_call_names
            missing_args = any(
                idx not in self._tool_call_args for idx in self._tool_call_names
            )
            should_reparse = False
            full_text = ""
            if missing_names or missing_args:
                full_text = "".join(self._tool_text_parts)
                # Skip the re-parse when the accumulated text has no
                # tool-call markers.  Avoids wasted `parse_non_stream`
                # work on plain-text responses (common when tools are
                # offered but the model replies without calling any) and
                # guards against detectors that raise on arbitrary input.
                should_reparse = bool(
                    full_text
                ) and self.tool_call_parser.has_tool_call(full_text)

            if should_reparse:
                if self._is_json_array_parser:
                    final_calls = _parse_json_array_buffer(full_text)
                    # Secondary fallback: when guided decoding did not
                    # constrain the output (e.g. the backend doesn't
                    # support it), the model may have produced tool calls
                    # in its native format.  Try the model-specific
                    # parser so we don't silently drop them.
                    if (
                        not final_calls
                        and self._tool_call_parser_name
                        and self._sglang_tools
                    ):
                        try:
                            fcp = FunctionCallParser(
                                tools=self._sglang_tools,
                                tool_call_parser=self._tool_call_parser_name,
                            )
                            _, final_calls = fcp.parse_non_stream(full_text)
                        except (
                            ValueError,
                            KeyError,
                            json.JSONDecodeError,
                            IndexError,
                        ) as e:
                            # Fallback path: model-native tool-call text is
                            # malformed. Log and return no tool calls rather
                            # than crashing the whole response — the primary
                            # JSON-array path has already failed, and the
                            # normal text is still usable.
                            logger.warning(
                                "Native tool-call fallback parse failed (parser=%r): %s",
                                self._tool_call_parser_name,
                                e,
                            )
                            final_calls = []
                else:
                    _, final_calls = self.tool_call_parser.parse_non_stream(full_text)
                # Filter to known tool names (reuse set from above).
                if known_names:
                    final_calls = [tc for tc in final_calls if tc.name in known_names]
                # Re-index sequentially so repeated calls to the same
                # tool get distinct indices (parse_non_stream may assign
                # indices based on the tool-definition position instead).
                # When the re-parse returns results, it is authoritative:
                # clear streaming state first so we don't mix a name from
                # the re-parse with args from streaming at the same index.
                if final_calls:
                    self._tool_call_ids.clear()
                    self._tool_call_names.clear()
                    self._tool_call_args.clear()
                    for seq_idx, tc in enumerate(final_calls):
                        self._tool_call_ids[seq_idx] = self._tool_call_id(
                            tc.name or "", seq_idx
                        )
                        if tc.name:
                            self._tool_call_names[seq_idx] = tc.name
                        if tc.parameters:
                            self._tool_call_args[seq_idx] = [tc.parameters]

            # Do not emit partial tool calls. A streaming parser can detect a
            # tool name before the model finishes malformed JSON; if the
            # finish-time re-parse cannot recover valid arguments, treat the
            # response as plain text instead of surfacing name + empty args.
            dropped_names = []
            for idx in list(self._tool_call_names):
                if not "".join(self._tool_call_args.get(idx, [])):
                    dropped_names.append(self._tool_call_names[idx])
                    del self._tool_call_names[idx]
                    self._tool_call_ids.pop(idx, None)
                    self._tool_call_args.pop(idx, None)
            if dropped_names:
                logger.warning(
                    "Dropping incomplete SGLang tool calls with no valid arguments: %s",
                    dropped_names,
                )

        if finish_reason and self._tool_call_names:
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

        # Rewrite finish_reason "stop" → "tool_calls" when tool calls were
        # detected, matching the OpenAI API spec and official SGLang behaviour.
        effective_finish = finish_reason
        if finish_reason == "stop" and self._tool_call_names:
            effective_finish = "tool_calls"

        if has_content or effective_finish:
            return {
                "index": 0,
                "delta": delta if has_content else {},
                "finish_reason": effective_finish,
                "logprobs": None,
            }

        return None
