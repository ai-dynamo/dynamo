#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
)
from vllm.parser.abstract_parser import DelegatingParser, Parser
from vllm.reasoning import ReasoningParser
from vllm.renderers import ChatParams
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.utils.async_utils import make_async


class _Renderer(Protocol):
    """Structural type for vLLM's chat-template renderer."""

    async def render_messages_async(
        self, messages: Any, params: ChatParams
    ) -> tuple[Any, dict[str, Any]]:
        ...


@dataclass
class PreprocessResult:
    request_for_sampling: ChatCompletionRequest
    tool_parser: ToolParser | None
    chat_template_kwargs: dict[str, Any]
    engine_prompt: dict[str, Any]
    prompt_token_ids: list[int]


_ASYNC_TOKENIZER_POOL: dict[int, Callable[..., Awaitable[Any]]] = {}
SKIP_REQUEST_VALIDATION = os.getenv("DYN_VLLM_SKIP_REQUEST_VALIDATION", "1") == "1"


def _get_async_tokenizer(tokenizer: TokenizerLike) -> Callable[..., Awaitable[Any]]:
    key = id(tokenizer)
    async_tokenizer = _ASYNC_TOKENIZER_POOL.get(key)
    if async_tokenizer is None:
        async_tokenizer = make_async(
            tokenizer, executor=ThreadPoolExecutor(max_workers=1)
        )
        _ASYNC_TOKENIZER_POOL[key] = async_tokenizer
    return async_tokenizer


def _materialize_assistant_tool_calls(
    messages: Sequence[Any],
) -> list[dict[str, Any] | Any]:
    # Mistral chat templating expects assistant tool_calls to be materialized
    # as a concrete list of dict-like values. Our validated message models may
    # still carry non-list sequence-like containers here, which can break or
    # mis-render when tokenize=True is used in-template. This helper converts
    # model objects to dicts and normalizes assistant.tool_calls to list when
    # possible, while preserving original values if they are not iterable.
    normalized: list[dict[str, Any] | Any] = []
    for message in messages:
        if hasattr(message, "model_dump"):
            msg: dict[str, Any] | Any = message.model_dump(exclude_none=False)
        else:
            msg = message

        if isinstance(msg, dict) and msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls is not None and not isinstance(tool_calls, list):
                try:
                    msg["tool_calls"] = list(tool_calls)
                except TypeError:
                    # Keep original object if it is not iterable.
                    pass

        normalized.append(msg)
    return normalized


class _OptionalReasoningDelegatingParser(DelegatingParser):
    """Delegate tools while allowing a generated ``<think>`` opener.

    vLLM's standard stream state decides whether reasoning is active entirely
    from the prompt. Dynamo's ``basic``/``nemotron_deci`` formats are optional:
    a marker-free prompt may be followed by either direct content/tool output
    or a generated ``<think>`` block. Keep the reasoner active until the first
    non-whitespace generated text resolves that ambiguity, then hand content
    to the normal DelegatingParser tool lifecycle.
    """

    _direct_output_started = False

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: Any,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        state = self._stream_state
        reasoner = self.reasoning_parser
        if not state.prompt_reasoning_checked:
            state.prompt_reasoning_checked = True
            if (
                reasoner is not None
                and prompt_token_ids is not None
                and not reasoner.is_reasoning_end(prompt_token_ids)
            ):
                reasoner.adjust_initial_state_from_prompt(prompt_token_ids)
            state.reasoning_ended = reasoner is None

        if reasoner is not None and not state.reasoning_ended:
            saw_start = reasoner.start_token_id in (
                state.previous_token_ids + delta_token_ids
            )
            prompt_started_reasoning = bool(
                getattr(reasoner, "_initial_in_reasoning", False)
            )
            self._direct_output_started = (
                not prompt_started_reasoning
                and not saw_start
                and bool(delta_text.strip())
            )

        return super().parse_delta(
            delta_text,
            delta_token_ids,
            request,
            prompt_token_ids=None,
            finished=finished,
        )

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        return self._direct_output_started or super().is_reasoning_end_streaming(
            input_ids, delta_ids
        )


def _combined_parser_class(
    reasoning_parser_class: type[ReasoningParser] | None,
    tool_parser_class: type[ToolParser] | None,
) -> type[Parser] | None:
    """Compose vLLM's request/response parser the same way as its server.

    The specialized Harmony and Mistral parser classes are not optional
    implementation details: their component parsers deliberately cannot be
    used independently, and Mistral owns its grammar and tool-call ID policy.
    Keep the composition request-local so parser class state cannot bleed
    between models hosted by the same frontend process.
    """
    if reasoning_parser_class is None and tool_parser_class is None:
        return None

    parser_base: type[Parser] = (
        _OptionalReasoningDelegatingParser
        if getattr(reasoning_parser_class, "dynamo_generated_reasoning_start", False)
        else DelegatingParser
    )
    is_harmony = False
    if reasoning_parser_class is not None or tool_parser_class is not None:
        from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
        from vllm.tool_parsers.gptoss_tool_parser import GptOssToolParser

        is_harmony = (
            reasoning_parser_class is not None
            and issubclass(reasoning_parser_class, GptOssReasoningParser)
        ) or (
            tool_parser_class is not None
            and issubclass(tool_parser_class, GptOssToolParser)
        )
        if is_harmony:
            from vllm.parser.harmony import HarmonyParser

            parser_base = HarmonyParser

    if tool_parser_class is not None:
        from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

        if not is_harmony and issubclass(tool_parser_class, MistralToolParser):
            from vllm.parser.mistral import MistralParser

            parser_base = MistralParser
            if reasoning_parser_class is not None:
                tool_parser_class = type(
                    "_DynamoReasoningMistralToolParser",
                    (tool_parser_class,),
                    {"model_can_reason": True},
                )

    return type(
        "_DynamoParser",
        (parser_base,),
        {
            "reasoning_parser_cls": reasoning_parser_class,
            "tool_parser_cls": tool_parser_class,
        },
    )


def _is_named_tool_choice(tool_choice: Any) -> bool:
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        return (
            tool_choice.get("type") == "function"
            and isinstance(function, dict)
            and bool(function.get("name"))
        )
    function = getattr(tool_choice, "function", None)
    return getattr(tool_choice, "type", None) == "function" and bool(
        getattr(function, "name", None)
    )


@dataclass(frozen=True)
class _GuidedJsonCallSnapshot:
    """The currently visible portion of one bare guided-JSON tool call."""

    name: str
    name_started: bool
    name_complete: bool
    arguments: str | None


def _skip_json_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _scan_json_string_prefix(text: str, position: int) -> tuple[str, int, bool]:
    """Decode the stable prefix of a possibly incomplete JSON string."""

    if position >= len(text) or text[position] != '"':
        return "", position, False

    decoded: list[str] = []
    position += 1
    while position < len(text):
        char = text[position]
        if char == '"':
            return "".join(decoded), position + 1, True
        if char != "\\":
            decoded.append(char)
            position += 1
            continue

        escape_start = position
        position += 1
        if position >= len(text):
            return "".join(decoded), escape_start, False
        escaped = text[position]
        simple_escapes = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        if escaped in simple_escapes:
            decoded.append(simple_escapes[escaped])
            position += 1
            continue
        if escaped != "u" or position + 4 >= len(text):
            return "".join(decoded), escape_start, False

        digits = text[position + 1 : position + 5]
        try:
            codepoint = int(digits, 16)
        except ValueError:
            return "".join(decoded), escape_start, False
        position += 5
        if 0xD800 <= codepoint <= 0xDBFF:
            if position + 5 >= len(text) or text[position : position + 2] != "\\u":
                return "".join(decoded), escape_start, False
            low_digits = text[position + 2 : position + 6]
            try:
                low = int(low_digits, 16)
            except ValueError:
                return "".join(decoded), escape_start, False
            if not 0xDC00 <= low <= 0xDFFF:
                return "".join(decoded), escape_start, False
            codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
            position += 6
        decoded.append(chr(codepoint))

    return "".join(decoded), position, False


def _scan_json_value_prefix(text: str, position: int) -> tuple[int, bool]:
    """Return the visible end and completeness of one JSON value."""

    if position >= len(text):
        return position, False
    opening = text[position]
    if opening == '"':
        _, end, complete = _scan_json_string_prefix(text, position)
        return (end if complete else len(text)), complete

    if opening in "[{":
        closing = {"[": "]", "{": "}"}
        stack = [closing[opening]]
        in_string = False
        escaped = False
        cursor = position + 1
        while cursor < len(text):
            char = text[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                cursor += 1
                continue
            if char == '"':
                in_string = True
            elif char in "[{":
                stack.append(closing[char])
            elif char in "]}":
                if char != stack[-1]:
                    return cursor, False
                stack.pop()
                if not stack:
                    return cursor + 1, True
            cursor += 1
        return len(text), False

    cursor = position
    while cursor < len(text) and text[cursor] not in ",]} \t\r\n":
        cursor += 1
    if cursor == len(text):
        return cursor, False
    candidate = text[position:cursor]
    try:
        json.loads(candidate)
    except (TypeError, ValueError):
        return cursor, False
    return cursor, True


def _normalize_reasoning_policy_kwargs(
    reasoning_policy_name: str | None,
    chat_template_kwargs: dict[str, Any],
) -> None:
    normalized = (
        reasoning_policy_name.strip().lower()
        if isinstance(reasoning_policy_name, str)
        else None
    )
    if normalized == "deepseek_r1":
        enabled: bool | None = None
        for key in ("thinking", "enable_thinking"):
            value = chat_template_kwargs.get(key)
            if type(value) is bool:
                enabled = value
                break
        if enabled is None and chat_template_kwargs.get("thinking_mode") == "chat":
            enabled = False
        if enabled is not None:
            chat_template_kwargs["enable_thinking"] = enabled
        return

    if normalized in {"minimax_m3", "minimax-m3"}:
        enabled = chat_template_kwargs.get("enable_thinking")
        if type(chat_template_kwargs.get("thinking")) is bool:
            enabled = chat_template_kwargs["thinking"]
        if chat_template_kwargs.get("thinking_mode") == "disabled":
            enabled = False
        if type(enabled) is bool:
            chat_template_kwargs["thinking"] = enabled
            chat_template_kwargs["enable_thinking"] = enabled
            chat_template_kwargs["thinking_mode"] = "enabled" if enabled else "disabled"
        return

    if normalized == "kimi_k25":
        enabled = chat_template_kwargs.get("enable_thinking")
        if type(chat_template_kwargs.get("thinking")) is bool:
            enabled = chat_template_kwargs["thinking"]
        if type(enabled) is bool:
            chat_template_kwargs["thinking"] = enabled
        return

    if normalized not in {"deepseek_v4", "deepseek-v4", "deepseekv4"}:
        return

    enabled: bool | None = None
    for key in ("thinking", "enable_thinking"):
        value = chat_template_kwargs.get(key)
        if type(value) is bool:
            enabled = value
            break
    if enabled is None:
        mode = chat_template_kwargs.get("thinking_mode")
        enabled = mode != "chat"
    chat_template_kwargs["thinking"] = enabled
    chat_template_kwargs["enable_thinking"] = enabled


def _uses_harmony_parser(
    reasoning_parser_class: type[ReasoningParser] | None,
    tool_parser_class: type[ToolParser] | None,
) -> bool:
    from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
    from vllm.tool_parsers.gptoss_tool_parser import GptOssToolParser

    return (
        reasoning_parser_class is not None
        and issubclass(reasoning_parser_class, GptOssReasoningParser)
    ) or (
        tool_parser_class is not None
        and issubclass(tool_parser_class, GptOssToolParser)
    )


def _prepare_request(
    request: dict[str, Any] | ChatCompletionRequest,
    *,
    tokenizer: TokenizerLike,
    tool_parser_class: type[ToolParser] | None,
    reasoning_parser_class: type[ReasoningParser] | None = None,
    reasoning_policy_name: str | None = None,
    model_config: Any | None = None,
    exclude_tools_when_tool_choice_none: bool = True,
    enable_auto_tool_choice: bool = False,
) -> tuple[ChatCompletionRequest, ToolParser | None, dict[str, Any], Any, ChatParams]:
    """Validate request and build arguments for template rendering.

    Returns:
        request_for_sampling: Validated ChatCompletionRequest.
        tool_parser: Instantiated tool parser, or None.
        chat_template_kwargs: Template kwargs (for PreprocessResult).
        messages_for_render: Messages to pass as first arg to render_messages.
        chat_params: ChatParams for render_messages / render_messages_async.
    """
    if isinstance(request, ChatCompletionRequest):
        request_for_sampling = request
    elif SKIP_REQUEST_VALIDATION:
        # Trusted fast path; caller must provide OpenAI-compatible payload.
        request_for_sampling = ChatCompletionRequest.model_construct(**request)
        if request_for_sampling.tools and any(
            not hasattr(tool, "model_dump") for tool in request_for_sampling.tools
        ):
            request_for_sampling = ChatCompletionRequest.model_validate(request)
    else:
        request_for_sampling = ChatCompletionRequest.model_validate(request)

    chat_template_kwargs = dict(request_for_sampling.chat_template_kwargs or {})
    legacy_template_args = (
        request.get("chat_template_args")
        if isinstance(request, dict)
        else getattr(request, "chat_template_args", None)
    )
    if isinstance(legacy_template_args, dict):
        chat_template_kwargs.update(legacy_template_args)

    def has_explicit_reasoning_toggle() -> bool:
        return any(
            key in chat_template_kwargs
            for key in ("thinking", "enable_thinking", "thinking_mode")
        )

    root_thinking = (
        request.get("thinking")
        if isinstance(request, dict)
        else getattr(request, "thinking", None)
    )
    if not has_explicit_reasoning_toggle():
        if isinstance(root_thinking, bool):
            chat_template_kwargs["thinking"] = root_thinking
            chat_template_kwargs["enable_thinking"] = root_thinking
            chat_template_kwargs["thinking_mode"] = (
                "enabled" if root_thinking else "disabled"
            )
        elif isinstance(root_thinking, dict):
            thinking_type = root_thinking.get("type")
            if thinking_type in {"enabled", "disabled"}:
                enabled = thinking_type == "enabled"
                chat_template_kwargs["thinking"] = enabled
                chat_template_kwargs["enable_thinking"] = enabled
                chat_template_kwargs["thinking_mode"] = (
                    "enabled" if enabled else "disabled"
                )
    if request_for_sampling.reasoning_effort is not None:
        if not has_explicit_reasoning_toggle():
            enabled = request_for_sampling.reasoning_effort != "none"
            chat_template_kwargs["thinking"] = enabled
            chat_template_kwargs["enable_thinking"] = enabled
            chat_template_kwargs["thinking_mode"] = "enabled" if enabled else "disabled"
        chat_template_kwargs["reasoning_effort"] = request_for_sampling.reasoning_effort
    _normalize_reasoning_policy_kwargs(reasoning_policy_name, chat_template_kwargs)
    if (
        _uses_harmony_parser(reasoning_parser_class, tool_parser_class)
        and chat_template_kwargs.get("reasoning_effort") == "none"
    ):
        raise ValueError("reasoning_effort='none' is not supported for Harmony")
    request_for_sampling.chat_template_kwargs = chat_template_kwargs

    tool_parser: ToolParser | None = None
    # With enable_auto_tool_choice the model may emit tool calls even when the
    # client did not supply an explicit `tools` list, so we activate the parser
    # whenever the tool_parser_class is available.
    has_tools = bool(request_for_sampling.tools)
    mistral_grammar_eligible = False
    harmony_parser = _uses_harmony_parser(reasoning_parser_class, tool_parser_class)
    if tool_parser_class is not None:
        from vllm.utils.mistral import is_mistral_tokenizer, is_mistral_tool_parser

        mistral_grammar_eligible = (
            is_mistral_tool_parser(tool_parser_class)
            and is_mistral_tokenizer(tokenizer)
            and bool(getattr(tokenizer, "supports_grammar", False))
        )
    response_tool_parser_class = (
        tool_parser_class
        if tool_parser_class
        and (
            has_tools
            or enable_auto_tool_choice
            or harmony_parser
            or mistral_grammar_eligible
        )
        else None
    )
    parser_class = _combined_parser_class(
        reasoning_parser_class, response_tool_parser_class
    )
    if parser_class is not None:
        parser_kwargs: dict[str, Any] = {"chat_template_kwargs": chat_template_kwargs}
        if model_config is not None:
            parser_kwargs["model_config"] = model_config
        parser = parser_class(
            tokenizer,
            request_for_sampling.tools,
            **parser_kwargs,
        )
        should_adjust_request = (
            reasoning_parser_class is not None
            or request_for_sampling.tool_choice != "none"
            or mistral_grammar_eligible
        )
        if should_adjust_request:
            request_for_sampling = parser.adjust_request(request_for_sampling)
        tool_parser = parser.tool_parser

    # Strip tools from the template when tool_choice=none so the model doesn't
    # see them and generate raw XML tool calls in its response.
    tool_dicts = (
        [tool.model_dump() for tool in request_for_sampling.tools]
        if request_for_sampling.tools
        and not (
            exclude_tools_when_tool_choice_none
            and request_for_sampling.tool_choice == "none"
        )
        else None
    )
    # Mistral warns that tokenize=False is unsafe for chat templates.
    is_mistral_tokenizer = (
        tokenizer.__class__.__name__ == "MistralTokenizer"
        or "tokenizers.mistral" in tokenizer.__class__.__module__
    )
    tokenize_in_template = is_mistral_tokenizer
    messages_for_render = (
        _materialize_assistant_tool_calls(request_for_sampling.messages)
        if is_mistral_tokenizer
        else request_for_sampling.messages
    )

    chat_params = ChatParams(
        chat_template=request_for_sampling.chat_template,
        chat_template_content_format="auto",
        chat_template_kwargs=dict(
            add_generation_prompt=request_for_sampling.add_generation_prompt,
            continue_final_message=request_for_sampling.continue_final_message,
            tools=tool_dicts,
            documents=request_for_sampling.documents,
            tokenize=tokenize_in_template,
            **chat_template_kwargs,
        ),
    )

    return (
        request_for_sampling,
        tool_parser,
        chat_template_kwargs,
        messages_for_render,
        chat_params,
    )


async def preprocess_chat_request(
    request: dict[str, Any] | ChatCompletionRequest,
    *,
    tokenizer: TokenizerLike,
    renderer: _Renderer,
    tool_parser_class: type[ToolParser] | None,
    reasoning_parser_class: type[ReasoningParser] | None = None,
    reasoning_policy_name: str | None = None,
    model_config: Any | None = None,
    exclude_tools_when_tool_choice_none: bool = True,
    enable_auto_tool_choice: bool = False,
) -> PreprocessResult:
    (
        request_for_sampling,
        tool_parser,
        chat_template_kwargs,
        messages,
        chat_params,
    ) = _prepare_request(
        request,
        tokenizer=tokenizer,
        tool_parser_class=tool_parser_class,
        reasoning_parser_class=reasoning_parser_class,
        reasoning_policy_name=reasoning_policy_name,
        model_config=model_config,
        exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
        enable_auto_tool_choice=enable_auto_tool_choice,
    )

    _, engine_prompt = await renderer.render_messages_async(messages, chat_params)

    if "prompt_token_ids" in engine_prompt:
        tokens = list(engine_prompt["prompt_token_ids"])
    else:
        async_tokenizer = _get_async_tokenizer(tokenizer)
        encoded = await async_tokenizer(
            engine_prompt["prompt"],
            add_special_tokens=request_for_sampling.add_special_tokens,
        )
        tokens = list(encoded.input_ids)

    return PreprocessResult(
        request_for_sampling=request_for_sampling,
        tool_parser=tool_parser,
        chat_template_kwargs=chat_template_kwargs,
        engine_prompt=engine_prompt,
        prompt_token_ids=tokens,
    )


class StreamingPostProcessor:
    def __init__(
        self,
        *,
        tokenizer: TokenizerLike,
        request_for_sampling: ChatCompletionRequest,
        sampling_params: SamplingParams,
        prompt_token_ids: Sequence[int],
        tool_parser: ToolParser | None,
        reasoning_parser_class: type[ReasoningParser] | None,
        chat_template_kwargs: dict[str, Any],
        stream_response: bool = True,
        tool_call_id_type: str = "random",
        model_config: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.request_for_sampling = request_for_sampling
        self.sampling_params = sampling_params
        self.tool_parser = tool_parser
        self.stream_response = stream_response
        self.prompt_token_ids = list(prompt_token_ids)
        # See https://github.com/ai-dynamo/dynamo/issues/8636 —
        # when the chat template runs with enable_thinking=False,
        # the reasoning open/close tags live in the prompt and the generated
        # output carries none — so is_reasoning_end_streaming() never fires,
        # reasoning_is_done stays false, and tool-call markup leaks into
        # reasoning_content. Skip the reasoning parser in that case.
        # `enable_thinking` is the convention adopted across the modern
        # reasoning-capable model families that vLLM supports; templates
        # that don't honor it simply leave it unset (no effect here).
        thinking_disabled = chat_template_kwargs.get("enable_thinking") is False
        response_reasoning_parser_class = (
            reasoning_parser_class if not thinking_disabled else None
        )
        self.reasoning_parser = (
            reasoning_parser_class(
                tokenizer,
                chat_template_kwargs=chat_template_kwargs,
            )
            if reasoning_parser_class and not thinking_disabled
            else None
        )
        if (
            self.reasoning_parser is not None
            and not self.reasoning_parser.is_reasoning_end(prompt_token_ids)
        ):
            self.reasoning_parser.adjust_initial_state_from_prompt(
                list(prompt_token_ids)
            )
        parser_tool_class = (
            type(tool_parser) if isinstance(tool_parser, ToolParser) else None
        )
        parser_class = _combined_parser_class(
            response_reasoning_parser_class, parser_tool_class
        )
        self._unified_parser: Parser | None = None
        self._completion_parser: Parser | None = None
        self._standard_guided_json = False
        self._guided_reasoning_complete = False
        self._streamed_guided_reasoning = ""
        self._emitted_guided_reasoning = ""
        self._guided_json_source = ""
        self._guided_json_emitted_calls: list[_GuidedJsonCallSnapshot] = []
        request_tools = getattr(request_for_sampling, "tools", None)
        request_messages = getattr(request_for_sampling, "messages", ())
        if parser_class is not None:
            parser_kwargs: dict[str, Any] = {
                "chat_template_kwargs": chat_template_kwargs
            }
            if model_config is not None:
                parser_kwargs["model_config"] = model_config
            self._unified_parser = parser_class(
                tokenizer,
                request_tools,
                **parser_kwargs,
            )
            structured_outputs = getattr(
                request_for_sampling, "structured_outputs", None
            )
            standard_guided_json = (
                self._unified_parser.tool_parser is not None
                and getattr(structured_outputs, "json", None) is not None
                and (
                    request_for_sampling.tool_choice == "required"
                    or _is_named_tool_choice(request_for_sampling.tool_choice)
                )
            )
            self._standard_guided_json = standard_guided_json
            if standard_guided_json:
                # The constraint defines the response shape. Strict-tool mode
                # may set the model parser's class flag false even though the
                # backend is producing standard required/named JSON here.
                self._unified_parser.tool_parser.supports_required_and_named = True
                self._completion_parser = parser_class(
                    tokenizer,
                    request_tools,
                    **parser_kwargs,
                )
                assert self._completion_parser.tool_parser is not None
                self._completion_parser.tool_parser.supports_required_and_named = True
            unified_reasoning_parser = self._unified_parser.reasoning_parser
            if (
                unified_reasoning_parser is not None
                and not unified_reasoning_parser.is_reasoning_end(prompt_token_ids)
            ):
                unified_reasoning_parser.adjust_initial_state_from_prompt(
                    list(prompt_token_ids)
                )
            completion_reasoning_parser = (
                self._completion_parser.reasoning_parser
                if self._completion_parser is not None
                else None
            )
            if (
                completion_reasoning_parser is not None
                and not completion_reasoning_parser.is_reasoning_end(prompt_token_ids)
            ):
                completion_reasoning_parser.adjust_initial_state_from_prompt(
                    list(prompt_token_ids)
                )
            self._unified_parser._stream_state.tool_call_id_type = tool_call_id_type
            self._unified_parser._stream_state.history_tool_call_cnt = (
                self._history_tool_call_count(request_messages)
            )
        self._tool_call_id_type = tool_call_id_type
        self._history_tool_calls_count = self._history_tool_call_count(request_messages)
        self._fast_plain_text = (
            self.tool_parser is None and self.reasoning_parser is None
        )

        self._control_markers = tuple(
            t for t in getattr(tokenizer, "all_special_tokens", ()) if t
        )

        self.previous_text = ""
        self.previous_token_ids: list[int] = []
        self.reasoning_is_done = False
        self.in_progress_tool_calls: dict[int, DeltaToolCall] = {}
        # Per-choice tracking (https://github.com/ai-dynamo/dynamo/issues/8636) of whether a tool_call delta was
        # emitted on that choice, keyed by `output.index`. Required because
        # `n > 1` requests stream multiple choices interleaved; a remap on
        # one choice must not bleed into another. See _remap_finish_reason().
        self._tool_call_choices_emitted: set[int] = set()
        # Buffer for post-reasoning tool text when </think> and <tool_call>
        # arrive in the same chunk.  The streaming tool parser cannot handle
        # this correctly, so we accumulate text here and fall back to the
        # non-streaming extract_tool_calls() once the buffer is complete.
        self._tool_text_buffer: str | None = None
        # Hold reasoning/tool boundary whitespace until the output shape is
        # known. Tool-only responses discard it; direct answers retain it.
        self._pending_content_whitespace = ""
        self._visible_content_since_tool = False

    @staticmethod
    def _history_tool_call_count(messages: Sequence[Any]) -> int:
        count = 0
        for message in messages:
            role = (
                message.get("role")
                if isinstance(message, dict)
                else getattr(message, "role", None)
            )
            if role != "assistant":
                continue
            tool_calls = (
                message.get("tool_calls")
                if isinstance(message, dict)
                else getattr(message, "tool_calls", None)
            )
            if tool_calls is not None:
                count += len(list(tool_calls))
        return count

    def _should_buffer_for_non_streaming_tool_parse(self) -> bool:
        return (
            not self.stream_response
            and self.tool_parser is not None
            and self.request_for_sampling.tool_choice != "none"
        )

    @staticmethod
    def _merge_tool_call(
        existing: DeltaToolCall | None, incoming: DeltaToolCall
    ) -> DeltaToolCall:
        if existing is None:
            if incoming.function and incoming.function.arguments is None:
                incoming.function.arguments = ""
            return incoming
        if incoming.id and not existing.id:
            existing.id = incoming.id
        if incoming.type and not existing.type:
            existing.type = incoming.type
        if incoming.function:
            if existing.function is None:
                existing.function = incoming.function
                if existing.function.arguments is None:
                    existing.function.arguments = ""
            else:
                if incoming.function.name and not existing.function.name:
                    existing.function.name = incoming.function.name
                if incoming.function.arguments:
                    if existing.function.arguments is None:
                        existing.function.arguments = ""
                    existing.function.arguments += incoming.function.arguments
        return existing

    def _is_control_only_content(self, content: str | None) -> bool:
        if not content:
            return True
        stripped = content
        for marker in self._control_markers:
            stripped = stripped.replace(marker, "")
        return stripped.strip() == ""

    def _boundary_whitespace(self, content: str | None) -> str | None:
        """Return visible whitespace for a control/whitespace-only delta.

        ``None`` means the delta contains visible non-whitespace content.  An
        empty string means it contained only control markers, which must be
        suppressed rather than replayed into a later direct answer.
        """
        if not content:
            return None
        visible = content
        for marker in self._control_markers:
            visible = visible.replace(marker, "")
        return visible if not visible or visible.isspace() else None

    def _resolve_streaming_content(
        self, content: str | None, output_index: int
    ) -> str | None:
        if self.tool_parser is None or not content:
            return content

        boundary_whitespace = self._boundary_whitespace(content)
        if boundary_whitespace is not None:
            if (
                boundary_whitespace
                and self._visible_content_since_tool
                and not self.in_progress_tool_calls
            ):
                return boundary_whitespace
            if output_index not in self._tool_call_choices_emitted:
                self._pending_content_whitespace += boundary_whitespace
            return None

        if self._pending_content_whitespace:
            if (
                output_index not in self._tool_call_choices_emitted
                and not self.in_progress_tool_calls
            ):
                content = self._pending_content_whitespace + content
            self._pending_content_whitespace = ""
        self._visible_content_since_tool = True
        return content

    def _should_parse_tools(self) -> bool:
        return (
            self.tool_parser is not None
            and self.request_for_sampling.tool_choice != "none"
        )

    def _tool_parser_terminal_markers(self, names: tuple[str, ...]) -> tuple[str, ...]:
        parser_engine = getattr(self.tool_parser, "_parser_engine", None)
        parser_engine_config = getattr(parser_engine, "parser_engine_config", None)
        terminals = getattr(parser_engine_config, "terminals", None)
        if not isinstance(terminals, dict):
            return ()

        markers: list[str] = []
        for name in names:
            marker = terminals.get(name)
            if isinstance(marker, str) and marker:
                markers.append(marker)
        return tuple(markers)

    def _tool_start_markers(self) -> tuple[str, ...]:
        markers = [
            getattr(self.tool_parser, "tool_call_start_token", None),
            # MistralToolParser names its [TOOL_CALLS] marker bot_token.
            getattr(self.tool_parser, "bot_token", None),
            *self._tool_parser_terminal_markers(("TOOL_START", "FUNC_PREFIX")),
        ]
        return tuple(
            dict.fromkeys(
                marker for marker in markers if isinstance(marker, str) and marker
            )
        )

    def _tool_end_markers(self) -> tuple[str, ...]:
        markers = [
            getattr(self.tool_parser, "tool_call_end_token", None),
            *self._tool_parser_terminal_markers(("TOOL_END", "FUNC_END")),
        ]
        return tuple(
            dict.fromkeys(
                marker for marker in markers if isinstance(marker, str) and marker
            )
        )

    @staticmethod
    def _compose_delta_message(
        reasoning: str | None, content: str | None
    ) -> DeltaMessage | None:
        delta_message = DeltaMessage(reasoning=reasoning, content=content)
        if not delta_message.reasoning and not delta_message.content:
            return None
        return delta_message

    def _add_tool_call_from_extracted(self, index: int, tool_call: Any) -> None:
        tool_delta = DeltaToolCall(
            index=index,
            type="function",
            id=(tool_call.id if tool_call.id else make_tool_call_id()),
            function=DeltaFunctionCall(
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            ),
        )
        existing = self.in_progress_tool_calls.get(index)
        self.in_progress_tool_calls[index] = self._merge_tool_call(existing, tool_delta)

    def _extract_tool_calls_from_text(
        self, text: str, *, saved_reasoning: str | None = None
    ) -> DeltaMessage | None:
        if self.tool_parser is None:
            return self._compose_delta_message(saved_reasoning, None)

        extracted = self.tool_parser.extract_tool_calls(text, self.request_for_sampling)
        if extracted.tools_called:
            for i, tool_call in enumerate(extracted.tool_calls):
                self._add_tool_call_from_extracted(i, tool_call)
            content = extracted.content or None
            if self._is_control_only_content(content):
                content = None
            return self._compose_delta_message(saved_reasoning, content)

        return self._compose_delta_message(saved_reasoning, extracted.content or None)

    def _extract_tool_calls_streaming(
        self,
        *,
        current_text: str,
        delta_text: str,
        delta_token_ids: list[int],
        current_token_ids: list[int],
    ) -> DeltaMessage | None:
        if self.tool_parser is None:
            return None
        return self.tool_parser.extract_tool_calls_streaming(
            previous_text=self.previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=self.previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=self.request_for_sampling,
        )

    def _merge_streaming_tool_calls(self, tool_calls: list[DeltaToolCall]) -> None:
        for tool_delta in tool_calls:
            existing = self.in_progress_tool_calls.get(tool_delta.index)
            merged = self._merge_tool_call(existing, tool_delta)
            self.in_progress_tool_calls[tool_delta.index] = merged

    def _dump_in_progress_tool_calls(self) -> list[dict[str, Any]]:
        return [
            tool_call.model_dump(exclude_none=True)
            for _, tool_call in self.in_progress_tool_calls.items()
        ]

    def _remap_finish_reason(
        self, output_index: int, finish_reason: str | None
    ) -> str | None:
        # Per https://github.com/ai-dynamo/dynamo/issues/8636 — OpenAI ChatCompletion finish_reason must be "tool_calls"
        # when the model called a tool. vLLM stops at <|im_end|> and reports
        # "stop"; remap once a tool_call delta has been emitted on THIS
        # choice. Per-choice tracking is required for `n > 1` requests —
        # choice 0 emitting tool_calls must not remap choice 1's stop.
        # Spec: https://github.com/openai/openai-openapi/blob/master/openapi.yaml
        if finish_reason == "stop" and output_index in self._tool_call_choices_emitted:
            return "tool_calls"
        return finish_reason

    def _emit_tool_calls_choice(self, output: Any) -> dict[str, Any]:
        self._tool_call_choices_emitted.add(output.index)
        self._pending_content_whitespace = ""
        self._visible_content_since_tool = False
        choice = {
            "index": output.index,
            "delta": {
                "role": "assistant",
                "tool_calls": self._dump_in_progress_tool_calls(),
            },
            "finish_reason": self._remap_finish_reason(
                output.index, output.finish_reason
            ),
            "logprobs": output.logprobs,
        }
        self.in_progress_tool_calls.clear()
        return choice

    def _build_choice(self, output: Any, delta: dict[str, Any]) -> dict[str, Any]:
        if delta.get("tool_calls"):
            self._tool_call_choices_emitted.add(output.index)
        return {
            "index": output.index,
            "delta": delta,
            "finish_reason": self._remap_finish_reason(
                output.index, output.finish_reason
            ),
            "logprobs": output.logprobs,
        }

    def _guided_tool_call_delta(self, index: int, tool_call: Any) -> dict[str, Any]:
        name = tool_call.name
        return DeltaToolCall(
            index=index,
            type="function",
            id=getattr(tool_call, "id", None)
            or make_tool_call_id(
                id_type=self._tool_call_id_type,
                func_name=name,
                idx=self._history_tool_calls_count + index,
            ),
            function=DeltaFunctionCall(
                name=name,
                arguments=tool_call.arguments,
            ),
        ).model_dump(exclude_none=True)

    def _parse_standard_guided_json(
        self, text: str
    ) -> tuple[list[FunctionCall] | None, str | None]:
        stripped = text.strip()
        if not stripped:
            return None, None
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            return None, text

        tool_choice = self.request_for_sampling.tool_choice
        if _is_named_tool_choice(tool_choice):
            function = getattr(tool_choice, "function", None)
            name = getattr(function, "name", None)
            if name is None and isinstance(tool_choice, dict):
                name = tool_choice["function"]["name"]
            return [FunctionCall(name=name, arguments=stripped)], None

        if tool_choice != "required" or not isinstance(parsed, list):
            return None, text

        calls: list[FunctionCall] = []
        for item in parsed:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                return None, text
            parameters = item.get("parameters", item.get("arguments"))
            if parameters is None:
                parameters = {}
            calls.append(
                FunctionCall(
                    name=item["name"],
                    arguments=json.dumps(parameters, ensure_ascii=False),
                )
            )
        return calls, None

    def _parse_complete_unified_response(
        self,
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        assert self._unified_parser is not None
        parser = self._completion_parser or self._unified_parser
        if not self._standard_guided_json:
            return parser.parse(
                self.previous_text,
                self.request_for_sampling,
                enable_auto_tools=True,
                model_output_token_ids=self.previous_token_ids,
            )

        from vllm.parser.harmony import HarmonyParser

        if not isinstance(parser, HarmonyParser):
            return parser.parse(
                self.previous_text,
                self.request_for_sampling,
                enable_auto_tools=True,
                model_output_token_ids=self.previous_token_ids,
            )

        # Harmony reasoning is a complete message terminated by <|end|>,
        # while reasoning-aware guided decoding emits standard bare JSON after
        # that boundary. Feeding the JSON tokens back to Harmony's FSM is an
        # invalid second message, so parse each wire format independently.
        split_index = self._harmony_guided_boundary(parser, self.previous_token_ids)

        if split_index >= 0:
            harmony_ids = self.previous_token_ids[: split_index + 1]
            harmony_text = self.tokenizer.decode(harmony_ids, skip_special_tokens=False)
            json_text = self.tokenizer.decode(
                self.previous_token_ids[split_index + 1 :],
                skip_special_tokens=True,
            )
        else:
            decoded = self.tokenizer.decode(
                self.previous_token_ids, skip_special_tokens=True
            )
            if decoded.lstrip().startswith(("[", "{")):
                harmony_ids = []
                harmony_text = ""
                json_text = decoded
            else:
                return parser.parse(
                    self.previous_text,
                    self.request_for_sampling,
                    enable_auto_tools=True,
                    model_output_token_ids=self.previous_token_ids,
                )

        reasoning = None
        content = None
        if harmony_ids:
            reasoning, content, harmony_calls = parser.parse(
                harmony_text,
                self.request_for_sampling,
                enable_auto_tools=True,
                model_output_token_ids=harmony_ids,
            )
            if harmony_calls:
                return reasoning, content, harmony_calls

        tool_calls, _ = self._parse_standard_guided_json(json_text)
        return reasoning, content, tool_calls

    def _harmony_guided_boundary(self, parser: Parser, token_ids: list[int]) -> int:
        reasoner = parser.reasoning_parser
        prefix = list(getattr(reasoner, "reasoning_end_token_ids_prefix", ()))
        suffix = list(getattr(reasoner, "reasoning_end_token_ids_suffix", ()))
        max_between = int(getattr(reasoner, "reasoning_max_num_between_tokens", 20))
        if prefix and suffix:
            for index in range(len(token_ids) - len(prefix) + 1):
                if token_ids[index : index + len(prefix)] != prefix:
                    continue
                suffix_start = index + len(prefix)
                suffix_stop = min(
                    len(token_ids) - len(suffix) + 1,
                    suffix_start + max_between,
                )
                for suffix_index in range(suffix_start, suffix_stop):
                    if token_ids[suffix_index : suffix_index + len(suffix)] == suffix:
                        return suffix_index + len(suffix) - 1

        # Older/non-aware backends can emit bare JSON immediately after the
        # analysis EOM without a final-channel header. Detect that shape from
        # token IDs; output.text may have stripped every Harmony marker.
        eom_token_id = self.tokenizer.get_vocab().get("<|end|>")
        if eom_token_id is not None:
            for index, token_id in enumerate(token_ids):
                if token_id != eom_token_id:
                    continue
                tail = self.tokenizer.decode(
                    token_ids[index + 1 :], skip_special_tokens=True
                ).lstrip()
                if tail.startswith(("[", "{")):
                    return index
        return -1

    def _stream_standard_guided_reasoning(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        *,
        finished: bool,
    ) -> str | None:
        assert self._unified_parser is not None
        if self._guided_reasoning_complete:
            return None

        from vllm.parser.harmony import HarmonyParser

        parser_text = delta_text
        parser_token_ids = delta_token_ids
        if isinstance(self._unified_parser, HarmonyParser):
            boundary = self._harmony_guided_boundary(
                self._unified_parser, self.previous_token_ids
            )
            if boundary >= 0:
                previous_length = len(self.previous_token_ids) - len(delta_token_ids)
                split_index = max(0, boundary - previous_length + 1)
                parser_token_ids = delta_token_ids[:split_index]
                parser_text = self.tokenizer.decode(
                    parser_token_ids, skip_special_tokens=False
                )
                self._guided_reasoning_complete = True

        delta_message = self._unified_parser.parse_delta(
            parser_text,
            parser_token_ids,
            self.request_for_sampling,
            self.prompt_token_ids,
            finished=finished and not self._guided_reasoning_complete,
        )
        if delta_message is None or not delta_message.reasoning:
            return None
        self._streamed_guided_reasoning += delta_message.reasoning
        return delta_message.reasoning

    def _required_guided_json_snapshots(
        self, source_text: str
    ) -> list[_GuidedJsonCallSnapshot]:
        """Parse every stable or partial call visible in a required array."""

        position = _skip_json_whitespace(source_text, 0)
        if position >= len(source_text) or source_text[position] != "[":
            return []
        position += 1
        calls: list[_GuidedJsonCallSnapshot] = []

        while True:
            position = _skip_json_whitespace(source_text, position)
            if position >= len(source_text) or source_text[position] == "]":
                break
            if source_text[position] != "{":
                break
            position += 1

            name = ""
            name_started = False
            name_complete = False
            arguments: str | None = None
            object_complete = False
            while True:
                position = _skip_json_whitespace(source_text, position)
                if position >= len(source_text):
                    break
                if source_text[position] == "}":
                    position += 1
                    object_complete = True
                    break

                key, key_end, key_complete = _scan_json_string_prefix(
                    source_text, position
                )
                if not key_complete:
                    break
                position = _skip_json_whitespace(source_text, key_end)
                if position >= len(source_text) or source_text[position] != ":":
                    break
                position = _skip_json_whitespace(source_text, position + 1)
                if position >= len(source_text):
                    break

                value_start = position
                if key == "name":
                    if source_text[position] != '"':
                        break
                    name, value_end, value_complete = _scan_json_string_prefix(
                        source_text, position
                    )
                    name_started = True
                    name_complete = value_complete
                else:
                    value_end, value_complete = _scan_json_value_prefix(
                        source_text, position
                    )
                    if key in {"parameters", "arguments"}:
                        arguments = source_text[value_start:value_end]

                if not value_complete:
                    position = len(source_text)
                    break
                position = _skip_json_whitespace(source_text, value_end)
                if position >= len(source_text):
                    break
                if source_text[position] == ",":
                    position += 1
                    continue
                if source_text[position] == "}":
                    position += 1
                    object_complete = True
                break

            if name_started:
                if object_complete and arguments is None:
                    arguments = "{}"
                calls.append(
                    _GuidedJsonCallSnapshot(
                        name=name,
                        name_started=True,
                        name_complete=name_complete,
                        arguments=arguments,
                    )
                )
            if not object_complete:
                break

            position = _skip_json_whitespace(source_text, position)
            if position >= len(source_text) or source_text[position] == "]":
                break
            if source_text[position] != ",":
                break
            position += 1

        return calls

    def _guided_json_snapshots(self, source_text: str) -> list[_GuidedJsonCallSnapshot]:
        if not _is_named_tool_choice(self.request_for_sampling.tool_choice):
            return self._required_guided_json_snapshots(source_text)

        function = getattr(self.request_for_sampling.tool_choice, "function", None)
        name = getattr(function, "name", None)
        if name is None and isinstance(self.request_for_sampling.tool_choice, dict):
            name = self.request_for_sampling.tool_choice["function"]["name"]
        position = _skip_json_whitespace(source_text, 0)
        if position >= len(source_text):
            return []
        value_end, _ = _scan_json_value_prefix(source_text, position)
        return [
            _GuidedJsonCallSnapshot(
                name=name,
                name_started=True,
                name_complete=True,
                arguments=source_text[position:value_end],
            )
        ]

    def _stream_standard_guided_json(self, source_text: str) -> list[dict[str, Any]]:
        """Emit only newly visible name/argument fragments from bare JSON."""

        if source_text.startswith(self._guided_json_source):
            self._guided_json_source = source_text
        elif not self._guided_json_source.startswith(source_text):
            # Non-engine parsers provide cumulative text. Engine-based parsers
            # provide deltas, so retain both shapes without duplicating a
            # cumulative prefix.
            self._guided_json_source += source_text

        snapshots = self._guided_json_snapshots(self._guided_json_source)
        deltas: list[dict[str, Any]] = []
        empty = _GuidedJsonCallSnapshot("", False, False, None)
        for index, snapshot in enumerate(snapshots):
            previous = (
                self._guided_json_emitted_calls[index]
                if index < len(self._guided_json_emitted_calls)
                else empty
            )
            if previous.name_started and not snapshot.name.startswith(previous.name):
                continue
            previous_arguments = previous.arguments or ""
            if snapshot.arguments is not None and not snapshot.arguments.startswith(
                previous_arguments
            ):
                continue

            first_delta = index >= len(self._guided_json_emitted_calls)
            if (
                first_delta
                and self._tool_call_id_type == "kimi_k2"
                and not snapshot.name_complete
            ):
                # Kimi IDs embed the function name. Delay the first delta
                # until the name is complete so the immutable ID is correct.
                continue
            name_delta = (
                snapshot.name[len(previous.name) :] if snapshot.name_started else None
            )
            arguments_delta = (
                snapshot.arguments[len(previous_arguments) :]
                if snapshot.arguments is not None
                else None
            )
            if not first_delta and not name_delta and not arguments_delta:
                continue

            deltas.append(
                DeltaToolCall(
                    index=index,
                    id=(
                        make_tool_call_id(
                            id_type=self._tool_call_id_type,
                            func_name=snapshot.name,
                            idx=self._history_tool_calls_count + index,
                        )
                        if first_delta
                        else None
                    ),
                    type="function" if first_delta else None,
                    function=DeltaFunctionCall(
                        name=name_delta or None,
                        arguments=arguments_delta or None,
                    ),
                ).model_dump(exclude_none=True)
            )
            if first_delta:
                self._guided_json_emitted_calls.append(snapshot)
            else:
                self._guided_json_emitted_calls[index] = snapshot
        return deltas

    def _harmony_guided_json_text(self, parser: Parser) -> str:
        split_index = self._harmony_guided_boundary(parser, self.previous_token_ids)
        if split_index >= 0:
            return self.tokenizer.decode(
                self.previous_token_ids[split_index + 1 :],
                skip_special_tokens=True,
            )
        decoded = self.tokenizer.decode(
            self.previous_token_ids, skip_special_tokens=True
        )
        stripped = decoded.lstrip()
        return stripped if stripped.startswith(("[", "{")) else ""

    def _process_standard_guided_generic_stream(
        self,
        output: Any,
        delta_text: str,
        delta_token_ids: list[int],
    ) -> dict[str, Any] | None:
        assert self._unified_parser is not None
        delta_message = self._unified_parser.parse_delta(
            delta_text,
            delta_token_ids,
            self.request_for_sampling,
            self.prompt_token_ids,
            finished=bool(output.finish_reason),
        )
        delta: dict[str, Any] = {"role": "assistant"}
        if (
            delta_message is not None
            and delta_message.reasoning
            and self.request_for_sampling.include_reasoning
        ):
            delta["reasoning_content"] = delta_message.reasoning
            self._emitted_guided_reasoning += delta_message.reasoning

        tool_phase_started = self._unified_parser._stream_state.reasoning_ended
        guided_tool_text = self._unified_parser._stream_state.previous_text
        if tool_phase_started and (
            tool_deltas := self._stream_standard_guided_json(guided_tool_text)
        ):
            delta["tool_calls"] = tool_deltas

        if output.finish_reason:
            reasoning, _, full_tool_calls = self._parse_complete_unified_response()
            if (
                reasoning
                and self.request_for_sampling.include_reasoning
                and reasoning.startswith(self._emitted_guided_reasoning)
                and (remainder := reasoning[len(self._emitted_guided_reasoning) :])
            ):
                delta["reasoning_content"] = (
                    delta.get("reasoning_content", "") + remainder
                )
            if full_tool_calls and not self._guided_json_emitted_calls:
                delta.setdefault("tool_calls", []).extend(
                    self._guided_tool_call_delta(index, tool_call)
                    for index, tool_call in enumerate(full_tool_calls)
                )

        if len(delta) == 1:
            if not output.finish_reason:
                return None
            delta = {}
        return self._build_choice(output, delta)

    def _process_unified_output(self, output: Any) -> dict[str, Any] | None:
        assert self._unified_parser is not None
        delta_token_ids = list(output.token_ids or [])
        delta_text = output.text or ""

        if not self.stream_response:
            self.previous_text += delta_text
            self.previous_token_ids.extend(delta_token_ids)
            if not output.finish_reason:
                return None
            reasoning, content, tool_calls = self._parse_complete_unified_response()
            delta: dict[str, Any] = {"role": "assistant"}
            if content:
                delta["content"] = content
            if reasoning and self.request_for_sampling.include_reasoning:
                delta["reasoning_content"] = reasoning
            if tool_calls and self.request_for_sampling.tool_choice != "none":
                delta["tool_calls"] = [
                    self._guided_tool_call_delta(index, tool_call)
                    for index, tool_call in enumerate(tool_calls)
                ]
            if len(delta) == 1:
                delta = {}
            return self._build_choice(output, delta)

        if self._standard_guided_json:
            self.previous_text += delta_text
            self.previous_token_ids.extend(delta_token_ids)
            from vllm.parser.harmony import HarmonyParser

            if not isinstance(self._unified_parser, HarmonyParser):
                return self._process_standard_guided_generic_stream(
                    output, delta_text, delta_token_ids
                )
            streamed_reasoning = self._stream_standard_guided_reasoning(
                delta_text,
                delta_token_ids,
                finished=bool(output.finish_reason),
            )
            tool_deltas = self._stream_standard_guided_json(
                self._harmony_guided_json_text(self._unified_parser)
            )
            if not output.finish_reason:
                delta: dict[str, Any] = {"role": "assistant"}
                if streamed_reasoning and self.request_for_sampling.include_reasoning:
                    self._emitted_guided_reasoning += streamed_reasoning
                    delta["reasoning_content"] = streamed_reasoning
                if tool_deltas:
                    delta["tool_calls"] = tool_deltas
                return self._build_choice(output, delta) if len(delta) > 1 else None

            reasoning, content, tool_calls = self._parse_complete_unified_response()
            if reasoning is None and streamed_reasoning:
                reasoning = streamed_reasoning
            if (
                reasoning
                and self._emitted_guided_reasoning
                and reasoning.startswith(self._emitted_guided_reasoning)
            ):
                reasoning = reasoning[len(self._emitted_guided_reasoning) :] or None
            delta: dict[str, Any] = {"role": "assistant"}
            if content:
                delta["content"] = content
            if reasoning and self.request_for_sampling.include_reasoning:
                delta["reasoning_content"] = reasoning
            if tool_deltas:
                delta["tool_calls"] = tool_deltas
            elif (
                tool_calls
                and self.request_for_sampling.tool_choice != "none"
                and not self._guided_json_emitted_calls
            ):
                delta["tool_calls"] = [
                    self._guided_tool_call_delta(index, tool_call)
                    for index, tool_call in enumerate(tool_calls)
                ]
            if len(delta) == 1:
                delta = {}
            return self._build_choice(output, delta)

        delta_message = self._unified_parser.parse_delta(
            delta_text,
            delta_token_ids,
            self.request_for_sampling,
            self.prompt_token_ids,
            finished=bool(output.finish_reason),
        )
        delta: dict[str, Any] = {"role": "assistant"}
        if delta_message is not None:
            content = self._resolve_streaming_content(
                delta_message.content, output.index
            )
            if content:
                delta["content"] = content
            if delta_message.reasoning and self.request_for_sampling.include_reasoning:
                self._pending_content_whitespace = ""
                delta["reasoning_content"] = delta_message.reasoning
            if (
                delta_message.tool_calls
                and self.request_for_sampling.tool_choice != "none"
            ):
                self._pending_content_whitespace = ""
                self._visible_content_since_tool = False
                delta["tool_calls"] = [
                    tool_call.model_dump(exclude_none=True)
                    for tool_call in delta_message.tool_calls
                ]
        if output.finish_reason and self._pending_content_whitespace:
            if (
                delta.get("tool_calls")
                or output.index in self._tool_call_choices_emitted
            ):
                self._pending_content_whitespace = ""
            elif "content" not in delta:
                delta["content"] = self._pending_content_whitespace
                self._pending_content_whitespace = ""
        if len(delta) == 1:
            if not output.finish_reason:
                return None
            delta = {}
        return self._build_choice(output, delta)

    def _process_non_streaming_tool_output(self, output: Any) -> dict[str, Any] | None:
        delta_token_ids = list(output.token_ids or [])
        delta_text = output.text or ""
        current_text = self.previous_text + delta_text
        current_token_ids = self.previous_token_ids + delta_token_ids

        self.previous_text = current_text
        self.previous_token_ids = current_token_ids
        if not output.finish_reason:
            return None

        saved_reasoning = None
        content = current_text
        if self.reasoning_parser:
            saved_reasoning, content = self.reasoning_parser.extract_reasoning(
                current_text,
                request=self.request_for_sampling,
            )
            if not self.request_for_sampling.include_reasoning:
                saved_reasoning = None

        delta_message = self._extract_tool_calls_from_text(
            content or "",
            saved_reasoning=saved_reasoning,
        )
        if delta_message is None:
            if self.in_progress_tool_calls:
                return self._emit_tool_calls_choice(output)
            return self._build_choice(output, {})

        delta: dict[str, Any] = {"role": "assistant"}
        if delta_message.content:
            delta["content"] = delta_message.content
        if delta_message.reasoning:
            delta["reasoning_content"] = delta_message.reasoning
        if self.in_progress_tool_calls:
            delta["tool_calls"] = self._dump_in_progress_tool_calls()
            self.in_progress_tool_calls.clear()
        if len(delta) == 1:
            delta = {}
        return self._build_choice(output, delta)

    def process_output(self, output: Any) -> dict[str, Any] | None:
        if self._unified_parser is not None:
            return self._process_unified_output(output)

        if self._should_buffer_for_non_streaming_tool_parse():
            return self._process_non_streaming_tool_output(output)

        delta_token_ids = list(output.token_ids or [])
        # vLLM output_processor already applies stop-token/stop-string trimming
        # to text. Re-detokenizing from token_ids can reintroduce stop markers.
        delta_text = output.text or ""
        delta: dict[str, Any] = {}
        if self._fast_plain_text:
            if delta_text:
                delta = {
                    "role": "assistant",
                    "content": delta_text,
                }
            elif output.finish_reason:
                delta = {}
            else:
                return None
            return self._build_choice(output, delta)

        current_text = self.previous_text + delta_text
        current_token_ids = self.previous_token_ids + delta_token_ids

        delta_message: DeltaMessage | None = DeltaMessage(content=delta_text)

        # ------------------------------------------------------------------
        # Drain the tool-text buffer (populated when </think> and <tool_call>
        # arrived in the same chunk).  The streaming tool parser cannot
        # handle that transition correctly, so we accumulate text here and
        # use the non-streaming extract_tool_calls() once complete.
        # ------------------------------------------------------------------
        if self._tool_text_buffer is not None:
            self._tool_text_buffer += delta_text
            buffer_complete = (
                any(
                    marker in self._tool_text_buffer
                    for marker in self._tool_end_markers()
                )
            ) or output.finish_reason
            if buffer_complete:
                buffered_text = self._tool_text_buffer
                self._tool_text_buffer = None
                delta_message = self._extract_tool_calls_from_text(buffered_text)
            else:
                # Still accumulating; emit nothing for this chunk.
                self.previous_text = current_text
                self.previous_token_ids = current_token_ids
                return None

        elif not self.reasoning_is_done and self.reasoning_parser:
            delta_message = self.reasoning_parser.extract_reasoning_streaming(
                self.previous_text,
                current_text,
                delta_text,
                self.previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

            # When reasoning ends in this chunk, reset accumulated state.
            # If there is post-reasoning content (e.g. <tool_call> markup),
            # buffer it for non-streaming extraction rather than feeding it
            # to the streaming tool parser which cannot handle the combined
            # reasoning-end + tool-start in a single chunk.
            if self.reasoning_parser.is_reasoning_end_streaming(
                current_token_ids, delta_token_ids
            ):
                self.reasoning_is_done = True
                saved_reasoning = delta_message.reasoning if delta_message else None
                post_content = (delta_message.content if delta_message else None) or ""

                self.previous_text = ""
                self.previous_token_ids = []
                current_text = ""
                current_token_ids = []

                tool_start_markers = self._tool_start_markers()
                if post_content and any(
                    marker in post_content for marker in tool_start_markers
                ):
                    # Tool call markup present — buffer for non-streaming
                    # extraction (streaming parser can't handle the combined
                    # reasoning-end + tool-start in a single chunk).
                    self._tool_text_buffer = post_content
                    if output.finish_reason:
                        # If finish_reason is already set, this is the final
                        # chunk; parse buffered text now instead of waiting for
                        # a later call that will never happen.
                        buffered_text = self._tool_text_buffer
                        self._tool_text_buffer = None
                        delta_message = self._extract_tool_calls_from_text(
                            buffered_text,
                            saved_reasoning=saved_reasoning,
                        )
                    else:
                        delta_message = self._compose_delta_message(
                            saved_reasoning,
                            None,
                        )
                else:
                    # Plain content (or no content) after reasoning end.
                    delta_message = self._compose_delta_message(
                        reasoning=saved_reasoning,
                        content=post_content if post_content else None,
                    )
            elif (
                delta_message
                and delta_message.content
                and not delta_message.reasoning
                and self._should_parse_tools()
            ):
                # Reasoning parser returned content (not reasoning).
                # The model may have skipped reasoning and gone straight
                # to tool calls (e.g. Mistral [TOOL_CALLS] without
                # [THINK]...[/THINK]).  Let the tool parser decide.
                delta_message = self._extract_tool_calls_streaming(
                    current_text=current_text,
                    delta_text=delta_text,
                    current_token_ids=current_token_ids,
                    delta_token_ids=delta_token_ids,
                )
        else:
            if self._should_parse_tools():
                no_prev_reasoning = (
                    delta_message
                    and delta_message.content
                    and not delta_message.reasoning
                )
                if self.reasoning_is_done or no_prev_reasoning:
                    delta_message = self._extract_tool_calls_streaming(
                        current_text=current_text,
                        delta_text=delta_text,
                        current_token_ids=current_token_ids,
                        delta_token_ids=delta_token_ids,
                    )

        choice = None
        if delta_message is None:
            if self.in_progress_tool_calls:
                choice = self._emit_tool_calls_choice(output)
            elif output.finish_reason:
                choice = self._build_choice(output, {})
        elif delta_message.tool_calls:
            self._pending_content_whitespace = ""
            self._merge_streaming_tool_calls(delta_message.tool_calls)
            if output.finish_reason and self.in_progress_tool_calls:
                # Tool calls and finish_reason arrived in the same chunk.
                # Emit now — there will be no subsequent process_output call
                # to drain the buffer.
                choice = self._emit_tool_calls_choice(output)
        elif delta_message.content or delta_message.reasoning:
            delta = {"role": "assistant"}
            content = self._resolve_streaming_content(
                delta_message.content, output.index
            )
            if content:
                delta["content"] = content
            if delta_message.reasoning:
                delta["reasoning_content"] = delta_message.reasoning
            if self.in_progress_tool_calls:
                self._pending_content_whitespace = ""
                self._visible_content_since_tool = False
                delta["tool_calls"] = self._dump_in_progress_tool_calls()
                self.in_progress_tool_calls.clear()
            if len(delta) > 1:
                choice = self._build_choice(output, delta)
        elif self.in_progress_tool_calls:
            choice = self._emit_tool_calls_choice(output)
        elif output.finish_reason:
            choice = self._build_choice(output, {})

        if output.finish_reason and self._pending_content_whitespace:
            if (
                output.index in self._tool_call_choices_emitted
                or self.in_progress_tool_calls
            ):
                self._pending_content_whitespace = ""
            elif choice is None or not choice.get("delta", {}).get("content"):
                pending = self._pending_content_whitespace
                self._pending_content_whitespace = ""
                if choice is None:
                    choice = self._build_choice(
                        output, {"role": "assistant", "content": pending}
                    )
                else:
                    choice.setdefault("delta", {})["role"] = "assistant"
                    choice["delta"]["content"] = pending

        if choice is None and output.finish_reason:
            choice = self._build_choice(output, {})

        self.previous_text = current_text
        self.previous_token_ids = current_token_ids
        return choice
