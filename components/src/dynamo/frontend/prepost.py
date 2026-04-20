#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.reasoning import ReasoningParser
from vllm.renderers import ChatParams
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.utils.async_utils import AsyncMicrobatchTokenizer

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    request_for_sampling: ChatCompletionRequest
    tool_parser: ToolParser | None
    chat_template_kwargs: dict[str, Any]
    engine_prompt: dict[str, Any]
    prompt_token_ids: list[int]


_ASYNC_TOKENIZER_POOL: dict[int, AsyncMicrobatchTokenizer] = {}
SKIP_REQUEST_VALIDATION = os.getenv("DYN_VLLM_SKIP_REQUEST_VALIDATION", "1") == "1"
DEBUG_FRONTEND = os.getenv("DYN_FRONTEND_DEBUG", "0") == "1"
DEBUG_FRONTEND_MAX_TEXT = int(os.getenv("DYN_FRONTEND_DEBUG_MAX_TEXT", "240"))


def _debug_enabled() -> bool:
    return DEBUG_FRONTEND


def _short_text(value: Any, limit: int = DEBUG_FRONTEND_MAX_TEXT) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<trimmed:{len(text) - limit}>"


def _hash_messages(messages: Sequence[Any]) -> str:
    payload: list[Any] = []
    for message in messages:
        if hasattr(message, "model_dump"):
            payload.append(message.model_dump(exclude_none=False))
        else:
            payload.append(message)
    return hashlib.sha256(repr(payload).encode("utf-8", "ignore")).hexdigest()[:16]


def _get_async_tokenizer(tokenizer: TokenizerLike) -> AsyncMicrobatchTokenizer:
    key = id(tokenizer)
    async_tokenizer = _ASYNC_TOKENIZER_POOL.get(key)
    if async_tokenizer is None:
        async_tokenizer = AsyncMicrobatchTokenizer(tokenizer)
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


def _prepare_request(
    request: dict[str, Any] | ChatCompletionRequest,
    *,
    tokenizer: TokenizerLike,
    tool_parser_class: type[ToolParser] | None,
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

    original_tools_count = len(request_for_sampling.tools or [])
    original_messages_hash = _hash_messages(request_for_sampling.messages)
    tool_parser: ToolParser | None = None
    # With enable_auto_tool_choice the model may emit tool calls even when the
    # client did not supply an explicit `tools` list, so we activate the parser
    # whenever the tool_parser_class is available.
    has_tools = bool(request_for_sampling.tools)
    if tool_parser_class and (has_tools or enable_auto_tool_choice):
        if request_for_sampling.tool_choice != "none":
            tool_parser = tool_parser_class(tokenizer)
            request_for_sampling = tool_parser.adjust_request(request_for_sampling)

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
    chat_template_kwargs = dict(request_for_sampling.chat_template_kwargs or {})
    chat_template_kwargs["reasoning_effort"] = request_for_sampling.reasoning_effort

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

    if _debug_enabled():
        logger.info(
            "[DYN_DEBUG preprocess] request_summary messages=%d messages_hash=%s tool_choice=%r "
            "tools_in=%d tools_exposed=%d tool_parser=%s auto_tool_choice=%s "
            "reasoning_effort=%r add_generation_prompt=%r continue_final_message=%r "
            "chat_template_kwargs_keys=%s tokenize_in_template=%s",
            len(request_for_sampling.messages),
            original_messages_hash,
            request_for_sampling.tool_choice,
            original_tools_count,
            len(tool_dicts or []),
            tool_parser.__class__.__name__ if tool_parser is not None else None,
            enable_auto_tool_choice,
            request_for_sampling.reasoning_effort,
            request_for_sampling.add_generation_prompt,
            request_for_sampling.continue_final_message,
            sorted(chat_template_kwargs.keys()),
            tokenize_in_template,
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
    renderer,
    tool_parser_class: type[ToolParser] | None,
    exclude_tools_when_tool_choice_none: bool = True,
    enable_auto_tool_choice: bool = False,
) -> PreprocessResult:
    t0 = time.monotonic()
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
        exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
        enable_auto_tool_choice=enable_auto_tool_choice,
    )

    t_prepare = time.monotonic()
    _, engine_prompt = await renderer.render_messages_async(messages, chat_params)
    t_render = time.monotonic()

    if "prompt_token_ids" in engine_prompt:
        tokens = list(engine_prompt["prompt_token_ids"])
    else:
        async_tokenizer = _get_async_tokenizer(tokenizer)
        encoded = await async_tokenizer(
            engine_prompt["prompt"],
            add_special_tokens=request_for_sampling.add_special_tokens,
        )
        tokens = list(encoded.input_ids)
    t_tokenize = time.monotonic()

    if _debug_enabled():
        prompt_text = engine_prompt.get("prompt")
        logger.info(
            "[DYN_DEBUG preprocess] render_summary prepare_ms=%.1f render_ms=%.1f tokenize_ms=%.1f "
            "prompt_tokens=%d has_prompt_text=%s prompt_text_hash=%s prompt_preview=%r",
            (t_prepare - t0) * 1000,
            (t_render - t_prepare) * 1000,
            (t_tokenize - t_render) * 1000,
            len(tokens),
            isinstance(prompt_text, str),
            (
                hashlib.sha256(prompt_text.encode("utf-8", "ignore")).hexdigest()[:16]
                if isinstance(prompt_text, str)
                else None
            ),
            _short_text(prompt_text) if isinstance(prompt_text, str) else None,
        )

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
        debug_request_id: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.request_for_sampling = request_for_sampling
        self.sampling_params = sampling_params
        self.tool_parser = tool_parser
        self.debug_request_id = debug_request_id
        self.reasoning_parser = (
            reasoning_parser_class(
                tokenizer,
                chat_template_kwargs=chat_template_kwargs,
            )
            if reasoning_parser_class
            else None
        )
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
        # Buffer for post-reasoning tool text when </think> and <tool_call>
        # arrive in the same chunk.  The streaming tool parser cannot handle
        # this correctly, so we accumulate text here and fall back to the
        # non-streaming extract_tool_calls() once the buffer is complete.
        self._tool_text_buffer: str | None = None
        self._debug_reasoning_started = False
        self._debug_logged_fast_path = False

    def _debug(self, message: str, *args: Any) -> None:
        if not _debug_enabled():
            return
        prefix = (
            f"[DYN_DEBUG stream rid={self.debug_request_id}] "
            if self.debug_request_id
            else "[DYN_DEBUG stream] "
        )
        logger.info(prefix + message, *args)

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

    def _should_parse_tools(self) -> bool:
        return (
            self.tool_parser is not None
            and self.request_for_sampling.tool_choice != "none"
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
        merged = self.in_progress_tool_calls[index]
        self._debug(
            "tool_call_extracted index=%d id=%r name=%r args_len=%d",
            index,
            merged.id,
            merged.function.name if merged.function else None,
            len(merged.function.arguments or "") if merged.function else 0,
        )

    def _extract_tool_calls_from_text(
        self, text: str, *, saved_reasoning: str | None = None
    ) -> DeltaMessage | None:
        if self.tool_parser is None:
            return self._compose_delta_message(saved_reasoning, None)

        extracted = self.tool_parser.extract_tool_calls(text, self.request_for_sampling)
        if extracted.tools_called:
            for i, tool_call in enumerate(extracted.tool_calls):
                self._add_tool_call_from_extracted(i, tool_call)
            return self._compose_delta_message(saved_reasoning, None)

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
            self._debug(
                "tool_call_stream index=%d id=%r name=%r args_len=%d",
                merged.index,
                merged.id,
                merged.function.name if merged.function else None,
                len(merged.function.arguments or "") if merged.function else 0,
            )

    def _dump_in_progress_tool_calls(self) -> list[dict[str, Any]]:
        return [
            tool_call.model_dump(exclude_none=True)
            for _, tool_call in self.in_progress_tool_calls.items()
        ]

    def _emit_tool_calls_choice(self, output: Any) -> dict[str, Any]:
        choice = {
            "index": output.index,
            "delta": {
                "role": "assistant",
                "tool_calls": self._dump_in_progress_tool_calls(),
            },
            "finish_reason": output.finish_reason,
            "logprobs": output.logprobs,
        }
        self.in_progress_tool_calls.clear()
        return choice

    @staticmethod
    def _build_choice(output: Any, delta: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": output.index,
            "delta": delta,
            "finish_reason": output.finish_reason,
            "logprobs": output.logprobs,
        }

    def process_output(self, output: Any) -> dict[str, Any] | None:
        delta_token_ids = list(output.token_ids or [])
        # vLLM output_processor already applies stop-token/stop-string trimming
        # to text. Re-detokenizing from token_ids can reintroduce stop markers.
        delta_text = output.text or ""
        delta: dict[str, Any] = {}
        if self._fast_plain_text:
            if not self._debug_logged_fast_path:
                self._debug("fast_plain_text enabled")
                self._debug_logged_fast_path = True
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
            tool_call_end = getattr(self.tool_parser, "tool_call_end_token", None)
            buffer_complete = (
                tool_call_end and tool_call_end in self._tool_text_buffer
            ) or output.finish_reason
            if buffer_complete:
                buffered_text = self._tool_text_buffer
                self._tool_text_buffer = None
                self._debug(
                    "tool_buffer_complete len=%d finish_reason=%r preview=%r",
                    len(buffered_text),
                    output.finish_reason,
                    _short_text(buffered_text),
                )
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
                self._debug(
                    "reasoning_end delta_preview=%r reasoning_len=%d content_len=%d finish_reason=%r",
                    _short_text(delta_text),
                    len(delta_message.reasoning or "") if delta_message else 0,
                    len(delta_message.content or "") if delta_message else 0,
                    output.finish_reason,
                )
                self.reasoning_is_done = True
                saved_reasoning = delta_message.reasoning if delta_message else None
                post_content = (delta_message.content if delta_message else None) or ""

                self.previous_text = ""
                self.previous_token_ids = []
                current_text = ""
                current_token_ids = []

                tool_call_start = getattr(
                    self.tool_parser, "tool_call_start_token", None
                )
                if post_content and tool_call_start and tool_call_start in post_content:
                    # Tool call markup present — buffer for non-streaming
                    # extraction (streaming parser can't handle the combined
                    # reasoning-end + tool-start in a single chunk).
                    self._tool_text_buffer = post_content
                    self._debug(
                        "tool_buffer_start len=%d finish_reason=%r preview=%r",
                        len(post_content),
                        output.finish_reason,
                        _short_text(post_content),
                    )
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
            elif delta_message and delta_message.reasoning and not self._debug_reasoning_started:
                self._debug_reasoning_started = True
                self._debug(
                    "reasoning_start len=%d preview=%r",
                    len(delta_message.reasoning or ""),
                    _short_text(delta_message.reasoning or delta_text),
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
            self._merge_streaming_tool_calls(delta_message.tool_calls)
            if output.finish_reason and self.in_progress_tool_calls:
                # Tool calls and finish_reason arrived in the same chunk.
                # Emit now — there will be no subsequent process_output call
                # to drain the buffer.
                choice = self._emit_tool_calls_choice(output)
        elif delta_message.content or delta_message.reasoning:
            delta = {"role": "assistant"}
            content = delta_message.content
            if self.in_progress_tool_calls and self._is_control_only_content(content):
                self._debug(
                    "control_only_content_suppressed preview=%r",
                    _short_text(content),
                )
                content = None
            if content:
                delta["content"] = content
            if delta_message.reasoning:
                delta["reasoning_content"] = delta_message.reasoning
            if self.in_progress_tool_calls:
                delta["tool_calls"] = self._dump_in_progress_tool_calls()
                self.in_progress_tool_calls.clear()
            if len(delta) > 1:
                choice = self._build_choice(output, delta)
        elif self.in_progress_tool_calls:
            choice = self._emit_tool_calls_choice(output)
        elif output.finish_reason:
            choice = self._build_choice(output, {})

        self.previous_text = current_text
        self.previous_token_ids = current_token_ids
        if choice is not None:
            delta_out = choice.get("delta") or {}
            self._debug(
                "emit_choice finish_reason=%r has_content=%s content_len=%d has_reasoning=%s "
                "reasoning_len=%d tool_calls=%d delta_tokens=%d",
                choice.get("finish_reason"),
                "content" in delta_out,
                len(delta_out.get("content") or ""),
                "reasoning_content" in delta_out,
                len(delta_out.get("reasoning_content") or ""),
                len(delta_out.get("tool_calls") or []),
                len(delta_token_ids),
            )
        return choice
