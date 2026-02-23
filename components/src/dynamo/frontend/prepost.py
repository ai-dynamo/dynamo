#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage, DeltaToolCall
from vllm.reasoning import ReasoningParser
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.utils.async_utils import AsyncMicrobatchTokenizer


@dataclass
class PreprocessResult:
    request_for_sampling: ChatCompletionRequest
    tool_parser: ToolParser | None
    chat_template_kwargs: dict[str, Any]
    engine_prompt: dict[str, Any]
    prompt_token_ids: list[int]


_ASYNC_TOKENIZER_POOL: dict[int, AsyncMicrobatchTokenizer] = {}
SKIP_REQUEST_VALIDATION = os.getenv("DYN_VLLM_SKIP_REQUEST_VALIDATION", "1") == "1"


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
) -> tuple[
    ChatCompletionRequest, ToolParser | None, dict[str, Any], Any, dict[str, Any]
]:
    """Validate request and build arguments for template rendering.

    Returns:
        request_for_sampling: Validated ChatCompletionRequest.
        tool_parser: Instantiated tool parser, or None.
        chat_template_kwargs: Template kwargs (for PreprocessResult).
        messages_for_render: Messages to pass as first arg to render_messages.
        render_kwargs: Keyword arguments for render_messages / render_messages_async.
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

    tool_parser: ToolParser | None = None
    if tool_parser_class and request_for_sampling.tools:
        if request_for_sampling.tool_choice != "none":
            tool_parser = tool_parser_class(tokenizer)
            request_for_sampling = tool_parser.adjust_request(request_for_sampling)

    tool_dicts = (
        [tool.model_dump() for tool in request_for_sampling.tools]
        if request_for_sampling.tools
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

    render_kwargs = dict(
        chat_template=request_for_sampling.chat_template,
        chat_template_content_format="auto",
        add_generation_prompt=request_for_sampling.add_generation_prompt,
        continue_final_message=request_for_sampling.continue_final_message,
        tools=tool_dicts,
        documents=request_for_sampling.documents,
        tokenize=tokenize_in_template,
        **chat_template_kwargs,
    )

    return (
        request_for_sampling,
        tool_parser,
        chat_template_kwargs,
        messages_for_render,
        render_kwargs,
    )


async def preprocess_chat_request(
    request: dict[str, Any] | ChatCompletionRequest,
    *,
    tokenizer: TokenizerLike,
    renderer,
    tool_parser_class: type[ToolParser] | None,
) -> PreprocessResult:
    (
        request_for_sampling,
        tool_parser,
        chat_template_kwargs,
        messages,
        render_kwargs,
    ) = _prepare_request(
        request, tokenizer=tokenizer, tool_parser_class=tool_parser_class
    )

    _, engine_prompt = await renderer.render_messages_async(messages, **render_kwargs)

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


def preprocess_chat_request_sync(
    request: dict[str, Any] | ChatCompletionRequest,
    *,
    tokenizer: TokenizerLike,
    renderer,
    tool_parser_class: type[ToolParser] | None,
) -> PreprocessResult:
    """Sync version of preprocess_chat_request for worker processes."""
    (
        request_for_sampling,
        tool_parser,
        chat_template_kwargs,
        messages,
        render_kwargs,
    ) = _prepare_request(
        request, tokenizer=tokenizer, tool_parser_class=tool_parser_class
    )

    _, engine_prompt = renderer.render_messages(messages, **render_kwargs)

    if "prompt_token_ids" in engine_prompt:
        tokens = list(engine_prompt["prompt_token_ids"])
    else:
        tokens = tokenizer.encode(
            engine_prompt["prompt"],
            add_special_tokens=request_for_sampling.add_special_tokens,
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
    ) -> None:
        self.tokenizer = tokenizer
        self.request_for_sampling = request_for_sampling
        self.sampling_params = sampling_params
        self.tool_parser = tool_parser
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

    def process_output(self, output: Any) -> dict[str, Any] | None:
        delta_token_ids = list(output.token_ids or [])
        # vLLM output_processor already applies stop-token/stop-string trimming
        # to text. Re-detokenizing from token_ids can reintroduce stop markers.
        delta_text = output.text or ""

        if self._fast_plain_text:
            if delta_text:
                delta: dict[str, Any] = {
                    "role": "assistant",
                    "content": delta_text,
                }
            elif output.finish_reason:
                delta = {}
            else:
                return None
            return {
                "index": output.index,
                "delta": delta,
                "finish_reason": output.finish_reason,
                "logprobs": output.logprobs,
            }

        current_text = self.previous_text + delta_text
        current_token_ids = self.previous_token_ids + delta_token_ids

        delta_message: DeltaMessage | None = DeltaMessage(content=delta_text)

        if not self.reasoning_is_done and self.reasoning_parser:
            delta_message = self.reasoning_parser.extract_reasoning_streaming(
                self.previous_text,
                current_text,
                delta_text,
                self.previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        should_parse_tools = (
            self.tool_parser is not None
            and self.request_for_sampling.tool_choice != "none"
        )
        if should_parse_tools:
            no_prev_reasoning = (
                delta_message and delta_message.content and not delta_message.reasoning
            )
            if self.reasoning_is_done or no_prev_reasoning:
                delta_message = self.tool_parser.extract_tool_calls_streaming(
                    previous_text=self.previous_text,
                    current_text=current_text,
                    delta_text=delta_text,
                    previous_token_ids=self.previous_token_ids,
                    current_token_ids=current_token_ids,
                    delta_token_ids=delta_token_ids,
                    request=self.request_for_sampling,
                )

        if (
            not self.reasoning_is_done
            and self.reasoning_parser
            and self.reasoning_parser.is_reasoning_end_streaming(
                current_token_ids, delta_token_ids
            )
        ):
            self.reasoning_is_done = True
            self.previous_text = ""
            self.previous_token_ids = []
            current_text = ""
            current_token_ids = []

        choice = None
        if delta_message is None:
            if self.in_progress_tool_calls:
                choice = {
                    "index": output.index,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            tool_call.model_dump(exclude_none=True)
                            for _, tool_call in sorted(
                                self.in_progress_tool_calls.items()
                            )
                        ],
                    },
                    "finish_reason": output.finish_reason,
                    "logprobs": output.logprobs,
                }
                self.in_progress_tool_calls.clear()
            elif output.finish_reason:
                choice = {
                    "index": output.index,
                    "delta": {},
                    "finish_reason": output.finish_reason,
                    "logprobs": output.logprobs,
                }
        elif delta_message.tool_calls:
            for tool_delta in delta_message.tool_calls:
                existing = self.in_progress_tool_calls.get(tool_delta.index)
                merged = self._merge_tool_call(existing, tool_delta)
                self.in_progress_tool_calls[tool_delta.index] = merged
        elif delta_message.content or delta_message.reasoning:
            delta: dict[str, Any] = {"role": "assistant"}
            content = delta_message.content
            if self.in_progress_tool_calls and self._is_control_only_content(content):
                content = None
            if content:
                delta["content"] = content
            if delta_message.reasoning:
                delta["reasoning_content"] = delta_message.reasoning
            if self.in_progress_tool_calls:
                delta["tool_calls"] = [
                    tool_call.model_dump(exclude_none=True)
                    for _, tool_call in sorted(self.in_progress_tool_calls.items())
                ]
                self.in_progress_tool_calls.clear()
            if len(delta) > 1:
                choice = {
                    "index": output.index,
                    "delta": delta,
                    "finish_reason": output.finish_reason,
                    "logprobs": output.logprobs,
                }
        elif self.in_progress_tool_calls:
            choice = {
                "index": output.index,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        tool_call.model_dump(exclude_none=True)
                        for _, tool_call in sorted(self.in_progress_tool_calls.items())
                    ],
                },
                "finish_reason": output.finish_reason,
                "logprobs": output.logprobs,
            }
            self.in_progress_tool_calls.clear()
        elif output.finish_reason:
            choice = {
                "index": output.index,
                "delta": {},
                "finish_reason": output.finish_reason,
                "logprobs": output.logprobs,
            }

        self.previous_text = current_text
        self.previous_token_ids = current_token_ids
        return choice
