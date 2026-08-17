#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.reasoning import ReasoningParser
from vllm.renderers import ChatParams, merge_kwargs
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.tool_parsers.utils import get_json_schema_from_tools
from vllm.utils.async_utils import make_async

from dynamo.llm.exceptions import InvalidArgument

from .thinking import apply_default_thinking_mode_to_template_kwargs


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
    guided_decoding: dict[str, Any] | None = None
    uses_dynamo_json_tool_call_fallback: bool = False


_ASYNC_TOKENIZER_POOL: dict[int, Callable[..., Awaitable[Any]]] = {}
SKIP_REQUEST_VALIDATION = os.getenv("DYN_VLLM_SKIP_REQUEST_VALIDATION", "1") == "1"


def _reject_non_finite_json(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def _is_named_tool_choice(tool_choice: Any) -> bool:
    """True only for a well-formed named tool choice.

    tool_choice arrives either validated (ChatCompletionNamedToolChoiceParam) or,
    on the DYN_VLLM_SKIP_REQUEST_VALIDATION fast path, as the raw client dict, so
    both shapes are checked. Testing only `not isinstance(tool_choice, str)` would
    classify malformed values such as {} or {"type": "function"} as a forced tool
    choice, which then gates the conflict check below.
    """
    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        return bool(tool_choice.function.name)
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        return (
            tool_choice.get("type") == "function"
            and isinstance(function, dict)
            and bool(function.get("name"))
        )
    return False


def _is_forced_tool_choice(tool_choice: Any) -> bool:
    return tool_choice == "required" or _is_named_tool_choice(tool_choice)


def _typed_tool_choice(tool_choice: Any) -> Any:
    """Normalize a raw named tool choice into the type vLLM's helpers expect.

    vLLM's get_json_schema_from_tools only recognizes a typed
    ChatCompletionNamedToolChoiceParam. On the DYN_VLLM_SKIP_REQUEST_VALIDATION
    fast path tool_choice can still be the raw client dict when the request
    already carries typed tools, and that helper silently returns None for it --
    leaving a named forced tool call with no constraint at all, which is the
    defect this whole path exists to fix. Callers reach this only after
    _is_named_tool_choice has confirmed the shape, so the name is present.
    """
    if isinstance(tool_choice, dict):
        return ChatCompletionNamedToolChoiceParam(
            type="function",
            function=ChatCompletionNamedFunction(name=tool_choice["function"]["name"]),
        )
    return tool_choice


def _has_explicit_output_constraint(request: ChatCompletionRequest) -> bool:
    """Whether the caller set a constraint over the whole generated token stream.

    CONTENT response_format variants (json_object, json_schema) are deliberately
    excluded. The OpenAI spec scopes those to the message the model returns to the
    user, "rather than when the model calls a tool", so a forced tool choice makes
    them unenforceable rather than contradictory -- vLLM drops them for exactly
    this case in ToolParser.adjust_request (response_format = None).

    response_format={"type": "structural_tag"} is NOT a content format: vLLM
    normalizes it into structured_outputs.structural_tag, a grammar over the whole
    token stream, which is the same slot the tool grammar needs. It is counted
    here so a forced choice rejects it instead of silently discarding it.

    guided_* and an explicit structured_outputs have no content scoping either.
    """
    if request.structured_outputs is not None:
        return True
    # Reads response_format too, so this catches the structural_tag variant
    # regardless of which field the caller used to express it.
    extracted = request.extract_structured_outputs()
    if extracted is not None and extracted.structural_tag is not None:
        return True
    request_extra = request.model_extra or {}
    if request_extra.get("guided_choice"):
        return True
    return any(
        request_extra.get(key) is not None
        for key in ("guided_json", "guided_regex", "guided_grammar")
    )


def _tool_is_strict(tool: Any) -> bool:
    return tool.function.strict is True


def _should_build_tool_call_guidance(
    request: ChatCompletionRequest,
    *,
    structural_tag_mode: str,
    structural_tag_scope: str,
) -> bool:
    tool_choice = request.tool_choice or "auto"
    if not request.tools:
        return False

    # TODO: preprocessor/structural_tag.rs installs a tool-call BAN structural tag
    # for tool_choice="none" when the mode is on and
    # exclude_tools_when_tool_choice_none is false (apply_tool_call_ban). Neither
    # Python path does. When tools stay in the rendered prompt the model can still
    # see them, so "none" is only a prompt-level suggestion here rather than an
    # enforced guarantee. sglang_prepost.py has the same gap.
    if tool_choice == "none":
        return False
    if _is_forced_tool_choice(tool_choice):
        return True
    if structural_tag_mode != "on":
        return False
    if tool_choice != "auto":
        return False
    if structural_tag_scope == "always":
        return True
    # An explicit single-call request is enough to attempt structural-tag
    # guidance. Forced-choice JSON guidance also bounds its array schema below.
    explicit_single_call = (
        "parallel_tool_calls" in request.model_fields_set
        and request.parallel_tool_calls is False
    )
    return explicit_single_call or any(_tool_is_strict(tool) for tool in request.tools)


def _request_for_vllm_structural_tag(
    request: ChatCompletionRequest,
    *,
    structural_tag_schema: str,
) -> ChatCompletionRequest:
    strict_schema = structural_tag_schema == "strict"
    tools = [
        tool.model_copy(
            update={
                "function": tool.function.model_copy(
                    update={
                        "strict": True if strict_schema else tool.function.strict,
                    }
                )
            }
        )
        for tool in request.tools or []
    ]
    return request.model_copy(update={"tools": tools})


def build_tool_call_guided_decoding(
    request: ChatCompletionRequest,
    tool_parser: ToolParser | None,
    *,
    parser_guided_decoding: dict[str, Any] | None = None,
    structural_tag_mode: str = "off",
    structural_tag_scope: str = "auto",
    structural_tag_schema: str = "auto",
) -> dict[str, Any] | None:
    """Build tool-call guidance through vLLM's configured tool parser."""
    if not _should_build_tool_call_guidance(
        request,
        structural_tag_mode=structural_tag_mode,
        structural_tag_scope=structural_tag_scope,
    ):
        return None

    if structural_tag_mode == "on" and tool_parser is not None:
        request_for_tag = _request_for_vllm_structural_tag(
            request,
            structural_tag_schema=structural_tag_schema,
        )
        structural_tag = tool_parser.get_structural_tag(request_for_tag)
        if structural_tag is not None:
            tag_value = (
                structural_tag.model_dump()
                if hasattr(structural_tag, "model_dump")
                else structural_tag
            )
            return {"structural_tag": tag_value}

    if parser_guided_decoding is not None:
        return parser_guided_decoding

    tool_choice = request.tool_choice or "auto"
    if _is_forced_tool_choice(tool_choice):
        # Parsers that require native tool syntax (e.g. Gemma4Engine, PoolsideV1)
        # leave structured_outputs unset for required/named choices on purpose;
        # forcing a JSON schema conflicts with their wire format and can crash
        # EngineCore under speculative decoding. Only fall back for parsers that
        # advertise JSON support (or when no parser is active).
        if tool_parser is not None and not getattr(
            tool_parser, "supports_required_and_named", True
        ):
            return None
        # tool_choice is already the typed named param by here; see
        # _validate_chat_completion_request.
        json_schema = get_json_schema_from_tools(tool_choice, request.tools)
        if json_schema is not None:
            if (
                request.parallel_tool_calls is False
                and json_schema.get("type") == "array"
            ):
                json_schema["maxItems"] = 1
            return {"json": json_schema}
    return None


# Convert vLLM structured-output parameters into Dynamo guided-decoding options.
def _guided_decoding_from_structured_outputs(
    structured_outputs: Any,
) -> dict[str, Any] | None:
    if structured_outputs is None:
        return None

    guided_decoding: dict[str, Any]
    if structured_outputs.json is not None:
        guided_decoding = {"json": structured_outputs.json}
    elif structured_outputs.regex is not None:
        guided_decoding = {"regex": structured_outputs.regex}
    elif structured_outputs.choice is not None:
        guided_decoding = {"choice": structured_outputs.choice}
    elif structured_outputs.grammar is not None:
        guided_decoding = {"grammar": structured_outputs.grammar}
    elif structured_outputs.json_object:
        guided_decoding = {"json": {"type": "object"}}
    elif structured_outputs.structural_tag is not None:
        guided_decoding = {"structural_tag": structured_outputs.structural_tag}
    else:
        return None

    if structured_outputs.whitespace_pattern is not None:
        guided_decoding["whitespace_pattern"] = structured_outputs.whitespace_pattern
    return guided_decoding


# Explicit assistant constraints take precedence over automatic tool guidance.
def _build_assistant_guided_decoding(
    request: ChatCompletionRequest,
) -> dict[str, Any] | None:
    guided_decoding = _guided_decoding_from_structured_outputs(
        request.extract_structured_outputs()
    )

    request_extra = request.model_extra or {}
    # Pick a single legacy guided_* constraint by precedence rather than merging
    # several keys into one dict, because guided_decoding carries exactly one
    # constraint and the elif chain above already honors that for
    # structured_outputs.
    #
    # TODO: first-match-wins silently discards the other constraints the caller
    # explicitly set, with no error and no annotation.
    # GuidedDecodingOptions::validate in protocols/common.rs rejects the same
    # request outright, so `guided_json` + `guided_regex` is a 400 through the Rust
    # frontend and a silent single-constraint request here. Rejecting is the
    # correct behavior; it is left as-is only to avoid adding a second new 400 to
    # this change. Note that validate() also counts whitespace_pattern toward its
    # exclusivity limit, so the {"json": ..., "whitespace_pattern": ...} pair built
    # below is accepted here and rejected there -- whitespace_pattern modifies a
    # grammar rather than being one, so that counter is the side that is wrong.
    legacy_guidance: dict[str, Any] = {}
    for key, value in (
        ("json", request_extra.get("guided_json")),
        ("regex", request_extra.get("guided_regex")),
        ("grammar", request_extra.get("guided_grammar")),
        ("choice", request_extra.get("guided_choice") or None),
    ):
        if value is not None:
            legacy_guidance = {key: value}
            break
    if legacy_guidance:
        # Legacy guided_* takes precedence over structured_outputs (prior
        # behavior), but as a single constraint.
        guided_decoding = legacy_guidance
        whitespace_pattern = request_extra.get("guided_whitespace_pattern")
        if whitespace_pattern is not None:
            guided_decoding["whitespace_pattern"] = whitespace_pattern
    return guided_decoding


def _tool_call_parsing_enabled(request: ChatCompletionRequest) -> bool:
    """Whether this request permits model output to be parsed as tool calls.

    A configured parser is a model capability, not a per-request instruction.
    Automatic tool choice yields to an explicit assistant-output constraint;
    forced choices retain the parser because their output must be a tool call.
    """
    if not request.tools:
        return False

    tool_choice = request.tool_choice or "auto"
    if tool_choice == "none":
        return False
    if _is_forced_tool_choice(tool_choice):
        return True
    return _build_assistant_guided_decoding(request) is None


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


# Fully validate nested fields only when the fast path leaves raw dictionaries.
def _validate_chat_completion_request(
    request: dict[str, Any] | ChatCompletionRequest,
) -> ChatCompletionRequest:
    if isinstance(request, ChatCompletionRequest):
        return request
    if not SKIP_REQUEST_VALIDATION:
        return ChatCompletionRequest.model_validate(request)

    validated_request = ChatCompletionRequest.model_construct(**request)
    has_unvalidated_tools = validated_request.tools and any(
        not hasattr(tool, "model_dump") for tool in validated_request.tools
    )
    if (
        has_unvalidated_tools
        or isinstance(validated_request.response_format, dict)
        or isinstance(validated_request.structured_outputs, dict)
    ):
        return ChatCompletionRequest.model_validate(request)
    # model_construct leaves tool_choice as the raw client dict. Normalize it
    # here, once, rather than at each consumer: vLLM's helpers branch on the
    # typed ChatCompletionNamedToolChoiceParam, and a dict makes
    # get_json_schema_from_tools silently return None while
    # get_structural_tag raises AttributeError on .model_dump().
    if _is_named_tool_choice(validated_request.tool_choice):
        validated_request.tool_choice = _typed_tool_choice(
            validated_request.tool_choice
        )
    return validated_request


def _prepare_request(
    request: dict[str, Any] | ChatCompletionRequest,
    *,
    tokenizer: TokenizerLike,
    tool_parser_class: type[ToolParser] | None,
    exclude_tools_when_tool_choice_none: bool = True,
    enable_auto_tool_choice: bool = False,
    default_chat_template_kwargs: dict[str, Any] | None = None,
    default_thinking_mode: str | None = None,
    validated_request: ChatCompletionRequest | None = None,
) -> tuple[ChatCompletionRequest, ToolParser | None, dict[str, Any], Any, ChatParams]:
    """Validate request and build arguments for template rendering.

    Args:
        request: The raw request. Keep it as the caller received it (a dict on
            the pythonized path) so transport-only aliases below are readable.
        validated_request: A pre-validated model when the caller already built
            one (avoids re-validating); semantic fields are read from it while
            transport-only aliases still come from the raw ``request`` dict.

    Returns:
        request_for_sampling: Validated ChatCompletionRequest.
        tool_parser: Instantiated tool parser, or None.
        chat_template_kwargs: Template kwargs (for PreprocessResult).
        messages_for_render: Messages to pass as first arg to render_messages.
        chat_params: ChatParams for render_messages / render_messages_async.
    """
    request_for_sampling = (
        validated_request
        if validated_request is not None
        else _validate_chat_completion_request(request)
    )

    tool_parser: ToolParser | None = None
    # enable_auto_tool_choice is retained for API compatibility; parser class
    # availability represents the configured server capability. Activation is
    # request-scoped so content-only output cannot be reclassified as a call.
    _ = enable_auto_tool_choice
    if tool_parser_class and _tool_call_parsing_enabled(request_for_sampling):
        tool_parser = tool_parser_class(tokenizer, request_for_sampling.tools)
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
    # serde's `alias` is deserialize-only, so pythonize emits the Rust field
    # name `chat_template_args`; read it too or client kwargs are dropped.
    raw_template_args = (
        request.get("chat_template_args") if isinstance(request, dict) else None
    )
    # Reuse vLLM's merge so unset request values (None/"auto") keep the server
    # default; copy the result since the default dict is shared across requests
    # and we mutate reasoning_effort below.
    chat_template_kwargs = dict(
        merge_kwargs(
            default_chat_template_kwargs,
            request_for_sampling.chat_template_kwargs or raw_template_args or {},
        )
    )
    # reasoning_effort is a request-level thinking control. Put an explicit
    # value into the kwargs before applying the deployment default so the two
    # cannot produce contradictory template controls.
    if request_for_sampling.reasoning_effort is not None:
        chat_template_kwargs["reasoning_effort"] = request_for_sampling.reasoning_effort
    chat_template_kwargs = apply_default_thinking_mode_to_template_kwargs(
        chat_template_kwargs,
        default_thinking_mode,
        request_has_root_thinking=(
            isinstance(request, dict) and request.get("thinking") is not None
        ),
    )
    # Don't let an absent top-level field clobber a nested reasoning_effort.
    if request_for_sampling.reasoning_effort is None:
        chat_template_kwargs.setdefault("reasoning_effort", None)

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
        # Renderer-managed keys last so a nested duplicate can't raise TypeError.
        chat_template_kwargs={
            **chat_template_kwargs,
            "add_generation_prompt": request_for_sampling.add_generation_prompt,
            "continue_final_message": request_for_sampling.continue_final_message,
            "tools": tool_dicts,
            "documents": request_for_sampling.documents,
            "tokenize": tokenize_in_template,
        },
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
    exclude_tools_when_tool_choice_none: bool = True,
    enable_auto_tool_choice: bool = False,
    default_chat_template_kwargs: dict[str, Any] | None = None,
    default_thinking_mode: str | None = None,
    structural_tag_mode: str = "off",
    structural_tag_scope: str = "auto",
    structural_tag_schema: str = "auto",
) -> PreprocessResult:
    validated_request = _validate_chat_completion_request(request)
    assistant_guided_decoding = _build_assistant_guided_decoding(validated_request)
    client_structured_guidance = deepcopy(
        _guided_decoding_from_structured_outputs(
            validated_request.extract_structured_outputs()
        )
    )
    is_forced_tool_choice = _is_forced_tool_choice(validated_request.tool_choice)
    if is_forced_tool_choice and not validated_request.tools:
        raise InvalidArgument("When using tool_choice, tools must be set.")
    # Must be read BEFORE _prepare_request: it calls ToolParser.adjust_request(),
    # which mutates this same request object in place and can set
    # structured_outputs itself. Reading it afterwards would treat a
    # parser-generated constraint as one the caller sent and reject the request.
    has_explicit_output_constraint = _has_explicit_output_constraint(validated_request)
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
        default_chat_template_kwargs=default_chat_template_kwargs,
        default_thinking_mode=default_thinking_mode,
        validated_request=validated_request,
    )

    adjusted_structured_guidance = _guided_decoding_from_structured_outputs(
        request_for_sampling.structured_outputs
    )
    parser_guided_decoding = (
        adjusted_structured_guidance
        if tool_parser is not None
        and adjusted_structured_guidance != client_structured_guidance
        else None
    )
    tool_guided_decoding = build_tool_call_guided_decoding(
        request_for_sampling,
        tool_parser,
        parser_guided_decoding=parser_guided_decoding,
        structural_tag_mode=structural_tag_mode,
        structural_tag_scope=structural_tag_scope,
        structural_tag_schema=structural_tag_schema,
    )
    # A forced tool choice already claims the decoder's single grammar slot, so a
    # caller-set guided_*/structured_outputs constraint over the same token stream
    # cannot also be honored. Reject it (InvalidArgument -> 400) rather than
    # silently drop it, matching has_explicit_guided_decoding in
    # preprocessor/tool_choice.rs.
    #
    # response_format is NOT part of that test: it is scoped to the message the
    # model returns to the user, not to tool calls, so it is dropped for a forced
    # choice instead of rejected. That matches both preprocessor/tool_choice.rs
    # (which clears gd.json) and vLLM's own ToolParser.adjust_request (which sets
    # response_format = None), and it keeps a request both the OpenAI spec and
    # vLLM accept from returning 400 here.
    if is_forced_tool_choice and has_explicit_output_constraint:
        raise InvalidArgument(
            "tool_choice forces a tool call and cannot be combined with an "
            "explicit guided_* or structured_outputs constraint."
        )
    guided_decoding = (
        assistant_guided_decoding
        if assistant_guided_decoding is not None and not is_forced_tool_choice
        else tool_guided_decoding
    )
    # build_tool_call_guided_decoding falls back to Dynamo's generic JSON schema
    # when a forced tool choice has no parser-provided grammar.  That can happen
    # both without a parser and with parsers (for example Hermes) whose
    # adjust_request() leaves structured_outputs unchanged.  In either case the
    # model emits Dynamo's bare JSON wire format, which must bypass the parser's
    # native-syntax decoder during postprocessing.
    uses_dynamo_json_tool_call_fallback = (
        is_forced_tool_choice
        and parser_guided_decoding is None
        and guided_decoding is tool_guided_decoding
        and isinstance(tool_guided_decoding, dict)
        and "json" in tool_guided_decoding
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
        guided_decoding=guided_decoding,
        uses_dynamo_json_tool_call_fallback=uses_dynamo_json_tool_call_fallback,
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
        uses_dynamo_json_tool_call_fallback: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.request_for_sampling = request_for_sampling
        self.sampling_params = sampling_params
        self.tool_parser = tool_parser
        self.stream_response = stream_response
        self._uses_dynamo_json_tool_call_fallback = uses_dynamo_json_tool_call_fallback
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
        self.reasoning_parser = (
            reasoning_parser_class(
                tokenizer,
                chat_template_kwargs=chat_template_kwargs,
            )
            if reasoning_parser_class and not thinking_disabled
            else None
        )
        self._fast_plain_text = (
            self.tool_parser is None
            and self.reasoning_parser is None
            and not self._uses_dynamo_json_tool_call_fallback
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
        self._dynamo_json_fallback_chunks: dict[int, list[str]] = {}

    def _decode_dynamo_json_fallback_tool_calls(
        self, text: str
    ) -> list[dict[str, Any]]:
        """Convert Dynamo's JSON fallback wire format into OpenAI tool calls."""
        try:
            tool_calls = json.loads(text, parse_constant=_reject_non_finite_json)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "Dynamo JSON tool-call fallback was not valid JSON"
            ) from exc

        available_tool_names = {
            tool.function.name for tool in self.request_for_sampling.tools or []
        }
        tool_choice = self.request_for_sampling.tool_choice
        required_tool_name = None
        if _is_named_tool_choice(tool_choice):
            if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
                required_tool_name = tool_choice.function.name
            else:
                required_tool_name = tool_choice["function"]["name"]
            tool_calls = [
                {
                    "name": required_tool_name,
                    "parameters": tool_calls,
                }
            ]
        elif not isinstance(tool_calls, list) or not tool_calls:
            raise ValueError(
                "Dynamo JSON tool-call fallback must contain a tool-call array"
            )

        if (
            self.request_for_sampling.parallel_tool_calls is False
            and len(tool_calls) > 1
        ):
            raise ValueError(
                "Dynamo JSON tool-call fallback returned multiple tool calls "
                "when parallel_tool_calls is false"
            )

        parsed_tool_calls: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                raise TypeError(
                    "Dynamo JSON tool-call fallback entries must be objects"
                )
            name = tool_call.get("name")
            if "parameters" not in tool_call:
                raise TypeError(
                    "Dynamo JSON tool-call fallback entries must include parameters"
                )
            parameters = tool_call["parameters"]
            if not isinstance(name, str) or name not in available_tool_names:
                raise ValueError(
                    "Dynamo JSON tool-call fallback named an unavailable tool"
                )
            if required_tool_name is not None and name != required_tool_name:
                raise ValueError(
                    "Dynamo JSON tool-call fallback did not use the required tool"
                )
            parsed_tool_calls.append(
                DeltaToolCall(
                    index=index,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=name,
                        arguments=json.dumps(parameters, separators=(",", ":")),
                    ),
                ).model_dump(exclude_none=True)
            )
        return parsed_tool_calls

    def _process_dynamo_json_fallback_tool_calls(
        self, output: Any
    ) -> dict[str, Any] | None:
        chunks = self._dynamo_json_fallback_chunks.setdefault(output.index, [])
        if output.text:
            chunks.append(output.text)
        if not output.finish_reason:
            return None

        text = "".join(self._dynamo_json_fallback_chunks.pop(output.index))
        delta: dict[str, Any]
        try:
            tool_calls = self._decode_dynamo_json_fallback_tool_calls(text)
        except (TypeError, ValueError):
            delta = {"role": "assistant", "content": text} if text else {}
            return self._build_choice(output, delta)

        delta = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        return self._build_choice(output, delta)

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
            return self._compose_delta_message(
                saved_reasoning, extracted.content or None
            )

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
        if self._uses_dynamo_json_tool_call_fallback:
            return self._process_dynamo_json_fallback_tool_calls(output)
        if self._should_buffer_for_non_streaming_tool_parse():
            return self._process_non_streaming_tool_output(output)

        # TODO: Emit the role only once per choice here instead of relying on
        # HTTP stream deduplication to remove repeated postprocessor roles.
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
        return choice
