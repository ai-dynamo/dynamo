# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct vLLM serving adapters and callbacks for Realtime handlers."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import aclosing
from importlib.util import find_spec
from typing import Any

import numpy as np

#: Adapt float32 audio chunks and generated-token feedback to vLLM StreamingInput.
StreamingInputFactory = Callable[
    [AsyncGenerator[np.ndarray, None], "asyncio.Queue[list[int]]"],
    AsyncGenerator[Any, None],
]
#: Await with chat messages, an optional output-token limit, and chat ``tools`` /
#: ``tool_choice`` options to obtain a stream of OpenAI chat-completion SSE frames.
#: The caller owns closing that stream.
ChatCompletionFactory = Callable[
    [list[dict[str, Any]], int | None, dict[str, Any]],
    Awaitable[AsyncGenerator[str, None]],
]
#: Consume cumulative user-text updates alongside committed chat history to warm
#: the prefix cache. Cancellation must complete engine cleanup before returning.
#: Takes the same tool options as ``ChatCompletionFactory`` so both render one prefix.
TextPrefillFactory = Callable[
    [list[dict[str, Any]], AsyncGenerator[str, None], dict[str, Any]],
    Coroutine[Any, Any, None],
]

logger = logging.getLogger(__name__)


def vllm_parser_names(
    tool_parser: str | None, reasoning_parser: str | None
) -> tuple[str | None, str | None]:
    """Keep only the parser names that vLLM registers.

    Realtime text parses with vLLM's chat serving, but the names come from the
    Dynamo parser flags, and some Dynamo parsers have no vLLM counterpart. vLLM
    rejects unknown names, so an unknown name disables that parsing instead of
    failing the worker.
    """
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    def known(name: str | None, kind: str, lookup: Callable[[str], Any]) -> bool:
        if not name:
            return False
        try:
            lookup(name)
        except KeyError:
            logger.warning(
                "%s parser %r is not a vLLM parser, so realtime text sessions "
                "run without it",
                kind,
                name,
            )
            return False
        return True

    return (
        tool_parser
        if known(tool_parser, "Tool call", ToolParserManager.get_tool_parser)
        else None,
        reasoning_parser
        if known(
            reasoning_parser, "Reasoning", ReasoningParserManager.get_reasoning_parser
        )
        else None,
    )


def _build_models(*, engine_client: Any, model_name: str, model_path: str) -> Any:
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels

    return OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=[BaseModelPath(name=model_name, model_path=model_path)],
        lora_modules=None,
    )


def build_realtime_serving(
    *,
    engine_client: Any,
    model_name: str,
    model_path: str,
) -> Any:
    """Build vLLM's OpenAI realtime serving adapter for one model."""
    from vllm.entrypoints.speech_to_text.realtime.serving import OpenAIServingRealtime

    models = _build_models(
        engine_client=engine_client,
        model_name=model_name,
        model_path=model_path,
    )
    return OpenAIServingRealtime(
        engine_client=engine_client,
        models=models,
        request_logger=None,
    )


def build_realtime_text_factories(
    *,
    engine_client: Any,
    model_name: str,
    model_path: str,
    chat_template_path: str | None,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
    default_chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[ChatCompletionFactory, TextPrefillFactory]:
    """Bind final-generation and prefix-warming callbacks to one vLLM engine.

    ``model_name`` is the served API name; ``model_path`` identifies its weights.
    ``chat_template_path`` optionally overrides the model's default chat template.
    Both callbacks use vLLM's chat renderer to apply that template and tokenize.
    ``tool_parser`` enables tool calls in the SSE stream. ``reasoning_parser``
    moves reasoning out of ``delta.content``. ``default_chat_template_kwargs``
    applies to every render, since Realtime sessions carry no template options.

    Awaiting the first callback with chat messages and an optional output-token
    limit returns an SSE stream that the caller must close. The second consumes
    cumulative pending-user text with committed chat history and warms stable,
    complete prefix-cache blocks through vLLM StreamingInput. Cancel and await
    this prefill callback before final generation, which renders the exact
    committed conversation independently. Without prefix caching, updates are
    consumed without starting speculative generation.
    """
    from vllm.engine.protocol import StreamingInput
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse
    from vllm.inputs import tokens_input
    from vllm.renderers.online_renderer import OnlineRenderer
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    # vLLM 0.32 moved the chat helpers to vllm.renderers.
    if find_spec("vllm.renderers.chat_utils") is not None:
        from vllm.renderers.chat_utils import load_chat_template
    else:
        from vllm.entrypoints.chat_utils import load_chat_template

    class RealtimeServingChat(OpenAIServingChat):
        async def chat_completion_stream_generator(
            self,
            request: ChatCompletionRequest,
            result_generator: Any,
            *args: Any,
            **kwargs: Any,
        ) -> AsyncGenerator[str, None]:
            # Closing the SSE generator alone does not close its engine iterator.
            async with aclosing(result_generator), aclosing(
                super().chat_completion_stream_generator(
                    request, result_generator, *args, **kwargs
                )
            ) as stream:
                async for frame in stream:
                    yield frame

    chat_template = load_chat_template(chat_template_path)
    online_renderer = OnlineRenderer(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        request_logger=None,
        chat_template=chat_template,
        chat_template_content_format="auto",
        enable_auto_tools=bool(tool_parser),
        tool_parser=tool_parser,
        reasoning_parser=reasoning_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
    )
    serving = RealtimeServingChat(
        engine_client=engine_client,
        models=_build_models(
            engine_client=engine_client,
            model_name=model_name,
            model_path=model_path,
        ),
        response_role="assistant",
        online_renderer=online_renderer,
        request_logger=None,
        chat_template=chat_template,
        chat_template_content_format="auto",
        reasoning_parser=reasoning_parser or "",
        enable_auto_tools=bool(tool_parser),
        tool_parser=tool_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
    )

    async def create_chat_completion(
        messages: list[dict[str, Any]],
        max_output_tokens: int | None,
        tool_options: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        request = ChatCompletionRequest(
            messages=messages,
            model=model_name,
            max_completion_tokens=max_output_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **tool_options,
        )
        response = await serving.create_chat_completion(request)
        if isinstance(response, ErrorResponse):
            raise ValueError(response.error.message)
        return response

    async def render_tokens(
        messages: list[dict[str, Any]], tool_options: dict[str, Any]
    ) -> list[int]:
        request = ChatCompletionRequest(
            messages=messages,
            model=model_name,
            add_generation_prompt=False,
            continue_final_message=True,
            **tool_options,
        )
        rendered = await serving.render_chat_request(request)
        if isinstance(rendered, ErrorResponse):
            raise ValueError(rendered.error.message)
        _, engine_inputs = rendered
        if len(engine_inputs) != 1:
            raise ValueError("Realtime text prefill requires one rendered prompt")
        token_ids = engine_inputs[0].get("prompt_token_ids")
        if token_ids is None:
            raise ValueError("Realtime text prefill requires tokenized chat input")
        return list(token_ids)

    async def prefill_text(
        messages: list[dict[str, Any]],
        updates: AsyncGenerator[str, None],
        tool_options: dict[str, Any],
    ) -> None:
        cache_config = engine_client.vllm_config.cache_config
        block_size = cache_config.block_size
        if not cache_config.enable_prefix_caching or not block_size:
            async for _ in updates:
                pass
            return

        emitted: list[int] = []
        sampling_params = SamplingParams.from_optional(
            temperature=0.0,
            max_tokens=1,
            output_kind=RequestOutputKind.DELTA,
            skip_clone=True,
        )

        async def streaming_input() -> AsyncGenerator[Any, None]:
            nonlocal emitted
            async for text in updates:
                token_ids = await render_tokens(
                    [*messages, {"role": "user", "content": text}],
                    tool_options,
                )
                if token_ids[: len(emitted)] != emitted:
                    # Appending text can change tokenizer boundaries. Stop this
                    # optimization instead of feeding an incorrect token stream;
                    # final generation independently renders the exact prompt.
                    return
                # Retokenizing appended text may alter the last few tokens.
                # Keep one cache block pending and submit only full blocks,
                # which also avoids engine work that the final request
                # cannot reuse through prefix caching.
                stable_end = max(0, len(token_ids) - block_size)
                end = stable_end // block_size * block_size
                if end > len(emitted):
                    delta = token_ids[len(emitted) : end]
                    emitted = token_ids[:end]
                    yield StreamingInput(tokens_input(delta))

        try:
            result_stream = engine_client.generate(
                prompt=streaming_input(),
                sampling_params=sampling_params,
                request_id=f"rt-prefill-{uuid.uuid4().hex}",
            )
            async for _ in result_stream:
                pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - prefill is an optional optimization
            logger.warning("realtime text prefill disabled for this turn: %s", exc)

    return create_chat_completion, prefill_text
