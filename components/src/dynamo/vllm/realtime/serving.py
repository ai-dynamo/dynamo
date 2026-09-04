# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared construction for vLLM realtime serving adapters."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from typing import Any

import numpy as np

StreamingInputFactory = Callable[
    [AsyncGenerator[np.ndarray, None], "asyncio.Queue[list[int]]"],
    AsyncGenerator[Any, None],
]
ChatCompletionFactory = Callable[
    [list[dict[str, str]], int | None],
    Awaitable[AsyncGenerator[str, None]],
]
TextPrefillFactory = Callable[
    [list[dict[str, str]], AsyncGenerator[tuple[str, bool], None]],
    Coroutine[Any, Any, None],
]

logger = logging.getLogger(__name__)


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
) -> tuple[ChatCompletionFactory, TextPrefillFactory]:
    """Build final-generation and incremental-prefill text adapters."""
    from vllm.engine.protocol import StreamingInput
    from vllm.entrypoints.chat_utils import load_chat_template
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.inputs import tokens_input
    from vllm.renderers.online_renderer import OnlineRenderer
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    chat_template = load_chat_template(chat_template_path)
    online_renderer = OnlineRenderer(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        request_logger=None,
        chat_template=chat_template,
        chat_template_content_format="auto",
    )
    serving = OpenAIServingChat(
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
    )

    async def create_chat_completion(
        messages: list[dict[str, str]],
        max_output_tokens: int | None,
    ) -> AsyncGenerator[str, None]:
        request = ChatCompletionRequest(
            messages=messages,
            model=model_name,
            max_completion_tokens=max_output_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        response = await serving.create_chat_completion(request)
        if not hasattr(response, "__aiter__"):
            message = getattr(response, "message", "Chat request failed")
            raise ValueError(message)
        return response

    async def render_tokens(
        messages: list[dict[str, str]], *, final: bool
    ) -> list[int]:
        request = ChatCompletionRequest(
            messages=messages,
            model=model_name,
            add_generation_prompt=final,
            continue_final_message=not final,
        )
        rendered = await serving.render_chat_request(request)
        if not isinstance(rendered, tuple):
            message = getattr(rendered, "message", "Chat prompt rendering failed")
            raise ValueError(message)
        _, engine_inputs = rendered
        if len(engine_inputs) != 1:
            raise ValueError("Realtime text prefill requires one rendered prompt")
        token_ids = engine_inputs[0].get("prompt_token_ids")
        if token_ids is None:
            raise ValueError("Realtime text prefill requires tokenized chat input")
        return list(token_ids)

    async def prefill_text(
        messages: list[dict[str, str]],
        updates: AsyncGenerator[tuple[str, bool], None],
    ) -> None:
        cache_config = engine_client.vllm_config.cache_config
        block_size = cache_config.block_size
        if not cache_config.enable_prefix_caching or not block_size:
            async for _ in updates:
                pass
            return

        emitted: list[int] = []
        text_parts: list[str] = []
        sampling_params = SamplingParams.from_optional(
            temperature=0.0,
            max_tokens=1,
            output_kind=RequestOutputKind.DELTA,
            skip_clone=True,
        )

        async def streaming_input() -> AsyncGenerator[Any, None]:
            nonlocal emitted
            async for text_delta, final in updates:
                text_parts.append(text_delta)
                token_ids = await render_tokens(
                    [*messages, {"role": "user", "content": "".join(text_parts)}],
                    final=final,
                )
                if token_ids[: len(emitted)] != emitted:
                    # Appending text can change tokenizer boundaries. Stop this
                    # optimization instead of feeding an incorrect token stream;
                    # final generation independently renders the exact prompt.
                    return
                if final:
                    end = len(token_ids)
                else:
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
