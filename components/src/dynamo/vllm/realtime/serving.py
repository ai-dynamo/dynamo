# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared construction for vLLM realtime serving adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
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


def build_chat_completion_factory(
    *,
    engine_client: Any,
    model_name: str,
    model_path: str,
    chat_template_path: str | None,
) -> ChatCompletionFactory:
    """Build a text-only chat adapter using vLLM's OpenAI serving stack."""
    from vllm.entrypoints.chat_utils import load_chat_template
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.renderers.online_renderer import OnlineRenderer

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

    return create_chat_completion
