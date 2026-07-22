# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared construction for vLLM realtime serving adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

import numpy as np

StreamingInputFactory = Callable[
    [AsyncGenerator[np.ndarray, None], "asyncio.Queue[list[int]]"],
    AsyncGenerator[Any, None],
]


def build_realtime_serving(
    *,
    engine_client: Any,
    model_name: str,
    model_path: str,
) -> Any:
    """Build vLLM's OpenAI realtime serving adapter for one model."""
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.entrypoints.speech_to_text.realtime.serving import OpenAIServingRealtime

    models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=[BaseModelPath(name=model_name, model_path=model_path)],
        lora_modules=None,
    )
    return OpenAIServingRealtime(
        engine_client=engine_client,
        models=models,
        request_logger=None,
    )
