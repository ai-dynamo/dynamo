# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model adapters producing append-only transcription deltas."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

import numpy as np

from .serving import StreamingInputFactory


@dataclass(frozen=True)
class TranscriptionDelta:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    input_text_tokens: int = 0


Transcribe = Callable[
    [AsyncGenerator[np.ndarray, None], str],
    AsyncGenerator[TranscriptionDelta, None],
]


class VllmRealtimeTranscriber:
    """Adapt vLLM's native streaming input and token feedback protocol."""

    def __init__(
        self,
        *,
        engine_client: Any,
        streaming_input_factory: StreamingInputFactory,
        sampling_params_factory: Callable[[], Any],
    ) -> None:
        self.engine_client = engine_client
        self._streaming_input_factory = streaming_input_factory
        self._sampling_params_factory = sampling_params_factory

    async def __call__(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        request_id: str,
    ) -> AsyncGenerator[TranscriptionDelta, None]:
        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        streaming_input = self._streaming_input_factory(audio_stream, input_stream)
        counted_input = False
        async with aclosing(
            self.engine_client.generate(
                prompt=streaming_input,
                sampling_params=self._sampling_params_factory(),
                request_id=request_id,
            )
        ) as results:
            async for result in results:
                if not result.outputs:
                    continue
                candidate = result.outputs[0]
                token_ids = list(candidate.token_ids)
                if token_ids:
                    input_stream.put_nowait(token_ids)
                input_tokens = 0
                if not counted_input:
                    input_tokens = len(result.prompt_token_ids or [])
                    counted_input = True
                yield TranscriptionDelta(
                    text=candidate.text,
                    input_tokens=input_tokens,
                    output_tokens=len(token_ids),
                )
