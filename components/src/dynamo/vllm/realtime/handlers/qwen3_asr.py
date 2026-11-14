# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transcribe Qwen3-ASR realtime turns as single-turn passes over all audio so far.

vLLM's ``Qwen3ASRRealtimeGeneration.buffer_realtime_audio`` sends each 5-second
segment as a new single-turn prompt on one resumable request. The scheduler
appends that prompt to everything the session already holds, so later segments
decode after a fake multi-turn conversation and the model repeats the opening
(vllm-project/vllm#56218). This follows the official Qwen3-ASR streaming loop
instead: each pass decodes all audio received so far as one user turn, with the
committed transcript prefix as the start of the assistant answer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import numpy as np

from dynamo._core import Context

from ..events import (
    input_audio_transcription_completed_event,
    input_audio_transcription_delta_event,
    input_audio_transcription_failed_event,
)

if TYPE_CHECKING:
    from .transcription import _Turn

logger = logging.getLogger(__name__)

# Same segment length as vLLM's native Qwen3-ASR realtime buffer.
SEGMENT_SECONDS = 5
# Official Qwen3-ASR streaming default (``unfixed_token_num``): the last tokens
# of each pass may still change once more audio arrives, so they stay uncommitted.
UNFIXED_TOKENS = 5

RenderPrompt = Callable[[np.ndarray, list[int]], Awaitable[Any]]


class Qwen3ASRTranscriber:
    """Run one realtime turn as a series of non-resumable engine requests."""

    def __init__(
        self,
        *,
        engine_client: Any,
        render_prompt: RenderPrompt,
        decode: Callable[[list[int]], str],
        sampling_params: Callable[[int], Any],
        sample_rate: int,
        max_tokens_per_segment: int,
    ) -> None:
        self.engine_client = engine_client
        self._render_prompt = render_prompt
        self._decode = decode
        self._sampling_params = sampling_params
        self._segment_samples = sample_rate * SEGMENT_SECONDS
        self._max_tokens_per_segment = max_tokens_per_segment

    async def run_turn(self, turn: _Turn, context: Context) -> None:
        chunks: list[np.ndarray] = []
        received = 0
        decoded = 0
        passes = 0
        committed: list[int] = []
        transcript = ""
        input_tokens = 0
        prefix_tokens = 0
        ended = False

        try:
            while not ended:
                chunk = await turn.audio.get()
                # One pass takes everything already queued, so passes never
                # fall behind audio that arrived while the previous one ran.
                while True:
                    if chunk is None:
                        ended = True
                        break
                    chunks.append(chunk)
                    received += len(chunk)
                    if turn.audio.empty():
                        break
                    chunk = turn.audio.get_nowait()
                if context.is_stopped():
                    return
                if not received or (
                    not ended and received - decoded < self._segment_samples
                ):
                    continue

                audio = np.concatenate(chunks)
                chunks = [audio]
                # vLLM's native path allows realtime_max_tokens per 5-second
                # segment; keep that budget for the new audio in this pass.
                segments = math.ceil((received - decoded) / self._segment_samples)
                result = await self._decode_pass(
                    audio,
                    committed,
                    request_id=f"{turn.request_id}-{passes}",
                    max_tokens=self._max_tokens_per_segment * max(segments, 1),
                    context=context,
                )
                if result is None:
                    return
                generated, input_tokens = result
                prefix_tokens = len(committed)
                decoded = received
                passes += 1

                output = committed + generated
                end = (
                    len(output) if ended else self._fixed_length(output, len(committed))
                )
                if end > len(committed):
                    committed = output[:end]
                    # Qwen's byte-level BPE decodes a token prefix to a text
                    # prefix once a trailing partial character is rolled back.
                    text = self._decode(committed)
                    delta = text[len(transcript) :]
                    transcript = text
                    if delta:
                        await turn.events.put(
                            input_audio_transcription_delta_event(turn.item_id, delta)
                        )

            # input_tokens is the last pass's prompt length, whose committed
            # transcript prefix counts as text; output_tokens counts the final
            # transcript, not tokens regenerated after a rollback.
            await turn.events.put(
                input_audio_transcription_completed_event(
                    turn.item_id,
                    transcript,
                    input_tokens=input_tokens,
                    output_tokens=len(committed),
                    text_tokens=prefix_tokens,
                )
            )
        except Exception as exc:  # noqa: BLE001 - isolate engine failures per turn
            logger.exception(
                "realtime transcription failed for request %s: %s", context.id(), exc
            )
            await turn.events.put(
                input_audio_transcription_failed_event(
                    turn.item_id, "Transcription failed"
                )
            )

    def _fixed_length(self, output: list[int], committed: int) -> int:
        """Return how many tokens of ``output`` are final; never below ``committed``."""
        end = max(committed, len(output) - UNFIXED_TOKENS)
        while end > committed and self._decode(output[:end]).endswith("\ufffd"):
            end -= 1
        return end

    async def _decode_pass(
        self,
        audio: np.ndarray,
        prefix: list[int],
        *,
        request_id: str,
        max_tokens: int,
        context: Context,
    ) -> tuple[list[int], int] | None:
        prompt = await self._render_prompt(audio, prefix)
        generated: list[int] = []
        input_tokens = 0
        finish_reason = None
        stop_reason = None
        async with aclosing(
            self.engine_client.generate(
                prompt=prompt,
                sampling_params=self._sampling_params(max_tokens),
                request_id=request_id,
            )
        ) as results:
            async for result in results:
                if context.is_stopped():
                    return None
                input_tokens = len(getattr(result, "prompt_token_ids", None) or [])
                outputs = getattr(result, "outputs", None)
                if not outputs:
                    continue
                candidate = outputs[0]
                generated.extend(getattr(candidate, "token_ids", None) or [])
                finish_reason = getattr(candidate, "finish_reason", None)
                stop_reason = getattr(candidate, "stop_reason", None)
        # vLLM keeps the EOS or stop token in token_ids; it must not become part
        # of the next pass's assistant prefix.
        if generated and finish_reason == "stop" and not isinstance(stop_reason, str):
            generated.pop()
        return generated, input_tokens


def build_qwen3_asr_transcriber(
    *,
    engine_client: Any,
    serving: Any,
    sample_rate: int,
) -> Qwen3ASRTranscriber:
    """Bind a transcriber to vLLM's realtime serving adapter for Qwen3-ASR."""
    from vllm.inputs import TokensPrompt
    from vllm.multimodal.processing.processor import cached_encode
    from vllm.renderers.inputs.preprocess import parse_model_prompt
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    model_cls = serving.model_cls
    tokenizer = serving.renderer.get_tokenizer()
    # Same single-turn template as Qwen3ASRRealtimeGeneration.buffer_realtime_audio.
    audio_placeholder = model_cls.get_placeholder_str("audio", 0)
    template_ids = cached_encode(
        tokenizer,
        f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n<|im_start|>assistant\n",
    )

    async def render_prompt(audio: np.ndarray, prefix: list[int]) -> Any:
        prompt = TokensPrompt(
            prompt_token_ids=template_ids + prefix,
            multi_modal_data={"audio": audio},
        )
        parsed = parse_model_prompt(serving.model_config, prompt)
        (engine_input,) = await serving.renderer.render_cmpl_async([parsed])
        return engine_input

    def decode(token_ids: list[int]) -> str:
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    def sampling_params(max_tokens: int) -> Any:
        return SamplingParams.from_optional(
            temperature=0.0,
            max_tokens=max_tokens,
            output_kind=RequestOutputKind.DELTA,
            skip_clone=True,
        )

    return Qwen3ASRTranscriber(
        engine_client=engine_client,
        render_prompt=render_prompt,
        decode=decode,
        sampling_params=sampling_params,
        sample_rate=sample_rate,
        max_tokens_per_segment=model_cls.realtime_max_tokens,
    )
