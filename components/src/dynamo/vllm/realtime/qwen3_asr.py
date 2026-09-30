# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental Qwen3-ASR decoding with a revisable, unreported text tail."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

import numpy as np
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tokenizers import cached_tokenizer_from_config

from .transcription import TranscriptionDelta

CHUNK_SECONDS = 2
UNFIXED_CHUNKS = 2
UNFIXED_TOKENS = 5


async def _audio_prefixes(
    audio_stream: AsyncGenerator[np.ndarray, None], chunk_samples: int
) -> AsyncGenerator[tuple[np.ndarray, bool], None]:
    """Yield accumulated audio at fixed boundaries, then the complete utterance."""
    chunks: list[np.ndarray] = []
    received = 0
    decoded = 0
    async for chunk in audio_stream:
        chunks.append(chunk)
        received += len(chunk)
        if received < decoded + chunk_samples:
            continue
        audio = np.concatenate(chunks)
        chunks = [audio]
        while decoded + chunk_samples <= received:
            decoded += chunk_samples
            yield audio[:decoded], False
    if received:
        yield np.concatenate(chunks), True


def _stable_prefix(tokenizer: Any, text: str, committed: str) -> str:
    """Retain a rollback window without retracting text already sent to clients."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    end = max(0, len(token_ids) - UNFIXED_TOKENS)
    prefix = tokenizer.decode(token_ids[:end])
    # A tokenizer boundary may split a multibyte character. Keep its bytes in the
    # revisable tail rather than emitting a replacement character to the client.
    while "\ufffd" in prefix and end:
        end -= 1
        prefix = tokenizer.decode(token_ids[:end])
    # A short continuation or retokenization can move the rollback boundary
    # behind the previous prefix. Previously emitted text is always immutable.
    return prefix if prefix.startswith(committed) else committed


class Qwen3ASRTranscriber:
    """Re-decode accumulated audio using Qwen's prefix rollback algorithm.

    The first two chunks may revise the whole hypothesis. Thereafter, withhold
    five tokens and force the already emitted prefix on subsequent requests.
    This adapts the Qwen SDK algorithm to append-only Realtime API deltas.
    """

    def __init__(self, *, engine_client: Any, serving: Any) -> None:
        self.engine_client = engine_client
        self.model_cls = serving.model_cls
        self.model_config = serving.model_config
        self.renderer = serving.renderer
        self.speech_config = self.model_cls.get_speech_to_text_config(
            self.model_config, "transcribe"
        )
        self.audio_token_id = self.model_config.hf_config.thinker_config.audio_token_id
        self.tokenizer = cached_tokenizer_from_config(self.model_config)
        self.chunk_samples = int(self.speech_config.sample_rate * CHUNK_SECONDS)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

    async def _decode(
        self, audio: np.ndarray, prefix: str, request_id: str
    ) -> TranscriptionDelta:
        # The session handler currently accepts English only. Forcing the
        # language keeps the model's language/<asr_text> header out of deltas.
        prompt = self.model_cls.get_generation_prompt(
            SpeechToTextParams(
                audio=audio,
                stt_config=self.speech_config,
                model_config=self.model_config,
                language="en",
            )
        )
        prompt["prompt_token_ids"] = [
            *prompt["prompt_token_ids"],
            *self.tokenizer.encode(prefix, add_special_tokens=False),
        ]
        parsed = parse_model_prompt(self.model_config, prompt)
        (engine_input,) = await self.renderer.render_cmpl_async([parsed])
        final = None
        async with aclosing(
            self.engine_client.generate(
                prompt=engine_input,
                sampling_params=self.sampling_params,
                request_id=request_id,
            )
        ) as results:
            async for result in results:
                if result.outputs:
                    final = result
        if final is None:
            raise RuntimeError("Qwen3-ASR returned no transcription output")
        output = final.outputs[0]
        if output.finish_reason == "length":
            raise RuntimeError("Qwen3-ASR reached the transcription token limit")
        prompt_token_ids = final.prompt_token_ids or []
        return TranscriptionDelta(
            text=prefix + output.text,
            input_tokens=len(prompt_token_ids),
            output_tokens=len(output.token_ids),
            input_text_tokens=(
                len(prompt_token_ids) - prompt_token_ids.count(self.audio_token_id)
            ),
        )

    async def __call__(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        request_id: str,
    ) -> AsyncGenerator[TranscriptionDelta, None]:
        committed = ""
        hypothesis = ""
        decoded_samples = 0
        chunk_index = 0
        async with aclosing(
            _audio_prefixes(audio_stream, self.chunk_samples)
        ) as prefixes:
            async for audio, final in prefixes:
                input_tokens = output_tokens = input_text_tokens = 0
                if len(audio) > decoded_samples:
                    result = await self._decode(
                        audio, committed, f"{request_id}_{chunk_index}"
                    )
                    hypothesis = result.text
                    input_tokens = result.input_tokens
                    output_tokens = result.output_tokens
                    input_text_tokens = result.input_text_tokens
                    decoded_samples = len(audio)
                    chunk_index += 1
                if final:
                    stable = hypothesis
                elif chunk_index >= UNFIXED_CHUNKS:
                    stable = _stable_prefix(self.tokenizer, hypothesis, committed)
                else:
                    stable = ""
                delta = stable[len(committed) :]
                committed = stable
                yield TranscriptionDelta(
                    delta, input_tokens, output_tokens, input_text_tokens
                )
