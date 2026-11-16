# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenAI audio/speech adapter for Speech NIM text-to-speech synthesis."""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import aclosing

import grpc
from riva.client import AudioEncoding

from dynamo._core import Context, InvalidArgument
from dynamo.common.protocols.audio_protocol import (
    AudioData,
    NvAudioSpeechResponse,
    NvCreateAudioSpeechRequest,
)
from dynamo.runtime import dynamo_endpoint

from ..riva import cancel_on_context_stop


class SpeechNimAudioSpeechBackend:
    """Stream Speech NIM TTS audio through Dynamo's OpenAI audio response contract."""

    def __init__(
        self,
        *,
        tts_service,
        model_name: str,
        voice: str,
        language_code: str,
        sample_rate_hz: int,
    ) -> None:
        self.tts_service = tts_service
        self.model_name = model_name
        self.voice = voice
        self.language_code = language_code
        self.sample_rate_hz = sample_rate_hz

    def _validate(self, request: NvCreateAudioSpeechRequest) -> None:
        if not request.input.strip():
            raise ValueError("input must contain text")
        if request.model not in (None, self.model_name):
            raise ValueError(f"model must be '{self.model_name}'")
        if request.response_format not in (None, "pcm"):
            raise ValueError("Speech NIM TTS currently supports response_format='pcm'")
        if request.data_source not in (None, "b64_json"):
            raise ValueError("Speech NIM TTS currently supports data_source='b64_json'")
        if request.speed not in (None, 1.0):
            raise ValueError("Speech NIM TTS does not support the speed parameter")
        if request.instructions:
            raise ValueError("Speech NIM TTS does not support instructions")
        for name, value in (
            ("task_type", request.task_type),
            ("ref_audio", request.ref_audio),
            ("ref_text", request.ref_text),
            ("max_new_tokens", request.max_new_tokens),
            ("nvext.cfg_scale", request.nvext.cfg_scale if request.nvext else None),
        ):
            if value is not None:
                raise ValueError(f"Speech NIM TTS does not support {name}")

    async def generate(
        self, request: NvCreateAudioSpeechRequest, context: Context
    ) -> AsyncGenerator[NvAudioSpeechResponse, None]:
        """Yield each Riva SDK ``SynthesizeOnline`` response as one PCM chunk."""
        self._validate(request)
        response_id = f"speech_{uuid.uuid4().hex}"
        created = int(time.time())
        try:
            # Starting the RPC is nonblocking; only reading its responses blocks.
            call = self.tts_service.synthesize_online(
                request.input,
                voice_name=request.voice or self.voice,
                language_code=request.language or self.language_code,
                encoding=AudioEncoding.LINEAR_PCM,
                sample_rate_hz=self.sample_rate_hz,
            )
            async with cancel_on_context_stop(context, call.cancel):
                responses = iter(call)
                while (
                    response := await asyncio.to_thread(next, responses, None)
                ) is not None:
                    yield NvAudioSpeechResponse(
                        id=response_id,
                        model=self.model_name,
                        created=created,
                        data=[
                            AudioData(
                                output_format="pcm",
                                b64_json=base64.b64encode(response.audio).decode(),
                            )
                        ],
                    )
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise InvalidArgument(
                    "Speech NIM rejected the synthesis parameters"
                ) from exc
            raise

    @dynamo_endpoint(NvCreateAudioSpeechRequest, NvAudioSpeechResponse)
    async def speech_endpoint(
        self, request: NvCreateAudioSpeechRequest, context: Context
    ):
        async with aclosing(self.generate(request, context)) as responses:
            async for response in responses:
                yield response.model_dump()
