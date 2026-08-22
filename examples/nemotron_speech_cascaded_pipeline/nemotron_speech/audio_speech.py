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
import queue
import threading
import time
import uuid
from collections.abc import AsyncGenerator

from riva.client import AudioEncoding

from dynamo.common.protocols.audio_protocol import (
    AudioData,
    NvAudioSpeechResponse,
    NvCreateAudioSpeechRequest,
)
from dynamo.runtime import dynamo_endpoint


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

    async def generate(
        self, request: NvCreateAudioSpeechRequest
    ) -> AsyncGenerator[NvAudioSpeechResponse, None]:
        """Yield each Riva SDK ``SynthesizeOnline`` response as one PCM chunk."""
        self._validate(request)
        output: queue.Queue[bytes | Exception | None] = queue.Queue(maxsize=32)
        stopped = threading.Event()
        call_holder = []
        response_id = f"speech_{uuid.uuid4().hex}"
        created = int(time.time())

        def put_output(item: bytes | Exception | None) -> bool:
            while not stopped.is_set():
                try:
                    output.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def synthesize() -> None:
            call = None
            try:
                call = self.tts_service.synthesize_online(
                    request.input,
                    voice_name=request.voice or self.voice,
                    language_code=request.language or self.language_code,
                    encoding=AudioEncoding.LINEAR_PCM,
                    sample_rate_hz=self.sample_rate_hz,
                )
                call_holder.append(call)
                for response in call:
                    if stopped.is_set():
                        break
                    if not put_output(response.audio):
                        break
            except Exception as exc:  # noqa: BLE001 - propagate gRPC failures
                put_output(exc)
            finally:
                if stopped.is_set() and call is not None:
                    call.cancel()
                else:
                    put_output(None)

        task = asyncio.create_task(asyncio.to_thread(synthesize))
        try:
            while (item := await asyncio.to_thread(output.get)) is not None:
                if isinstance(item, Exception):
                    raise item
                yield NvAudioSpeechResponse(
                    id=response_id,
                    model=self.model_name,
                    created=created,
                    data=[
                        AudioData(
                            output_format="pcm",
                            b64_json=base64.b64encode(item).decode(),
                        )
                    ],
                )
        finally:
            stopped.set()
            if call_holder:
                call_holder[0].cancel()
            while not output.empty():
                output.get_nowait()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    @dynamo_endpoint(NvCreateAudioSpeechRequest, NvAudioSpeechResponse)
    async def speech_endpoint(self, request: NvCreateAudioSpeechRequest):
        async for response in self.generate(request):
            yield response.model_dump()
