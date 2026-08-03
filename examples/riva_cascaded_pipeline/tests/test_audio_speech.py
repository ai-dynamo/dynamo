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

"""Unit tests for the Riva OpenAI audio/speech adapter."""

import base64
from types import SimpleNamespace

import grpc
import pytest
from riva.client import AudioEncoding
from riva_nim.audio_speech import RivaAudioSpeechBackend
from riva_nim.riva_client import wait_for_service_ready

from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

MODEL = "magpie-tts-multilingual"


class _FakeCall:
    def __init__(self, chunks: list[bytes]) -> None:
        self.responses = [SimpleNamespace(audio=chunk) for chunk in chunks]
        self.cancelled = False

    def __iter__(self):
        return iter(self.responses)

    def cancel(self) -> None:
        self.cancelled = True


class _FakeTtsService:
    def __init__(self, chunks: list[bytes]) -> None:
        self.call = _FakeCall(chunks)
        self.args = None
        self.kwargs = None

    def synthesize_online(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self.call


def _backend(service: _FakeTtsService) -> RivaAudioSpeechBackend:
    return RivaAudioSpeechBackend(
        tts_service=service,
        model_name=MODEL,
        voice="Magpie-Multilingual.EN-US.Aria",
        language_code="en-US",
        sample_rate_hz=24_000,
    )


async def test_streams_each_riva_response_as_pcm():
    service = _FakeTtsService([b"first", b"second"])
    responses = [
        response
        async for response in _backend(service).generate(
            NvCreateAudioSpeechRequest(
                input="hello world",
                model=MODEL,
                voice="Magpie-Multilingual.EN-US.Aria",
                response_format="pcm",
            )
        )
    ]

    assert service.args == ("hello world",)
    assert service.kwargs == {
        "voice_name": "Magpie-Multilingual.EN-US.Aria",
        "language_code": "en-US",
        "encoding": AudioEncoding.LINEAR_PCM,
        "sample_rate_hz": 24_000,
    }
    assert len({response.id for response in responses}) == 1
    assert [base64.b64decode(response.data[0].b64_json) for response in responses] == [
        b"first",
        b"second",
    ]
    assert all(response.data[0].output_format == "pcm" for response in responses)


@pytest.mark.parametrize(
    "update, message",
    [
        ({"model": "wrong-model"}, "model must be"),
        ({"response_format": "wav"}, "response_format='pcm'"),
        ({"speed": 1.5}, "speed parameter"),
        ({"instructions": "whisper"}, "instructions"),
    ],
)
async def test_rejects_unsupported_openai_parameters(update, message):
    request = {
        "input": "hello",
        "model": MODEL,
        "response_format": "pcm",
        **update,
    }

    with pytest.raises(ValueError, match=message):
        await anext(
            _backend(_FakeTtsService([])).generate(
                NvCreateAudioSpeechRequest(**request)
            )
        )


async def test_closing_output_cancels_riva_call():
    service = _FakeTtsService([b"first", b"second"])
    output = _backend(service).generate(
        NvCreateAudioSpeechRequest(
            input="hello",
            model=MODEL,
            response_format="pcm",
        )
    )

    await anext(output)
    await output.aclose()

    assert service.call.cancelled


async def test_waits_for_riva_service_before_registration(monkeypatch):
    calls = []

    class _Future:
        def result(self, *, timeout):
            calls.append(timeout)

    service = SimpleNamespace(auth=SimpleNamespace(channel=object(), uri="riva:50051"))
    monkeypatch.setattr(grpc, "channel_ready_future", lambda _channel: _Future())

    await wait_for_service_ready(service, 0.1)

    assert calls == [0.1]


async def test_riva_readiness_timeout_is_actionable(monkeypatch):
    class _Future:
        def result(self, *, timeout):
            raise grpc.FutureTimeoutError()

    service = SimpleNamespace(auth=SimpleNamespace(channel=object(), uri="riva:50051"))
    monkeypatch.setattr(grpc, "channel_ready_future", lambda _channel: _Future())

    with pytest.raises(TimeoutError, match="riva:50051.*0.1 seconds"):
        await wait_for_service_ready(service, 0.1)
