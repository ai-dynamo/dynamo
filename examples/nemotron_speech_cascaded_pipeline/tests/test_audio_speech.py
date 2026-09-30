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

"""Unit tests for the Speech NIM OpenAI audio/speech adapter."""

import asyncio
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

import grpc
import pytest

pytest.importorskip(
    "riva.client", reason="NVIDIA Riva client is an example-only dependency"
)

from nemotron_speech.riva import wait_for_service_ready  # noqa: E402
from nemotron_speech.tts.adapter import SpeechNimAudioSpeechBackend  # noqa: E402
from riva.client import AudioEncoding  # noqa: E402

from dynamo._core import Context  # noqa: E402
from dynamo.common.protocols.audio_protocol import (  # noqa: E402
    NvCreateAudioSpeechRequest,
)

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


def _backend(service: _FakeTtsService) -> SpeechNimAudioSpeechBackend:
    return SpeechNimAudioSpeechBackend(
        tts_service=service,
        model_name=MODEL,
        voice="Magpie-Multilingual.EN-US.Aria",
        language_code="en-US",
        sample_rate_hz=24_000,
    )


async def test_streams_each_speech_nim_response_as_pcm():
    service = _FakeTtsService([b"first", b"second"])
    responses = [
        response
        async for response in _backend(service).generate(
            NvCreateAudioSpeechRequest(
                input="hello world",
                model=MODEL,
                voice="Magpie-Multilingual.EN-US.Aria",
                response_format="pcm",
            ),
            Context(),
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
                NvCreateAudioSpeechRequest(**request), Context()
            )
        )


@pytest.mark.parametrize("endpoint", [False, True])
async def test_closing_output_cancels_speech_nim_call(endpoint):
    service = _FakeTtsService([b"first", b"second"])
    backend = _backend(service)
    request = NvCreateAudioSpeechRequest(
        input="hello", model=MODEL, response_format="pcm"
    )
    # Test the example wrapper separately from the runtime decorator's lifetime.
    output = (
        backend.speech_endpoint.__wrapped__(backend, request, Context())
        if endpoint
        else backend.generate(request, Context())
    )

    await anext(output)
    await output.aclose()

    assert service.call.cancelled


@pytest.mark.parametrize("after_first_chunk", [False, True])
async def test_rpc_failure_propagates_and_cancels_call(after_first_chunk):
    class FailedCall(_FakeCall):
        def __iter__(self):
            if after_first_chunk:
                yield SimpleNamespace(audio=b"first")
            raise grpc.RpcError("synthesis failed")

    service = _FakeTtsService([])
    service.call = FailedCall([])
    output = _backend(service).generate(
        NvCreateAudioSpeechRequest(input="hello", response_format="pcm"), Context()
    )
    if after_first_chunk:
        await anext(output)
    with pytest.raises(grpc.RpcError, match="synthesis failed"):
        await anext(output)
    assert service.call.cancelled


@pytest.mark.timeout(5)
async def test_concurrent_requests_do_not_starve_the_thread_pool(monkeypatch):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=2) as executor:

        async def to_thread(function, /, *args, **kwargs):
            return await loop.run_in_executor(
                executor, partial(function, *args, **kwargs)
            )

        monkeypatch.setattr(asyncio, "to_thread", to_thread)

        async def collect():
            return [
                response
                async for response in _backend(_FakeTtsService([b"first"])).generate(
                    NvCreateAudioSpeechRequest(input="hello", response_format="pcm"),
                    Context(),
                )
            ]

        responses = await asyncio.wait_for(asyncio.gather(collect(), collect()), 1)

    assert [len(items) for items in responses] == [1, 1]


@pytest.mark.timeout(5)
@pytest.mark.parametrize("cooperative", [False, True])
async def test_cancellation_stops_a_blocked_speech_nim_call(cooperative):
    loop = asyncio.get_running_loop()
    started, finished = asyncio.Event(), asyncio.Event()
    cancelled = threading.Event()

    class StalledCall(_FakeCall):
        def __iter__(self):
            return self

        def __next__(self):
            loop.call_soon_threadsafe(started.set)
            try:
                if not cancelled.wait(timeout=2):
                    raise TimeoutError("RPC was not cancelled")
                raise StopIteration
            finally:
                loop.call_soon_threadsafe(finished.set)

        def cancel(self):
            cancelled.set()

    service = _FakeTtsService([])
    service.call = StalledCall([])
    context = Context()
    output = _backend(service).generate(
        NvCreateAudioSpeechRequest(input="hello", response_format="pcm"), context
    )
    task = asyncio.create_task(anext(output))
    try:
        await asyncio.wait_for(started.wait(), 1)
        if cooperative:
            context.stop_generating()
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(task, 1)
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert cancelled.is_set()
        await asyncio.wait_for(finished.wait(), 1)
    finally:
        cancelled.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_waits_for_speech_nim_before_registration(monkeypatch):
    calls = []

    class _Future:
        def result(self, *, timeout):
            calls.append(timeout)

    service = SimpleNamespace(
        auth=SimpleNamespace(channel=object(), uri="speech-nim:50051")
    )
    monkeypatch.setattr(grpc, "channel_ready_future", lambda _channel: _Future())

    await wait_for_service_ready(service, 0.1)

    assert calls == [0.1]


async def test_speech_nim_readiness_timeout_is_actionable(monkeypatch):
    class _Future:
        def result(self, *, timeout):
            raise grpc.FutureTimeoutError()

    service = SimpleNamespace(
        auth=SimpleNamespace(channel=object(), uri="speech-nim:50051")
    )
    monkeypatch.setattr(grpc, "channel_ready_future", lambda _channel: _Future())

    with pytest.raises(TimeoutError, match="speech-nim:50051.*0.1 seconds"):
        await wait_for_service_ready(service, 0.1)
