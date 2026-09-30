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

"""Unit tests for the Speech NIM OpenAI realtime transcription adapter."""

import asyncio
import base64
import threading
from contextlib import aclosing
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "riva.client", reason="NVIDIA Riva client is an example-only dependency"
)

from nemotron_speech.realtime_asr import (  # noqa: E402
    OPENAI_PCM_SAMPLE_RATE,
    PCM16_BYTES_PER_SAMPLE,
    SpeechNimRealtimeTranscriptionHandler,
    _AudioTurn,
)
from riva.client import ASRService, AudioEncoding  # noqa: E402

from dynamo._core import Context  # noqa: E402

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

MODEL = "nemotron-asr-streaming"


class _FakeCall:
    def __init__(self, responses):
        self.responses = iter(responses)

    def __iter__(self):
        return self.responses

    def cancel(self):
        pass


class _FakeAsrService(ASRService):
    def __init__(self, responses=None) -> None:
        self.auth = SimpleNamespace(get_auth_metadata=lambda: [])
        self.stub = SimpleNamespace(StreamingRecognize=self._recognize)
        self.audio = b""
        self.streaming_config = None
        self.responses = responses or [
            _response("hello", final=False),
            _response("hello world", final=True),
        ]

    def _recognize(self, requests, *, metadata):
        def responses():
            self.streaming_config = next(requests).streaming_config
            for request in requests:
                self.audio += request.audio_content
            yield from self.responses

        return _FakeCall(responses())


def _response(transcript: str, *, final: bool):
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                is_final=final,
                alternatives=[SimpleNamespace(transcript=transcript)],
            )
        ]
    )


def _session() -> dict:
    return {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24_000},
                "transcription": {"model": MODEL, "language": "en"},
                "turn_detection": None,
            }
        },
    }


async def _drive(handler, events):
    async def request_stream():
        for event in events:
            yield event

    return [event async for event in handler.generate(request_stream(), Context())]


def _handler(
    service: _FakeAsrService, *, commit_padding_ms: int = 0
) -> SpeechNimRealtimeTranscriptionHandler:
    return SpeechNimRealtimeTranscriptionHandler(
        asr_service=service,
        model_name=MODEL,
        nim_model="",
        language_code="en-US",
        commit_padding_ms=commit_padding_ms,
        timeout_s=1.0,
    )


@pytest.mark.timeout(5)
@pytest.mark.parametrize("stop", ["task", "context", "timeout"])
async def test_stalled_rpc_is_cancelled_on_disconnect_or_timeout(stop):
    started = threading.Event()
    cancelled = threading.Event()

    class StalledCall:
        def __iter__(self):
            started.set()
            cancelled.wait(timeout=2)
            return iter(())

        def cancel(self):
            cancelled.set()

    service = _FakeAsrService()
    service.stub.StreamingRecognize = lambda *args, **kwargs: StalledCall()
    handler = _handler(service)
    handler.timeout_s = 0.01
    turn = _AudioTurn()
    context = Context()
    task = asyncio.create_task(handler._run_turn(turn, context))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        if stop == "timeout":
            turn.close()
            await asyncio.wait_for(task, 1)
        elif stop == "context":
            context.stop_generating()
            await asyncio.wait_for(task, 1)
            assert turn.events.empty()
        else:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert cancelled.is_set()
    finally:
        cancelled.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.timeout(5)
async def test_context_stop_closes_uncommitted_turn_after_rpc_finishes():
    loop = asyncio.get_running_loop()
    finished = asyncio.Event()

    class FinishedCall(_FakeCall):
        def __iter__(self):
            yield _response("hello", final=True)
            loop.call_soon_threadsafe(finished.set)

    service = _FakeAsrService()
    service.stub.StreamingRecognize = lambda *args, **kwargs: FinishedCall(())
    context, turn = Context(), _AudioTurn()
    task = asyncio.create_task(_handler(service)._run_turn(turn, context))
    try:
        await asyncio.wait_for(finished.wait(), 1)
        assert not task.done()
        context.stop_generating()
        await asyncio.wait_for(task, 1)
        assert turn.closed
        assert turn.events.empty()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.timeout(5)
async def test_context_stop_releases_backpressured_transcript_producer():
    blocked = asyncio.Event()

    class OutputQueue(asyncio.Queue):
        async def put(self, item):
            blocked.set()
            await super().put(item)

    service = _FakeAsrService()
    service.stub.StreamingRecognize = lambda *args, **kwargs: _FakeCall(
        [_response("hello", final=False)]
    )
    context, turn = Context(), _AudioTurn()
    turn.events = OutputQueue(maxsize=1)
    turn.events.put_nowait({"type": "queued_delta"})
    task = asyncio.create_task(_handler(service)._run_turn(turn, context))
    try:
        await asyncio.wait_for(blocked.wait(), 1)
        context.stop_generating()
        await asyncio.wait_for(task, 1)
        assert turn.closed
        assert turn.events.empty()
    finally:
        while not turn.events.empty():
            turn.events.get_nowait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.timeout(5)
async def test_rpc_failure_before_commit_allows_next_turn():
    class FailedCall(_FakeCall):
        def __iter__(self):
            raise RuntimeError("backend unavailable")

    service = _FakeAsrService()
    recognize = service.stub.StreamingRecognize
    calls = []

    def fail_first_request(requests, *, metadata):
        calls.append(None)
        if len(calls) == 1:
            return FailedCall(())
        return recognize(requests, metadata=metadata)

    service.stub.StreamingRecognize = fail_first_request
    requests = asyncio.Queue()

    async def request_stream():
        while (event := await requests.get()) is not None:
            yield event

    append = {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(b"\x00\x01" * 320).decode(),
    }
    async with aclosing(
        _handler(service).generate(request_stream(), Context())
    ) as responses:
        await requests.put({"type": "session.update", "session": _session()})
        await requests.put(append)
        assert (await anext(responses))["type"] == "session.updated"
        failure = await asyncio.wait_for(anext(responses), 1)
        assert failure["type"] == "conversation.item.input_audio_transcription.failed"

        await requests.put(append)
        await requests.put({"type": "input_audio_buffer.commit"})
        await requests.put(None)
        remaining = [event async for event in responses]

    assert len(calls) == 2
    assert (
        remaining[-1]["type"] == "conversation.item.input_audio_transcription.completed"
    )
    assert remaining[-1]["transcript"] == "hello world"
    assert remaining[-1]["item_id"] != failure["item_id"]


async def test_streams_pcm_and_emits_canonical_transcription_events():
    service = _FakeAsrService()
    pcm = b"\x00\x01" * 320
    result = await _drive(
        _handler(service),
        [
            {"type": "session.update", "session": _session()},
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            },
            {"type": "input_audio_buffer.commit"},
        ],
    )

    event_types = [event["type"] for event in result]
    assert event_types[0] == "session.updated"
    assert "input_audio_buffer.committed" in event_types
    assert "conversation.item.input_audio_transcription.delta" in event_types
    assert event_types[-1] == ("conversation.item.input_audio_transcription.completed")
    assert result[-1]["transcript"] == "hello world"
    item_ids = {event["item_id"] for event in result if "item_id" in event}
    assert len(item_ids) == 1
    assert service.audio == pcm
    config = service.streaming_config.config
    assert config.encoding == AudioEncoding.LINEAR_PCM
    assert config.sample_rate_hertz == 24_000
    assert config.language_code == "en-US"


async def test_appends_configured_silence_before_closing_speech_nim_stream():
    service = _FakeAsrService()
    pcm = b"\x00\x01" * 320
    padding_ms = 20

    result = await _drive(
        _handler(service, commit_padding_ms=padding_ms),
        [
            {"type": "session.update", "session": _session()},
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            },
            {"type": "input_audio_buffer.commit"},
        ],
    )

    padding_bytes = OPENAI_PCM_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * padding_ms // 1000
    assert service.audio == pcm + bytes(padding_bytes)
    assert result[-1]["transcript"] == "hello world"


async def test_does_not_append_revised_interim_hypothesis():
    service = _FakeAsrService(
        [
            _response("recognize", final=False),
            _response("recognize wreck", final=False),
            _response("recognize speech", final=False),
            _response("recognize speech", final=True),
        ]
    )
    pcm = b"\x00\x01" * 320

    result = await _drive(
        _handler(service),
        [
            {"type": "session.update", "session": _session()},
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            },
            {"type": "input_audio_buffer.commit"},
        ],
    )

    deltas = [
        event["delta"]
        for event in result
        if event["type"] == "conversation.item.input_audio_transcription.delta"
    ]
    assert "".join(deltas) == "recognize wreck"


@pytest.mark.parametrize(
    "event, message",
    [
        (
            {"type": "input_audio_buffer.append", "audio": "not-base64"},
            "valid base64",
        ),
        (
            {"type": "input_audio_buffer.append", "audio": 123},
            "base64 string",
        ),
        ({"type": "input_audio_buffer.commit"}, "buffer is empty"),
    ],
)
async def test_invalid_audio_returns_recoverable_error(event, message):
    result = await _drive(
        _handler(_FakeAsrService()),
        [{"type": "session.update", "session": _session()}, event],
    )

    errors = [item for item in result if item["type"] == "error"]
    assert len(errors) == 1
    assert message in errors[0]["error"]["message"]


async def test_rejects_server_vad_without_starting_speech_nim():
    service = _FakeAsrService()
    session = _session()
    session["audio"]["input"]["turn_detection"] = {"type": "server_vad"}

    result = await _drive(
        _handler(service),
        [{"type": "session.update", "session": session}],
    )

    assert [event["type"] for event in result] == ["error"]
    assert "local VAD" in result[0]["error"]["message"]
    assert service.streaming_config is None
