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

import base64
from types import SimpleNamespace

import pytest
from nemotron_speech.realtime_asr import (
    OPENAI_PCM_SAMPLE_RATE,
    PCM16_BYTES_PER_SAMPLE,
    SpeechNimRealtimeTranscriptionHandler,
)
from riva.client import AudioEncoding

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

MODEL = "nemotron-asr-streaming"


class _Context:
    def is_stopped(self) -> bool:
        return False


class _FakeAsrService:
    def __init__(self, responses=None) -> None:
        self.audio = b""
        self.streaming_config = None
        self.responses = responses or [
            _response("hello", final=False),
            _response("hello world", final=True),
        ]

    def streaming_response_generator(self, *, audio_chunks, streaming_config):
        self.streaming_config = streaming_config
        for chunk in audio_chunks:
            self.audio += chunk
        yield from self.responses


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

    return [event async for event in handler.generate(request_stream(), _Context())]


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
