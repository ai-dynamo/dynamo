# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import numpy as np
import pytest

from dynamo.vllm.realtime_transcription import RealtimeTranscriptionHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

MODEL = "test/realtime-asr"
LOCAL_VAD_PLACEHOLDER = {
    "type": "server_vad",
    "create_response": False,
    "interrupt_response": False,
    "prefix_padding_ms": 0,
    "silence_duration_ms": 0,
    "threshold": 1.0,
}


class _Context:
    def __init__(self) -> None:
        self.stopped = False

    def is_stopped(self) -> bool:
        return self.stopped


class _FakeEngine:
    def __init__(self) -> None:
        self.audio: list[np.ndarray] = []
        self.request_ids: list[str] = []

    async def generate(self, *, prompt, sampling_params, request_id):
        del sampling_params
        self.request_ids.append(request_id)
        async for chunk in prompt:
            self.audio.append(chunk)
        yield SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            outputs=[SimpleNamespace(text="hello ", token_ids=[10, 11])],
        )
        yield SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            outputs=[SimpleNamespace(text="world", token_ids=[12])],
        )


async def _stream_audio(audio_stream, input_stream):
    del input_stream
    async for chunk in audio_stream:
        yield chunk


def _handler(engine: _FakeEngine) -> RealtimeTranscriptionHandler:
    return RealtimeTranscriptionHandler(
        engine_client=engine,
        model_name=MODEL,
        model_sample_rate=16_000.0,
        streaming_input_factory=_stream_audio,
        sampling_params_factory=lambda: object(),
    )


def _session(*, turn_detection=LOCAL_VAD_PLACEHOLDER) -> dict:
    return {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24_000},
                "transcription": {"model": MODEL, "language": "en"},
                "noise_reduction": None,
                "turn_detection": turn_detection,
            }
        },
    }


async def _drive(handler, events):
    async def request_stream():
        for event in events:
            yield event

    return [event async for event in handler.generate(request_stream(), _Context())]


def test_transcription_session_streams_canonical_events_and_resamples_audio():
    pcm = np.linspace(-12_000, 12_000, 2_400, dtype=np.int16).tobytes()
    engine = _FakeEngine()
    events = [
        {"type": "session.update", "session": _session()},
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode(),
        },
        {"type": "input_audio_buffer.commit"},
    ]

    result = asyncio.run(_drive(_handler(engine), events))
    event_types = [event["type"] for event in result]

    assert event_types == [
        "session.updated",
        "input_audio_buffer.committed",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed",
    ]
    committed = result[1]
    assert all(event.get("item_id") == committed["item_id"] for event in result[2:])
    assert "".join(event["delta"] for event in result[2:4]) == "hello world"
    assert result[-1]["transcript"] == "hello world"
    assert result[-1]["usage"] == {
        "type": "tokens",
        "input_tokens": 3,
        "output_tokens": 3,
        "total_tokens": 6,
        "input_token_details": {"audio_tokens": 3, "text_tokens": 0},
    }
    assert len(engine.audio) == 1
    assert abs(engine.audio[0].size - 1_600) <= 1


@pytest.mark.parametrize(
    "event, message",
    [
        ({"type": "input_audio_buffer.append", "audio": "not base64"}, "valid base64"),
        ({"type": "input_audio_buffer.append", "audio": 123}, "base64 string"),
        ({"type": "input_audio_buffer.commit"}, "buffer is empty"),
    ],
)
def test_invalid_audio_events_return_recoverable_errors(event, message):
    result = asyncio.run(
        _drive(
            _handler(_FakeEngine()),
            [{"type": "session.update", "session": _session()}, event],
        )
    )

    errors = [item for item in result if item["type"] == "error"]
    assert len(errors) == 1
    assert message in errors[0]["error"]["message"]


def test_invalid_audio_format_returns_recoverable_session_error():
    session = _session()
    session["audio"]["input"]["format"] = "pcm16"

    result = asyncio.run(
        _drive(
            _handler(_FakeEngine()),
            [{"type": "session.update", "session": session}],
        )
    )

    assert [event["type"] for event in result] == ["error"]
    assert result[0]["error"]["code"] == "invalid_session"
    assert "format must be an object" in result[0]["error"]["message"]


@pytest.mark.parametrize(
    "audio_input_update, transcription_update, message",
    [
        ({"format": {"type": "audio/pcm", "rate": 16_000}}, {}, "24000 Hz"),
        ({}, {"language": "fr"}, "only English"),
        ({}, {"prompt": "Dynamo vocabulary"}, "prompts are not supported"),
        ({"noise_reduction": {"type": "near_field"}}, {}, "not supported"),
    ],
)
def test_unsupported_session_options_are_rejected(
    audio_input_update, transcription_update, message
):
    session = _session()
    audio_input = session["audio"]["input"]
    audio_input.update(audio_input_update)
    audio_input["transcription"].update(transcription_update)

    result = asyncio.run(
        _drive(
            _handler(_FakeEngine()),
            [{"type": "session.update", "session": session}],
        )
    )

    assert [event["type"] for event in result] == ["error"]
    assert message in result[0]["error"]["message"]


def test_server_vad_is_rejected_until_worker_supports_it():
    server_vad = {
        "type": "server_vad",
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
        "threshold": 0.5,
    }
    result = asyncio.run(
        _drive(
            _handler(_FakeEngine()),
            [
                {
                    "type": "session.update",
                    "session": _session(turn_detection=server_vad),
                }
            ],
        )
    )

    assert [event["type"] for event in result] == ["error"]
    assert "server turn detection is not supported" in result[0]["error"]["message"]


def test_clear_cancels_uncommitted_turn_and_allows_next_utterance():
    pcm = base64.b64encode(np.ones(480, dtype=np.int16).tobytes()).decode()
    engine = _FakeEngine()
    result = asyncio.run(
        _drive(
            _handler(engine),
            [
                {"type": "session.update", "session": _session()},
                {"type": "input_audio_buffer.append", "audio": pcm},
                {"type": "input_audio_buffer.clear"},
                {"type": "input_audio_buffer.append", "audio": pcm},
                {"type": "input_audio_buffer.commit"},
            ],
        )
    )

    assert "input_audio_buffer.cleared" in [event["type"] for event in result]
    completed = [
        event
        for event in result
        if event["type"] == "conversation.item.input_audio_transcription.completed"
    ]
    assert len(completed) == 1
