# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import aiohttp
import pytest
import smoke_speech_loop
from smoke_speech_loop import (
    _complete_chat,
    _complete_realtime,
    _RealtimeTextInput,
    _transcribe,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class _WebSocket:
    def __init__(self, events):
        self.events = iter(events)
        self.sent = []

    async def send_json(self, event):
        self.sent.append(event)

    async def receive(self):
        return SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(next(self.events)),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class _WebSocketSession:
    def __init__(self, websocket):
        self.websocket = websocket

    def ws_connect(self, url, **kwargs):
        return self.websocket


async def test_realtime_llm_streams_complete_transcript_and_collects_text():
    websocket = _WebSocket(
        [
            {"type": "conversation.item.done"},
            {"type": "response.created"},
            {"type": "response.output_text.delta", "delta": "hello "},
            {"type": "response.output_text.delta", "delta": "world"},
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )

    text, ttft, total = await _complete_realtime(
        websocket,
        SimpleNamespace(timeout=1.0),
        "A completed transcript",
    )

    assert text == "hello world"
    assert 0 <= ttft <= total
    assert websocket.sent == [
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "A completed transcript"}],
            },
        },
        {"type": "response.create"},
    ]


@pytest.mark.parametrize(
    "deltas,final_text",
    [
        (["recognize", " wreck"], "recognize speech"),
        (["hello"], "hello world"),
        ([], "hello"),
    ],
)
async def test_realtime_llm_sends_final_only_text_without_more_warming(
    deltas, final_text
):
    websocket = _WebSocket([])
    text_input = _RealtimeTextInput(websocket)

    for delta in deltas:
        await text_input.append(delta)
    await text_input.commit(final_text)

    assert websocket.sent == [
        *[{"type": "input_text.append", "text": delta} for delta in deltas],
        *([{"type": "input_text.clear"}] if deltas else []),
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": final_text}],
            },
        },
    ]


async def test_transcription_forwards_deltas_to_realtime_llm_before_commit():
    asr_websocket = _WebSocket(
        [
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "delta": "hello",
            },
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "hello",
            },
        ]
    )
    llm_websocket = _WebSocket([])
    args = SimpleNamespace(
        base_url="http://dynamo:8000",
        asr_model="test/asr",
        language="en",
        timeout=1.0,
        chunk_bytes=2,
    )

    transcript, started, first_delta, completed = await _transcribe(
        _WebSocketSession(asr_websocket),
        args,
        b"\x00\x00",
        _RealtimeTextInput(llm_websocket),
    )

    assert transcript == "hello"
    assert started <= first_delta <= completed
    assert llm_websocket.sent == [
        {"type": "input_text.append", "text": "hello"},
        {"type": "input_text.commit"},
    ]


class _Content:
    def __init__(self, lines):
        self.lines = iter(lines)

    async def readline(self):
        return next(self.lines, b"")


class _Response:
    def __init__(self, lines):
        self.content = _Content(lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self, lines):
        self.response = _Response(lines)
        self.request = None

    def post(self, url, *, json):
        self.request = (url, json)
        return self.response


async def test_chat_llm_streams_same_transcript_and_collects_text():
    session = _Session(
        [
            b'data: {"choices":[{"delta":{"content":"hello "}}]}\n',
            b'data: {"choices":[{"delta":{"content":"world"}}]}\n',
            b"data: [DONE]\n",
        ]
    )
    args = SimpleNamespace(
        base_url="http://dynamo:8000",
        llm_model="test/model",
        llm_instructions="Be concise.",
        max_output_tokens=32,
    )

    text, ttft, total = await _complete_chat(session, args, "A completed transcript")

    assert text == "hello world"
    assert 0 <= ttft <= total
    assert session.request == (
        "http://dynamo:8000/v1/chat/completions",
        {
            "model": "test/model",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "A completed transcript"},
            ],
            "max_completion_tokens": 32,
            "stream": True,
        },
    )


async def test_report_measures_from_asr_final_including_handoff(monkeypatch, capsys):
    async def synthesize(session, args):
        return b"\x01\x00", 0.01

    async def transcribe(session, args, pcm, text_input):
        return "hello", 10.0, 11.0, 12.0

    async def complete_chat(session, args, transcript):
        return "answer", 12.4, 13.0

    monkeypatch.setattr(smoke_speech_loop, "_synthesize", synthesize)
    monkeypatch.setattr(smoke_speech_loop, "_transcribe", transcribe)
    monkeypatch.setattr(smoke_speech_loop, "_complete_chat", complete_chat)

    await smoke_speech_loop.run(
        SimpleNamespace(timeout=1.0, llm_transport="chat", min_rms=0.0)
    )

    report = json.loads(capsys.readouterr().out)
    assert report["asr_first_transcript_ms"] == 1000.0
    assert report["asr_completed_ms"] == 2000.0
    assert report["llm_ttft_from_asr_final_ms"] == 400.0
    assert report["llm_total_from_asr_final_ms"] == 1000.0
    assert report["asr_start_to_llm_first_token_ms"] == 2400.0
