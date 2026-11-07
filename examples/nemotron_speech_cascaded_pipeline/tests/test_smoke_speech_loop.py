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

import asyncio
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

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.asyncio,
]


class _WebSocket:
    def __init__(self, events):
        self.events = iter(events)
        self.sent = []
        self.closed = False

    async def send_json(self, event):
        self.sent.append(event)

    async def receive(self):
        event = next(self.events)
        if isinstance(event, BaseException):
            raise event
        return SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(event),
        )

    async def close(self):
        self.closed = True

    def __await__(self):
        return self.__aenter__().__await__()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class _WebSocketSession:
    def __init__(self, *websockets):
        self.websockets = iter(websockets)

    def ws_connect(self, url, **kwargs):
        return next(self.websockets)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


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


@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_realtime_llm_rejects_unsuccessful_response_after_partial_output(status):
    websocket = _WebSocket(
        [
            {"type": "response.output_text.delta", "delta": "partial"},
            {"type": "response.done", "response": {"status": status}},
        ]
    )

    with pytest.raises(RuntimeError, match=status):
        await _complete_realtime(websocket, SimpleNamespace(timeout=1.0), "hello")


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


@pytest.mark.parametrize("partial", ["", "discard this hypothesis"])
async def test_empty_final_transcript_never_commits_speculative_text(partial):
    websocket = _WebSocket([])
    text_input = _RealtimeTextInput(websocket)

    await text_input.append(partial)
    await text_input.commit("")

    assert websocket.sent == (
        [
            {"type": "input_text.append", "text": partial},
            {"type": "input_text.clear"},
        ]
        if partial
        else []
    )


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


@pytest.mark.parametrize(
    "event_type,after_delta",
    [("error", False), ("conversation.item.input_audio_transcription.failed", True)],
)
async def test_transcription_failure_stops_upload_without_committing(
    event_type, after_delta
):
    events = [{"type": event_type, "error": "ASR unavailable"}]
    if after_delta:
        events.insert(
            0,
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "delta": "partial",
            },
        )
    asr_websocket = _WebSocket(events)
    llm_websocket = _WebSocket([])
    args = SimpleNamespace(
        base_url="http://dynamo",
        asr_model="test/asr",
        language="en",
        timeout=1.0,
        chunk_bytes=2,
    )

    with pytest.raises(RuntimeError, match="ASR unavailable"):
        await _transcribe(
            _WebSocketSession(asr_websocket),
            args,
            b"\x00\x00" * 4,
            _RealtimeTextInput(llm_websocket),
        )

    assert llm_websocket.sent == (
        [{"type": "input_text.append", "text": "partial"}] if after_delta else []
    )
    sent_types = [event["type"] for event in asr_websocket.sent]
    assert "input_audio_buffer.commit" not in sent_types
    assert sent_types.count("input_audio_buffer.append") < 4


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


@pytest.mark.parametrize(
    "outcome", ["completed", "asr_error", "llm_error", "cancelled", "empty_final"]
)
async def test_realtime_run_orders_handoff_and_closes_connections(monkeypatch, outcome):
    llm_websocket = _WebSocket(
        [
            {"type": "session.created"},
            {"type": "session.updated"},
            {"type": "response.output_text.delta", "delta": "answer"},
            {
                "type": "response.done",
                "response": {
                    "status": "failed" if outcome == "llm_error" else "completed"
                },
            },
        ]
    )
    asr_events = [
        {"type": "conversation.item.input_audio_transcription.delta", "delta": delta}
        for delta in ("hello", " world")
    ]
    asr_events.append(
        asyncio.CancelledError()
        if outcome == "cancelled"
        else (
            {"type": "error", "error": "ASR unavailable"}
            if outcome == "asr_error"
            else {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "" if outcome == "empty_final" else "hello world",
            }
        )
    )
    asr_websocket = _WebSocket(asr_events)
    session = _WebSocketSession(llm_websocket, asr_websocket)

    async def synthesize(session, args):
        return b"\x01\x00" * 4, 0.01

    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kwargs: session)
    monkeypatch.setattr(smoke_speech_loop, "_synthesize", synthesize)
    args = SimpleNamespace(
        base_url="http://dynamo",
        asr_model="test/asr",
        llm_model="test/llm",
        language="en",
        timeout=1.0,
        chunk_bytes=2,
        llm_transport="realtime",
        llm_instructions="Be concise.",
        max_output_tokens=32,
        min_rms=0.0,
    )
    if outcome == "completed":
        await smoke_speech_loop.run(args)
    else:
        error = asyncio.CancelledError if outcome == "cancelled" else RuntimeError
        with pytest.raises(error):
            await smoke_speech_loop.run(args)

    assert llm_websocket.closed and asr_websocket.closed
    assert [event["type"] for event in llm_websocket.sent] == [
        "session.update",
        "input_text.append",
        "input_text.append",
        *(
            ["input_text.commit", "response.create"]
            if outcome in {"completed", "llm_error"}
            else ["input_text.clear"]
            if outcome == "empty_final"
            else []
        ),
    ]


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
