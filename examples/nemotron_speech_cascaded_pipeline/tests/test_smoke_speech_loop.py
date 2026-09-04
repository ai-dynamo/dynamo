# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import aiohttp
import pytest
from smoke_speech_loop import _complete_chat, _complete_realtime

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
