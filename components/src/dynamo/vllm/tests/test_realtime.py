# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from dynamo.vllm.realtime import RealtimeHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class _RecordingHandler:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def generate(self, request_stream, context):
        del context
        async for event in request_stream:
            self.events.append(event)
        yield {"type": "session.updated"}


async def _drive(handler, events):
    async def request_stream():
        for event in events:
            yield event

    return [event async for event in handler.generate(request_stream(), Mock())]


def test_dispatches_session_and_replays_initial_update():
    transcription = _RecordingHandler()
    handler = RealtimeHandler({"transcription": transcription})
    events = [
        {
            "type": "session.update",
            "event_id": "event_1",
            "session": {"type": "transcription"},
        },
        {"type": "input_audio_buffer.commit"},
    ]

    result = asyncio.run(_drive(handler, events))

    assert result == [{"type": "session.updated"}]
    assert transcription.events == events


def test_rejects_unsupported_session_type():
    transcription = _RecordingHandler()
    handler = RealtimeHandler({"transcription": transcription})

    result = asyncio.run(
        _drive(
            handler,
            [
                {
                    "type": "session.update",
                    "event_id": "event_1",
                    "session": {"type": "realtime"},
                }
            ],
        )
    )

    assert result[0]["type"] == "error"
    assert result[0]["error"]["code"] == "unsupported_session"
    assert result[0]["error"]["event_id"] == "event_1"
    assert transcription.events == []
