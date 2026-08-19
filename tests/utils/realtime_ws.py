# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side driver for the ``/v1/realtime`` WebSocket endpoint.

Shared by the realtime tests so the turn protocol -- open a session, commit a
buffer of audio, collect the response envelope -- is implemented once. Tests
supply the audio and assert on the returned ``TurnResult``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Server events from one committed turn, grouped by kind."""

    response_id: str | None = None
    audio_b64_parts: list[str] = field(default_factory=list)
    transcript_parts: list[str] = field(default_factory=list)
    saw_audio_done: bool = False
    status: str | None = None

    @property
    def transcript(self) -> str:
        return "".join(self.transcript_parts)

    @property
    def audio_bytes(self) -> bytes:
        return b"".join(base64.b64decode(p) for p in self.audio_b64_parts)

    @property
    def audio_pcm16(self) -> np.ndarray:
        return np.frombuffer(self.audio_bytes, dtype=np.int16)


async def recv_json(ws: aiohttp.ClientWebSocketResponse, timeout_s: float) -> dict:
    msg = await asyncio.wait_for(ws.receive(), timeout=timeout_s)
    if msg.type is not aiohttp.WSMsgType.TEXT:
        raise AssertionError(f"unexpected websocket frame: {msg.type!r} {msg.data!r}")
    return json.loads(msg.data)


async def drain_until(
    ws: aiohttp.ClientWebSocketResponse, expected_type: str, timeout_s: float = 5.0
) -> dict:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        remaining = deadline - loop.time()
        event = await recv_json(ws, max(remaining, 0.01))
        if event.get("type") == expected_type:
            return event
    raise AssertionError(
        f"timed out waiting for a {expected_type!r} frame on the websocket"
    )


async def open_session(
    ws: aiohttp.ClientWebSocketResponse, session: dict, timeout_s: float = 5.0
) -> dict:
    """Await ``session.created``, send ``session.update``, await the echo."""
    await drain_until(ws, "session.created", timeout_s)
    await ws.send_str(json.dumps({"type": "session.update", "session": session}))
    return await drain_until(ws, "session.updated", timeout_s)


async def commit_audio(ws: aiohttp.ClientWebSocketResponse, pcm16: bytes) -> None:
    """Append one PCM16 buffer and close the turn."""
    await ws.send_str(
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16).decode("utf-8"),
            }
        )
    )
    await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))


async def collect_turn(
    ws: aiohttp.ClientWebSocketResponse,
    timeout_s: float,
    *,
    allow_unknown_events: bool = False,
) -> TurnResult:
    """Collect one turn's server events, up to and including ``response.done``.

    An ``error`` frame always raises -- a turn that fails is never a pass.
    ``allow_unknown_events`` relaxes only the check on *unrecognized* event
    types, for callers driving a real engine that may emit more of them.
    """
    result = TurnResult()
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s

    while result.status is None:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise AssertionError(
                "timed out before response.done; "
                f"audio_parts={len(result.audio_b64_parts)}, "
                f"saw_audio_done={result.saw_audio_done}"
            )
        event = await recv_json(ws, remaining)
        etype = event.get("type")

        if etype == "error":
            raise AssertionError(f"server error event: {event}")
        elif etype == "response.created":
            result.response_id = event["response"]["id"]
        elif etype == "response.output_audio_transcript.delta":
            result.transcript_parts.append(event["delta"])
            assert event["response_id"] == result.response_id, event
        elif etype == "response.output_audio.delta":
            result.audio_b64_parts.append(event["delta"])
            assert event["response_id"] == result.response_id, event
        elif etype == "response.output_audio.done":
            result.saw_audio_done = True
            assert event["response_id"] == result.response_id, event
        elif etype == "response.done":
            result.status = event["response"]["status"]
            assert event["response"]["id"] == result.response_id, event
        elif allow_unknown_events:
            logger.info("ignoring unasserted event type %r", etype)
        else:
            raise AssertionError(f"unexpected event type {etype!r}: {event}")

    assert result.response_id is not None
    return result
