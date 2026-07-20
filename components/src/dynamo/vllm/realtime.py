# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch OpenAI Realtime sessions to model-specific handlers."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Protocol

from dynamo._core import Context


class RealtimeSessionHandler(Protocol):
    def generate(
        self,
        request_stream: AsyncGenerator[Any, None],
        context: Context,
    ) -> AsyncGenerator[dict, None]:
        ...


class RealtimeHandler:
    """Select one session handler from the initial ``session.update`` event."""

    def __init__(self, handlers: Mapping[str, RealtimeSessionHandler]) -> None:
        self._handlers = dict(handlers)

    async def generate(
        self,
        request_stream: AsyncGenerator[Any, None],
        context: Context,
    ) -> AsyncGenerator[dict, None]:
        try:
            first_event = await anext(request_stream)
        except StopAsyncIteration:
            return

        if (
            not isinstance(first_event, dict)
            or first_event.get("type") != "session.update"
        ):
            yield _error_event(
                "invalid_event",
                "first event must be session.update",
                event_id=(
                    first_event.get("event_id")
                    if isinstance(first_event, dict)
                    else None
                ),
            )
            return

        session = first_event.get("session")
        session_type = session.get("type") if isinstance(session, dict) else None
        handler = (
            self._handlers.get(session_type) if isinstance(session_type, str) else None
        )
        if handler is None:
            yield _error_event(
                "unsupported_session",
                f"unsupported session type: {session_type!r}",
                event_id=first_event.get("event_id"),
            )
            return

        async def replay() -> AsyncGenerator[Any, None]:
            yield first_event
            async for event in request_stream:
                yield event

        async for event in handler.generate(replay(), context):
            yield event


def _error_event(code: str, message: str, *, event_id: str | None) -> dict:
    return {
        "type": "error",
        "event_id": f"event_{uuid.uuid4().hex}",
        "error": {
            "type": "invalid_request_error",
            "code": code,
            "message": message,
            "event_id": event_id,
        },
    }
