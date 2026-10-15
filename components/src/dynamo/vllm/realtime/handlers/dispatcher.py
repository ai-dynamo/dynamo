# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch OpenAI Realtime sessions to a handler for their session type."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from typing import Any, Protocol

from dynamo._core import Context

from ..events import invalid_request_error_event


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
            yield invalid_request_error_event(
                "invalid_event",
                "first event must be session.update",
                client_event_id=(
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
            yield invalid_request_error_event(
                "unsupported_session",
                f"unsupported session type: {session_type!r}",
                client_event_id=first_event.get("event_id"),
            )
            return

        async def replay() -> AsyncGenerator[Any, None]:
            yield first_event
            async for event in request_stream:
                yield event

        async for event in handler.generate(replay(), context):
            yield event
