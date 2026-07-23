# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from dynamo.vllm.realtime.connection import RealtimeConnection, RealtimeTurn

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class _Context:
    def is_stopped(self) -> bool:
        return False


class _Turn(RealtimeTurn):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


async def _collect(stream: AsyncGenerator[dict, None]) -> list[dict]:
    return [event async for event in stream]


def test_concurrent_turn_output_is_yielded_in_creation_order():
    async def scenario() -> list[dict]:
        second_finished = asyncio.Event()
        connection: RealtimeConnection[_Turn]

        async def run_turn(turn: _Turn, context: _Context) -> None:
            del context
            if turn.name == "first":
                await second_finished.wait()
            await connection.emit_for_turn(turn, {"turn": turn.name})
            if turn.name == "second":
                second_finished.set()

        connection = RealtimeConnection(
            context=_Context(),
            run_turn=run_turn,
            max_concurrent_turns=2,
        )

        async def request_stream():
            for name in ("first", "second"):
                yield name

        async def handle_event(name, active_connection):
            await active_connection.ensure_turn(lambda: _Turn(name))
            active_connection.finish_active_turn()

        return await asyncio.wait_for(
            _collect(
                connection.generate(
                    request_stream(),
                    handle_event=handle_event,
                    close_active_turn=lambda turn: None,
                )
            ),
            timeout=1,
        )

    assert asyncio.run(scenario()) == [{"turn": "first"}, {"turn": "second"}]


def test_cancelling_active_turn_releases_capacity():
    async def scenario() -> tuple[list[dict], bool]:
        first_started = asyncio.Event()
        first_cancelled = asyncio.Event()
        never = asyncio.Event()
        connection: RealtimeConnection[_Turn]

        async def run_turn(turn: _Turn, context: _Context) -> None:
            del context
            if turn.name == "first":
                first_started.set()
                try:
                    await never.wait()
                finally:
                    first_cancelled.set()
                return
            await connection.emit_for_turn(turn, {"turn": turn.name})

        connection = RealtimeConnection(
            context=_Context(),
            run_turn=run_turn,
            max_concurrent_turns=1,
        )

        async def request_stream():
            for event in ("first", "cancel", "second", "commit"):
                yield event

        async def handle_event(event, active_connection):
            if event in {"first", "second"}:
                await active_connection.ensure_turn(lambda: _Turn(event))
                if event == "first":
                    await first_started.wait()
            elif event == "cancel":
                active_connection.cancel_active_turn()
            else:
                active_connection.finish_active_turn()

        result = await asyncio.wait_for(
            _collect(
                connection.generate(
                    request_stream(),
                    handle_event=handle_event,
                    close_active_turn=lambda turn: None,
                )
            ),
            timeout=1,
        )
        return result, first_cancelled.is_set()

    result, first_cancelled = asyncio.run(scenario())

    assert result == [{"turn": "second"}]
    assert first_cancelled


def test_closing_output_cancels_pump_and_turn_tasks():
    async def scenario() -> tuple[bool, bool]:
        turn_cancelled = asyncio.Event()
        never = asyncio.Event()
        created_turns: list[_Turn] = []
        connection: RealtimeConnection[_Turn]

        async def run_turn(turn: _Turn, context: _Context) -> None:
            del context
            try:
                await connection.emit_for_turn(turn, {"type": "started"})
                await never.wait()
            finally:
                turn_cancelled.set()

        connection = RealtimeConnection(
            context=_Context(),
            run_turn=run_turn,
            max_concurrent_turns=1,
        )

        async def request_stream():
            yield "start"
            await never.wait()

        async def handle_event(event, active_connection):
            turn = await active_connection.ensure_turn(lambda: _Turn(event))
            created_turns.append(turn)

        output = connection.generate(
            request_stream(),
            handle_event=handle_event,
            close_active_turn=lambda turn: None,
        )
        assert await asyncio.wait_for(anext(output), timeout=1) == {"type": "started"}

        await asyncio.wait_for(output.aclose(), timeout=1)

        return turn_cancelled.is_set(), created_turns[0].task.done()

    turn_cancelled, task_done = asyncio.run(scenario())

    assert turn_cancelled
    assert task_done
