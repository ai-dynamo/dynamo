# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-neutral orchestration for bidirectional realtime connections."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Generic, TypeVar, cast

from dynamo._core import Context


class RealtimeTurn:
    """Connection-managed state shared by one realtime input/output turn."""

    def __init__(self) -> None:
        self.events: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=256)
        self.task: asyncio.Task[None] | None = None


TurnT = TypeVar("TurnT", bound=RealtimeTurn)


class RealtimeConnection(Generic[TurnT]):
    """Coordinate input pumping, turn tasks, and ordered server output."""

    def __init__(
        self,
        *,
        context: Context,
        run_turn: Callable[[TurnT, Context], Awaitable[None]],
        max_concurrent_turns: int,
        max_queued_turns: int = 0,
    ) -> None:
        if max_concurrent_turns < 1:
            raise ValueError("max_concurrent_turns must be at least 1")
        if max_queued_turns < 0:
            raise ValueError("max_queued_turns cannot be negative")
        self._context = context
        self._run_turn = run_turn
        self._engine_slots = asyncio.Semaphore(max_concurrent_turns)
        self._turn_capacity = asyncio.Semaphore(max_concurrent_turns + max_queued_turns)
        self._out_stream: asyncio.Queue[dict | RealtimeTurn | None] = asyncio.Queue()
        self._active_turn: TurnT | None = None
        self._turns: list[TurnT] = []

    @property
    def active_turn(self) -> TurnT | None:
        return self._active_turn

    def emit(self, event: dict) -> None:
        """Queue a standalone server event in client-event order."""
        self._out_stream.put_nowait(event)

    async def emit_for_turn(self, turn: TurnT, event: dict) -> None:
        """Queue an event in one turn's ordered, backpressured output."""
        await turn.events.put(event)

    async def ensure_turn(self, factory: Callable[[], TurnT]) -> TurnT:
        """Return the open input turn, creating and scheduling it if needed."""
        turn = self._active_turn
        if turn is not None:
            return turn

        await self._turn_capacity.acquire()
        try:
            turn = factory()
            turn.task = asyncio.create_task(self._drive_turn(turn))
        except Exception:
            self._turn_capacity.release()
            raise

        turn.task.add_done_callback(self._release_turn_capacity)
        self._active_turn = turn
        self._turns.append(turn)
        self._out_stream.put_nowait(turn)
        return turn

    def _release_turn_capacity(self, _: asyncio.Task[None]) -> None:
        self._turn_capacity.release()

    def finish_active_turn(self) -> TurnT | None:
        """Detach and return the active turn after its input is closed."""
        turn = self._active_turn
        self._active_turn = None
        return turn

    def cancel_active_turn(self) -> TurnT | None:
        """Detach, cancel, and close the active uncommitted turn."""
        turn = self.finish_active_turn()
        if turn is not None and turn.task is not None:
            turn.task.cancel()
            # A task cancelled before its coroutine starts cannot run its finally
            # block, so close the output explicitly. Drop partial uncommitted output.
            drain_queue(turn.events)
            turn.events.put_nowait(None)
        return turn

    async def _drive_turn(self, turn: TurnT) -> None:
        try:
            async with self._engine_slots:
                if not self._context.is_stopped():
                    await self._run_turn(turn, self._context)
        finally:
            await turn.events.put(None)

    async def generate(
        self,
        request_stream: AsyncGenerator[Any, None],
        *,
        handle_event: Callable[[Any, "RealtimeConnection[TurnT]"], Awaitable[None]],
        close_active_turn: Callable[[TurnT], None],
    ) -> AsyncGenerator[dict, None]:
        """Run both directions until input and all queued turns complete."""

        async def pump() -> None:
            try:
                async for event in request_stream:
                    if self._context.is_stopped():
                        break
                    await handle_event(event, self)
            finally:
                try:
                    turn = self.finish_active_turn()
                    if turn is not None:
                        close_active_turn(turn)
                finally:
                    self._out_stream.put_nowait(None)

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                item = await self._out_stream.get()
                if item is None:
                    await pump_task
                    break
                if isinstance(item, RealtimeTurn):
                    turn = cast(TurnT, item)
                    while (event := await turn.events.get()) is not None:
                        yield event
                    if turn.task is not None:
                        result = (
                            await asyncio.gather(turn.task, return_exceptions=True)
                        )[0]
                        if isinstance(result, BaseException) and not isinstance(
                            result, asyncio.CancelledError
                        ):
                            raise result
                    self._turns.remove(turn)
                else:
                    yield item
        finally:
            pump_task.cancel()
            pending = [turn.task for turn in self._turns if turn.task is not None]
            for task in pending:
                task.cancel()
            for turn in self._turns:
                drain_queue(turn.events)
            await asyncio.gather(pump_task, *pending, return_exceptions=True)


def drain_queue(queue: asyncio.Queue[Any]) -> None:
    """Discard all currently buffered queue items without blocking."""
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break
