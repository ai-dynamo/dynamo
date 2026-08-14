# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class PrefillResultDrain:
    """Track accepted prefill work until its result stream reaches terminal state."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def pending_count(self) -> int:
        return len(self._tasks)

    def create_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            logger.error(
                "Prefill result consumer failed",
                exc_info=(type(exception), exception, exception.__traceback__),
            )

    async def drain(self) -> None:
        """Wait for all tracked work without letting waiter cancellation cancel it."""
        while tasks := tuple(self._tasks):
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks), return_exceptions=True
            )

    def cancel(self) -> None:
        """Cancel tracked work as a final cleanup fallback after drain timeout."""
        for task in tuple(self._tasks):
            task.cancel()
