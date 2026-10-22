# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancellation across decision preparation, queued admission, and engine work."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

_T = TypeVar("_T")


async def _wait_for_callback(cancelled: Callable[[], bool]) -> None:
    """Support protocol test contexts that do not expose the runtime waker."""
    while not cancelled():
        await asyncio.sleep(0.01)


async def run_with_context_cancellation(
    operation: Awaitable[_T], context: Any, cancelled: Callable[[], bool]
) -> _T:
    """Cancel and drain owned Python work when the runtime context terminates.

    This does not establish physical GPU abort settlement. Actual admitted
    children still use the handler's abort monitor and explicit engine cleanup.
    """
    task = asyncio.ensure_future(operation)
    signal = None
    try:
        if cancelled():
            raise asyncio.CancelledError
        # The PyO3 runtime returns an awaitable, not necessarily a coroutine.
        signal = asyncio.ensure_future(
            context.async_killed_or_stopped()
            if hasattr(context, "async_killed_or_stopped")
            else _wait_for_callback(cancelled)
        )
        done, _ = await asyncio.wait(
            (task, signal), return_when=asyncio.FIRST_COMPLETED
        )
        if signal in done:
            await signal
            raise asyncio.CancelledError
        return await task
    finally:
        owned = [task] if signal is None else [task, signal]
        for pending in owned:
            if not pending.done():
                pending.cancel()
        await asyncio.gather(*owned, return_exceptions=True)
