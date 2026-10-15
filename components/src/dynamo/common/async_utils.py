# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancellation-safe helpers for offloaded and admitted work."""

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, TypeVar

T = TypeVar("T")


async def run_sync_in_thread(
    function: Callable[..., T], /, *args: Any, **kwargs: Any
) -> T:
    """Run synchronous work without abandoning it when the caller is cancelled.

    Python cannot stop a worker created by :func:`asyncio.to_thread`. Wait for
    that worker before propagating cancellation so a caller never assumes a
    mutating operation has stopped while it can still commit in the background.
    """

    worker = asyncio.get_running_loop().run_in_executor(
        None, partial(function, *args, **kwargs)
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.wait({worker})
            except asyncio.CancelledError:
                continue
        if not worker.cancelled():
            worker.exception()
        raise


async def run_async_to_completion(awaitable: Awaitable[T]) -> T:
    """Finish an admitted async operation before propagating cancellation."""

    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                continue
        if not task.cancelled():
            task.exception()
        raise
