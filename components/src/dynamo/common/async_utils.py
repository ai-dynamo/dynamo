# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for safely dispatching synchronous work from asyncio."""

import asyncio
from collections.abc import Callable
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

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
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
