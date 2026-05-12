# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deferred abort for decode-mode disaggregated serving.

Cancelling a decode request before its first generation result races
the prefill peer's in-flight NIXL KV transfer. Each backend subclasses
`DeferredAbort` with its native abort handle; the bookkeeping lives here.
"""

from __future__ import annotations

import abc
import asyncio
import logging

logger = logging.getLogger(__name__)


class DeferredAbort(abc.ABC):
    """Defer engine abort until the first generation result arrives.

    Usage in `generate()`::

        guard = MyDeferredAbort(...)
        self._active_aborts[request_id] = guard
        try:
            async for chunk in stream:
                guard.signal_first_token()
                yield chunk
        finally:
            self._active_aborts.pop(request_id, None)
            await guard.close()

    `abort()` is non-blocking and idempotent; it fires now-or-on-
    first-token.
    """

    def __init__(self) -> None:
        self._first_token_event = asyncio.Event()
        self._abort_requested = False
        # Strong reference so a parked wait task isn't GC'd.
        self._abort_task: asyncio.Task[None] | None = None

    def signal_first_token(self) -> None:
        self._first_token_event.set()

    def abort(self) -> None:
        if self._abort_requested:
            return
        self._abort_requested = True
        self._abort_task = asyncio.create_task(self._wait_and_abort())

    async def _wait_and_abort(self) -> None:
        # event may have fired between abort() and task pickup
        if not self._first_token_event.is_set():
            try:
                await self._wait_for_first_token()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "DeferredAbort: wait raised; firing abort anyway",
                    exc_info=True,
                )
        try:
            await self._do_abort_now()
        except Exception:
            logger.warning("DeferredAbort: engine abort raised", exc_info=True)

    async def _wait_for_first_token(self) -> None:
        """Override when the engine has a side channel for first-token
        arrival (e.g. TRT-LLM's `GenerationResult.aqueue`) — useful
        when the main generate loop may be cancelled before yielding."""
        await self._first_token_event.wait()

    @abc.abstractmethod
    async def _do_abort_now(self) -> None:
        """Fire the engine-specific abort call."""

    async def close(self) -> None:
        """Cancel a parked wait task if the request finished without
        being aborted. Idempotent."""
        if self._abort_task is None or self._abort_task.done():
            return
        if not self._first_token_event.is_set():
            self._abort_task.cancel()
        try:
            await self._abort_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning("DeferredAbort: close raised", exc_info=True)
