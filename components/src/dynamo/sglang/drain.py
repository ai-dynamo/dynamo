# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reversible routing withdrawal without engine completion tracking."""

import asyncio
import math
from collections.abc import Callable, Mapping
from contextlib import aclosing
from functools import wraps

from dynamo._core import WorkerDraining
from dynamo.health_check import HEALTH_CHECK_KEY


class DrainConflict(RuntimeError):
    """The worker lifecycle does not permit the requested control."""


class WorkerDrain:
    def __init__(
        self,
        endpoint,
        *,
        lock: asyncio.Lock,
        shutdown_event: asyncio.Event,
        terminal_event: asyncio.Event,
        ready_event: asyncio.Event,
        is_paused: Callable[[], bool],
        grace_period: float,
    ) -> None:
        if not math.isfinite(grace_period) or grace_period < 0:
            raise ValueError("drain grace period must be finite and non-negative")
        self._endpoint = endpoint
        self._lock = lock
        self._shutdown_event = shutdown_event
        self._terminal_event = terminal_event
        self._ready_event = ready_event
        self._is_paused = is_paused
        self._grace_period = grace_period
        self._state = "starting"
        self._accepting = True
        self._withdrawn = False
        self._last_error: str | None = None
        self._grace_task: asyncio.Task | None = None
        self._operations: set[asyncio.Task] = set()
        self._close_task: asyncio.Task | None = None

    def _is_stopping(self) -> bool:
        if self._terminal_event.is_set() or self._shutdown_event.is_set():
            self.stop()
        elif self._state == "starting" and self._ready_event.is_set():
            self._state = "serving"
        return self._state == "stopping"

    @property
    def withdrawn(self) -> bool:
        return self._is_stopping() or self._withdrawn

    def stop(self) -> None:
        self._state = "stopping"
        self._terminal_event.set()
        if self._shutdown_event.is_set():
            self._accepting = False
        self._cancel_grace()

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(self._operation_finished)
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await asyncio.shield(self._close_task)
            raise

    async def _close(self) -> None:
        grace = self._grace_task
        self._shutdown_event.set()
        self.stop()
        self._accepting = False
        tasks = set(self._operations)
        if grace is not None:
            tasks.add(grace)
        for task in tasks:
            task.cancel()
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=5)
            if pending:
                raise RuntimeError("worker drain controls did not stop before cleanup")
        try:
            async with asyncio.timeout(5):
                await self._endpoint.unregister_endpoint_instance()
        except Exception as error:
            self._last_error = str(error)
            raise
        self._withdrawn = True

    def _cancel_grace(self) -> None:
        if self._grace_task is not None:
            self._grace_task.cancel()
            self._grace_task = None

    async def status(self, body: dict) -> dict:
        self._validate_body(body)
        self._is_stopping()
        return {
            "state": self._state,
            "admin_withdrawn": self._withdrawn,
            "accepting_requests": self._ready_event.is_set() and self._accepting,
            "last_error": self._last_error,
        }

    @staticmethod
    def _validate_body(body: dict) -> None:
        if body:
            raise ValueError("worker drain controls require an empty object")

    def _operation_finished(self, task: asyncio.Task) -> None:
        self._operations.discard(task)
        if not task.cancelled():
            task.exception()

    async def _owned(self, operation):
        task = asyncio.create_task(operation)
        self._operations.add(task)
        task.add_done_callback(self._operation_finished)
        return await asyncio.shield(task)

    async def drain(self, body: dict) -> dict:
        self._validate_body(body)
        return await self._owned(self._drain())

    async def _drain(self) -> dict:
        async with self._lock:
            if self._is_stopping():
                raise DrainConflict("worker shutdown has started")
            if not self._ready_event.is_set():
                raise DrainConflict("worker registration is not ready")
            if self._state in ("withdrawing", "withdrawn"):
                return await self.status({})
            self._state = "withdrawing"
            self._last_error = None
            try:
                await self._endpoint.unregister_endpoint_instance()
            except Exception as error:
                self._last_error = str(error)
                if not self._is_stopping():
                    self._state = "serving"
                raise
            self._withdrawn = True
            if not self._is_stopping():
                if self._grace_period == 0:
                    self._accepting = False
                    self._state = "withdrawn"
                else:
                    self._grace_task = asyncio.create_task(self._close_after_grace())
            return await self.status({})

    async def _close_after_grace(self) -> None:
        await asyncio.sleep(self._grace_period)
        async with self._lock:
            if self._state == "withdrawing":
                self._accepting = False
                self._state = "withdrawn"

    async def resume(self, body: dict) -> dict:
        self._validate_body(body)
        return await self._owned(self._resume())

    async def _resume(self) -> dict:
        async with self._lock:
            if self._is_stopping():
                raise DrainConflict("worker shutdown has started")
            if not self._ready_event.is_set():
                raise DrainConflict("worker registration is not ready")
            if self._is_paused():
                raise DrainConflict(
                    "continue engine generation before resuming worker routing"
                )
            if self._state == "serving":
                return await self.status({})
            self._cancel_grace()
            self._accepting = True
            try:
                await self._endpoint.register_endpoint_instance()
            except Exception as error:
                self._accepting = False
                if not self._is_stopping():
                    self._state = "withdrawn"
                self._last_error = str(error)
                raise
            self._withdrawn = False
            if self._is_stopping():
                try:
                    await self._endpoint.unregister_endpoint_instance()
                except Exception as error:
                    self._last_error = str(error)
                    raise
                self._withdrawn = True
                raise DrainConflict("worker shutdown superseded resume")
            self._state = "serving"
            self._last_error = None
            return await self.status({})

    def wrap(self, handler):
        @wraps(handler)
        async def admitted(*args, **kwargs):
            request = args[0] if args else kwargs.get("request")
            health_check = False
            if isinstance(request, Mapping):
                inner = request.get("request", request)
                health_check = bool(request.get(HEALTH_CHECK_KEY)) or (
                    isinstance(inner, Mapping) and bool(inner.get(HEALTH_CHECK_KEY))
                )
            self._is_stopping()
            if self._shutdown_event.is_set() or (
                not health_check
                and (not self._ready_event.is_set() or not self._accepting)
            ):
                raise WorkerDraining("worker is not accepting new requests")
            async with aclosing(handler(*args, **kwargs)) as stream:
                async for item in stream:
                    yield item

        return admitted
