# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reversible request draining for an SGLang worker."""

import asyncio
import math
from collections.abc import Callable, Coroutine, Mapping
from contextlib import aclosing
from functools import wraps
from time import monotonic
from typing import Any

from dynamo._core import WorkerDraining
from dynamo.health_check import HEALTH_CHECK_KEY


class DrainConflict(RuntimeError):
    pass


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
        pending_engine_requests: Callable[[], int | None] | None = None,
        scheduler_idle: Callable[[], Coroutine[Any, Any, bool]] | None = None,
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
        self._pending_engine_requests = pending_engine_requests
        self._scheduler_idle = scheduler_idle
        self._idle_probe: asyncio.Task[bool] | None = None
        self._canaries = 0
        # None resumes admission; infinity keeps it open until withdrawal is confirmed.
        self._admission_deadline: float | None = None
        self._last_error: str | None = None
        self._operations: set[asyncio.Task] = set()
        self._close_task: asyncio.Task | None = None

    def _is_stopping(self) -> bool:
        return self._terminal_event.is_set() or self._shutdown_event.is_set()

    def _accepts_requests(self) -> bool:
        return not self._shutdown_event.is_set() and (
            self._admission_deadline is None or monotonic() < self._admission_deadline
        )

    def check_generation_control(self) -> None:
        """Check the generation transition while the lifecycle lock is held."""
        if self._is_stopping():
            raise DrainConflict("worker shutdown has started")
        if not self._ready_event.is_set():
            raise DrainConflict("worker registration is not ready")
        if self._admission_deadline is not None:
            raise DrainConflict(
                "resume worker routing before changing generation state"
            )

    @property
    def withdrawn(self) -> bool:
        return self._is_stopping() or self._admission_deadline is not None

    def _reset_idle_probe(self) -> None:
        if self._idle_probe is not None:
            self._idle_probe.cancel()
            self._idle_probe = None

    def stop(self) -> None:
        self._terminal_event.set()
        self._reset_idle_probe()

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
        self._shutdown_event.set()
        self.stop()
        tasks = set(self._operations)
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

    async def status(self, body: dict) -> dict:
        self._validate_body(body)
        deadline = self._admission_deadline
        accepting = self._ready_event.is_set() and self._accepts_requests()
        reason = None
        if self._is_stopping():
            state = "stopping"
        elif not self._ready_event.is_set():
            state = "starting"
        elif deadline is None:
            state = "serving"
        elif deadline == math.inf:
            state, reason = "withdrawing", "discovery_unregister_pending"
        elif accepting:
            state, reason = "withdrawing", "routing_grace"
        else:
            state = "withdrawn"
            reason = self._local_work_reason()
            if reason is None:
                reason = self._scheduler_idle_reason()
                if reason is None:
                    state = "drained"
            else:
                self._reset_idle_probe()
        return {
            "state": state,
            "reason": reason,
            "admin_withdrawn": self.withdrawn,
            "accepting_requests": accepting,
            "last_error": self._last_error,
        }

    def _local_work_reason(self) -> str | None:
        if self._pending_engine_requests is None:
            return "completion_check_unavailable"
        inflight = self._endpoint.inflight_requests()
        if inflight is None:
            return "completion_check_unavailable"
        if inflight or self._canaries:
            return "dynamo_inflight"
        pending = self._pending_engine_requests()
        if pending is None:
            return "completion_check_unavailable"
        if pending:
            return "sglang_inflight"
        return None

    def _scheduler_idle_reason(self) -> str | None:
        if self._scheduler_idle is None:
            return None
        if self._idle_probe is None:
            probe = asyncio.create_task(self._scheduler_idle())
            self._idle_probe = probe
            self._operations.add(probe)
            probe.add_done_callback(self._operation_finished)
        if not self._idle_probe.done():
            return "scheduler_idle_pending"
        if not self._idle_probe.cancelled() and self._idle_probe.result():
            return None
        self._reset_idle_probe()
        return "scheduler_idle_unavailable"

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
            if self._admission_deadline is not None and math.isfinite(
                self._admission_deadline
            ):
                return await self.status({})
            if self._is_paused():
                raise DrainConflict(
                    "continue engine generation before draining the worker"
                )
            self._admission_deadline = math.inf
            self._last_error = None
            try:
                await self._endpoint.unregister_endpoint_instance()
            except Exception as error:
                self._last_error = str(error)
                raise
            if self._is_stopping():
                raise DrainConflict("worker shutdown superseded drain")
            self._admission_deadline = monotonic() + self._grace_period
            return await self.status({})

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
            self._reset_idle_probe()
            self._admission_deadline = None
            self._last_error = None
            try:
                await self._endpoint.register_endpoint_instance()
            except Exception as error:
                self._last_error = str(error)
                if self._is_stopping():
                    await self._endpoint.unregister_endpoint_instance()
                raise
            if self._is_stopping():
                try:
                    await self._endpoint.unregister_endpoint_instance()
                except Exception as error:
                    self._last_error = str(error)
                    raise
                raise DrainConflict("worker shutdown superseded resume")
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
            if self._shutdown_event.is_set() or (
                not health_check
                and (not self._ready_event.is_set() or not self._accepts_requests())
            ):
                raise WorkerDraining("worker is not accepting new requests")
            track_canary = health_check and self._scheduler_idle is not None
            if track_canary:
                # Local canaries bypass TCP ingress accounting.
                self._reset_idle_probe()
                self._canaries += 1
            try:
                async with aclosing(handler(*args, **kwargs)) as stream:
                    async for item in stream:
                        yield item
            finally:
                if track_canary:
                    self._canaries -= 1

        return admitted
