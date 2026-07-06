# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Idle sleep TTL ladder for vLLM workers.

Tracks per-worker request activity and, once the worker has been idle
(zero in-flight requests) for longer than a configurable TTL, invokes the
same transition as the external ``control/sleep`` route (unregister from
discovery -> drain -> engine sleep). An optional escalation TTL deepens
the sleep level after a further period asleep. Waking stays external via
the ``control/wake_up`` route; a successful wake resets the idle timer.

This module intentionally has no vLLM/torch imports so it can be unit
tested standalone.
"""

import asyncio
import enum
import logging
import time
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

SleepFn = Callable[[int], Awaitable[dict]]
EscalateFn = Callable[[int, int], Awaitable[dict]]


class IdleSleepState(enum.Enum):
    WARM = "warm"
    ASLEEP = "asleep"


class IdleSleepMonitor:
    """Per-worker idle monitor implementing a WARM -> ASLEEP TTL ladder.

    All transitions are single-flighted: the monitor runs one background
    task, and the injected ``sleep_fn`` / ``escalate_fn`` are expected to
    serialize against external sleep/wake via the handler's pause lock.
    A sleep is only attempted when ``in_flight == 0`` and the idle time
    exceeds the TTL; requests arriving mid-transition are drained by the
    sleep path itself, same as an externally requested sleep.
    """

    def __init__(
        self,
        *,
        sleep_fn: SleepFn,
        timeout: float,
        level: int = 1,
        escalate_fn: Optional[EscalateFn] = None,
        escalate_timeout: Optional[float] = None,
        escalate_level: int = 2,
        shutdown_event: Optional[asyncio.Event] = None,
        poll_interval: Optional[float] = None,
        time_source: Callable[[], float] = time.monotonic,
    ):
        if timeout <= 0:
            raise ValueError("idle sleep timeout must be > 0")
        if escalate_timeout is not None and escalate_timeout <= 0:
            raise ValueError("idle sleep escalate timeout must be > 0")
        if escalate_timeout is not None and escalate_fn is None:
            raise ValueError("escalate_timeout requires escalate_fn")

        self._sleep_fn = sleep_fn
        self._escalate_fn = escalate_fn
        self._timeout = timeout
        self._level = level
        self._escalate_timeout = escalate_timeout
        self._escalate_level = escalate_level
        self._shutdown_event = shutdown_event
        self._poll_interval = (
            poll_interval
            if poll_interval is not None
            else max(0.01, min(1.0, timeout / 4.0))
        )
        self._now = time_source

        self._state = IdleSleepState.WARM
        self._in_flight = 0
        self._last_activity = self._now()
        self._slept_at: Optional[float] = None
        self._current_level: Optional[int] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def state(self) -> IdleSleepState:
        return self._state

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def current_level(self) -> Optional[int]:
        return self._current_level

    def start(self) -> None:
        """Start the background idle monitoring task."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())
        logger.info(
            "[IdleSleep] Monitor started (timeout=%.1fs, level=%d, "
            "escalate_timeout=%s, escalate_level=%d)",
            self._timeout,
            self._level,
            f"{self._escalate_timeout:.1f}s" if self._escalate_timeout else "off",
            self._escalate_level,
        )

    def stop(self) -> None:
        """Cancel the background idle monitoring task."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def record_request_started(self) -> None:
        self._in_flight += 1
        self._last_activity = self._now()

    def record_request_finished(self) -> None:
        if self._in_flight > 0:
            self._in_flight -= 1
        self._last_activity = self._now()

    @contextmanager
    def track_request(self) -> Iterator[None]:
        """Track one in-flight request for idle accounting."""
        self.record_request_started()
        try:
            yield
        finally:
            self.record_request_finished()

    def notify_slept(self, level: int) -> None:
        """Record that the engine went to sleep (auto or external)."""
        self._state = IdleSleepState.ASLEEP
        self._slept_at = self._now()
        self._current_level = level

    def notify_woken(self) -> None:
        """Record that the engine woke up; resets the idle timer."""
        self._state = IdleSleepState.WARM
        self._slept_at = None
        self._current_level = None
        self._last_activity = self._now()

    async def _run(self) -> None:
        while True:
            try:
                if self._shutdown_event and self._shutdown_event.is_set():
                    logger.info(
                        "[IdleSleep] Shutdown event detected, stopping idle monitor."
                    )
                    break

                await self._tick()

                if self._shutdown_event:
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(), timeout=self._poll_interval
                        )
                        logger.info(
                            "[IdleSleep] Shutdown event detected, stopping idle monitor."
                        )
                        break
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                logger.debug("[IdleSleep] Idle monitor task cancelled.")
                raise
            except Exception:
                logger.exception("[IdleSleep] Unexpected error in idle monitor loop")
                raise

    async def _tick(self) -> None:
        if self._state is IdleSleepState.WARM:
            await self._maybe_sleep()
        else:
            await self._maybe_escalate()

    async def _maybe_sleep(self) -> None:
        if self._in_flight > 0:
            return
        idle_for = self._now() - self._last_activity
        if idle_for < self._timeout:
            return

        result = await self._sleep_fn(self._level)
        if _is_ok(result):
            # notify_slept() may already have run via the sleep path; keep
            # the transition idempotent by recording state here as well.
            self.notify_slept(self._level)
            logger.info(
                "[IdleSleep] Worker auto-slept after %.1fs idle (level=%d)",
                idle_for,
                self._level,
            )
        else:
            # Back off a full TTL before retrying so a persistent failure
            # cannot hot-loop the sleep path.
            self._last_activity = self._now()
            logger.warning(
                "[IdleSleep] Auto-sleep failed, retrying after %.1fs: %s",
                self._timeout,
                _message(result),
            )

    async def _maybe_escalate(self) -> None:
        if (
            self._escalate_timeout is None
            or self._escalate_fn is None
            or self._slept_at is None
        ):
            return
        from_level = self._current_level if self._current_level is not None else 0
        if from_level >= self._escalate_level:
            return
        if self._now() - self._slept_at < self._escalate_timeout:
            return

        result = await self._escalate_fn(from_level, self._escalate_level)
        if _is_ok(result):
            self._current_level = self._escalate_level
            logger.info(
                "[IdleSleep] Sleep escalated (level=%d->%d)",
                from_level,
                self._escalate_level,
            )
        else:
            # Back off a full escalation TTL before retrying.
            self._slept_at = self._now()
            logger.warning(
                "[IdleSleep] Sleep escalation failed, retrying after %.1fs: %s",
                self._escalate_timeout,
                _message(result),
            )


def _is_ok(result: Any) -> bool:
    return isinstance(result, dict) and result.get("status") == "ok"


def _message(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("message", result))
    return str(result)
