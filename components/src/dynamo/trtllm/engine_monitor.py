# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM executor health monitor."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

from dynamo.common.engine_monitor import EngineHealthMonitorConfig

if TYPE_CHECKING:
    from dynamo.runtime import DistributedRuntime
    from dynamo.trtllm.engine import TensorRTLLMEngine

logger = logging.getLogger(__name__)

# Upper bound on how often the watchdog thread reads the executor's fatal
# error; the read is a cheap attribute lookup.
_WATCHDOG_MAX_POLL_SECONDS = 0.5
# A zero health-check interval is valid, but Event.wait(0) returns at once and
# would spin the watchdog thread.
_WATCHDOG_MIN_POLL_SECONDS = 0.05

# Held by the first fatal-exit caller and never released, so later callers
# return at once instead of starting a second teardown.
_fatal_exit_lock = threading.Lock()


def exit_worker_on_fatal_error(
    engine: Optional["TensorRTLLMEngine"],
    runtime: Optional[Any],
    shutdown_timeout: Optional[float] = None,
) -> None:
    """Shut down the worker after a fatal engine error and exit with code 1.

    Safe to call from any thread; it never waits on the asyncio event loop.
    Only the first call acts. Runtime shutdown stops admission before engine
    teardown begins; a runtime shutdown error does not skip engine cleanup.
    When the TRT-LLM executor has recorded a fatal
    error, its worker world is dead and TRT-LLM teardown is skipped: that
    teardown sends to and joins the dead ranks, and its IPC sockets may be in
    use by a thread that is already blocked on them. Otherwise the teardown
    runs on a daemon thread and is bounded by ``shutdown_timeout`` seconds
    (``DYN_ENGINE_HEALTH_SHUTDOWN_TIMEOUT`` when None; 0 means no bound).
    """
    if not _fatal_exit_lock.acquire(blocking=False):
        return
    try:
        if runtime is not None:
            try:
                logger.warning("Initiating Dynamo Runtime shutdown.")
                runtime.shutdown()
            except Exception as exc:
                logger.warning(
                    "Dynamo Runtime shutdown failed during TRT-LLM fatal shutdown: %r",
                    exc,
                    exc_info=True,
                )
        if engine is not None:
            fatal_error = engine.get_health_check_fatal_error()
            if fatal_error is not None:
                logger.error(
                    "Skipping TRT-LLM engine shutdown; the engine is dead: %r",
                    fatal_error,
                )
            else:
                if shutdown_timeout is None:
                    shutdown_timeout = (
                        EngineHealthMonitorConfig.from_env().shutdown_timeout
                    )
                _shutdown_engine(engine, shutdown_timeout)
    except Exception as exc:
        logger.warning(
            "Error during TRT-LLM fatal shutdown: %r",
            exc,
            exc_info=True,
        )
    finally:
        logger.critical("Forcing process exit after fatal TRT-LLM engine error.")
        os._exit(1)


def _shutdown_engine(engine: "TensorRTLLMEngine", timeout: float) -> None:
    """Run engine shutdown on a daemon thread and wait at most ``timeout``.

    Stops waiting early if the executor records a fatal error meanwhile: the
    teardown then talks to dead ranks and will not finish.
    """

    def run() -> None:
        try:
            engine.shutdown()
        except Exception as exc:
            logger.warning("TRT-LLM engine shutdown failed: %r", exc, exc_info=True)

    thread = threading.Thread(target=run, name="trtllm-fatal-shutdown", daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout if timeout > 0 else None
    while thread.is_alive():
        if engine.get_health_check_fatal_error() is not None:
            logger.error(
                "TRT-LLM engine became fatal during shutdown; exiting without waiting."
            )
            return
        wait = _WATCHDOG_MAX_POLL_SECONDS
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = min(wait, remaining)
        thread.join(wait)
    if thread.is_alive():
        logger.error(
            "TRT-LLM engine shutdown did not finish within %.1f s; exiting anyway.",
            timeout,
        )


def _watchdog_poll_seconds(interval: float) -> float:
    return max(_WATCHDOG_MIN_POLL_SECONDS, min(interval, _WATCHDOG_MAX_POLL_SECONDS))


class TrtllmEngineMonitor:
    """Poll TRT-LLM's internal health API and stop the worker on fatal state."""

    def __init__(
        self,
        engine: "TensorRTLLMEngine",
        runtime: Optional["DistributedRuntime"] = None,
        shutdown_event: Optional[asyncio.Event] = None,
        *,
        interval: Optional[float] = None,
        check_timeout: Optional[float] = None,
        shutdown_timeout: Optional[float] = None,
    ) -> None:
        self.engine = engine
        self.runtime = runtime
        self.shutdown_event = shutdown_event
        health_config = EngineHealthMonitorConfig.from_env(
            interval=interval,
            check_timeout=check_timeout,
            shutdown_timeout=shutdown_timeout,
        )
        self.interval = health_config.interval
        self.check_timeout = health_config.check_timeout
        self.shutdown_timeout = health_config.shutdown_timeout
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._stop_watchdog = threading.Event()

        if not engine.supports_health_check():
            logger.info(
                "TRT-LLM health monitor disabled; installed TRT-LLM does not "
                "expose _check_health() or executor.check_health()."
            )
            return

        self._monitor_task = asyncio.create_task(self._check_engine_health())
        # An IPC send to a dead rank can block the event loop and this monitor;
        # the thread only reads the recorded fatal error, so it can still exit.
        threading.Thread(
            target=self._watch_fatal_error,
            name="trtllm-fatal-watchdog",
            daemon=True,
        ).start()
        logger.info("TRT-LLM engine health monitor started.")

    async def stop(self) -> None:
        self._stop_watchdog.set()
        if self._monitor_task is None:
            return
        if self._monitor_task is asyncio.current_task():
            return
        self._monitor_task.cancel()
        try:
            await self._monitor_task
        except asyncio.CancelledError:
            pass
        finally:
            self._monitor_task = None

    async def _check_engine_health(self) -> None:
        while True:
            try:
                if self.shutdown_event is not None and self.shutdown_event.is_set():
                    logger.info(
                        "TRT-LLM health monitor stopping because shutdown_event is set."
                    )
                    break

                try:
                    healthy = await self._run_health_check()
                except Exception as exc:
                    logger.error("TRT-LLM health check raised: %r", exc, exc_info=True)
                    healthy = False

                if not healthy:
                    fatal_error = self.engine.get_health_check_fatal_error()
                    logger.error(
                        "TRT-LLM engine is unhealthy: %r",
                        fatal_error if fatal_error else "unknown",
                    )
                    await self._shutdown_worker()
                    return

                if self.shutdown_event is not None:
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(), timeout=self.interval
                        )
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                logger.debug("TRT-LLM health monitor cancelled.")
                break

    def _watch_fatal_error(self) -> None:
        poll = _watchdog_poll_seconds(self.interval)
        while True:
            self._stop_watchdog.wait(poll)
            fatal_error = self.engine.get_health_check_fatal_error()
            if fatal_error is not None:
                logger.error("TRT-LLM engine recorded a fatal error: %r", fatal_error)
                exit_worker_on_fatal_error(
                    self.engine, self.runtime, self.shutdown_timeout
                )
                return
            if self._stop_watchdog.is_set() or (
                self.shutdown_event is not None and self.shutdown_event.is_set()
            ):
                return

    async def _run_health_check(self) -> bool:
        health_check = asyncio.to_thread(self.engine.check_health)
        if self.check_timeout > 0:
            return await asyncio.wait_for(health_check, timeout=self.check_timeout)
        return await health_check

    async def _shutdown_worker(self) -> None:
        self._stop_watchdog.set()
        await asyncio.to_thread(
            exit_worker_on_fatal_error,
            self.engine,
            self.runtime,
            self.shutdown_timeout,
        )
