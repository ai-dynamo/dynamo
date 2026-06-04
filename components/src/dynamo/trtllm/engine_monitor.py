# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM executor health monitor."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dynamo.runtime import DistributedRuntime
    from dynamo.trtllm.engine import TensorRTLLMEngine

logger = logging.getLogger(__name__)

HEALTH_CHECK_INTERVAL = 2.0
HEALTH_CHECK_TIMEOUT = 30.0
HEALTH_SHUTDOWN_TIMEOUT = 60.0
HEALTH_CHECK_INTERVAL_ENV = "DYN_TRTLLM_HEALTH_CHECK_INTERVAL"
HEALTH_CHECK_TIMEOUT_ENV = "DYN_TRTLLM_HEALTH_CHECK_TIMEOUT"
HEALTH_SHUTDOWN_TIMEOUT_ENV = "DYN_TRTLLM_HEALTH_SHUTDOWN_TIMEOUT"


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %.1f", name, value, default)
        return default
    if parsed < 0:
        logger.warning("Negative %s=%r; using 0", name, value)
        return 0.0
    return parsed


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
        self.interval = (
            _env_float(HEALTH_CHECK_INTERVAL_ENV, HEALTH_CHECK_INTERVAL)
            if interval is None
            else interval
        )
        self.check_timeout = (
            _env_float(HEALTH_CHECK_TIMEOUT_ENV, HEALTH_CHECK_TIMEOUT)
            if check_timeout is None
            else check_timeout
        )
        self.shutdown_timeout = (
            _env_float(HEALTH_SHUTDOWN_TIMEOUT_ENV, HEALTH_SHUTDOWN_TIMEOUT)
            if shutdown_timeout is None
            else shutdown_timeout
        )
        self._monitor_task: Optional[asyncio.Task[None]] = None

        if not engine.supports_health_check():
            logger.info(
                "TRT-LLM health monitor disabled; installed TRT-LLM does not "
                "expose _check_health() or executor.check_health()."
            )
            return

        self._monitor_task = asyncio.create_task(self._check_engine_health())
        logger.info("TRT-LLM engine health monitor started.")

    async def stop(self) -> None:
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

                threw_exception = False
                healthy = True
                try:
                    healthy = await self._run_health_check()
                except asyncio.TimeoutError:
                    logger.error(
                        "TRT-LLM health check timed out after %.1fs.",
                        self.check_timeout,
                    )
                    threw_exception = True
                except Exception as exc:
                    logger.error("TRT-LLM health check raised: %r", exc, exc_info=True)
                    threw_exception = True

                if threw_exception or not healthy:
                    fatal_error = self.engine.get_health_check_fatal_error()
                    if fatal_error is not None:
                        logger.error("TRT-LLM engine is unhealthy: %r", fatal_error)
                    else:
                        logger.error("TRT-LLM engine is unhealthy.")
                    self._shutdown_worker()
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

    async def _run_health_check(self) -> bool:
        health_check = asyncio.to_thread(self.engine.check_health)
        if self.check_timeout > 0:
            return await asyncio.wait_for(health_check, timeout=self.check_timeout)
        return await health_check

    def _shutdown_worker(self) -> None:
        self._shutdown_engine()
        try:
            if self.runtime is not None:
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
        except Exception as exc:
            logger.warning(
                "Dynamo Runtime shutdown failed during TRT-LLM fatal health path: %r",
                exc,
                exc_info=True,
            )
        finally:
            os._exit(1)

    def _shutdown_engine(self) -> None:
        """Shutdown the TRT-LLM engine on crash scenarios to free resources."""

        def timeout_handler(signum, frame):
            raise TimeoutError("TRT-LLM engine shutdown timed out")

        if self.shutdown_timeout > 0:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(math.ceil(self.shutdown_timeout))

        try:
            self.engine.shutdown()
        except Exception as exc:
            logger.warning("TRT-LLM engine shutdown failed: %r", exc, exc_info=True)
        finally:
            signal.alarm(0)
