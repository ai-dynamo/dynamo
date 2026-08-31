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

from dynamo.common.engine_monitor import EngineHealthMonitorConfig

if TYPE_CHECKING:
    from dynamo.runtime import DistributedRuntime
    from dynamo.trtllm.engine import TensorRTLLMEngine

logger = logging.getLogger(__name__)


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

        previous_handler = None
        if self.shutdown_timeout > 0:
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(math.ceil(self.shutdown_timeout))

        try:
            self.engine.shutdown()
        except Exception as exc:
            logger.warning("TRT-LLM engine shutdown failed: %r", exc, exc_info=True)
        finally:
            if self.shutdown_timeout > 0:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)
