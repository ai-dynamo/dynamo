# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import traceback
from collections.abc import Callable

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo.common.engine_monitor import EngineHealthMonitorConfig
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.process_monitor import LocalEngineProcesses
from dynamo.vllm.shutdown_watchdog import ShutdownWatchdog

configure_dynamo_logging
logger = logging.getLogger(__name__)


class VllmEngineMonitor:
    """
    Monitors the health of the vLLM engine and initiates a shutdown if the engine is dead.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        engine_client: AsyncLLM,
        shutdown_event: asyncio.Event | None = None,
        *,
        generate_endpoint=None,
        is_busy: Callable[[], bool] = lambda: False,
    ):
        if not isinstance(runtime, DistributedRuntime):
            raise ValueError(
                f"{self.__class__.__name__} requires an instance of DistributedRuntime."
            )
        if not isinstance(engine_client, AsyncLLM):
            raise ValueError(
                f"{self.__class__.__name__} requires an instance of AsyncLLM."
            )

        self.runtime = runtime
        self.engine_client = engine_client
        self.shutdown_event = shutdown_event
        self.generate_endpoint = generate_endpoint
        self.is_busy = is_busy
        self.failed = False
        self._fatal_task = None
        self.health_config = EngineHealthMonitorConfig.from_env()
        self.processes = LocalEngineProcesses(engine_client)
        self._monitor_task = asyncio.create_task(self._check_engine_health())
        self._process_task = asyncio.create_task(self._check_engine_processes())
        self._stats_task = asyncio.create_task(self._periodic_log_stats())

        logger.info(
            f"{self.__class__.__name__} initialized and health check task started."
        )

    def __del__(self):
        for name in ("_monitor_task", "_stats_task", "_process_task"):
            task = getattr(self, name, None)
            if task is not None:
                task.cancel()

    def _shutdown_engine(self):
        """
        Shutdown the vLLM engine on crash scenarios to free resources.
        """

        try:
            self.engine_client.shutdown()
        except Exception as e:
            logger.warning(f"vLLM engine shutdown failed: {e}")

    def request_shutdown(self):
        if not self.failed:
            self.failed = True
            self._fatal_task = asyncio.create_task(self._shutdown_worker())
        return self._fatal_task

    async def _shutdown_worker(self):
        """Withdraw discovery before cleanup, with an independent exit deadline."""
        watchdog = None
        try:
            if self.health_config.shutdown_timeout > 0:
                watchdog = ShutdownWatchdog(self.health_config.shutdown_timeout)
            if self.generate_endpoint is not None:
                try:
                    await asyncio.wait_for(
                        self.generate_endpoint.unregister_endpoint_instance(),
                        timeout=self.health_config.check_timeout or None,
                    )
                except Exception:
                    logger.exception("Failed to withdraw vLLM endpoint from discovery")
            self._shutdown_engine()
            self.runtime.shutdown()
        except Exception:
            logger.exception("Dynamo Runtime shutdown failed")
        finally:
            try:
                os._exit(1)
            finally:
                if watchdog is not None:
                    watchdog.close()

    async def _check_engine_processes(self):
        while not self.failed:
            if self.shutdown_event and self.shutdown_event.is_set():
                return
            try:
                self.processes.check()
                if self.engine_client.errored:
                    raise EngineDeadError()
            except Exception:
                if self.shutdown_event and self.shutdown_event.is_set():
                    return
                logger.exception("Independent vLLM process health check failed")
                self.request_shutdown()
                return
            await asyncio.sleep(self.health_config.interval)

    async def _check_engine_health(self):
        """
        Continuously check engine health until:
        1. Engine or worker health check fails or times out - initiate shutdown
        2. Shutdown event is triggered - stop monitoring gracefully
        3. Task is cancelled - cleanup
        """
        while not self.failed:
            try:
                # Check if shutdown event was triggered - stop monitoring
                if self.shutdown_event and self.shutdown_event.is_set():
                    logger.info(
                        f"{self.__class__.__name__}: Shutdown event detected, stopping engine health monitoring."
                    )
                    break

                await self._run_health_check()

                # Sleep with shutdown event awareness for faster response
                if self.shutdown_event:
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(),
                            timeout=self.health_config.interval,
                        )
                        # Shutdown event was set during sleep
                        logger.info(
                            f"{self.__class__.__name__}: Shutdown event detected, stopping engine health monitoring."
                        )
                        break
                    except asyncio.TimeoutError:
                        # Normal timeout, continue monitoring
                        pass
                else:
                    # No shutdown event, just sleep normally
                    await asyncio.sleep(self.health_config.interval)

            except Exception as e:
                if self.shutdown_event and self.shutdown_event.is_set():
                    logger.warning(
                        "%s: %s while worker shutdown is in progress; "
                        "stopping health monitoring.",
                        self.__class__.__name__,
                        type(e).__name__,
                    )
                    break
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error(f"vLLM AsyncLLM health check failed: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.request_shutdown()
                return
            except asyncio.CancelledError:
                logger.debug(f"{self.__class__.__name__}: Health check task cancelled.")
                break

    async def _run_health_check(self):
        if self.is_busy():
            return
        health_check = self.engine_client.check_health()
        try:
            if self.health_config.check_timeout > 0:
                return await asyncio.wait_for(
                    health_check, timeout=self.health_config.check_timeout
                )
            return await health_check
        except asyncio.TimeoutError:
            if self.is_busy():
                return
            raise

    async def _periodic_log_stats(self):
        """Periodically flush vLLM engine stats (throughput, cache usage, etc.)."""
        try:
            interval = float(os.environ.get("VLLM_LOG_STATS_INTERVAL", "10.0"))
        except ValueError:
            logger.warning(
                "Invalid VLLM_LOG_STATS_INTERVAL value: %r, using default 10.0",
                os.environ.get("VLLM_LOG_STATS_INTERVAL"),
            )
            interval = 10.0
        if interval <= 0:
            return
        if not getattr(self.engine_client, "log_stats", True):
            return

        while True:
            try:
                if self.shutdown_event and self.shutdown_event.is_set():
                    break

                if self.shutdown_event:
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(), timeout=interval
                        )
                        break
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(interval)

                await self.engine_client.do_log_stats()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Error in periodic stats logging", exc_info=True)
