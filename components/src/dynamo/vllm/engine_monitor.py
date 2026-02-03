# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import signal
import traceback

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging
logger = logging.getLogger(__name__)

HEALTH_CHECK_INTERVAL = 2
ENGINE_SHUTDOWN_TIMEOUT = 30  # seconds


class VllmEngineMonitor:
    """
    Monitors the health of the vLLM engine and initiates a shutdown if the engine is dead.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        engine_client: AsyncLLM,
        shutdown_event: asyncio.Event = None,
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
        self._monitor_task = asyncio.create_task(self._check_engine_health())

        logger.info(
            f"{self.__class__.__name__} initialized and health check task started."
        )

    def __del__(self):
        self._monitor_task.cancel()

    def _shutdown_engine(self):
        """
        Shutdown the vLLM engine on crash scenarios to free resources.
        """

        # Has timeout protection via SIGALRM
        def timeout_handler(signum, frame):
            raise TimeoutError("Engine shutdown timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(ENGINE_SHUTDOWN_TIMEOUT)

        try:
            self.engine_client.shutdown()
        except Exception as e:
            logger.warning(f"vLLM engine shutdown failed: {e}")
        finally:
            signal.alarm(0)

    async def _check_engine_health(self):
        """
        Continuously check engine health until:
        1. Engine dies (EngineDeadError) - initiate shutdown
        2. Shutdown event is triggered - stop monitoring gracefully
        3. Task is cancelled - cleanup
        """
        while True:
            try:
                # Check if shutdown event was triggered - stop monitoring
                if self.shutdown_event and self.shutdown_event.is_set():
                    logger.info(
                        f"{self.__class__.__name__}: Shutdown event detected, stopping engine health monitoring."
                    )
                    break

                await self.engine_client.check_health()

                # Sleep with shutdown event awareness for faster response
                if self.shutdown_event:
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(), timeout=HEALTH_CHECK_INTERVAL
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
                    await asyncio.sleep(HEALTH_CHECK_INTERVAL)

            except EngineDeadError as e:
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error(f"vLLM AsyncLLM health check failed: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self._shutdown_engine()
                self.runtime.shutdown()
                os._exit(1)
            except asyncio.CancelledError:
                logger.debug(f"{self.__class__.__name__}: Health check task cancelled.")
                break
