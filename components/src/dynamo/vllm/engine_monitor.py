# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import traceback

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging
logger = logging.getLogger(__name__)

HEALTH_CHECK_INTERVAL = 2


class VllmEngineMonitor:
    """
    Monitors the health of the vLLM engine and initiates a shutdown if the engine is dead.
    """

    def __init__(self, runtime: DistributedRuntime, engine_client: AsyncLLM):
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
        self._monitor_task = asyncio.create_task(self._check_engine_health())

        logger.info(
            f"{self.__class__.__name__} initialized and health check task started."
        )

    def __del__(self):
        self._monitor_task.cancel()

    async def _check_engine_health(self):
        while True:
            try:
                await self.engine_client.check_health()
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            except EngineDeadError as e:
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error(f"vLLM AsyncLLM health check failed: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)
            except asyncio.CancelledError:
                pass
