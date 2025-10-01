# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import time

import sglang as sgl
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

# Configuration constants
HEALTH_CHECK_INTERVAL = int(os.getenv("SGLANG_HEALTH_CHECK_INTERVAL", "2"))
MAX_FAILURES_BEFORE_SHUTDOWN = int(os.getenv("SGLANG_MAX_FAILURES_BEFORE_SHUTDOWN", "3"))


class SglangEngineMonitor:
    """
    Monitors the health of the SGLang engine and initiates a shutdown if the engine is unresponsive.
    """

    def __init__(self, runtime: DistributedRuntime, engine: sgl.Engine, endpoint_name: str = "generate"):
        if not isinstance(runtime, DistributedRuntime):
            raise ValueError(
                f"{self.__class__.__name__} requires an instance of DistributedRuntime."
            )
        # Note: We can't use isinstance with sgl.Engine since it's a LazyImport
        # The type of sgl.Engine is <class 'sglang.utils.LazyImport'>
        # Just check if it has the expected method
        if not hasattr(engine, 'get_server_info'):
            raise ValueError(
                f"{self.__class__.__name__} requires an SGLang Engine with get_server_info method."
            )

        self.runtime = runtime
        self.engine = engine
        self.endpoint_name = endpoint_name
        self.consecutive_failures = 0
        self.last_health_status = None  # Initialize to None so first update works
        self._monitor_task = asyncio.create_task(self._check_engine_health())

        # Mark endpoint as ready initially
        self._update_health_status("Ready")

        logger.info(
            f"{self.__class__.__name__} initialized for endpoint '{endpoint_name}' and health check task started."
        )

    def __del__(self):
        if hasattr(self, '_monitor_task') and self._monitor_task:
            self._monitor_task.cancel()

    def _update_health_status(self, status: str):
        """Update the endpoint health status in the DistributedRuntime."""
        if status == self.last_health_status:
            return  # No change needed
            
        try:
            logger.info(
                f"Updating endpoint '{self.endpoint_name}' health status: "
                f"{self.last_health_status} -> {status}"
            )
            self.runtime.set_endpoint_health_status(self.endpoint_name, status)
        except Exception as e:
            logger.error(f"Failed to update health status: {e}")
        finally:
            # Always update internal state, even if runtime call fails
            self.last_health_status = status

    async def _check_engine_health(self):
        """Periodically check if the SGLang engine is responsive."""
        while True:
            try:
                # Use get_server_info() to check if engine is responsive
                # This will raise an exception if the engine is dead
                loop = asyncio.get_event_loop()
                server_info = await loop.run_in_executor(None, self.engine.get_server_info)
                
                # Check server info status if available
                # SGLang uses ServerStatus enum: "Up", "Starting", "UnHealthy"
                # Note: internal_states might be a list if multiple DP ranks
                internal_states = server_info.get("internal_states", {})
                if isinstance(internal_states, list):
                    if internal_states:
                        # Take the first DP rank's status if available
                        internal_states = internal_states[0] if internal_states[0] else {}
                    else:
                        # Empty list - no internal states available
                        internal_states = {}
                
                server_status = internal_states.get("status", "Up") if isinstance(internal_states, dict) else "Up"
                
                # Check if server status indicates a problem
                # "Up" and "ready" are healthy, "Starting" and "UnHealthy" are not
                if server_status not in ["Up", "ready"]:
                    logger.warning(f"SGLang engine reporting unhealthy status: {server_status}")
                    self._update_health_status("NotReady")
                else:
                    self.consecutive_failures = 0
                    self._update_health_status("Ready")
                
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                # This is expected when the monitor is being shut down
                logger.debug("Health check loop cancelled, shutting down gracefully")
                break
            except Exception as e:
                self.consecutive_failures += 1
                logger.error(f"SGLang engine health check failed (attempt {self.consecutive_failures}): {e}")
                self._update_health_status("NotReady")
                if self.consecutive_failures >= MAX_FAILURES_BEFORE_SHUTDOWN:
                    logger.warning(f"SGLang engine unresponsive after {self.consecutive_failures} attempts")
                    logger.warning("Initiating Dynamo Runtime shutdown.")
                    self.runtime.shutdown()
                    os._exit(1)
                    
                # Wait before retrying
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def stop(self):
        """Stop the monitor gracefully."""
        logger.info("Stopping SGLang engine monitor")
        
        self._update_health_status("NotReady")
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                logger.debug("Monitor task cancelled successfully")
                pass
        logger.info("SGLang engine monitor stopped")