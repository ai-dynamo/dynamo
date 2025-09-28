# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Optional

from dynamo._core import VirtualConnectorCoordinator
from dynamo.planner.defaults import WORKER_COMPONENT_NAMES
from dynamo.planner.planner_connector import PlannerConnector
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

# Constants for scaling readiness check and waiting
SCALING_CHECK_INTERVAL = int(
    os.environ.get("SCALING_CHECK_INTERVAL", 10)
)  # Check every 10 seconds
SCALING_MAX_WAIT_TIME = int(
    os.environ.get("SCALING_MAX_WAIT_TIME", 1800)
)  # Maximum wait time: 30 minutes (1800 seconds)
SCALING_MAX_RETRIES = SCALING_MAX_WAIT_TIME // SCALING_CHECK_INTERVAL  # 180 retries


class VirtualConnector(PlannerConnector):
    """
    This is a virtual connector for planner to output scaling decisions to non-native environments
    This virtual connector does not actually scale the deployment, instead, it communicates with the non-native environment through dynamo-runtime's VirtualConnectorCoordinator.
    The deployment environment needs to use VirtualConnectorClient (in the Rust/Python bindings) to read from the scaling decisions and update report scaling status.
    """

    def __init__(
        self, runtime: DistributedRuntime, dynamo_namespace: str, backend: str
    ):
        self.connector = VirtualConnectorCoordinator(
            runtime,
            dynamo_namespace,
            SCALING_CHECK_INTERVAL,
            SCALING_MAX_WAIT_TIME,
            SCALING_MAX_RETRIES,
        )

        self.backend = backend
        self.worker_component_names = WORKER_COMPONENT_NAMES[backend]

    async def _async_init(self):
        """Async initialization that must be called after __init__"""
        await self.connector.async_init()

    async def _update_scaling_decision(
        self, num_prefill: Optional[int] = None, num_decode: Optional[int] = None
    ):
        """Update scaling decision"""
        await self.connector.update_scaling_decision(num_prefill, num_decode)

    async def _wait_for_scaling_completion(self):
        """Wait for the deployment environment to report that scaling is complete"""
        await self.connector.wait_for_scaling_completion()

    def _component_to_worker_type(self, component_name: str) -> Optional[str]:
        """Map component name to worker type (prefill or decode)"""
        if component_name == self.worker_component_names.prefill_worker_k8s_name:
            return "prefill"
        elif component_name == self.worker_component_names.decode_worker_k8s_name:
            return "decode"
        else:
            return None

    async def add_component(self, component_name: str, blocking: bool = True):
        """Add a component by increasing its replica count by 1"""
        worker_type = self._component_to_worker_type(component_name)
        if worker_type is None:
            logger.warning(f"Unknown component name: {component_name}, skipping")
            return

        state = self.connector.read_state()

        if worker_type == "prefill":
            await self._update_scaling_decision(
                num_prefill=state.num_prefill_workers + 1
            )
        elif worker_type == "decode":
            await self._update_scaling_decision(num_decode=state.num_decode_workers + 1)

        if blocking:
            await self._wait_for_scaling_completion()

    async def remove_component(self, component_name: str, blocking: bool = True):
        """Remove a component by decreasing its replica count by 1"""
        worker_type = self._component_to_worker_type(component_name)
        if worker_type is None:
            logger.warning(f"Unknown component name: {component_name}, skipping")
            return

        state = self.connector.read_state()

        if worker_type == "prefill":
            new_count = max(0, state.num_prefill_workers - 1)
            await self._update_scaling_decision(num_prefill=new_count)
        elif worker_type == "decode":
            new_count = max(0, state.num_decode_workers - 1)
            await self._update_scaling_decision(num_decode=new_count)

        if blocking:
            await self._wait_for_scaling_completion()

    async def set_component_replicas(
        self, target_replicas: dict[str, int], blocking: bool = True
    ):
        """Set the replicas for multiple components at once"""
        if not target_replicas:
            raise ValueError("target_replicas cannot be empty")

        num_prefill = None
        num_decode = None

        for component_name, replicas in target_replicas.items():
            worker_type = self._component_to_worker_type(component_name)
            if worker_type is None:
                logger.warning(f"Unknown component name: {component_name}, skipping")
                continue

            if worker_type == "prefill":
                num_prefill = replicas
            elif worker_type == "decode":
                num_decode = replicas

        if num_prefill is None and num_decode is None:
            return

        # Update scaling decision if there are any changes
        await self._update_scaling_decision(
            num_prefill=num_prefill, num_decode=num_decode
        )

        if blocking:
            await self._wait_for_scaling_completion()
