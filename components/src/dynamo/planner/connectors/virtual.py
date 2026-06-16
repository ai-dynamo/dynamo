# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Optional

from dynamo._core import VirtualConnectorCoordinator
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import PlannerConnector, WorkerInfoProvider
from dynamo.planner.errors import EmptyTargetReplicasError
from dynamo.planner.monitoring.worker_info import WorkerInfo
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
        self,
        runtime: DistributedRuntime,
        dynamo_namespace: str,
        worker_info_provider: WorkerInfoProvider,
        model_name: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.coord = VirtualConnectorCoordinator(
            runtime,
            dynamo_namespace,
            SCALING_CHECK_INTERVAL,
            SCALING_MAX_WAIT_TIME,
            SCALING_MAX_RETRIES,
        )

        if model_name is None:
            raise ValueError("Model name is required for virtual connector")

        self.model_name = model_name.lower()  # normalize model name to lowercase (MDC)

        self.dynamo_namespace = dynamo_namespace
        self.worker_info_provider = worker_info_provider

    def get_worker_info(
        self,
        sub_component_type: SubComponentType,
        backend: str = "vllm",
    ) -> WorkerInfo:
        return self.worker_info_provider.get_worker_info(sub_component_type, backend)

    async def async_init(self):
        """Async initialization that must be called after __init__"""
        await self.coord.async_init()

    async def _update_scaling_decision(
        self, num_prefill: Optional[int] = None, num_decode: Optional[int] = None
    ):
        """Update scaling decision"""
        await self.coord.update_scaling_decision(num_prefill, num_decode)

    async def _wait_for_scaling_completion(self):
        """Wait for the deployment environment to report that scaling is complete"""
        await self.coord.wait_for_scaling_completion()

    async def add_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """Add a component by increasing its replica count by 1"""
        state = self.coord.read_state()

        if sub_component_type == SubComponentType.PREFILL:
            await self._update_scaling_decision(
                num_prefill=state.num_prefill_workers + 1
            )
        elif sub_component_type == SubComponentType.DECODE:
            await self._update_scaling_decision(num_decode=state.num_decode_workers + 1)

        if blocking:
            await self._wait_for_scaling_completion()

    async def remove_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """Remove a component by decreasing its replica count by 1"""
        state = self.coord.read_state()

        if sub_component_type == SubComponentType.PREFILL:
            new_count = max(0, state.num_prefill_workers - 1)
            await self._update_scaling_decision(num_prefill=new_count)
        elif sub_component_type == SubComponentType.DECODE:
            new_count = max(0, state.num_decode_workers - 1)
            await self._update_scaling_decision(num_decode=new_count)

        if blocking:
            await self._wait_for_scaling_completion()

    async def set_component_replicas(
        self, target_replicas: list[TargetReplica], blocking: bool = True
    ):
        """Set the replicas for multiple components at once"""
        if not target_replicas:
            raise EmptyTargetReplicasError()

        num_prefill = None
        num_decode = None

        for target_replica in target_replicas:
            if target_replica.sub_component_type == SubComponentType.PREFILL:
                num_prefill = target_replica.desired_replicas
            elif target_replica.sub_component_type == SubComponentType.DECODE:
                num_decode = target_replica.desired_replicas

        if num_prefill is None and num_decode is None:
            return

        # Update scaling decision if there are any changes
        await self._update_scaling_decision(
            num_prefill=num_prefill, num_decode=num_decode
        )

        if blocking:
            await self._wait_for_scaling_completion()

    async def validate_deployment(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
        require_prefill: bool = True,
        require_decode: bool = True,
    ):
        """Validate the deployment"""
        pass

    async def wait_for_deployment_ready(self, include_planner: bool = True):
        """Wait for the deployment to be ready"""
        del include_planner
        await self._wait_for_scaling_completion()

    def get_worker_runtime_namespace(self, base_dynamo_namespace: str) -> str:
        return base_dynamo_namespace

    def get_actual_worker_counts(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> tuple[int, int, bool]:
        """Read worker counts reported by the virtual connector client."""
        del prefill_component_name, decode_component_name
        state = self.coord.read_state()
        return state.num_prefill_workers, state.num_decode_workers, True

    def get_model_name(
        self, require_prefill: bool = True, require_decode: bool = True
    ) -> str:
        """Get the model name from the deployment"""
        return self.model_name

    def get_gpu_counts(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> tuple[Optional[int], Optional[int]]:
        """Virtual deployments do not expose GPU shape through the coordinator."""
        del require_prefill, require_decode
        return None, None
