# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
from typing import Any, Optional

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
    """Coordinate planner scaling decisions for non-native environments.

    The connector does not scale a deployment directly. It publishes decisions
    through the Dynamo runtime's ``VirtualConnectorCoordinator``; the deployment
    environment consumes them with ``VirtualConnectorClient`` and reports scaling
    status back to the coordinator.

    Virtual deployments do not have a Kubernetes API from which to derive worker
    component and endpoint metadata. They therefore require a
    ``worker_info_provider`` that resolves ``WorkerInfo`` from runtime MDC. The
    planner factory normally supplies its ``RuntimeFpmProvider``, sharing the same
    runtime discovery source used for forward-pass metrics.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        dynamo_namespace: str,
        worker_info_provider: WorkerInfoProvider,
        model_name: Optional[str] = None,
    ):
        """Initialize a virtual deployment connector.

        Args:
            runtime: Distributed runtime used for coordination and discovery.
            dynamo_namespace: Namespace containing the virtual deployment.
            worker_info_provider: Required source of runtime WorkerInfo/MDC used
                to locate worker endpoints. ``construct_environment`` normally
                provides a ``RuntimeFpmProvider``.
            model_name: Model name reported by the deployment.
        """
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

        self.runtime = runtime
        self.dynamo_namespace = dynamo_namespace
        self.worker_info_provider = worker_info_provider
        self._worker_info: dict[SubComponentType, WorkerInfo] = {}
        self._worker_clients: dict[SubComponentType, tuple[str, Any]] = {}

    def get_worker_info(
        self,
        sub_component_type: SubComponentType,
        backend: str = "vllm",
    ) -> WorkerInfo:
        worker_info = self.worker_info_provider.get_worker_info(
            sub_component_type, backend
        )
        self._worker_info[sub_component_type] = worker_info
        return worker_info

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
            current = state.num_prefill_workers
            if current < 0:
                current = await self._get_actual_worker_count(sub_component_type)
            await self._update_scaling_decision(num_prefill=current + 1)
        elif sub_component_type == SubComponentType.DECODE:
            current = state.num_decode_workers
            if current < 0:
                current = await self._get_actual_worker_count(sub_component_type)
            await self._update_scaling_decision(num_decode=current + 1)

        if blocking:
            await self._wait_for_scaling_completion()

    async def remove_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """Remove a component by decreasing its replica count by 1"""
        state = self.coord.read_state()

        if sub_component_type == SubComponentType.PREFILL:
            current = state.num_prefill_workers
            if current < 0:
                current = await self._get_actual_worker_count(sub_component_type)
            new_count = max(0, current - 1)
            await self._update_scaling_decision(num_prefill=new_count)
        elif sub_component_type == SubComponentType.DECODE:
            current = state.num_decode_workers
            if current < 0:
                current = await self._get_actual_worker_count(sub_component_type)
            new_count = max(0, current - 1)
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

    async def get_actual_worker_counts(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> tuple[int, int, bool]:
        """Read active workers from discovery and scaling status from the client ack.

        Coordinator worker counts are desired targets, not observations. Runtime
        endpoint discovery is the source of truth for active workers, while the
        coordinator's decision acknowledgement indicates whether scaling is still
        in progress.
        """
        prefill_count = 0
        decode_count = 0
        if prefill_component_name is not None:
            prefill_count = await self._get_actual_worker_count(
                SubComponentType.PREFILL
            )
        if decode_component_name is not None:
            decode_count = await self._get_actual_worker_count(SubComponentType.DECODE)
        state = self.coord.read_state()
        stable = await self.coord.is_scaling_ready()
        if prefill_component_name is not None and state.num_prefill_workers >= 0:
            stable = stable and prefill_count == state.num_prefill_workers
        if decode_component_name is not None and state.num_decode_workers >= 0:
            stable = stable and decode_count == state.num_decode_workers
        return prefill_count, decode_count, stable

    async def _get_actual_worker_count(
        self, sub_component_type: SubComponentType
    ) -> int:
        worker_info = self._worker_info.get(sub_component_type)
        if worker_info is None:
            worker_info = self.get_worker_info(sub_component_type)
        if not worker_info.component_name or not worker_info.endpoint:
            logger.warning(
                "WorkerInfo missing component or endpoint for %s; reporting zero workers",
                sub_component_type.value,
            )
            return 0

        endpoint_name = (
            f"{self.dynamo_namespace}.{worker_info.component_name}."
            f"{worker_info.endpoint}"
        )
        cached = self._worker_clients.get(sub_component_type)
        if cached is None or cached[0] != endpoint_name:
            client = await self.runtime.endpoint(endpoint_name).client()
            self._worker_clients[sub_component_type] = (endpoint_name, client)
            # Runtime discovery populates a new client's initial instance snapshot
            # asynchronously. Preserve the pre-refactor settling window before the
            # first read so existing workers are not transiently reported as zero.
            await asyncio.sleep(0.1)
        else:
            client = cached[1]
        return len(client.instance_ids())

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
