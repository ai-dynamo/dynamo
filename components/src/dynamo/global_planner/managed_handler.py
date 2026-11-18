# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed request validation; capacity execution stays in the controller."""

from dynamo.global_planner.allocation_policy import PoolAllocationPolicy
from dynamo.global_planner.managed_config import ManagedAllocationConfig
from dynamo.global_planner.managed_controller import ManagedAllocationController
from dynamo.planner.connectors.protocol import (
    AllocationStatusRequest,
    ScaleRequest,
    ScaleResponse,
    ScaleStatus,
)
from dynamo.runtime import dynamo_endpoint


class ManagedRequestHandler:
    """Bind each request to its enrolled pool, then delegate to the controller."""

    def __init__(
        self, config: ManagedAllocationConfig, controller: ManagedAllocationController
    ):
        self.config = config
        self.controller = controller
        self._participants = {pool.participant_id: pool for pool in config.policy.pools}

    def _pool(
        self, request: ScaleRequest | AllocationStatusRequest
    ) -> PoolAllocationPolicy:
        participant = f"{request.k8s_namespace}/{request.graph_deployment_name}"
        pool = self._participants.get(participant)
        if (
            pool is None
            or self.config.caller_namespaces[pool.pool_id] != request.caller_namespace
        ):
            raise ValueError("caller and deployment do not match an enrolled pool")
        # Caller namespace is a cooperative binding, not authenticated identity.
        # The runtime must be restricted to trusted Local Planner participants.
        return pool

    @dynamo_endpoint(ScaleRequest, ScaleResponse)
    async def scale_request(self, request: ScaleRequest):
        try:
            pool = self._pool(request)
            if len(request.target_replicas) != 1:
                raise ValueError("managed pools require exactly one replica target")
            target = request.target_replicas[0]
            if (
                target.sub_component_type.value != pool.role
                or target.component_name not in (None, pool.component_name)
            ):
                raise ValueError(
                    "replica target does not match the enrolled component and role"
                )
            if request.request_id is None:
                raise ValueError(
                    "managed requests require request_id; upgrade the Local Planner"
                )
            response = self.controller.submit(
                pool.pool_id, target.desired_replicas, request.request_id
            )
        except ValueError as exc:
            response = _error(exc)
        yield response.model_dump(mode="json")

    @dynamo_endpoint(AllocationStatusRequest, ScaleResponse)
    async def allocation_status(self, request: AllocationStatusRequest):
        try:
            response = self.controller.status(self._pool(request).pool_id)
        except ValueError as exc:
            response = _error(exc)
        yield response.model_dump(mode="json")


def _error(exc: ValueError) -> ScaleResponse:
    """Reject an invalid request without touching retained demand."""
    return ScaleResponse(
        status=ScaleStatus.ERROR, message=str(exc), current_replicas={}
    )
