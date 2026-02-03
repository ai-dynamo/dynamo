# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for calling remote planner's scale_request endpoint."""

import logging

from dynamo.planner.scale_protocol import ScaleRequest, ScaleResponse
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


class RemotePlannerClient:
    """Client for delegating scaling requests to centralized planner"""

    def __init__(
        self,
        runtime: DistributedRuntime,
        central_namespace: str,
        central_component: str,
    ):
        self.runtime = runtime
        self.central_namespace = central_namespace
        self.central_component = central_component
        self._client = None

    async def _ensure_client(self):
        """Lazy initialization of endpoint client"""
        if self._client is None:
            endpoint = (
                self.runtime.namespace(self.central_namespace)
                .component(self.central_component)
                .endpoint("scale_request")
            )
            self._client = await endpoint.client()
            await self._client.wait_for_instances()
            logger.info(
                f"Connected to centralized planner at {self.central_namespace}.{self.central_component}"
            )

    async def send_scale_request(self, request: ScaleRequest) -> ScaleResponse:
        """Send scale request to centralized planner"""
        await self._ensure_client()

        logger.info(
            f"Sending scale request to centralized planner: "
            f"prefill={[r.desired_replicas for r in request.target_replicas if r.sub_component_type == 'prefill']}, "
            f"decode={[r.desired_replicas for r in request.target_replicas if r.sub_component_type == 'decode']}"
        )

        # Send request to single endpoint
        request_json = request.model_dump_json()

        response_data = await self._client.scale_request(request_json)

        if response_data is None:
            raise RuntimeError("No response from centralized planner")

        # Parse response
        response = ScaleResponse(**response_data)
        logger.info(f"Scale request response: {response.status} - {response.message}")

        return response
