# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for calling remote planner's scale_request endpoint."""

import asyncio
import logging

from dynamo._core import Client
from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.connectors.protocol import (
    AllocationStatusRequest,
    ScaleRequest,
    ScaleResponse,
)
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


class RemotePlannerClient:
    """Client for delegating scaling requests to centralized planner"""

    def __init__(
        self,
        runtime: DistributedRuntime,
        central_namespace: str,
        central_component: str,
        connection_timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.runtime = runtime
        self.central_namespace = central_namespace
        self.central_component = central_component
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        self._client: Client | None = None
        self._status_client: Client | None = None

    async def _ensure_client(self):
        """Lazy initialization of endpoint client with retry mechanism"""
        if self._client is None:
            self._client = await self._connect("scale_request")

    async def _connect(self, endpoint_name: str) -> Client:
        """Resolve either endpoint with the same bounded connection retries."""
        address = f"{self.central_namespace}.{self.central_component}.{endpoint_name}"
        endpoint = self.runtime.endpoint(address)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    "Connecting to GlobalPlanner at %s (attempt %s/%s)",
                    address,
                    attempt + 1,
                    self.max_retries,
                )
                client = await endpoint.client()
                await asyncio.wait_for(
                    client.wait_for_instances(), timeout=self.connection_timeout
                )
                logger.info("Connected to GlobalPlanner at %s", address)
                return client
            except Exception as e:
                # Runtime discovery can fail before a request is submitted.
                # Retry connection only; never replay a mutation implicitly.
                last_error = e
                logger.warning("Connection attempt %s failed: %s", attempt + 1, e)
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Failed to connect to GlobalPlanner at {address} after "
                        f"{self.max_retries} attempts. Last error: {last_error}"
                    ) from e
            await asyncio.sleep(2**attempt)
        raise ValueError("max_retries must be positive")

    async def send_scale_request(self, request: ScaleRequest) -> ScaleResponse:
        """Send scale request to centralized planner"""
        await self._ensure_client()

        logger.info(
            "Sending scale request to centralized planner: prefill=%s, decode=%s",
            [
                r.desired_replicas
                for r in request.target_replicas
                if r.sub_component_type == SubComponentType.PREFILL
            ],
            [
                r.desired_replicas
                for r in request.target_replicas
                if r.sub_component_type == SubComponentType.DECODE
            ],
        )

        # Send request via the runtime client's generate method (the correct API for
        # calling any dynamo endpoint, regardless of its registered name)
        if self._client is None:
            raise RuntimeError("Scale request client was not initialized")
        response = await self._send(self._client, request.model_dump_json())
        logger.info(
            "Scale request response: %s - %s", response.status, response.message
        )
        return response

    async def get_allocation_status(
        self, request: AllocationStatusRequest
    ) -> ScaleResponse:
        """Read a pool's allocation without submitting or replacing demand."""

        async def read_status() -> ScaleResponse:
            if self._status_client is None:
                self._status_client = await self._connect("allocation_status")
            return await self._send(self._status_client, request.model_dump_json())

        # Bound discovery and response streaming together. This observational
        # request must not indefinitely delay independent worker readiness reads.
        return await asyncio.wait_for(read_status(), timeout=self.connection_timeout)

    @staticmethod
    async def _send(client: Client, request_json: str) -> ScaleResponse:
        stream = await client.generate(request_json)

        response_data = None
        async for output in stream:
            response_data = output.data() if hasattr(output, "data") else output
            break  # Both planner endpoints yield a single response.

        if response_data is None:
            raise RuntimeError("No response from centralized planner")

        return ScaleResponse(**response_data)
