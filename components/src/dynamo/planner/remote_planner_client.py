# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for calling remote planner's scale_request endpoint."""

import asyncio
import logging

from dynamo.planner.defaults import SubComponentType
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
        connection_timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.runtime = runtime
        self.central_namespace = central_namespace
        self.central_component = central_component
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        self._client = None

    async def _ensure_client(self):
        """Lazy initialization of endpoint client with retry mechanism"""
        if self._client is None:
            endpoint = self.runtime.endpoint(
                f"{self.central_namespace}.{self.central_component}.scale_request"
            )

            # Retry logic with exponential backoff
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    logger.info(
                        f"Attempting to connect to GlobalPlanner at "
                        f"{self.central_namespace}.{self.central_component} "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )

                    self._client = await endpoint.client()

                    # Wait for instances with timeout
                    await asyncio.wait_for(
                        self._client.wait_for_instances(),
                        timeout=self.connection_timeout,
                    )

                    logger.info(
                        f"Successfully connected to centralized planner at "
                        f"{self.central_namespace}.{self.central_component}"
                    )
                    return

                except asyncio.TimeoutError as e:
                    last_error = e
                    logger.warning(
                        f"Connection attempt {attempt + 1} timed out after "
                        f"{self.connection_timeout}s"
                    )
                    self._client = None

                except Exception as e:
                    last_error = e
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                    self._client = None

                # Exponential backoff before retry (except on last attempt)
                if attempt < self.max_retries - 1:
                    backoff = 2**attempt  # 1s, 2s, 4s, ...
                    logger.info(f"Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)

            # All retries exhausted
            raise RuntimeError(
                f"Failed to connect to GlobalPlanner at "
                f"{self.central_namespace}.{self.central_component} after "
                f"{self.max_retries} attempts. Last error: {last_error}"
            )

    async def send_scale_request(self, request: ScaleRequest) -> ScaleResponse:
        """Send scale request to centralized planner"""
        await self._ensure_client()

        logger.info(
            f"Sending scale request to centralized planner: "
            f"prefill={[r.desired_replicas for r in request.target_replicas if r.sub_component_type == SubComponentType.PREFILL]}, "
            f"decode={[r.desired_replicas for r in request.target_replicas if r.sub_component_type == SubComponentType.DECODE]}"
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
