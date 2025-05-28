# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Router Component with Universal X-Request-Id Support

This example shows how router components can use Dynamo SDK's built-in
request tracing for automatic request ID propagation.
"""

import logging
from typing import Optional, Tuple

from dynamo.sdk import (
    RequestTracingMixin,
    endpoint,
    get_current_request_id,
    service,
)

logger = logging.getLogger(__name__)


@service(
    dynamo={"enabled": True, "namespace": "dynamo"},
    resources={"cpu": "2", "memory": "4Gi"},
)
class Router(RequestTracingMixin):
    """
    Router component with automatic X-Request-Id support.

    Benefits of using RequestTracingMixin:
    - ensure_request_id(): Automatic request ID management
    - log_with_request_id(): Consistent logging with request ID
    - get_current_request_id(): Access request ID anywhere in call stack
    """

    def __init__(self):
        self.worker_loads = {}
        self.worker_count = 3

    @endpoint()
    async def route(
        self, request_data: str, request_id: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Route requests to optimal workers with automatic request ID tracking.

        The RequestTracingMixin automatically handles request ID management.
        """
        request_id = self.ensure_request_id(request_id)

        self.log_with_request_id("info", "Routing request to optimal worker")

        optimal_worker = await self._find_optimal_worker(request_data)
        prefix_hit_rate = await self._calculate_prefix_hit_rate(
            request_data, optimal_worker
        )

        self.log_with_request_id(
            "debug", f"Selected worker {optimal_worker} with hit rate {prefix_hit_rate}"
        )

        return optimal_worker, prefix_hit_rate

    async def _find_optimal_worker(self, request_data: str) -> str:
        """
        Internal method that can access request ID from context.
        """
        current_request_id = get_current_request_id()
        if current_request_id:
            logger.debug(f"Finding optimal worker for request: {current_request_id}")

        optimal_worker_id = hash(request_data) % self.worker_count
        return f"worker-{optimal_worker_id}"

    async def _calculate_prefix_hit_rate(
        self, request_data: str, worker_id: str
    ) -> float:
        """
        Calculate expected prefix hit rate for the selected worker.
        """
        self.log_with_request_id(
            "debug", f"Calculating prefix hit rate for {worker_id}"
        )

        return 0.75

    @endpoint()
    async def update_load(
        self, worker_id: str, load: float, request_id: Optional[str] = None
    ):
        """
        Update worker load information with request tracking.
        """
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("debug", f"Updating load for {worker_id}: {load}")

        self.worker_loads[worker_id] = load
