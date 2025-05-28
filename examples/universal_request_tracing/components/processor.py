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
Main Processor Component with Universal X-Request-Id Support

This example shows how the main processor component can use Dynamo SDK's built-in
request tracing for automatic request ID propagation across components.
"""

import logging
from typing import Any, Dict, List, Optional
import asyncio

from dynamo.sdk import (
    RequestTracingMixin,
    endpoint,
    get_current_request_id,
    service,
    with_request_id,
    async_on_start,
)
from dynamo.client import DynamoClient

logger = logging.getLogger(__name__)


@service(
    dynamo={"enabled": True, "namespace": "dynamo"},
    resources={"cpu": "2", "memory": "4Gi"},
)
class Processor(RequestTracingMixin):
    """
    Main processor component with automatic X-Request-Id support.

    Benefits of using RequestTracingMixin:
    - ensure_request_id(): Automatic request ID management
    - log(): Consistent logging with request ID
    - get_current_request_id(): Access request ID anywhere in call stack
    """

    def __init__(self):
        self.prefiller_client: Optional[DynamoClient] = None
        self.decoder_client: Optional[DynamoClient] = None
        self.request_count = 0

    @async_on_start
    async def async_init(self):
        """Asynchronous initialization method that runs on service startup."""
        self.log("debug", "Initializing component clients")
        self.prefiller_client = await DynamoClient.create("prefiller")
        self.decoder_client = await DynamoClient.create("decoder")
        self.log("debug", "Component clients initialized successfully")

    @endpoint(is_api=True)
    @with_request_id()
    async def process(
        self, request_text: str, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process request with automatic request ID tracking.

        Args:
            request_text: The text to process
            request_id: Request ID parameter. The @with_request_id decorator
                       ensures it's a non-None str inside the function body.

        Returns:
            A dict containing the processing results
        """
        self.log("info", "Processing request")
        self.request_count += 1

        try:
            # Prefill KV cache
            self.log("debug", "Calling prefiller service")
            prefill_result = await self.prefiller_client.prefill(
                request_text, request_id=request_id
            )

            # Pass hidden states to decoder
            hidden_states = self._compute_hidden_states(request_text)
            self.log("debug", "Calling decoder service")
            token_results = []
            async for token in self.decoder_client.decode(
                hidden_states, request_id=request_id
            ):
                token_results.append(token)

            # Format and return results
            self.log("info", "Request processing completed")
            return {
                "input": request_text,
                "request_id": request_id,
                "prefill_status": prefill_result["status"],
                "tokens": token_results,
            }

        except Exception as e:
            self.log("error", f"Processing failed: {e}")
            raise

    def _compute_hidden_states(self, text: str) -> List[float]:
        """
        Simplified hidden states computation.
        In a real system, this would be a more complex operation.
        """
        self.log("debug", "Computing hidden states")
        return [ord(c) / 256.0 for c in text]  # Simple mock computation

    @endpoint(is_api=True)
    @with_request_id()
    async def get_system_stats(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics from all components with request tracking.

        Args:
            request_id: Request ID parameter. The @with_request_id decorator
                       ensures it's a non-None str inside the function body.

        Returns:
            A dict containing stats from all components
        """
        self.log("info", "Retrieving system statistics")

        prefiller_stats = await self.prefiller_client.get_stats(request_id=request_id)
        decoder_stats = await self.decoder_client.get_stats(request_id=request_id)

        self.log("debug", "All stats retrieved successfully")

        return {
            "processor": {"request_count": self.request_count},
            "prefiller": prefiller_stats,
            "decoder": decoder_stats,
        }
