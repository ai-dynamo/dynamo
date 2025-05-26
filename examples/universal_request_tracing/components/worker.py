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
Worker Component with Universal X-Request-Id Support

This example shows how worker components can use Dynamo SDK's built-in
request tracing for automatic request ID propagation and logging.
"""

import asyncio
import logging
from typing import AsyncIterator, Optional

from dynamo.sdk import (
    RequestTracingMixin,
    dynamo_endpoint,
    service,
    get_current_request_id,
)

logger = logging.getLogger(__name__)


@service(
    dynamo={"enabled": True, "namespace": "dynamo"},
    resources={"cpu": "8", "memory": "32Gi", "gpu": "1"},
)
class Worker(RequestTracingMixin):
    """
    Worker component with automatic X-Request-Id support.
    
    Benefits of using RequestTracingMixin:
    - ensure_request_id(): Automatic request ID management
    - log_with_request_id(): Consistent logging with request ID
    - get_current_request_id(): Access request ID anywhere in call stack
    """

    def __init__(self):
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.current_requests = {}

    @dynamo_endpoint(name="generate")
    async def generate(self, request_data: str, request_id: Optional[str] = None) -> AsyncIterator[str]:
        """
        Generate text with automatic request ID tracking.
        
        The RequestTracingMixin automatically handles request ID management.
        """
        request_id = self.ensure_request_id(request_id)
        
        self.log_with_request_id("info", f"Starting generation on {self.model_name}")
        
        try:
            self.current_requests[request_id] = {"status": "generating", "model": self.model_name}
            
            async for token in self._generate_tokens(request_data):
                yield token
                
        except Exception as e:
            self.log_with_request_id("error", f"Generation failed: {e}")
            raise
        finally:
            self.current_requests.pop(request_id, None)
            self.log_with_request_id("info", "Generation completed")

    async def _generate_tokens(self, request_data: str) -> AsyncIterator[str]:
        """
        Internal token generation method that can access request ID from context.
        """
        current_request_id = get_current_request_id()
        if current_request_id:
            logger.debug(f"Generating tokens for request: {current_request_id}")
        
        tokens = ["Hello", " world", "!", " This", " is", " a", " test", " response", "."]
        
        for i, token in enumerate(tokens):
            self.log_with_request_id("debug", f"Generated token {i+1}/{len(tokens)}: '{token}'")
            
            await asyncio.sleep(0.1)
            yield token

    @dynamo_endpoint(name="get_status")
    async def get_status(self, request_id: Optional[str] = None) -> dict:
        """
        Get worker status with request tracking.
        """
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("debug", "Retrieving worker status")
        
        return {
            "model": self.model_name,
            "active_requests": len(self.current_requests),
            "request_details": self.current_requests,
            "status": "healthy"
        }

    @dynamo_endpoint(name="prefill")
    async def prefill(self, request_data: str, request_id: Optional[str] = None) -> AsyncIterator[str]:
        """
        Prefill operation with request tracking.
        """
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("info", "Starting prefill operation")
        
        await self._prefill_kv_cache(request_data)
        
        async for token in self._generate_tokens(request_data):
            yield token

    async def _prefill_kv_cache(self, request_data: str):
        """
        Simulate KV cache prefilling with request tracking.
        """
        current_request_id = get_current_request_id()
        
        self.log_with_request_id("debug", "Prefilling KV cache")
        
        await asyncio.sleep(0.2)
        
        self.log_with_request_id("debug", "KV cache prefill completed")
