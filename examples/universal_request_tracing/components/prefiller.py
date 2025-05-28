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
Prefiller Component with Universal X-Request-Id Support

This example shows how prefiller components can use Dynamo SDK's built-in
request tracing for automatic request ID propagation and KV cache management.
"""

import asyncio
import logging
from typing import AsyncIterator, Dict, Optional

from dynamo.sdk import RequestTracingMixin, endpoint, get_current_request_id, service

logger = logging.getLogger(__name__)


@service(
    dynamo={"enabled": True, "namespace": "dynamo"},
    resources={"cpu": "4", "memory": "16Gi", "gpu": "1"},
)
class Prefiller(RequestTracingMixin):
    """
    Prefiller component with automatic X-Request-Id support.

    Benefits of using RequestTracingMixin:
    - ensure_request_id(): Automatic request ID management
    - log_with_request_id(): Consistent logging with request ID
    - get_current_request_id(): Access request ID anywhere in call stack
    """

    def __init__(self):
        self.kv_cache = {}
        self.prefill_queue = []
        self.cache_stats = {"hits": 0, "misses": 0, "prefills": 0}

    @endpoint()
    async def prefill(
        self, request_data: str, request_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Prefill KV cache with automatic request ID tracking.

        The RequestTracingMixin automatically handles request ID management.
        """
        request_id = self.ensure_request_id(request_id)

        self.log_with_request_id("info", "Starting KV cache prefill operation")

        try:
            cache_key = self._generate_cache_key(request_data)
            if cache_key in self.kv_cache:
                self.log_with_request_id("debug", "Found existing KV cache entry")
                self.cache_stats["hits"] += 1
                return {"status": "cache_hit", "cache_key": cache_key}

            prefill_result = await self._perform_prefill(request_data, cache_key)
            self.cache_stats["prefills"] += 1

            self.log_with_request_id(
                "info", f"Prefill completed for cache key: {cache_key}"
            )
            return prefill_result

        except Exception as e:
            self.log_with_request_id("error", f"Prefill failed: {e}")
            raise

    async def _perform_prefill(
        self, request_data: str, cache_key: str
    ) -> Dict[str, any]:
        """
        Internal prefill operation that can access request ID from context.
        """
        current_request_id = get_current_request_id()
        if current_request_id:
            logger.debug(f"Performing prefill for request: {current_request_id}")

        self.log_with_request_id(
            "debug", f"Computing KV states for cache key: {cache_key}"
        )

        await asyncio.sleep(0.5)

        kv_states = {
            "key_states": f"key_states_for_{cache_key}",
            "value_states": f"value_states_for_{cache_key}",
            "attention_mask": f"mask_for_{cache_key}",
            "position_ids": list(range(len(request_data.split()))),
        }

        self.kv_cache[cache_key] = {
            "kv_states": kv_states,
            "request_id": current_request_id,
            "timestamp": asyncio.get_event_loop().time(),
        }

        self.log_with_request_id("debug", "KV states computed and cached")

        return {"status": "prefilled", "cache_key": cache_key, "kv_states": kv_states}

    @endpoint()
    async def get_cache(
        self, cache_key: str, request_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Retrieve cached KV states with request tracking.
        """
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("debug", f"Retrieving cache for key: {cache_key}")

        if cache_key in self.kv_cache:
            self.cache_stats["hits"] += 1
            self.log_with_request_id("debug", "Cache hit")
            return self.kv_cache[cache_key]
        else:
            self.cache_stats["misses"] += 1
            self.log_with_request_id("debug", "Cache miss")
            return {"status": "cache_miss"}

    @endpoint()
    async def batch_prefill(
        self, requests: list, request_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, any]]:
        """
        Batch prefill operation with request tracking.
        """
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id(
            "info", f"Starting batch prefill for {len(requests)} requests"
        )

        for i, request_data in enumerate(requests):
            self.log_with_request_id(
                "debug", f"Processing batch item {i+1}/{len(requests)}"
            )

            cache_key = self._generate_cache_key(request_data)
            result = await self._perform_prefill(request_data, cache_key)

            yield {"batch_index": i, "request_data": request_data, **result}

    def _generate_cache_key(self, request_data: str) -> str:
        """
        Generate cache key for request data.
        """
        current_request_id = get_current_request_id()

        import hashlib

        content_hash = hashlib.md5(request_data.encode()).hexdigest()[:8]

        if current_request_id:
            logger.debug(
                f"Generated cache key {content_hash} for request {current_request_id}"
            )

        return f"cache_{content_hash}"

    @endpoint()
    async def get_stats(self, request_id: Optional[str] = None) -> Dict[str, any]:
        """
        Get prefiller statistics with request tracking.
        """
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("debug", "Retrieving prefiller statistics")

        return {
            "cache_stats": self.cache_stats,
            "cache_size": len(self.kv_cache),
            "queue_size": len(self.prefill_queue),
            "hit_rate": self.cache_stats["hits"]
            / max(1, self.cache_stats["hits"] + self.cache_stats["misses"]),
        }
