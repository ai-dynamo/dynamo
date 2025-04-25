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


import logging
from difflib import SequenceMatcher
from typing import AsyncIterator

from components.utils import check_required_workers
from components.worker import DummyWorker

from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

WorkerId = str

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo-demo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Router:
    """
    Request handler for the generate endpoint
    """

    kv_cache: dict[str, str] = {}
    threshold = 0.6
    worker = depends(DummyWorker)

    def __init__(self):
        self.min_workers = 2

    @async_on_start
    async def async_init(self):
        print("in kv router async_init")
        self.runtime = dynamo_context["runtime"]
        self.workers_client = (
            await self.runtime.namespace("dynamo-demo")
            .component("DummyWorker")
            .endpoint("worker_generate")
            .client()
        )

        await check_required_workers(self.workers_client, self.min_workers, "kv router")

        print("KV Router initialized")

    def _cost_function(self, request_prompt):
        worker_ids = self.workers_client.endpoint_ids()
        num_workers = len(worker_ids)
        max_hit_rate = -1.0
        for curr_id in self.kv_cache.keys():
            # Estimate hit rate by string matching
            hit_rate = SequenceMatcher(
                None, self.kv_cache[curr_id], request_prompt
            ).ratio()
            if hit_rate > max_hit_rate:
                max_hit_rate = hit_rate
                max_id = curr_id
        print(f"{max_hit_rate=},{len(self.kv_cache.keys())=}")
        if max_hit_rate > self.threshold:
            # Found the hit rate larger than the threshold
            return max_id, max_hit_rate
        elif len(self.kv_cache.keys()) == num_workers:
            # Cache is already full, return the max rate
            return max_id, max_hit_rate
        else:
            # Add current request into the cache
            for curr_id in worker_ids:
                if curr_id not in self.kv_cache.keys():
                    self.kv_cache[curr_id] = request_prompt
                    break
            return curr_id, -1

    # A dummy hit rate checking endpoint
    # The actual worker selection is based on custom cost function
    # See details at examples/llm/components/kv_router.py
    @dynamo_endpoint()
    async def check_hit_rate(self, request_prompt: str) -> AsyncIterator[WorkerId]:
        max_id, max_hit_rate = self._cost_function(request_prompt)
        yield f"{max_id}_{max_hit_rate}"
