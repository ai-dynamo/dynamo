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

from dynamo.runtime import EtcdKvCache
from dynamo.sdk import dynamo_context

logger = logging.getLogger(__name__)


class PyDisaggregatedRouter:
    def __init__(
        self,
        runtime,
        namespace,
        max_local_prefill_length=1000,
        max_prefill_queue_size=2,
    ):
        self.runtime = runtime
        self.namespace = namespace
        self.max_local_prefill_length = max_local_prefill_length
        self.max_prefill_queue_size = max_prefill_queue_size

    async def async_init(self):
        runtime = dynamo_context["runtime"]
        self.etcd_kv_cache = await EtcdKvCache.create(
            runtime.etcd_client(),
            f"/{self.namespace}/disagg_router/",
            {
                "max_local_prefill_length": str(self.max_local_prefill_length),
                "max_prefill_queue_size": str(self.max_prefill_queue_size),
            },
        )

    async def prefill_remote(
        self, prompt_length: int, prefix_hit_rate: float, queue_size: int
    ):
        max_local_prefill_length = int(
            await self.etcd_kv_cache.get("max_local_prefill_length")
        )
        max_prefill_queue_size = int(
            await self.etcd_kv_cache.get("max_prefill_queue_size")
        )
        absolute_prefill_length = int(prompt_length * (1 - prefix_hit_rate))
        # TODO: consider size of each request in the queue when making the decision
        decision = (
            absolute_prefill_length > max_local_prefill_length
            and queue_size < max_prefill_queue_size
        )
        logger.info(
            f"Remote prefill: {decision} (prefill length: {absolute_prefill_length}/{max_local_prefill_length}, prefill queue size: {queue_size}/{max_prefill_queue_size})"
        )
        return decision
