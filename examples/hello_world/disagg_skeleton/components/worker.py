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
import os
import random
import socket

from components.utils import (
    GeneralRequest,
    GeneralResponse,
    NixlMetadataStore,
    PrefillQueue,
    RemotePrefillRequest,
)
from vllm.distributed.device_communicators.nixl import NixlMetadata

from dynamo.llm import KvMetricsPublisher
from dynamo.sdk import async_on_start, dynamo_context, endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo-demo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class DummyWorker:
    def __init__(self):
        self.hostname = socket.gethostname()

        self.do_remote_prefill = True
        self.model_name = "DummyLLM"
        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )
        self._prefill_queue_stream_name = self.model_name
        logger.info(
            f"Prefill queue: {self._prefill_queue_nats_server}:{self._prefill_queue_stream_name}"
        )
        self.component = dynamo_context["component"]
        self.metrics_publisher = KvMetricsPublisher()
        # Register an endpoint for consumers of the KV Metrics
        # (KvMetricsAggregator in kv_router) to listen/gather on.
        self.metrics_publisher.create_endpoint(self.component)
        # Initialize some metrics for the worker/class to track
        self.request_active_slots = 0
        self.request_total_slots = 1024
        self.kv_active_blocks = 0
        self.kv_total_blocks = 1024
        self.num_requests_waiting = 0
        self.gpu_cache_usage_perc = 0.0
        self.gpu_prefix_cache_hit_rate = 0.0

        # Publish some initial metrics to register
        # this worker as a candidate for KV Routing.
        self.metrics_publisher.publish(
            self.request_active_slots,
            self.request_total_slots,
            self.kv_active_blocks,
            self.kv_total_blocks,
            self.num_requests_waiting,
            self.gpu_cache_usage_perc,
            self.gpu_prefix_cache_hit_rate,
        )

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]

        if self.do_remote_prefill:
            # Create dummy Nixl meta data
            metadata = NixlMetadata(
                engine_id=self.hostname,
                agent_metadata=[],
                kv_caches_base_addr=[[]],
                num_blocks=0,
            )
            metadata_store = NixlMetadataStore("dynamo-nixl", runtime)
            await metadata_store.put(metadata.engine_id, metadata)

        self.disaggregated_router = "DummyDisaggregateRouter"
        logger.info("DummyWorker has been initialized")

    def get_remote_prefill_request_callback(self):
        # TODO: integrate prefill_queue to dynamo endpoint
        async def callback(request: RemotePrefillRequest):
            logger.info(
                f"enqueue request {self._prefill_queue_nats_server}, \
                  {self._prefill_queue_stream_name},{request.engine_id=}"
            )
            async with PrefillQueue.get_instance(
                nats_server=self._prefill_queue_nats_server,
                stream_name=self._prefill_queue_stream_name,
            ) as prefill_queue:
                await prefill_queue.enqueue_prefill_request(request)

        return callback

    def publish_kv_metrics(self):
        # Populate the frequently changing metrics with random data for
        # demonstration. These values should be tracked by the implementation,
        # or queried from the underlying inference framework.
        self.kv_active_blocks = random.randint(0, 1024)
        self.num_requests_waiting = random.randint(0, 100)
        self.gpu_cache_usage_perc = random.uniform(0, 1.0)
        self.gpu_prefix_cache_hit_rate = random.uniform(0, 1.0)

        # Publish the metrics with the current state
        self.metrics_publisher.publish(
            self.request_active_slots,
            self.request_total_slots,
            self.kv_active_blocks,
            self.kv_total_blocks,
            self.num_requests_waiting,
            self.gpu_cache_usage_perc,
            self.gpu_prefix_cache_hit_rate,
        )

    @endpoint()
    async def worker_generate(self, request: GeneralRequest):
        # TODO: consider prefix hit when deciding prefill locally or remotely

        if self.disaggregated_router is not None:
            # decision = (
            #   absolute_prefill_length > self.max_local_prefill_length
            #   and queue_size < self.max_prefill_queue_size )
            # Disagg router decision is based on prefill length and queue size
            # Always set to True in this demo (see details at disagg_router.py)
            disagg_router_decision = True
        else:
            # always prefill remotely if no disaggregated router is provided
            disagg_router_decision = True

        if self.do_remote_prefill and disagg_router_decision:
            ## Mimic the process of enqueue request for prefill
            prefill_request = RemotePrefillRequest(
                engine_id=self.hostname, request_id=request.request_id
            )
            callback = self.get_remote_prefill_request_callback()
            await callback(prefill_request)

        logger.info(f"{self.hostname}: Worker invoked")
        self.publish_kv_metrics()
        yield GeneralResponse(
            request_id=request.request_id,
            worker_output=request.prompt + "_GeneratedBy_" + self.hostname,
        ).model_dump_json()
