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
import socket

from components.utils import (
    GeneralRequest,
    GeneralResponse,
    NixlMetadataStore,
    PrefillQueue,
    RemotePrefillRequest,
)
from vllm.distributed.device_communicators.nixl import NixlMetadata

from dynamo.sdk import async_on_start, dynamo_context, dynamo_endpoint, service

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
        logger.info("VllmWorker has been initialized")

    def get_remote_prefill_request_callback(self):
        # TODO: integrate prefill_queue to dynamo endpoint
        async def callback(request: RemotePrefillRequest):
            print(
                f"enqueue request {self._prefill_queue_nats_server}, \
                  {self._prefill_queue_stream_name},{request.engine_id=}"
            )
            async with PrefillQueue.get_instance(
                nats_server=self._prefill_queue_nats_server,
                stream_name=self._prefill_queue_stream_name,
            ) as prefill_queue:
                await prefill_queue.enqueue_prefill_request(request)

        return callback

    @dynamo_endpoint()
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

        print(f"{self.hostname}: Worker invoked")
        yield GeneralResponse(
            request_id=request.request_id,
            worker_output=request.prompt + "_GeneratedBy_" + self.hostname,
        ).model_dump_json()
