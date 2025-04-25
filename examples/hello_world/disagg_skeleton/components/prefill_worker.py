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


import asyncio
import logging
import os
import socket
import sys

from components.utils import NixlMetadataStore, PrefillQueue, RemotePrefillRequest
from vllm.distributed.device_communicators.nixl import NixlMetadata

from dynamo.sdk import async_on_start, dynamo_context, dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo-demo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class PrefillWorker:
    def __init__(self):
        self._loaded_metadata = set()
        self.initialized = False
        self.hostname = socket.gethostname()
        self.engine_id = self.hostname

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        # create dummy meta data
        metadata = NixlMetadata(
            engine_id=self.engine_id,
            agent_metadata=[],
            kv_caches_base_addr=[[]],
            num_blocks=0,
        )
        self._metadata_store = NixlMetadataStore("dynamo-nixl", runtime)
        await self._metadata_store.put(metadata.engine_id, metadata)
        task = asyncio.create_task(self.prefill_queue_handler())

        def prefill_queue_handler_cb(fut):
            try:
                fut.result()
                print("prefill queue handler exited successfully")
            except Exception as e:
                print(f"[ERROR] prefill queue handler failed: {e!r}")
                sys.exit(1)

        task.add_done_callback(prefill_queue_handler_cb)
        print("PrefillWorker initialized")

    async def prefill_queue_handler(self):
        print("Prefill queue handler entered")
        prefill_queue_nats_server = os.getenv("NATS_SERVER", "nats://localhost:4222")
        prefill_queue_stream_name = "DummyLLM"
        print(f"Prefill queue: {prefill_queue_nats_server}:{prefill_queue_stream_name}")
        self.initialized = True
        # TODO: integrate prefill_queue to a dynamo endpoint
        async with PrefillQueue.get_instance(
            nats_server=prefill_queue_nats_server,
            stream_name=prefill_queue_stream_name,
        ) as prefill_queue:
            print("prefill queue handler started")
            while True:
                # TODO: this might add a small overhead to pull prefill from nats
                # need to test and check how much overhead it is
                prefill_request = await prefill_queue.dequeue_prefill_request()
                if prefill_request is not None:
                    print(f"Dequeued prefill request: {prefill_request.request_id}")
                    async for _ in self.prefill_generate(prefill_request):
                        pass

    async def prefill_generate(self, request: RemotePrefillRequest):
        # TODO check if metadata has changed
        # and reload - currently only loading once
        print(f"prefill invoked {request.engine_id}{self._loaded_metadata=}")
        if request.engine_id not in self._loaded_metadata:
            remote_metadata = await self._metadata_store.get(request.engine_id)
            # await self.engine_client.add_remote_nixl_metadata(remote_metadata)
            print(f"Received nixl metadata from host {remote_metadata.engine_id}")
            self._loaded_metadata.add(remote_metadata.engine_id)

        print("Prefill invoked and will read KV cache from worker and write it back")
        yield "prefill invoked"

    @dynamo_endpoint()
    async def mock(self, req: RemotePrefillRequest):
        yield f"mock_response: {req}"
