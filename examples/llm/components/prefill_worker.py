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
import sys

from pydantic import BaseModel
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.vllm import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest

from dynamo.sdk import async_on_start, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.service import LeaseConfig

logger = logging.getLogger(__name__)


class RequestType(BaseModel):
    text: str


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
        "custom_lease": LeaseConfig(ttl=1),  # 1 second
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class PrefillWorker:
    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self._loaded_metadata = set()
        self.initialized = False
        if self.engine_args.enable_chunked_prefill is not False:
            logger.info("Chunked prefill is not supported yet, setting to False")
            self.engine_args.enable_chunked_prefill = False

        if self.engine_args.pipeline_parallel_size != 1:
            logger.info("Pipeline parallel size is not supported yet, setting to 1")
            self.engine_args.pipeline_parallel_size = 1

        if self.engine_args.disable_async_output_proc is not True:
            logger.info("Async output processing is not supported yet, setting to True")
            self.engine_args.disable_async_output_proc = True

        if self.engine_args.enforce_eager is not True:
            logger.info("Prefill must be done eagerly, setting to True")
            self.engine_args.enforce_eager = True

        if self.engine_args.enable_prefix_caching is not False:
            logger.info(
                "Prefix caching is not supported yet in prefill worker, setting to False"
            )
            self.engine_args.enable_prefix_caching = False

    @async_on_start
    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")
        runtime = dynamo_context["runtime"]
        metadata = self.engine_client.nixl_metadata
        self._metadata_store = NixlMetadataStore("dynamo", runtime)
        await self._metadata_store.put(metadata.engine_id, metadata)
        self.task = asyncio.create_task(self.prefill_queue_handler())

        def prefill_queue_handler_cb(fut):
            try:
                fut.result()
                logger.info("prefill queue handler exited successfully")
            except Exception as e:
                logger.error(f"[ERROR] prefill queue handler failed: {e!r}")
                sys.exit(1)

        self.task.add_done_callback(prefill_queue_handler_cb)
        self.lease = dynamo_context["lease"]
        logger.info("PrefillWorker initialized")

    def shutdown_vllm_engine(self):
        """Shutdown the background loop"""
        logger.info("Shutting down vllm engine")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
            logger.info("PrefillWorker shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    async def prefill_queue_handler(self):
        logger.info("Prefill queue handler entered")
        prefill_queue_nats_server = os.getenv("NATS_SERVER", "nats://localhost:4222")
        prefill_queue_stream_name = (
            self.engine_args.served_model_name
            if self.engine_args.served_model_name is not None
            else "vllm"
        )
        logger.info(
            f"Prefill queue: {prefill_queue_nats_server}:{prefill_queue_stream_name}"
        )
        self.initialized = True
        # TODO: integrate prefill_queue to a dynamo endpoint
        async with PrefillQueue.get_instance(
            nats_server=prefill_queue_nats_server,
            stream_name=prefill_queue_stream_name,
        ) as prefill_queue:
            logger.info("prefill queue handler started")
            while True:
                # TODO: this might add a small overhead to pull prefill from nats
                # need to test and check how much overhead it is
                prefill_request = await prefill_queue.dequeue_prefill_request()
                if prefill_request is not None:
                    logger.info(
                        f"Dequeued prefill request: {prefill_request.request_id}"
                    )
                    async for _ in self.generate(prefill_request):
                        pass
                is_valid = await self.lease.is_valid()
                if not is_valid:
                    logger.info(
                        "Shutdown requested, checking if engine has any pending prefill sending requests"
                    )
                    while True:
                        if not await self.engine_client.has_unfinished_requests():
                            break
                        logger.info(
                            "Engine has pending prefill sending requests, rechecking in 1 second..."
                        )
                        await asyncio.sleep(1)
                    self.shutdown_vllm_engine()
                    break

    async def generate(self, request: RemotePrefillRequest):
        sampling_params = request.sampling_params
        sampling_params.max_tokens = 1
        sampling_params.min_tokens = 1

        remote_prefill_params = RemotePrefillParams(
            is_remote_decode=True,
            decode_block_ids=request.block_ids,
            decode_engine_id=request.engine_id,
            decode_computed_block_ids=request.computed_block_ids,
        )

        # TODO check if metadata has changed
        # and reload - currently only loading once
        if request.engine_id not in self._loaded_metadata:
            remote_metadata = await self._metadata_store.get(request.engine_id)
            await self.engine_client.add_remote_nixl_metadata(remote_metadata)
            logger.info(
                f"Loaded nixl metadata from engine {request.engine_id} into "
                f"engine {self.engine_client.nixl_metadata.engine_id}"
            )
            self._loaded_metadata.add(request.engine_id)

        async for _ in self.engine_client.generate(
            request_id=request.request_id,
            prompt=TokensPrompt(prompt_token_ids=request.prompt_token_ids),
            sampling_params=sampling_params,
            remote_prefill_params=remote_prefill_params,
        ):
            yield

    @dynamo_endpoint()
    async def mock(self, req: RequestType):
        yield f"mock_response: {req}"
