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
import os

import uvloop
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.vllm import parse_vllm_args
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.logger import logger as vllm_logger
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest

from dynamo.runtime import DistributedRuntime, dynamo_worker


class RequestHandler:
    def __init__(self, engine_client, metadata_store):
        self.engine_client = engine_client
        self._metadata_store = metadata_store
        self._loaded_metadata = set()
        print("RequestHandler initialized")

    async def generate(self, request: RemotePrefillRequest):
        sampling_params = request.sampling_params
        sampling_params.max_tokens = 1
        sampling_params.min_tokens = 1

        remote_prefill_params = RemotePrefillParams(
            is_remote_decode=True,
            decode_block_ids=request.block_ids,
            decode_engine_id=request.engine_id,
        )

        # TODO check if metadata has changed
        # and reload - currently only loading once

        if request.engine_id not in self._loaded_metadata:
            remote_metadata = await self._metadata_store.get(request.engine_id)
            await self.engine_client.add_remote_nixl_metadata(remote_metadata)
            print(
                f"Loaded nixl metadata from engine {request.engine_id} into engine {self.engine_client.nixl_metadata.engine_id}"
            )
            self._loaded_metadata.add(request.engine_id)

        async for _ in self.engine_client.generate(
            request_id=request.request_id,
            prompt=TokensPrompt(prompt_token_ids=request.prompt_token_ids),
            sampling_params=sampling_params,
            remote_prefill_params=remote_prefill_params,
        ):
            yield


@dynamo_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    # TODO: we don't need it now, but will need it after the queue is integrated to the runtime
    component = runtime.namespace("dynamo-init").component("prefill")
    await component.create_service()

    async with build_async_engine_client_from_engine_args(engine_args) as engine_client:
        metadata = engine_client.nixl_metadata
        metadata_store = NixlMetadataStore("dynamo-init", runtime)
        await metadata_store.put(metadata.engine_id, metadata)

        # TODO: move this to prefill_queue.py
        prefill_queue_nats_server = os.getenv("NATS_SERVER", "nats://localhost:4222")
        prefill_queue_stream_name = (
            engine_args.served_model_name
            if engine_args.served_model_name is not None
            else "vllm"
        )
        vllm_logger.info(
            f"Prefill queue: {prefill_queue_nats_server}:{prefill_queue_stream_name}"
        )

        request_handler = RequestHandler(engine_client, metadata_store)

        # TODO: integrate prefill_queue to a dynamo endpoint
        async with PrefillQueue.get_instance(
            nats_server=prefill_queue_nats_server,
            stream_name=prefill_queue_stream_name,
        ) as prefill_queue:
            while True:
                # TODO: this might add a small overhead to pull prefill from nats
                # need to test and check how much overhead it is
                prefill_request = await prefill_queue.dequeue_prefill_request()
                if prefill_request is not None:
                    vllm_logger.info(f"Dequeued prefill request: {prefill_request}")
                    async for _ in request_handler.generate(prefill_request):
                        pass


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()

    if engine_args.enable_chunked_prefill is not False:
        print("Chunked prefill is not supported yet, setting to False")
        engine_args.enable_chunked_prefill = False

    if engine_args.pipeline_parallel_size != 1:
        print("Pipeline parallel size is not supported yet, setting to 1")
        engine_args.pipeline_parallel_size = 1

    if engine_args.disable_async_output_proc is not True:
        print("Async output processing is not supported yet, setting to True")
        engine_args.disable_async_output_proc = True

    if engine_args.enforce_eager is not True:
        print("Prefill must be done eagerly, setting to True")
        engine_args.enforce_eager = True

    asyncio.run(worker(engine_args))
