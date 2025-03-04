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
from utils.nixl import find_remote_metadata
from utils.prefill_queue import PrefillQueue
from utils.vllm import parse_vllm_args
from vllm.distributed.device_communicators.nixl import NixlMetadata
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.logger import logger as vllm_logger
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest

from triton_distributed.runtime import DistributedRuntime, triton_worker


class RequestHandler:
    def __init__(self, engine_client):
        self.engine_client = engine_client
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

        async for _ in self.engine_client.generate(
            request_id=request.request_id,
            prompt=TokensPrompt(prompt_token_ids=request.prompt_token_ids),
            sampling_params=sampling_params,
            remote_prefill_params=remote_prefill_params,
        ):
            yield


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    component = runtime.namespace("triton-init").component("prefill")
    await component.create_service()

    async with build_async_engine_client_from_engine_args(engine_args) as engine_client:
        # This should be replaced with etcd
        metadata = engine_client.nixl_metadata

        print(f"Waiting for remote metadata for engine {metadata.engine_id}")
        remote_metadata: list[NixlMetadata] = []
        while not remote_metadata:
            await asyncio.sleep(1)
            remote_metadata = find_remote_metadata(metadata.engine_id)

        print(
            f"Found {len(remote_metadata)} remote metadata for engine {metadata.engine_id}"
        )
        for remote_metadata in remote_metadata:
            await engine_client.add_remote_nixl_metadata(remote_metadata)

        prefill_queue_nats_server = os.getenv("NATS_SERVER", "nats://localhost:4222")
        prefill_queue_stream_name = engine_args.model
        vllm_logger.info(
            f"Prefill queue: {prefill_queue_nats_server}:{prefill_queue_stream_name}"
        )

        # TODO: integrate prefill_queue to an triton_distributed endpoint
        async with PrefillQueue.get_instance(
            nats_server=prefill_queue_nats_server,
            stream_name=prefill_queue_stream_name,
        ) as prefill_queue:
            while True:
                prefill_request = await prefill_queue.dequeue_prefill_request()
                if prefill_request is not None:
                    vllm_logger.info(f"Dequeued prefill request: {prefill_request}")
                    async for _ in RequestHandler(engine_client).generate(
                        prefill_request
                    ):
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
