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


import os

import msgspec
from utils.nixl import NixlMetadataStore
from utils.vllm import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest

from dynamo.sdk import (
    async_on_start,
    dynamo_context,
    dynamo_endpoint,
    server_context,
    service,
)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo-init",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class PrefillWorkerRouterLess:
    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        gpu_idx = (
            self.engine_args.cuda_visible_device_offset
            + server_context.worker_index
            - 1
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_idx}"
        self._loaded_metadata = set()
        self.initialized = False
        if self.engine_args.enable_chunked_prefill is not False:
            print("Chunked prefill is not supported yet, setting to False")
            self.engine_args.enable_chunked_prefill = False

        if self.engine_args.pipeline_parallel_size != 1:
            print("Pipeline parallel size is not supported yet, setting to 1")
            self.engine_args.pipeline_parallel_size = 1

        if self.engine_args.disable_async_output_proc is not True:
            print("Async output processing is not supported yet, setting to True")
            self.engine_args.disable_async_output_proc = True

        if self.engine_args.enforce_eager is not True:
            print("Prefill must be done eagerly, setting to True")
            self.engine_args.enforce_eager = True
        print("PrefillWorkerRouterLess initialized")

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
        self._metadata_store = NixlMetadataStore("dynamo-init", runtime)
        await self._metadata_store.put(metadata.engine_id, metadata)

    @dynamo_endpoint()
    async def generate(self, raw_request: str):
        request: RemotePrefillRequest = msgspec.json.decode(
            raw_request.encode("utf-8"), type=RemotePrefillRequest
        )

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
