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
SGLang disaggregated serving flow is

Processor -> PrefillWorker -> DecodeWorker

This is different from how we've implemented the vLLM disaggregated flow.

For now - the SGLangWorker will be responsible for aggreagted and prefill and we will
have a separate DecodeWorker.
"""

import asyncio
import logging
import random
import socket

import sglang as sgl
from sglang.srt.server_args import ServerArgs
from components.decode_worker import SGLangDecodeWorker
from sglang.srt.utils import get_ip
from dynamo.runtime import DistributedRuntime, dynamo_worker
from utils.protocol import DisaggPreprocessedRequest, PreprocessedRequest
from utils.args import parse_sglang_args

from dynamo.llm import ModelType, register_llm
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1},
    workers=1,
)
class SGLangWorker:
    decode_worker = depends(SGLangDecodeWorker)

    def __init__(self, engine_args: ServerArgs, decode_client=None):
        self.engine_args = engine_args
        self.engine = sgl.Engine(server_args=self.engine_args)
        self.decode_client = decode_client
        
        # Initialize bootstrap info if needed
        if self.engine_args.disaggregation_mode:
            self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info()
        
        logger.info("SGLangWorker initialized")

    def _get_bootstrap_info(self):
        """
        Bootstrap info is stored in the worker's tokenizer manager. We use it to
        add servers to the bootstrap_room
        """
        inner_tm = self.engine.tokenizer_manager
        bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port

        # multinode check
        if inner_tm.server_args.dist_init_addr:
            bootstrap_host = socket.gethostbyname(
                inner_tm.server_args.dist_init_addr.split(":")[0]
            )
        else:
            bootstrap_host = get_ip()

        return bootstrap_host, bootstrap_port

    def _build_sampling_params(self, request: PreprocessedRequest) -> dict:
        # TODO: maintain a full mapping from PreprocessedRequest to SGLang's SamplingParams
        sampling_params = {}
        if request.sampling_options.temperature:
            sampling_params["temperature"] = request.sampling_options.temperature
        if request.sampling_options.top_p:
            sampling_params["top_p"] = request.sampling_options.top_p
        if request.sampling_options.top_k:
            sampling_params["top_k"] = request.sampling_options.top_k
        sampling_params["max_new_tokens"] = request.stop_conditions.max_tokens
        if request.stop_conditions.ignore_eos:
            sampling_params["ignore_eos"] = request.stop_conditions.ignore_eos
        return sampling_params

    async def generate(self, request: PreprocessedRequest):
        print("request", request)
        request = PreprocessedRequest.model_validate(request)
        sampling_params = self._build_sampling_params(request)

        if self.engine_args.disaggregation_mode != "null":
            bootstrap_room = self._generate_bootstrap_room()

            # Create disaggregation request
            disagg_request = DisaggPreprocessedRequest(
                request=request,
                sampling_params=sampling_params,
                bootstrap_host=self.bootstrap_host,
                bootstrap_port=self.bootstrap_port,
                bootstrap_room=bootstrap_room,
            )

            # prefill response is not used
            prefill = await self.engine.async_generate(
                input_ids=request.token_ids,
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=self.bootstrap_host,
                bootstrap_port=self.bootstrap_port,
                bootstrap_room=bootstrap_room,
            )
            prefill_task = asyncio.create_task(self._prefill_generator(prefill))

            decode = await self.decode_client.generate(disagg_request.model_dump_json())

            async for out in self._process_stream(decode, unpack=True):
                yield out

            await prefill_task
        else:
            print("Generating without disaggregation")
            g = await self.engine.async_generate(
                input_ids=request.token_ids,
                sampling_params=sampling_params,
                stream=True,
            )

            async for out in self._process_stream(g, unpack=False):
                yield out

    async def _process_stream(self, stream_source, unpack: bool):
        num_output_tokens_so_far = 0
        async for res in stream_source:
            # Debug print to see response structure
            print(f"Stream response: {res}")
            
            # Rest of your code
            data = res.data() if unpack else res
            print(f"Processed data: {data}")
            
            # Access output_ids or equivalent field safely
            output_ids = data.get("output_ids", [])
            if not output_ids and "outputs" in data:
                # Try alternative field names
                output_ids = data.get("outputs", [])
            
            finish_reason = data.get("meta_info", {}).get("finish_reason")
            if finish_reason:
                # Don't forward the stop token
                out = {"token_ids": [], "finish_reason": finish_reason["type"]}
            else:
                next_total_toks = len(output_ids)
                out = {"token_ids": output_ids[num_output_tokens_so_far:]}
            yield out
            num_output_tokens_so_far = next_total_toks

    def _generate_bootstrap_room(self):
        return random.randint(0, 2**63 - 1)

    async def _prefill_generator(self, prefill):
        async for _ in prefill:
            pass

if __name__ == "__main__":
    import asyncio
    import uvloop
    from dynamo.runtime import DistributedRuntime, dynamo_worker

    engine_args: ServerArgs = parse_sglang_args()

    @dynamo_worker()
    async def worker(runtime: DistributedRuntime, engine_args):
        component = runtime.namespace("dynamo").component("comp")
        await component.create_service()

        endpoint = component.endpoint("generate")

        if engine_args.disaggregation_mode:
            decode_client = (
                await runtime.namespace("dynamo")
                .component(SGLangDecodeWorker.__name__)
                .endpoint("generate")
                .client()
            )
        else:
            decode_client = None

        worker = SGLangWorker(engine_args, decode_client)

        await register_llm(
            ModelType.Backend,
            endpoint,
            engine_args.model_path,
            engine_args.served_model_name,
        )

        print("serving endpoint")
        await endpoint.serve_endpoint(worker.generate)

    uvloop.install()
    asyncio.run(worker(engine_args))
