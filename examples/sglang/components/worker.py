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
from components.decode_worker import SGLangDecodeWorker
from sglang.srt.utils import get_ip
from utils.protocol import DisaggPreprocessedRequest, PreprocessedRequest
from utils.sglang import parse_sglang_args

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

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_sglang_args(class_name, "")
        self.engine = sgl.Engine(server_args=self.engine_args)

        logger.info("SGLangWorker initialized")

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        logger.info("Registering LLM for discovery")
        comp_ns, comp_name = SGLangWorker.dynamo_address()  # type: ignore
        endpoint = runtime.namespace(comp_ns).component(comp_name).endpoint("generate")
        await register_llm(
            ModelType.Backend,
            endpoint,
            self.engine_args.model_path,
            self.engine_args.served_model_name,
        )
        if self.engine_args.disaggregation_mode:
            self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info()
            comp_ns, comp_name = SGLangDecodeWorker.dynamo_address()  # type: ignore
            self.decode_client = (
                await runtime.namespace(comp_ns)
                .component(comp_name)
                .endpoint("generate")
                .client()
            )

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

    @dynamo_endpoint()
    async def generate(self, request: PreprocessedRequest):
        # TODO: maintain a mapping from SGLang's Ouput struct to LLMEngineOuput
        sampling_params = self._build_sampling_params(request)

        if self.engine_args.disaggregation_mode != "null":
            bootstrap_room = self._generate_bootstrap_room()

            # decode worker request
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
            data = res.data() if unpack else res
            finish_reason = data["meta_info"]["finish_reason"]
            if finish_reason:
                # Don't forward the stop token
                out = {"token_ids": [], "finish_reason": finish_reason["type"]}
            else:
                next_total_toks = len(data["output_ids"])
                out = {"token_ids": data["output_ids"][num_output_tokens_so_far:]}
            yield out
            num_output_tokens_so_far = next_total_toks

    def _generate_bootstrap_room(self):
        return random.randint(0, 2**63 - 1)

    async def _prefill_generator(self, prefill):
        async for _ in prefill:
            pass
