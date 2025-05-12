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

import logging
import signal

import sglang as sgl
from utils.protocol import PreprocessedRequest
from utils.sglang import parse_sglang_args

from dynamo.llm import ModelType, register_llm
from dynamo.sdk import async_on_start, dynamo_context, dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1},
    workers=1,
)
class SGLangWorker:
    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_sglang_args(class_name, "")
        self.engine = sgl.Engine(server_args=self.engine_args)

        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self.shutdown_sglang_engine)

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

    def shutdown_sglang_engine(self, signum, frame):
        self.engine.shutdown()
        logger.info("SGLang engine shutdown")

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
        g = await self.engine.async_generate(
            input_ids=request.token_ids,
            sampling_params=sampling_params,
            stream=True,
        )
        num_output_tokens_so_far = 0
        async for res in g:
            finish_reason = res["meta_info"]["finish_reason"]
            if finish_reason:
                # Don't forward the stop token
                out = {"token_ids": [], "finish_reason": finish_reason["type"]}
            else:
                next_total_toks = len(res["output_ids"])
                out = {"token_ids": res["output_ids"][num_output_tokens_so_far:]}
            yield out
            num_output_tokens_so_far = next_total_toks
