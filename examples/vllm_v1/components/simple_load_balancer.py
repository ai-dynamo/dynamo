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

import copy
import logging
import uuid
from typing import AsyncGenerator, Optional

from components.worker import VllmDecodeWorker, VllmPrefillWorker
from utils.args import parse_vllm_args
from utils.protocol import MyRequestOutput, PreprocessedRequest, vLLMGenerateRequest
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

from dynamo.llm import ModelType, register_llm
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class SimpleLoadBalancer:
    prefill_worker = depends(VllmPrefillWorker)
    decode_worker = depends(VllmDecodeWorker)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        model_config = self.engine_args.create_model_config()
        self.default_sampling_params = model_config.get_diff_sampling_param()
        self.enable_disagg = self.engine_args.enable_disagg

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        logger.info("Registering LLM for discovery")
        comp_ns, comp_name = SimpleLoadBalancer.dynamo_address()  # type: ignore
        endpoint_name = "generate"
        for served_model_name in self.engine_args.served_model_name:
            logger.info(
                f"Registering endpoint {endpoint_name} with model {self.engine_args.model} and served_model_name {served_model_name}"
            )
            endpoint = (
                runtime.namespace(comp_ns).component(comp_name).endpoint(endpoint_name)
            )
            await register_llm(
                ModelType.Backend,
                endpoint,
                self.engine_args.model,
                served_model_name,
            )

        comp_ns, comp_name = VllmDecodeWorker.dynamo_address()  # type: ignore
        self.decode_worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )

        comp_ns, comp_name = VllmPrefillWorker.dynamo_address()  # type: ignore
        self.prefill_worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )

        logger.info("SimpleLoadBalancer has been initialized")

    async def send_request_to_prefill(
        self, request: vLLMGenerateRequest
    ) -> MyRequestOutput:
        logger.debug("Sending request to prefill")

        prefill_request = copy.deepcopy(request)
        extra_args = prefill_request.sampling_params.extra_args or {}
        extra_args["kv_transfer_params"] = {
            "do_remote_decode": True,
        }
        prefill_request.sampling_params.extra_args = extra_args
        prefill_request.sampling_params.max_tokens = 1
        prefill_request.sampling_params.min_tokens = 1

        logger.debug("Prefill request: %s", prefill_request.model_dump_json())

        async for prefill_response in await self.prefill_worker_client.round_robin(
            prefill_request.model_dump_json()
        ):
            return MyRequestOutput.model_validate_json(prefill_response.data())

    async def send_request_to_decode(
        self,
        request: vLLMGenerateRequest,
        prefill_response: Optional[MyRequestOutput] = None,
    ) -> AsyncGenerator[MyRequestOutput, None]:
        logger.debug("Sending request to decode")

        decode_request = copy.deepcopy(request)

        if prefill_response:
            extra_args = decode_request.sampling_params.extra_args or {}
            extra_args["kv_transfer_params"] = prefill_response.kv_transfer_params
            decode_request.sampling_params.extra_args = extra_args

        logger.debug("Decode request: %s", decode_request.model_dump_json())

        async for decode_response in await self.decode_worker_client.round_robin(
            decode_request.model_dump_json()
        ):
            yield MyRequestOutput.model_validate_json(decode_response.data())

    @dynamo_endpoint()
    async def generate(self, request: PreprocessedRequest):
        logger.debug(
            "Processor received completion request: %s", request.model_dump_json()
        )

        vllm_request = self._create_vllm_request(request)

        logger.debug("VLLM request: %s", vllm_request.model_dump_json())

        if self.enable_disagg:
            prefill_response = await self.send_request_to_prefill(vllm_request)

            logger.debug("Prefill response: %s", prefill_response.model_dump_json())
        else:
            prefill_response = None

        gen = self.send_request_to_decode(vllm_request, prefill_response)
        async for res in self._stream_response(gen):
            yield res

    def _create_vllm_request(self, request: PreprocessedRequest) -> vLLMGenerateRequest:
        request_id = str(uuid.uuid4().hex)

        prompt = TokensPrompt(prompt_token_ids=request.token_ids)

        sampling_params = SamplingParams(**self.default_sampling_params)
        for key, value in request.sampling_options.model_dump().items():
            if not value:
                continue
            if hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        max_tokens = request.stop_conditions.max_tokens
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        return vLLMGenerateRequest(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )

    async def _stream_response(self, gen: AsyncGenerator[MyRequestOutput, None]):
        num_output_tokens_so_far = 0
        async for res in gen:
            logger.debug("Decode response: %s", res.model_dump_json())
            # res is our MyRequestOutput

            # This is the expected way for a request to end.
            # The new token ID will be eos, don't forward it.
            if res.finished:
                yield {"finish_reason": "stop", "token_ids": []}
                break

            if not res.outputs:
                yield {"finish_reason": "error", "token_ids": []}
                break

            output = res.outputs[0]
            next_total_toks = len(output.token_ids)
            out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}
            if output.finish_reason:
                out["finish_reason"] = output.finish_reason
            if output.stop_reason:
                out["stop_reason"] = output.stop_reason
            yield out
            num_output_tokens_so_far = next_total_toks
