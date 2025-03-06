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

import uuid
from typing import AsyncIterator

import bentoml
from sdk_kv_router.router import Router
from sdk_kv_router.worker import VllmEngine

with bentoml.importing():
    from transformers import AutoTokenizer
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    from vllm.outputs import RequestOutput
    from vllm.transformers_utils.tokenizer import AnyTokenizer
    from common.chat_processor import ChatProcessor, ProcessMixIn
    from common.protocol import MyRequestOutput, Tokens, vLLMGenerateRequest

from compoundai import depends, nova_endpoint, service, tdist_context


@service(
    nova={
        "enabled": True,
        "namespace": "triton-init",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Processor(ProcessMixIn):
    """
    vLLM pre and post processing
    """

    workers = depends(VllmEngine)
    router = depends(Router)

    def __init__(self):
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.engine_args = AsyncEngineArgs(
            model=model,
            tokenizer=model,
            enable_prefix_caching=True,
            block_size=64,
            max_model_len=16384,
        )
        self.model_config = self.engine_args.create_model_config()
        self.tokenizer = self._create_tokenizer()
        self.chat_processor = ChatProcessor(self.tokenizer, self.model_config)

    def _create_tokenizer(self) -> AnyTokenizer:
        """Create a TokenizerGroup using engine arguments similar to VLLM's approach"""
        model_path = self.engine_args.model

        # Create the base tokenizer with VLLM's typical settings
        base_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            truncation_side="left",
            use_fast=True,  # VLLM might use the fast tokenizer for efficiency
        )
        return base_tokenizer

    async def generate_responses(
        self, engine_generator
    ) -> AsyncIterator[RequestOutput]:
        async for resp in engine_generator:
            # Deserialize the response from the engine
            # Creates correct vLLM objects for each field
            output = MyRequestOutput.model_validate_json(resp.data())
            # OpenAIServingChat.chat_completion_stream_generator() method expects a RequestOutput object
            yield RequestOutput(
                request_id=output.request_id,
                prompt=output.prompt,
                prompt_token_ids=output.prompt_token_ids,
                prompt_logprobs=output.prompt_logprobs,
                outputs=output.outputs,
                finished=output.finished,
                metrics=output.metrics,
            )

    @nova_endpoint()
    async def generate(self, raw_request: ChatCompletionRequest):
        request_id = str(uuid.uuid4())
        (
            request,
            conversation,
            prompt,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(raw_request)
        worker_id = None
        async for worker in self.router.generate(
            Tokens(tokens=engine_prompt["prompt_token_ids"]).model_dump_json()
        ):
            worker_id = worker
            break
        runtime = tdist_context["runtime"]
        comp_ns, comp_name = VllmEngine.nova_address()
        worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )
        if worker_id == "":
            engine_generator = await worker_client.generate(
                vLLMGenerateRequest(
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ).model_dump_json()
            )
        else:
            engine_generator = await worker_client.direct(
                vLLMGenerateRequest(
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ).model_dump_json(),
                uuid.UUID(worker_id).int,
            )
        output = self.generate_responses(engine_generator)

        async for response in await self._stream_response(
            request, output, request_id, conversation
        ):
            yield response
