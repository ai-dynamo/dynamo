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
import uuid
from enum import Enum
from typing import AsyncIterator, Tuple, Union

import uvloop
from transformers import AutoTokenizer
from utils.chat_processor import ChatProcessor, CompletionsProcessor, ProcessMixIn
from utils.protocol import MyRequestOutput, Tokens, vLLMGenerateRequest
from utils.vllm import parse_vllm_args
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionStreamResponse,
)
from vllm.logger import logger as vllm_logger
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer

from dynamo.runtime import Client, DistributedRuntime, dynamo_endpoint, dynamo_worker


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


class Processor(ProcessMixIn):
    """
    vLLM pre and post processing
    """

    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        router_client: Client,
        workers_client: Client,
    ):
        self.engine_args = engine_args
        self.model_config = self.engine_args.create_model_config()
        self.tokenizer = self._create_tokenizer(engine_args)
        self.chat_processor = ChatProcessor(self.tokenizer, self.model_config)
        self.completions_processor = CompletionsProcessor(
            self.tokenizer, self.model_config
        )
        self.router_client = router_client
        self.workers_client = workers_client
        self.router_mode = engine_args.router

    def _create_tokenizer(self, engine_args: AsyncEngineArgs) -> AnyTokenizer:
        """Create a TokenizerGroup using engine arguments similar to VLLM's approach"""
        model_path = engine_args.model

        # Create the base tokenizer with VLLM's typical settings
        base_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            truncation_side="left",
            use_fast=True,  # VLLM might use the fast tokenizer for efficiency
        )
        return base_tokenizer

    async def _generate(
        self,
        raw_request: Union[CompletionRequest, ChatCompletionRequest],
        request_type: RequestType,
    ):
        request_id = str(uuid.uuid4())
        vllm_logger.debug(f"Got raw request: {raw_request}")
        (
            request,
            conversation,
            prompt,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(raw_request)

        if self.router_mode == "kv":
            worker_id_generator: AsyncIterator = await self.router_client.generate(
                Tokens(tokens=engine_prompt["prompt_token_ids"]).model_dump_json()
            )

            route_response = (
                await worker_id_generator.__anext__()
            )  # only one worker id is returned
            worker_id, prefix_hit_rate = route_response.data().split("_")
            prefix_hit_rate = float(prefix_hit_rate)
            vllm_logger.info(
                f"Worker ID: {worker_id} with estimated prefix hit rate: {prefix_hit_rate}"
            )

            if worker_id == "":
                engine_generator = await self.workers_client.random(
                    vLLMGenerateRequest(
                        engine_prompt=engine_prompt,
                        sampling_params=sampling_params,
                        request_id=request_id,
                        prefix_hit_rate=prefix_hit_rate,
                    ).model_dump_json()
                )
            else:
                engine_generator = await self.workers_client.direct(
                    vLLMGenerateRequest(
                        engine_prompt=engine_prompt,
                        sampling_params=sampling_params,
                        request_id=request_id,
                        prefix_hit_rate=prefix_hit_rate,
                    ).model_dump_json(),
                    int(worker_id),
                )
        elif self.router_mode == "random":
            engine_generator = await self.workers_client.random(
                vLLMGenerateRequest(
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ).model_dump_json()
            )
        elif self.router_mode == "round-robin":
            engine_generator = await self.workers_client.round_robin(
                vLLMGenerateRequest(
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ).model_dump_json()
            )

        output = self._generate_responses(engine_generator, request_type)

        async for response in await self._stream_response(
            request, output, request_id, conversation
        ):
            yield response

    async def _generate_responses(
        self, engine_generator: AsyncIterator[RequestOutput], request_type: RequestType
    ) -> AsyncIterator[Union[RequestOutput, Tuple[int, RequestOutput]]]:
        prompt_idx = 0
        async for resp in engine_generator:
            # Deserialize the response from the engine
            # Creates correct vLLM objects for each field
            output = MyRequestOutput.model_validate_json(resp.data())

            # OpenAIServingChat.chat_completion_stream_generator() method expects a RequestOutput object
            request_output = RequestOutput(
                request_id=output.request_id,
                prompt=output.prompt,
                prompt_token_ids=output.prompt_token_ids,
                prompt_logprobs=output.prompt_logprobs,
                outputs=output.outputs,
                finished=output.finished,
                metrics=output.metrics,
            )

            if request_type == RequestType.CHAT:
                # For chat requests, yield the request_output directly.
                yield request_output
            elif request_type == RequestType.COMPLETION:
                # Completion requests can have multiple prompts and stream generator requires the prompt index
                yield (prompt_idx, request_output)
            else:
                raise NotImplementedError(
                    f"Request type {request_type} not implemented"
                )

    @dynamo_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate_chat(self, raw_request):
        async for response in self._generate(raw_request, RequestType.CHAT):
            yield response

    @dynamo_endpoint(CompletionRequest, CompletionStreamResponse)
    async def generate_completions(self, raw_request):
        async for response in self._generate(raw_request, RequestType.COMPLETION):
            yield response


@dynamo_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Set up clients to the router and workers.
    Serve the dynamo-init.process.chat/completions endpoint.
    """
    workers_client = (
        await runtime.namespace("dynamo-init")
        .component("vllm")
        .endpoint("generate")
        .client()
    )

    router_client = (
        await runtime.namespace("dynamo-init")
        .component("router")
        .endpoint("generate")
        .client()
    )

    preprocess_component = runtime.namespace("dynamo-init").component("process")
    await preprocess_component.create_service()

    chat_endpoint = preprocess_component.endpoint("chat/completions")
    completions_endpoint = preprocess_component.endpoint("completions")

    processor = Processor(engine_args, router_client, workers_client)

    await asyncio.gather(
        chat_endpoint.serve_endpoint(processor.generate_chat),
        completions_endpoint.serve_endpoint(processor.generate_completions),
    )


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
