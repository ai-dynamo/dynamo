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
import uuid
from enum import Enum
from typing import AsyncIterator, Tuple, Union

from transformers import AutoTokenizer
from utils.chat_processor import ChatProcessor, CompletionsProcessor, ProcessMixIn
from utils.protocol import MyRequestOutput, Tokens, vLLMGenerateRequest
from utils.vllm import parse_vllm_args
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from vllm.logger import logger as vllm_logger
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import FlexibleArgumentParser
from dynamo.sdk import (
    depends,
    dynamo_endpoint,
    service,
    dynamo_context,
)
from dynamo.sdk.lib.config import ServiceConfig
from sdk_vllm.worker import VllmWorker
from sdk_vllm.kv_router import Router

os.environ["VLLM_LOG_LEVEL"] = "DEBUG"

class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


def parse_args(service_name, prefix) -> AsyncEngineArgs:
    config = ServiceConfig.get_instance()
    vllm_args = config.as_args(service_name, prefix=prefix)
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--router",
        type=str,
        choices=["random", "round-robin", "kv"],
        default="random",
        help="Router type to use for scheduling requests to workers",
    )
    parser.add_argument(
        "--remote-prefill", action="store_true", help="Enable remote prefill"
    )
    parser.add_argument(
        "--conditional-disagg",
        action="store_true",
        help="Use disaggregated router to decide whether to prefill locally or remotely",
    )
    parser.add_argument(
        "--max-local-prefill-length",
        type=int,
        default=1000,
        help="Maximum length of local prefill",
    )
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args(vllm_args)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.router = args.router
    engine_args.remote_prefill = args.remote_prefill
    engine_args.conditional_disagg = args.conditional_disagg
    engine_args.max_local_prefill_length = args.max_local_prefill_length
    return engine_args

@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo-init",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Processor(ProcessMixIn):
    """
    vLLM pre and post processing
    """

    worker = depends(VllmWorker)
    router = depends(Router)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_args(class_name, "vllm_")
        self.model_config = self.engine_args.create_model_config()
        print(f"[Processor] self.model_config: {self.model_config}")
        print(f"[Processor] self.engine_args: {self.engine_args}")
        self.tokenizer = self._create_tokenizer(self.engine_args)
        self.chat_processor = ChatProcessor(self.tokenizer, self.model_config)
        self.completions_processor = CompletionsProcessor(
            self.tokenizer, self.model_config
        )
        self.router_mode = self.engine_args.router

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
        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = VllmWorker.dynamo_address()  # type: ignore
        print(f"[Processor] comp_ns: {comp_ns}, comp_name: {comp_name}")
        worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )
        if self.router_mode == "kv":
            async for route_response in self.router.generate(
                Tokens(tokens=engine_prompt["prompt_token_ids"]).model_dump_json()
            ):
                worker_id, prefix_hit_rate = route_response.split("_")
                prefix_hit_rate = float(prefix_hit_rate)
                vllm_logger.info(
                    f"Worker ID: {worker_id} with estimated prefix hit rate: {prefix_hit_rate}"
                )
                break

            if worker_id == "":
                engine_generator = await worker_client.generate(
                    vLLMGenerateRequest(
                        engine_prompt=engine_prompt,
                        sampling_params=sampling_params,
                        request_id=request_id,
                        prefix_hit_rate=prefix_hit_rate,
                    ).model_dump_json()
                )
            else:
                engine_generator = await worker_client.direct(
                    vLLMGenerateRequest(
                        engine_prompt=engine_prompt,
                        sampling_params=sampling_params,
                        request_id=request_id,
                        prefix_hit_rate=prefix_hit_rate,
                    ).model_dump_json(),
                    int(worker_id),
                )
        elif self.router_mode == "random":
            engine_generator = await worker_client.generate(
                vLLMGenerateRequest(
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ).model_dump_json()
            )
        # TODO: add round-robin mode
        # elif self.router_mode == "round-robin":
        #     engine_generator = await self.worker.round_robin(
        #         vLLMGenerateRequest(
        #             engine_prompt=engine_prompt,
        #             sampling_params=sampling_params,
        #             request_id=request_id,
        #         ).model_dump_json()
        #     )

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

    @dynamo_endpoint()
    async def chat_completions(self, raw_request: ChatCompletionRequest):
        async for response in self._generate(raw_request, RequestType.CHAT):
            yield response

    # @dynamo_endpoint()
    # async def completions(self, raw_request: CompletionRequest):
    #     async for response in self._generate(raw_request, RequestType.COMPLETION):
    #         yield response
