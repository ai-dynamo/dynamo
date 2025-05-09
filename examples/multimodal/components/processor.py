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

import json
import logging
import uuid
from enum import Enum
from typing import AsyncIterator, Tuple, Union

from components.decode_worker import VllmDecodeWorker
from transformers import AutoTokenizer
from utils.chat_processor import ChatProcessor, CompletionsProcessor, ProcessMixIn
from utils.logging import check_required_workers
from utils.protocol import MultiModalRequest, MyRequestOutput, vLLMMultimodalRequest
from utils.vllm import parse_vllm_args
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer

from dynamo.runtime import EtcdKvCache
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

logger = logging.getLogger(__name__)


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Processor(ProcessMixIn):
    """
    vLLM pre and post processing
    """

    worker = depends(VllmDecodeWorker)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.model_config = self.engine_args.create_model_config()
        self.tokenizer = self._create_tokenizer(self.engine_args)
        self.chat_processor = ChatProcessor(self.tokenizer, self.model_config)
        self.completions_processor = CompletionsProcessor(
            self.tokenizer, self.model_config
        )
        self.min_workers = 1

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

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = VllmDecodeWorker.dynamo_address()  # type: ignore
        self.worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )

        await check_required_workers(self.worker_client, self.min_workers)

        self.etcd_kv_cache = await EtcdKvCache.create(
            runtime.etcd_client(),
            "/dynamo/processor/",
            {"router": self.engine_args.router},
        )

    # Main method to parse the request and send the request to the vllm worker.
    async def _generate(
        self,
        raw_request: Union[CompletionRequest, ChatCompletionRequest],
        image: str,
        request_type: RequestType,
    ):
        request_id = str(uuid.uuid4())
        logger.debug(f"Got raw request: {raw_request}")
        (
            request,
            conversation,
            prompt,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(raw_request)

        worker_request = vLLMMultimodalRequest(
            engine_prompt=engine_prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            image_url=image,
        )
        router_mode = (await self.etcd_kv_cache.get("router")).decode()
        if router_mode == "kv":
            # The current KV router does not support multimodal requests because
            # it performs cache lookup based solely on prompt tokens. At this stage,
            # multimodal data (e.g., image features) is not yet available, so the router
            # cannot select the optimal worker using both prompt and image inputs.
            raise NotImplementedError(
                "Multimodal requests are not supported for kv router mode"
            )

        if router_mode == "random":
            response_generator = await self.worker_client.generate(
                worker_request.model_dump_json()
            )
        elif router_mode == "round-robin":
            response_generator = await self.worker_client.round_robin(
                worker_request.model_dump_json()
            )
        else:
            raise NotImplementedError(f"Router mode {router_mode} not implemented")

        output = self._generate_responses(response_generator, request_type)

        # TODO: This is a temporary solution to combine the content from the engine generator.
        # After having the multimodal support in OpenAI compatible frontend, we can use that directly without the need to manually combine the content.
        combined_content = ""
        async for response in await self._stream_response(
            request, output, request_id, conversation
        ):
            if "choices" in response and len(response["choices"]) > 0:
                delta = response["choices"][0].get("delta", {})
                content = delta.get("content", "")
                combined_content += content

                # Yield complete content on final response
                if response["choices"][0].get("finish_reason") is not None:
                    yield combined_content

    # This method is used to process the responses from the engine generator.
    async def _generate_responses(
        self,
        response_generator: AsyncIterator[RequestOutput],
        request_type: RequestType,
    ) -> AsyncIterator[Union[RequestOutput, Tuple[int, RequestOutput]]]:
        prompt_idx = 0
        async for resp in response_generator:
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

    # The generate endpoint will be used by the frontend to handle incoming requests.
    @dynamo_endpoint()
    async def generate(self, request: MultiModalRequest):
        # TODO: After having the multimodal support in OpenAI compatible frontend, we can use that directly and remove the custom endpoint.
        msg = {
            "role": "user",
            "content": "USER: <image>\nQuestion:" + request.prompt + " Answer:",
        }

        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=[msg],
            stream=True,
            max_tokens=request.max_tokens,
            request_id=str(uuid.uuid4()),
        )

        async for response in self._generate(
            chat_request, request.image, RequestType.CHAT
        ):
            yield json.dumps(response)
