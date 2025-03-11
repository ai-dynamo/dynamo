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
import json
import signal
import uuid

import uvloop
from common.base_engine import BaseTensorrtLLMEngine, TensorrtLLMEngineConfig
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.processor import merge_promises, parse_chat_message_content
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionStreamResponse,
)

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker

logger.set_level("debug")


class TensorrtLLMEngine(BaseTensorrtLLMEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, trt_llm_engine_config: TensorrtLLMEngineConfig):
        super().__init__(trt_llm_engine_config)

    @dynamo_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate_chat(self, request):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        logger.debug(f"Received chat request: {request}")
        request_id = str(uuid.uuid4())
        self._ongoing_request_count += 1

        try:
            conversation = []
            for message in request.messages:
                conversation.extend(parse_chat_message_content(message))
            tool_dicts = (
                None
                if request.tools is None
                else [tool.model_dump() for tool in request.tools]
            )
            prompt: str = self._tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
                tools=tool_dicts,
                documents=request.documents,
                chat_template=request.chat_template,
                **(request.chat_template_kwargs or {}),
            )
            sampling_params = request.to_sampling_params()

            promise = self._llm_engine.generate_async(
                prompt,
                sampling_params,
                streaming=request.stream,
            )
            # NOTE: somehow stream and non-stream is working with the same path
            response_generator = self.chat_processor.stream_response(
                request, request_id, conversation, promise
            )
            async for response in response_generator:
                yield response

            self._ongoing_request_count -= 1
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            raise RuntimeError("Failed to generate: " + str(e))

    @dynamo_endpoint(CompletionRequest, CompletionStreamResponse)
    async def generate_completion(self, request):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        self._ongoing_request_count += 1
        logger.debug(f"Received completion request: {request}")

        if isinstance(request.prompt, str) or (
            isinstance(request.prompt, list) and isinstance(request.prompt[0], int)
        ):
            prompts = [request.prompt]
        else:
            prompts = request.prompt

        promises = []
        sampling_params = request.to_sampling_params()

        try:
            for prompt in prompts:
                promise = self._llm_engine.generate_async(
                    prompt,
                    sampling_params,
                    streaming=request.stream,
                )
                promises.append(promise)

            generator = merge_promises(promises)
            num_choices = (
                len(prompts) if request.n is None else len(prompts) * request.n
            )

            # NOTE: always send `stream: true` to the worker, and decide whether to aggregate  or not before sending the response back to client.
            response_generator = self.completions_processor.create_completion_generator(
                request, generator, num_choices
            )
            async for response in response_generator:
                yield json.loads(response)

            self._ongoing_request_count -= 1
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            raise RuntimeError("Failed to generate: " + str(e))


@dynamo_worker()
async def worker(runtime: DistributedRuntime, engine_config: LLMAPIConfig):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    namespace_str = "dynamo"
    component_str = "tensorrt-llm"

    component = runtime.namespace(namespace_str).component(component_str)
    await component.create_service()

    completions_endpoint = component.endpoint("completions")
    chat_completions_endpoint = component.endpoint("chat/completions")

    trt_llm_engine_config = TensorrtLLMEngineConfig(
        namespace_str=namespace_str,
        component_str=component_str,
        engine_config=engine_config,
    )
    engine = TensorrtLLMEngine(trt_llm_engine_config)

    await asyncio.gather(
        completions_endpoint.serve_endpoint(engine.generate_completion),
        chat_completions_endpoint.serve_endpoint(engine.generate_chat),
    )


if __name__ == "__main__":
    uvloop.install()
    args, engine_config = parse_tensorrt_llm_args()
    asyncio.run(worker(engine_config))
