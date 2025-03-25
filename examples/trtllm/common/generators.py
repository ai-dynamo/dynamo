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
import os
import signal
import uuid

from common.base_engine import BaseTensorrtLLMEngine
from common.processor import merge_promises, parse_chat_message_content
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.logger import logger

logger.set_level(os.getenv("DYNAMO_TRTLLM_LOG_LEVEL", "info"))


# TODO: support disaggregated as well
async def chat_generator(engine: BaseTensorrtLLMEngine, request):
    if engine._llm_engine is None:
        raise RuntimeError("Engine not initialized")

    logger.debug(f"Received chat request: {request}")
    request_id = str(uuid.uuid4())
    engine._ongoing_request_count += 1

    try:
        conversation = []
        for message in request.messages:
            conversation.extend(parse_chat_message_content(message))
        tool_dicts = (
            None
            if request.tools is None
            else [tool.model_dump() for tool in request.tools]
        )
        prompt: str = engine._tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=request.add_generation_prompt,
            tools=tool_dicts,
            documents=request.documents,
            chat_template=request.chat_template,
            **(request.chat_template_kwargs or {}),
        )
        sampling_params = request.to_sampling_params()

        promise = engine._llm_engine.generate_async(
            prompt,
            sampling_params,
            streaming=request.stream,
        )
        response_generator = engine.chat_processor.stream_response(
            request, request_id, conversation, promise
        )
        async for response in response_generator:
            yield response

        engine._ongoing_request_count -= 1
    except CppExecutorError:
        # If internal executor error is raised, shutdown the server
        signal.raise_signal(signal.SIGINT)
    except Exception as e:
        raise RuntimeError("Failed to generate: " + str(e))


async def completion_generator(engine: BaseTensorrtLLMEngine, request):
    if engine._llm_engine is None:
        raise RuntimeError("Engine not initialized")

    engine._ongoing_request_count += 1
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
            promise = engine._llm_engine.generate_async(
                prompt,
                sampling_params,
                streaming=request.stream,
            )
            promises.append(promise)

        generator = merge_promises(promises)
        num_choices = len(prompts) if request.n is None else len(prompts) * request.n

        # NOTE: always send `stream: true` to the worker, and decide whether to aggregate  or not before sending the response back to client.
        response_generator = engine.completions_processor.create_completion_generator(
            request, generator, num_choices
        )
        async for response in response_generator:
            yield json.loads(response)

        engine._ongoing_request_count -= 1
    except CppExecutorError:
        # If internal executor error is raised, shutdown the server
        signal.raise_signal(signal.SIGINT)
    except Exception as e:
        raise RuntimeError("Failed to generate: " + str(e))
