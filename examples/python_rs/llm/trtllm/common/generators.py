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

from dataclasses import asdict
from common.processor import (
    ChatProcessor,
    CompletionsProcessor,
    merge_promises,
    parse_chat_message_content,
)
from common.protocol import DisaggregatedTypeConverter, Tokens, TRTLLMWorkerRequest, TRTLLMWorkerResponse, TRTLLMWorkerResponseOutput
from common.utils import ServerType
from tensorrt_llm.logger import logger
from tensorrt_llm.executor.result import CompletionOutput

logger.set_level("debug")


async def chat_preprocessor(request, tokenizer):
    conversation = []
    for message in request.messages:
        conversation.extend(parse_chat_message_content(message))
    tool_dicts = (
        None if request.tools is None else [tool.model_dump() for tool in request.tools]
    )
    prompt: str = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=request.add_generation_prompt,
        tools=tool_dicts,
        documents=request.documents,
        chat_template=request.chat_template,
        **(request.chat_template_kwargs or {}),
    )
    sampling_params = request.to_sampling_params()

    return TRTLLMWorkerRequest(
        id=request.id,
        prompt=prompt,
        sampling_params=asdict(sampling_params),
        conversation=conversation,
        disaggregated_params=request.disaggregated_params,
        # NOTE: dont include the first token (e.g. <s>) when searching for a prefix match. We might want to exclude all special tokens at some point.
        tokens=Tokens(tokens=tokenizer.encode(prompt)[1:]),
    )


async def chat_postprocessor(
    engine_generator,
    request,
    conversation,
    server_type: ServerType,
    chat_processor: ChatProcessor,
):
    async for raw_response in engine_generator:
        response = TRTLLMWorkerResponse.model_validate_json(raw_response.data())
        response.outputs = [TRTLLMWorkerResponseOutput(**response.outputs[0])]

        if request.disaggregated_params is not None and server_type == ServerType.CTX:
            response_data = chat_processor.yield_first_chat(
                request, request.id, response
            )
        else:
            response_data = chat_processor.create_chat_stream_response(
                request,
                request.id,
                response,
                conversation,
                first_iteration=(not request.disaggregated_params is not None),
            )
        logger.debug(f"[postprocessor] Response: {response_data}")
        yield response_data


async def completion_preprocessor(request, tokenizer):
    if isinstance(request.prompt, str) or (
        isinstance(request.prompt, list)
        and all(isinstance(x, int) for x in request.prompt)
    ):
        prompt = request.prompt
    else:
        raise ValueError(
            "Invalid prompt type. Only string or list of integers are supported."
        )

    sampling_params = request.to_sampling_params()

    return TRTLLMWorkerRequest(
        id=request.id,
        prompt=prompt,
        sampling_params=asdict(sampling_params),
        disaggregated_params=request.disaggregated_params,
        tokens=Tokens(tokens=tokenizer.encode(prompt)[1:]), 
    )


async def completion_postprocessor(
    engine_generator, request, completions_processor: CompletionsProcessor
):
    async for raw_response in engine_generator:
        response = TRTLLMWorkerResponse.model_validate_json(raw_response.data())
        response.outputs = [TRTLLMWorkerResponseOutput(**response.outputs[0])]

        response_data = completions_processor.create_completion_stream_response(
            request,
            response,
        )
        logger.debug(f"[postprocessor] Response: {response_data}")
        yield response_data
