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

import time
import logging
import uuid
from enum import Enum
from typing import AsyncIterator, Tuple, Union
from components.worker import SGLangWorker

from utils.chat_processor import ChatProcessor, CompletionsProcessor, ProcessMixIn
from utils.protocol import SGLangGenerateRequest, MyRequestOutput

from sglang.srt.openai_api.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ChatCompletionResponse,
    CompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
)
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

logger = logging.getLogger(__name__)

from utils.sglang import parse_sglang_args


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Processor(ProcessMixIn):
    """
    SGLang pre and post processing
    """

    sglang_worker = depends(SGLangWorker)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_sglang_args(class_name, "") 
        self.chat_processor = ChatProcessor()
        self.completions_processor = CompletionsProcessor()
        # init tokenizer manager
        super().__init__()

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = SGLangWorker.dynamo_address()  # type: ignore
        self.worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )
        logger.warning("processor initialized")

    async def _generate(
        self,
        raw_request: Union[CompletionRequest, ChatCompletionRequest],
        request_type: RequestType,
    ):
        request_id = str(uuid.uuid4())
        logger.debug(f"Got raw request: {raw_request}")
        request, sampling_params = await self._parse_raw_request(raw_request)

        logger.warning(sampling_params)
        logger.warning(request)

        request_obj = SGLangGenerateRequest(
            request_id=request_id,
            input_ids=request.input_ids,
            sampling_params=sampling_params,
        ).model_dump_json()

        logger.warning(request_obj)

        # Get the async generator from worker
        output_generator = await self.worker_client.generate(request_obj)
        
        # Properly yield each result from the generator
        async for result in output_generator:
            yield result

    @dynamo_endpoint(name="chat/completions")
    async def chat_completions(self, raw_request: ChatCompletionRequest):
        # Get a stream generator from _generate
        sglang_generator = self._generate(raw_request, RequestType.CHAT)
        
        # Process the stream using generate_stream_response
        async for response in self.generate_stream_response(raw_request, sglang_generator):
            yield response

    async def generate_stream_response(self, request: ChatCompletionRequest, sglang_generator):
        """Converts SGLang stream outputs to OpenAI-compatible chat completion responses"""
        is_first = True
        stream_buffer = ""
        created = int(time.time())
        content_id = str(uuid.uuid4())
        
        async for response in sglang_generator:
            try:
                # Get the data from the response
                response_data = response.data()
                
                # Parse the MyRequestOutput object
                output = MyRequestOutput.model_validate_json(response_data)
                content = output.text
                
                if is_first:
                    # First chunk with role
                    is_first = False
                    # Send a message with just the role
                    choice = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant"),
                        finish_reason=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content_id,
                        created=created,
                        choices=[choice],
                        model=request.model,
                    )
                    # Convert to serialized JSON format for streaming
                    yield chunk.model_dump()
                
                # Get the output_ids from content
                output_ids = content.get("output_ids", [])
                meta_info = content.get("meta_info", {})
                
                # Decode the tokens to get text
                if hasattr(self.tokenizer_manager, 'tokenizer') and self.tokenizer_manager.tokenizer:
                    curr_text = self.tokenizer_manager.tokenizer.decode(output_ids)
                else:
                    curr_text = str(output_ids)  # Fallback
                    
                # Calculate delta (new content since last chunk)
                delta = curr_text[len(stream_buffer):]
                stream_buffer = curr_text
                
                # Get finish reason from meta info
                finish_reason = meta_info.get("finish_reason")
                finish_reason_type = finish_reason.get("type") if finish_reason else None
                
                # Create OpenAI-compatible response
                if delta:  # Only send non-empty deltas
                    choice = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=delta),
                        finish_reason=None,  # Only set on final chunk
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=meta_info.get("id", content_id),
                        created=created,
                        choices=[choice],
                        model=request.model,
                    )
                    # Convert to serialized JSON format for streaming
                    yield chunk.model_dump()
                    
            except Exception as e:
                logger.error(f"Error processing response: {e}")
                # If we get an error, try to create a valid response
                error_choice = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=f"Error: {str(e)}"),
                    finish_reason="error"
                )
                error_chunk = ChatCompletionStreamResponse(
                    id=content_id,
                    created=created,
                    choices=[error_choice],
                    model=request.model
                )
                yield error_chunk.model_dump()
        
        # Send a final chunk with finish_reason
        choice = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(),  # Empty delta for final chunk
            finish_reason="stop",  # Or get from the last chunk
        )
        final_chunk = ChatCompletionStreamResponse(
            id=content_id,
            created=created,
            choices=[choice],
            model=request.model,
        )
        yield final_chunk.model_dump()
        
        # Send DONE marker (optional based on client expectations)
        # This might be needed depending on your client implementation
        # yield "data: [DONE]\n\n"

    # @dynamo_endpoint()
    # async def completions(self, raw_request: CompletionRequest):
    #     async for response in self._generate(raw_request, RequestType.COMPLETION):
    #         yield response
