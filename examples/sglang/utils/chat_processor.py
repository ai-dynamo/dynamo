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

import logging
import time
import uuid
from typing import AsyncIterator, Dict, Union

from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.openai_api.adapter import (
    to_openai_style_logprobs,
    v1_chat_generate_request,
    v1_generate_request,
)
from sglang.srt.openai_api.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
)
from sglang.srt.server_args import ServerArgs
from utils.protocol import MyRequestOutput
from utils.tokenizer_manager import PatchedTokenizerManager

logger = logging.getLogger(__name__)


class ProcessMixIn:
    """
    Mixin for pre and post processing for SGLang
    """

    engine_args: ServerArgs
    chat_processor: "ChatProcessor | None"
    completions_processor: "CompletionsProcessor | None"

    def __init__(self):
        self.tokenizer_manager = PatchedTokenizerManager(self.engine_args)
        print("tokenizer manager initialized?")

    def _get_processor(
        self, raw_request: Union[ChatCompletionRequest, CompletionRequest]
    ):
        return (
            self.chat_processor
            if isinstance(raw_request, ChatCompletionRequest)
            else self.completions_processor
        )

    async def _parse_raw_request(
        self, raw_request: Union[ChatCompletionRequest, CompletionRequest]
    ):
        processor = self._get_processor(raw_request)
        if processor is None:
            raise RuntimeError("Processor has not been initialized")
        preprocess_result, sampling_params = await processor.preprocess(
            raw_request, self.tokenizer_manager
        )

        return preprocess_result, sampling_params


class ChatProcessor:
    def parse_raw_request(
        self, raw_request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        return ChatCompletionRequest.model_validate(raw_request)

    async def preprocess(
        self,
        raw_request: ChatCompletionRequest,
        tokenizer_manager: PatchedTokenizerManager,
    ) -> GenerateReqInput:
        chat_generate_req_input, _ = v1_chat_generate_request(
            [raw_request], tokenizer_manager
        )
        return chat_generate_req_input, chat_generate_req_input.sampling_params

    async def generate_stream_response(
        self,
        request: ChatCompletionRequest,
        sglang_generator: AsyncIterator,
        tokenizer_manager=None,
    ):
        """Converts SGLang stream outputs to OpenAI-compatible chat completion responses"""
        is_first = True
        stream_buffer = ""
        created = int(time.time())
        content_id = str(uuid.uuid4())

        async for response in sglang_generator:
            try:
                response_data = response.data()

                output = MyRequestOutput.model_validate_json(response_data)
                content = output.text

                if is_first:
                    is_first = False
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
                    yield chunk.model_dump()

                output_ids = content.get("output_ids", [])
                meta_info = content.get("meta_info", {})

                if (
                    tokenizer_manager
                    and hasattr(tokenizer_manager, "tokenizer")
                    and tokenizer_manager.tokenizer
                ):
                    curr_text = tokenizer_manager.tokenizer.decode(output_ids)
                else:
                    curr_text = str(output_ids)

                delta = curr_text[len(stream_buffer) :]
                stream_buffer = curr_text

                if delta:
                    choice = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=delta),
                        finish_reason=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=meta_info.get("id", content_id),
                        created=created,
                        choices=[choice],
                        model=request.model,
                    )
                    yield chunk.model_dump()

            except Exception as e:
                logger.error(f"Error processing response: {e}")
                error_choice = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=f"Error: {str(e)}"),
                    finish_reason="error",
                )
                error_chunk = ChatCompletionStreamResponse(
                    id=content_id,
                    created=created,
                    choices=[error_choice],
                    model=request.model,
                )
                yield error_chunk.model_dump()

        choice = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(),
            finish_reason="stop",
        )
        final_chunk = ChatCompletionStreamResponse(
            id=content_id,
            created=created,
            choices=[choice],
            model=request.model,
        )
        yield final_chunk.model_dump()


class CompletionsProcessor:
    def parse_raw_request(self, raw_request: CompletionRequest) -> CompletionRequest:
        return CompletionRequest.model_validate(raw_request)

    # the v1_generate_request does not tokenize under the hood so we return the TokenizedGenerateReqInput along with sampling params as a dict
    # because the sgl.Engine runs SamplingParams.__init__()
    async def preprocess(
        self, raw_request: CompletionRequest, tokenizer_manager: PatchedTokenizerManager
    ) -> TokenizedGenerateReqInput:
        generate_req_input, _ = v1_generate_request([raw_request])
        tokenized_generate_req_input = await tokenizer_manager._tokenize_one_request(
            generate_req_input
        )

        return tokenized_generate_req_input, generate_req_input.sampling_params

    async def generate_stream_response(
        self,
        request: CompletionRequest,
        sglang_generator: AsyncIterator,
        tokenizer_manager=None,
    ):
        stream_buffers: Dict[int, str] = {}
        n_prev_tokens: Dict[int, int] = {}
        created = int(time.time())
        content_id = str(uuid.uuid4())

        async for response in sglang_generator:
            try:
                response_data = response.data()

                output = MyRequestOutput.model_validate_json(response_data)
                content = output.text

                index = content.get("index", 0)
                stream_buffer = stream_buffers.get(index, "")
                n_prev_token = n_prev_tokens.get(index, 0)

                output_ids = content.get("output_ids", [])

                if (
                    tokenizer_manager
                    and hasattr(tokenizer_manager, "tokenizer")
                    and tokenizer_manager.tokenizer
                ):
                    curr_text = tokenizer_manager.tokenizer.decode(output_ids)
                else:
                    curr_text = str(output_ids)

                delta = curr_text[len(stream_buffer) :]
                stream_buffer = curr_text

                if request.logprobs is not None:
                    if not stream_buffer and request.echo:
                        input_token_logprobs = content["meta_info"].get(
                            "input_token_logprobs"
                        )
                        input_top_logprobs = content["meta_info"].get(
                            "input_top_logprobs"
                        )
                    else:
                        input_token_logprobs = None
                        input_top_logprobs = None

                    logprobs = to_openai_style_logprobs(
                        input_token_logprobs=input_token_logprobs,
                        input_top_logprobs=input_top_logprobs,
                        output_token_logprobs=content["meta_info"].get(
                            "output_token_logprobs", []
                        )[n_prev_token:],
                        output_top_logprobs=content["meta_info"].get(
                            "output_top_logprobs", []
                        )[n_prev_token:],
                    )
                    n_prev_token = len(
                        content["meta_info"].get("output_token_logprobs", [])
                    )
                else:
                    logprobs = None

                if delta:
                    choice_data = CompletionResponseStreamChoice(
                        index=index,
                        text=delta,
                        logprobs=logprobs,
                        finish_reason=content["meta_info"].get("finish_reason")["type"]
                        if content["meta_info"].get("finish_reason")
                        else None,
                        matched_stop=(
                            content["meta_info"].get("finish_reason")["matched"]
                            if content["meta_info"].get("finish_reason")
                            and "matched" in content["meta_info"].get("finish_reason")
                            else None
                        ),
                    )

                    chunk = CompletionStreamResponse(
                        id=content["meta_info"].get("id", content_id),
                        created=created,
                        object="text_completion",
                        choices=[choice_data],
                        model=request.model,
                    )

                    stream_buffers[index] = stream_buffer
                    n_prev_tokens[index] = n_prev_token

                    yield chunk.model_dump()

            except Exception as e:
                logger.error(
                    f"Error processing completion response: {e}", exc_info=True
                )
                error_choice = CompletionResponseStreamChoice(
                    index=0,
                    text=f"Error: {str(e)}",
                    logprobs=None,
                    finish_reason="error",
                )
                error_chunk = CompletionStreamResponse(
                    id=content_id,
                    created=created,
                    object="text_completion",
                    choices=[error_choice],
                    model=request.model,
                )
                yield error_chunk.model_dump()

        # Final chunk with finish_reason="stop"
        final_choice = CompletionResponseStreamChoice(
            index=0,
            text="",
            logprobs=None,
            finish_reason="stop",
        )
        final_chunk = CompletionStreamResponse(
            id=content_id,
            created=created,
            object="text_completion",
            choices=[final_choice],
            model=request.model,
        )
        yield final_chunk.model_dump()
