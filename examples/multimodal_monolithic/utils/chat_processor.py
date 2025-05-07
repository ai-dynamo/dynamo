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
import time
from typing import AsyncIterator, List, Optional, Protocol, Union, runtime_checkable

from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.chat_utils import ConversationMessage
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import RequestPrompt
from vllm.inputs.data import TokensPrompt
from vllm.transformers_utils.tokenizer import AnyTokenizer


@runtime_checkable
class ProcessMixInRequired(Protocol):
    engine_args: AsyncEngineArgs
    chat_processor: "ChatProcessor | None"
    completions_processor: "CompletionsProcessor | None"
    model_config: ModelConfig


class ProcessMixIn(ProcessMixInRequired):
    """
    Mixin for pre and post processing for vLLM
    Requires engine_args, engine_client, processor, model_config to be initialized
    """

    engine_args: AsyncEngineArgs
    chat_processor: "ChatProcessor | None"
    completions_processor: "CompletionsProcessor | None"
    model_config: ModelConfig

    def __init__(self):
        pass

    def _get_processor(
        self, raw_request: Union[CompletionRequest, ChatCompletionRequest]
    ):
        # Determine the processor type based on the request structure
        return (
            self.chat_processor
            if isinstance(raw_request, ChatCompletionRequest)
            else self.completions_processor
        )

    async def _parse_raw_request(
        self, raw_request: Union[CompletionRequest, ChatCompletionRequest]
    ):
        processor = self._get_processor(raw_request)
        if processor is None:
            raise RuntimeError("Processor has not been initialized")
        request = processor.parse_raw_request(raw_request)
        preprocess_result = await processor.preprocess(raw_request)

        default_max_tokens = self.model_config.max_model_len - len(
            preprocess_result.engine_prompt["prompt_token_ids"]
        )
        default_sampling_params = self.model_config.get_diff_sampling_param()
        sampling_params = request.to_sampling_params(
            default_max_tokens,
            self.model_config.logits_processor_pattern,
            default_sampling_params,
        )
        return (
            request,
            preprocess_result.conversation,
            preprocess_result.request_prompt,
            preprocess_result.engine_prompt,
            sampling_params,
        )

    async def _stream_response(self, request, generator, request_id, conversation):
        processor = self._get_processor(request)
        if processor is None:
            raise RuntimeError("processor has not been initialized")
        return processor.stream_response(
            request,
            generator,
            request_id,
            conversation,
        )


class PreprocessResult:
    def __init__(
        self,
        conversation: Optional[ConversationMessage],
        request_prompt: RequestPrompt,
        engine_prompt: TokensPrompt,
    ):
        self.conversation = conversation
        self.request_prompt = request_prompt
        self.engine_prompt = engine_prompt


class ChatProcessor:
    def __init__(self, tokenizer: AnyTokenizer, model_config: ModelConfig):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.openai_serving = OpenAIServingChat(
            engine_client=None,
            model_config=model_config,
            models=None,
            request_logger=None,
            response_role="assistant",
            chat_template=None,
            chat_template_content_format="auto",
        )

    def parse_raw_request(
        self, raw_request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        return ChatCompletionRequest.parse_obj(raw_request)

    async def preprocess(self, raw_request: ChatCompletionRequest) -> PreprocessResult:
        request = self.parse_raw_request(raw_request)

        # TODO: Revisit this later when adding multi-modal support for the frontend.
        # If no chat template is provided and tokenizer doesn't have one,
        # use a simple format that just concatenates messages
        if not request.chat_template and not self.tokenizer.chat_template:
            chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}\n{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n{% endif %}{% endfor %}Assistant:"
        else:
            chat_template = request.chat_template or self.tokenizer.chat_template

        (
            conversation,
            request_prompts,
            engine_prompts,
        ) = await self.openai_serving._preprocess_chat(
            request,
            self.tokenizer,
            request.messages,
            chat_template=chat_template,
            chat_template_content_format=self.openai_serving.chat_template_content_format,
            add_generation_prompt=request.add_generation_prompt,
            continue_final_message=request.continue_final_message,
            tool_dicts=None,
            documents=request.documents,
            chat_template_kwargs=request.chat_template_kwargs,
            tool_parser=self.openai_serving.tool_parser,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            add_special_tokens=request.add_special_tokens,
        )

        return PreprocessResult(conversation[0], request_prompts[0], engine_prompts[0])

    async def stream_response(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator,
        request_id: str,
        conversation: List,
    ):
        request_metadata = RequestResponseMetadata(request_id=request_id)
        if not request.stream:
            raise ValueError("Only streaming responses are supported")
        async for raw_response in self.openai_serving.chat_completion_stream_generator(
            request,
            result_generator,
            request_id,
            request.model,
            conversation,
            self.tokenizer,
            request_metadata,
        ):
            if raw_response.startswith("data: [DONE]"):
                break
            response = json.loads(raw_response.lstrip("data: "))
            yield response


class CompletionsProcessor:
    def __init__(self, tokenizer: AnyTokenizer, model_config: ModelConfig):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.openai_serving = OpenAIServingCompletion(
            engine_client=None,
            model_config=model_config,
            models=None,
            request_logger=None,
        )

    def parse_raw_request(self, raw_request: CompletionRequest) -> CompletionRequest:
        return CompletionRequest.parse_obj(raw_request)

    async def preprocess(self, raw_request: CompletionRequest) -> PreprocessResult:
        request = self.parse_raw_request(raw_request)

        (
            request_prompts,
            engine_prompts,
        ) = await self.openai_serving._preprocess_completion(
            request,
            self.tokenizer,
            input_or_inputs=request.prompt,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            add_special_tokens=request.add_special_tokens,
        )

        return PreprocessResult(None, request_prompts[0], engine_prompts[0])

    async def stream_response(
        self,
        request: CompletionRequest,
        result_generator: AsyncIterator,
        request_id: str,
        conversation: Optional[List[ConversationMessage]] = None,
    ):
        request_metadata = RequestResponseMetadata(request_id=request_id)
        if not request.stream:
            raise ValueError("Only streaming responses are supported")
        async for raw_response in self.openai_serving.completion_stream_generator(
            request,
            result_generator,
            request_id,
            int(time.time()),  # created_time
            request.model,
            1,  # num_prompts
            self.tokenizer,
            request_metadata,
        ):
            if raw_response.startswith("data: [DONE]"):
                break
            response = json.loads(raw_response.lstrip("data: "))

            yield response
