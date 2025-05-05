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
from dataclasses import asdict
from typing import Any, Dict, List, Union

from common.parser import LLMAPIConfig
from common.protocol import (
    DynamoTRTLLMChatCompletionResponseStreamChoice,
    DynamoTRTLLMChatCompletionStreamResponse,
    DynamoTRTLLMCompletionResponseStreamChoice,
    DynamoTRTLLMCompletionStreamResponse,
    Tokens,
    TRTLLMWorkerRequest,
    TRTLLMWorkerResponse,
    TRTLLMWorkerResponseOutput,
)
from common.utils import ConversationMessage
from openai.types.chat import ChatCompletionMessageParam
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    DeltaMessage,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class ChatProcessorMixin:
    def __init__(
        self, engine_config: LLMAPIConfig, using_engine_generator: bool = False
    ):
        self._engine_config = engine_config
        logger.info(f"Using LLM API config: {self._engine_config.to_dict()}")
        # model name for chat processor
        self._model_name = self._engine_config.model_name
        logger.info(f"Set model name: {self._model_name}")
        # model for LLMAPI input
        self._model = self._model_name
        if self._engine_config.model_path:
            self._model = self._engine_config.model_path
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._engine_config.model_path
            )
            logger.info(f"Using model from path: {self._engine_config.model_path}")
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._engine_config.model_name
            )
        if self._engine_config.extra_args.get("tokenizer", None):
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._engine_config.extra_args.get("tokenizer", None)
            )
        self.chat_processor = ChatProcessor(
            self._model_name, self._tokenizer, using_engine_generator
        )
        self.completions_processor = CompletionsProcessor(
            self._model_name, self._tokenizer
        )


def parse_chat_message_content(
    message: ChatCompletionMessageParam,
) -> Union[ConversationMessage, List[ConversationMessage], List[None]]:
    role = message["role"]
    content = message.get("content")

    if content is None:
        return []
    if isinstance(content, str):
        return [ConversationMessage(role=role, content=content)]

    texts: List[str] = []
    for part in content:
        part_type = part["type"]
        if part_type == "text":
            text = part["text"]  # type: ignore
            texts.append(text)
        else:
            raise NotImplementedError(f"{part_type} is not supported")

    text_prompt = "\n".join(texts)
    return [ConversationMessage(role=role, content=text_prompt)]


class BaseChatProcessor:
    def __init__(
        self,
        model: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        self.model = model
        self.tokenizer = tokenizer

    def _get_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            role = "assistant"
        else:
            role = request.messages[-1]["role"]
        return role

    def _stream_usage_info(
        self, request: ChatCompletionRequest, prompt_tokens: int, completion_tokens: int
    ):
        if (
            request.stream_options
            and request.stream_options.include_usage
            and request.stream_options.continuous_usage_stats
        ):
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        else:
            usage = None
        return usage

    def _create_logprobs(
        self, token_ids: List[int], logprobs: List[float]
    ) -> ChatCompletionLogProbs:
        assert len(token_ids) == len(
            logprobs
        ), "token_ids and logprobs have different lengths"
        content: List[ChatCompletionLogProbsContent] = []
        for token_id, logprob in zip(token_ids, logprobs):
            token = self.tokenizer.decode(token_id)
            # returning multiple logprobs is not supported
            first_logprob = ChatCompletionLogProbsContent(
                token=token,
                # NOTE: min logprob -9999.0 for probabilities extremely close to 0
                logprob=max(logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            content.append(first_logprob)
        chat_logprobs = ChatCompletionLogProbs(content=content)
        return chat_logprobs


class ChatProcessor(BaseChatProcessor):
    def __init__(
        self,
        model: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        using_engine_generator: bool = False,
    ):
        super().__init__(model, tokenizer)
        self.using_engine_generator = using_engine_generator

    def yield_first_chat(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        response: RequestOutput,
        content: str | None = None,
    ):
        role = self._get_role(request)
        num_choices = 1 if request.n is None else request.n
        num_tokens = len(response.prompt_token_ids)
        content = response.outputs[0].text_diff

        for i in range(num_choices):
            choice = DynamoTRTLLMChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role=role, content=content),
                finish_reason=None,
            )
            if response.outputs[0].disaggregated_params is not None:
                # Do not include the disaggregated params in response
                # from processor.
                pass

            chunk = DynamoTRTLLMChatCompletionStreamResponse(
                id=request_id,
                choices=[choice],
                model=self.model,
            )
            chunk.usage = self._stream_usage_info(request, num_tokens, 0)

            return chunk.model_dump_json()

    def create_chat_stream_response(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        response: RequestOutput,
        conversation: List[Dict[str, Any]],
        first_iteration: bool = True,
    ) -> str:
        num_choices = 1 if request.n is None else request.n
        finish_reason_sent = [False] * num_choices
        role = self._get_role(request)

        prompt_tokens = len(response.prompt_token_ids)
        if first_iteration:
            return self.yield_first_chat(request, request_id, response)

            # TODO: Fix this
            if request.echo:
                last_msg_content = ""
                if (
                    conversation
                    and conversation[-1].get("content")
                    and conversation[-1].get("role") == role
                ):
                    last_msg_content = conversation[-1]["content"]

                if last_msg_content:
                    return self.yield_first_chat(
                        request, request_id, response, content=last_msg_content
                    )
        first_iteration = False

        for output in response.outputs:
            i = output.index

            if finish_reason_sent[i]:
                continue

            delta_text = output.text_diff
            if (
                request.tool_choice
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            ):
                delta_message = DeltaMessage(
                    tool_calls=[
                        ToolCall(
                            function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=delta_text,
                            )
                        )
                    ]
                )
            else:
                delta_message = DeltaMessage(content=delta_text, role=role)

            choice = DynamoTRTLLMChatCompletionResponseStreamChoice(
                index=i, delta=delta_message, finish_reason=None
            )
            if request.logprobs:
                logprobs = output.logprobs_diff
                token_ids = output.token_ids_diff
                choice.logprobs = self._create_logprobs(token_ids, logprobs)
            if output.finish_reason is not None:
                choice.finish_reason = output.finish_reason
                choice.stop_reason = output.stop_reason
                finish_reason_sent[i] = True
            if output.disaggregated_params is not None:
                # Block the disaggregated params at processor level
                pass

            chunk = DynamoTRTLLMChatCompletionStreamResponse(
                id=request_id,
                choices=[choice],
                model=self.model,
            )
            logger.debug(f"[processor] Chunk: {chunk}")
            chunk.usage = self._stream_usage_info(request, prompt_tokens, output.length)
            return chunk.model_dump_json()

        # TODO: make request.stream_options.include_usage = True when stream=False in rust
        if request.stream_options and request.stream_options.include_usage:
            completion_tokens = sum(output.length for output in response.outputs)
            final_usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            final_usage_chunk = DynamoTRTLLMChatCompletionStreamResponse(
                id=request_id,
                choices=[],
                model=self.model,
                usage=final_usage,
            )
            return final_usage_chunk.model_dump_json()
        return "data: [DONE]\n\n"

    async def preprocess(self, request):
        conversation: List[Any] = []
        for message in request.messages:
            conversation.extend(parse_chat_message_content(message))
        tool_dicts = (
            None
            if request.tools is None
            else [tool.model_dump() for tool in request.tools]
        )
        prompt = self.tokenizer.apply_chat_template(
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
            model=request.model,
            prompt=prompt,
            sampling_params=asdict(sampling_params),
            conversation=conversation,
            disaggregated_params=request.disaggregated_params,
            # NOTE: dont include the first token (e.g. <s>) when searching for a prefix match. We might want to exclude all special tokens at some point.
            tokens=Tokens(tokens=self.tokenizer.encode(prompt)[1:]),
        )

    async def postprocess(
        self,
        engine_generator,
        request,
        conversation,
    ):
        first_iteration = True
        last_text_len = 0
        last_token_ids_len = 0
        async for raw_response in engine_generator:
            if self.using_engine_generator:
                response = TRTLLMWorkerResponse(
                    request_id=request.id,
                    prompt=raw_response.prompt,
                    prompt_token_ids=raw_response.prompt_token_ids,
                    outputs=[asdict(raw_response.outputs[0])],
                    finished=raw_response.finished,
                )
                response.outputs = [TRTLLMWorkerResponseOutput(**response.outputs[0])]
            else:
                response = TRTLLMWorkerResponse.model_validate_json(raw_response.data())
                response.outputs[0]["text"] = self.tokenizer.decode(
                    response.outputs[0]["token_ids"]
                )
                # Need to keep track of the last text and token ids length
                # to calculate the diff.
                # TODO: This is a hack to get the diff. We should identify why
                # the diff is not being calculated in the worker.
                response.outputs[0]["_last_text_len"] = last_text_len
                response.outputs[0]["_last_token_ids_len"] = last_token_ids_len
                last_text_len = len(response.outputs[0]["text"])
                last_token_ids_len = len(response.outputs[0]["token_ids"])
                response.outputs = [TRTLLMWorkerResponseOutput(**response.outputs[0])]

            response_data = self.create_chat_stream_response(
                request,
                request.id,
                response,
                conversation,
                first_iteration=first_iteration,
            )
            first_iteration = False
            logger.debug(f"[postprocessor] Response: {response_data}")
            yield response_data


class CompletionsProcessor:
    def __init__(
        self,
        model: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        self.model = model
        self.tokenizer = tokenizer

    def create_completion_stream_response(self, request, response):
        num_choices = 1 if request.n is None else request.n
        echoed = [False] * num_choices

        # len(response.outputs) is always 1
        for gen_idx, output in enumerate(response.outputs):
            delta_text = output.text_diff
            if request.echo and not echoed[gen_idx]:
                delta_text = request.prompt + delta_text
                echoed[gen_idx] = True
            choice = DynamoTRTLLMCompletionResponseStreamChoice(
                index=gen_idx,
                text=delta_text,
                stop_reason=output.stop_reason,
                finish_reason=output.finish_reason,
            )
            if output.disaggregated_params is not None:
                # Block the disagg_params
                pass

            chunk = DynamoTRTLLMCompletionStreamResponse(
                model=self.model,
                choices=[choice],
            )
            return chunk.model_dump_json()

    async def preprocess(self, request):
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
            model=request.model,
            prompt=prompt,
            sampling_params=asdict(sampling_params),
            disaggregated_params=request.disaggregated_params,
            tokens=Tokens(tokens=self.tokenizer.encode(prompt)[1:]),
        )

    async def postprocess(
        self,
        engine_generator,
        request,
    ):
        async for raw_response in engine_generator:
            response = TRTLLMWorkerResponse.model_validate_json(raw_response.data())
            response.outputs = [TRTLLMWorkerResponseOutput(**response.outputs[0])]

            response_data = self.create_completion_stream_response(
                request,
                response,
            )
            logger.debug(f"[postprocessor] Response: {response_data}")
            yield response_data
