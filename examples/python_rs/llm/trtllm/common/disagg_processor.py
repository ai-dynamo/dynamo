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
from typing import Any, Dict, List, TypedDict, Union

from common.protocol import DisaggChatCompletionStreamResponse
from openai.types.chat import ChatCompletionMessageParam
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
    DeltaMessage,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from transformers import AutoTokenizer


class ConversationMessage(TypedDict):
    role: str
    content: str


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


class ChatProcessor:
    def __init__(
        self, model: str, tokenizer: AutoTokenizer, request: ChatCompletionRequest
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.request = request
        self.num_choices = 1 if self.request.n is None else self.request.n
        self.finish_reason_sent = [False] * self.num_choices
        self.role = self._get_role(self.request)

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

    def get_chat_stream_response(
        self,
        request_id: str,
        res: RequestOutput,
        first_iteration: bool,
    ) -> DisaggChatCompletionStreamResponse:
        def get_first_chat(
            num_tokens: int, role: str | None = None, content: str | None = None
        ):
            for i in range(self.num_choices):
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(role=role, content=content),
                    finish_reason=None,
                )
                chunk = DisaggChatCompletionStreamResponse(
                    id=request_id,
                    created=int(time.time()),
                    object="chat.completion.chunk",
                    choices=[choice_data],
                    model=self.model,
                )
                chunk.usage = self._stream_usage_info(
                    self.request, num_tokens, completion_tokens=0
                )

                return chunk

        prompt_tokens = len(res.prompt_token_ids)
        if first_iteration:
            return get_first_chat(prompt_tokens, role=self.role)

        for output in res.outputs:
            i = output.index

            if self.finish_reason_sent[i]:
                continue

            delta_text = output.text_diff
            if (
                self.request.tool_choice
                and type(self.request.tool_choice) is ChatCompletionNamedToolChoiceParam
            ):
                delta_message = DeltaMessage(
                    tool_calls=[
                        ToolCall(
                            function=FunctionCall(
                                name=self.request.tool_choice.function.name,
                                arguments=delta_text,
                            )
                        )
                    ]
                )
            else:
                delta_message = DeltaMessage(content=delta_text)

            choice = ChatCompletionResponseStreamChoice(
                index=i, delta=delta_message, finish_reason=None
            )
            if self.request.logprobs:
                logprobs = output.logprobs_diff
                token_ids = output.token_ids_diff
                choice.logprobs = self._create_logprobs(token_ids, logprobs)
            if output.finish_reason is not None:
                choice.finish_reason = output.finish_reason
                choice.stop_reason = output.stop_reason
                self.finish_reason_sent[i] = True
            chunk = DisaggChatCompletionStreamResponse(
                id=request_id,
                created=int(time.time()),
                object="chat.completion.chunk",
                choices=[choice],
                model=self.model,
            )
            chunk.usage = self._stream_usage_info(
                self.request, prompt_tokens, output.length
            )
            return chunk

    def create_final_stream_response(
        self,
        request_id: str,
        final_result: RequestOutput,
    ) -> DisaggChatCompletionStreamResponse:
        prompt_tokens = len(final_result.prompt_token_ids)
        completion_tokens = sum(output.length for output in final_result.outputs)
        final_usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        final_usage_chunk = DisaggChatCompletionStreamResponse(
            id=request_id,
            created=int(time.time()),
            object="chat.completion",
            choices=[],
            model=self.model,
            usage=final_usage,
        )
        return final_usage_chunk

    async def create_chat_response(
        self,
        request: ChatCompletionRequest,
        conversation: List[Dict[str, Any]],
        model: str,
        promise: RequestOutput,
    ) -> ChatCompletionResponse:
        await promise.aresult()
        choices: List[ChatCompletionResponseChoice] = []
        role = self._get_role(request)
        for output in promise.outputs:
            if request.tool_choice and isinstance(
                request.tool_choice, ChatCompletionNamedToolChoiceParam
            ):
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=[
                        ToolCall(
                            function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=output.text,
                            )
                        )
                    ],
                )
            else:
                message = ChatMessage(role=role, content=output.text)
            choice = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason,
            )

            if request.logprobs:
                choice.logprobs = self._create_logprobs(
                    output.token_ids, output.logprobs
                )
            choices.append(choice)

        if request.echo:
            last_msg_content = ""
            if (
                conversation
                and conversation[-1].get("content")
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"]
            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(promise.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in promise.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            model=model,
            choices=choices,
            usage=usage,
        )
        return response
