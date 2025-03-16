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
import time
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    List,
    Tuple,
    TypedDict,
    Union,
)

from common.protocol import (
    DisaggCompletionResponseStreamChoice,
    DisaggCompletionStreamResponse,
    DisaggregatedTypeConverter,
)
from openai.types.chat import ChatCompletionMessageParam
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from transformers import AutoTokenizer

logger.set_level("debug")


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
    def __init__(self, model: str, tokenizer: AutoTokenizer):
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
                logprob=max(logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            content.append(first_logprob)
        chat_logprobs = ChatCompletionLogProbs(content=content)
        return chat_logprobs

    async def _chat_stream_generator(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        conversation: List[Dict[str, Any]],
        promise: RequestOutput,
    ) -> AsyncGenerator[str, None]:
        first_iteration = True
        num_choices = 1 if request.n is None else request.n
        finish_reason_sent = [False] * num_choices
        role = self._get_role(request)

        def yield_first_chat(
            num_tokens: int, role: str | None = None, content: str | None = None
        ):
            for i in range(num_choices):
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(role=role, content=content),
                    finish_reason=None,
                )
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=int(time.time()),
                    object="chat.completion.chunk",
                    choices=[choice_data],
                    model=self.model,
                )
                chunk.usage = self._stream_usage_info(request, num_tokens, 0)

                data = chunk.model_dump_json(exclude_unset=True)
                return data

        async for res in promise:
            prompt_tokens = len(res.prompt_token_ids)
            if first_iteration:
                yield f"data: {yield_first_chat(prompt_tokens, role=role)} \n\n"

                if request.echo:
                    last_msg_content = ""
                    if (
                        conversation
                        and conversation[-1].get("content")
                        and conversation[-1].get("role") == role
                    ):
                        last_msg_content = conversation[-1]["content"]

                    if last_msg_content:
                        yield f"data: {yield_first_chat(prompt_tokens, content=last_msg_content)}\n\n"
            first_iteration = False

            for output in res.outputs:
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
                    delta_message = DeltaMessage(content=delta_text)

                choice = ChatCompletionResponseStreamChoice(
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
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=int(time.time()),
                    object="chat.completion.chunk",
                    choices=[choice],
                    model=self.model,
                )
                chunk.usage = self._stream_usage_info(
                    request, prompt_tokens, output.length
                )
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"

        if request.stream_options and request.stream_options.include_usage:
            completion_tokens = sum(output.length for output in promise.outputs)
            final_usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            final_usage_chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=int(time.time()),
                object="chat.completion",
                choices=[],
                model=self.model,
                usage=final_usage,
            )
            final_usage_data = final_usage_chunk.model_dump_json()
            yield f"data: {final_usage_data}\n\n"
        yield "data: [DONE]\n\n"

    async def stream_response(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        conversation: List[Dict[str, Any]],
        promise: RequestOutput,
    ) -> AsyncGenerator[str, None]:
        assert request.stream, "Only stream is supported"
        async for raw_response in self._chat_stream_generator(
            request, request_id, conversation, promise
        ):
            if raw_response.startswith("data: [DONE]"):
                break
            response = json.loads(raw_response.lstrip("data: "))
            yield response

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


def merge_promises(
    promises: List[RequestOutput],
) -> AsyncIterator[Tuple[int, RequestOutput]]:
    outputs = asyncio.Queue()  # type: ignore
    finished = [False] * len(promises)

    async def producer(i: int, promise: RequestOutput):
        async for output in promise:
            await outputs.put((i, output))
        finished[i] = True

    _tasks = [
        asyncio.create_task(producer(i, promise)) for i, promise in enumerate(promises)
    ]

    async def consumer():
        while not all(finished) or not outputs.empty():
            item = await outputs.get()
            yield item
        await asyncio.gather(*_tasks)

    return consumer()


class CompletionsProcessor:
    def __init__(self, model: str):
        self.model = model

    def _post_process(self, request, prompt_idx, num_choices, requst_output):
        res = []
        echoed = [False] * num_choices
        num_repsonse_per_request = 1 if request.n is None else request.n
        for gen_idx, output in enumerate(requst_output.outputs):
            response_idx = prompt_idx * num_repsonse_per_request + gen_idx
            delta_text = output.text_diff
            if request.echo and not echoed[response_idx]:
                delta_text = request.prompt + delta_text
                echoed[response_idx] = True
            choice = DisaggCompletionResponseStreamChoice(
                index=response_idx,
                text=delta_text,
                stop_reason=output.stop_reason,
                finish_reason=output.finish_reason,
            )
            if output.disaggregated_params is not None:
                choice.disaggregated_params = (
                    DisaggregatedTypeConverter.to_oai_disaggregated_params(
                        output.disaggregated_params
                    )
                )
            chunk = DisaggCompletionStreamResponse(
                model=self.model,
                choices=[choice],
            )
            res.append(chunk.model_dump_json())
        return res

    async def create_completion_generator(
        self,
        request: CompletionRequest,
        generator: AsyncIterator[Tuple[int, RequestOutput]],
        num_choices: int,
    ):
        async for prompt_idx, requst_output in generator:
            pp_res = self._post_process(request, prompt_idx, num_choices, requst_output)
            for _p in pp_res:
                yield _p

    async def create_completion_response(
        self,
        request: CompletionRequest,
        generator: AsyncIterator[Tuple[int, RequestOutput]],
        num_choices: int,
    ):
        choices = [None] * num_choices
        num_repsonse_per_request = 1 if request.n is None else request.n
        num_prompt_tokens = num_gen_tokens = 0
        async for prompt_idx, request_output in generator:
            num_prompt_tokens += len(request_output.prompt_token_ids)
            for gen_idx, output in enumerate(request_output.outputs):
                num_gen_tokens += len(output.token_ids)
                output_text = output.text
                if request.echo:
                    output_text = request_output.prompt + output_text
                idx = prompt_idx * num_repsonse_per_request + gen_idx

                disaggregated_params = CompletionResponseChoice.to_disaggregated_params(
                    output.disaggregated_params
                )
                choice = CompletionResponseChoice(
                    index=idx,
                    text=output_text,
                    stop_reason=output.stop_reason,
                    finish_reason=output.finish_reason,
                    disaggregated_params=disaggregated_params,
                )
                choices[idx] = choice

        usage_info = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_gen_tokens,
            total_tokens=num_gen_tokens + num_prompt_tokens,
        )
        response = CompletionResponse(
            model=self.model,
            choices=choices,
            usage=usage_info,
        )
        return response
