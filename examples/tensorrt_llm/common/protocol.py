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

import base64
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Union

import torch
from common.utils import ConversationMessage
from pydantic import BaseModel, ConfigDict, Field
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    CompletionRequest,
    CompletionResponseStreamChoice,
    DisaggregatedParams,
    UsageInfo,
)


# The max_tokens is being deprecated in favor of max_completion_tokens.
# However, TRTLLM protocol might still refer it as max_tokens.
class DynamoTRTLLMCompletionRequest(CompletionRequest):
    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4().hex)}")
    max_completion_tokens: Optional[int] = None


class DynamoTRTLLMChatCompletionRequest(ChatCompletionRequest):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)


class Tokens(BaseModel):
    tokens: list[int]


class Request(BaseModel):
    prompt: str
    sampling_params: dict
    streaming: bool


class TRTLLMWorkerRequest(BaseModel):
    model: str
    id: str
    prompt: str | None = None
    sampling_params: dict
    streaming: bool = True
    conversation: Optional[List[ConversationMessage]] = Field(default=None)
    tokens: Optional[Tokens] = Field(default=None)
    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)


@dataclass
class TRTLLMWorkerResponseOutput:
    index: int
    text: str
    token_ids: list[int]
    logprobs: Optional[List[float]] = None
    cumulative_logprob: Optional[float] = None
    finish_reason: Optional[Literal["stop", "length", "timeout", "cancelled"]] = None
    stop_reason: Optional[Union[int, str]] = None
    generation_logits: Optional[torch.Tensor] = None
    disaggregated_params: Optional[DisaggregatedParams] = None

    _last_text_len: int = field(default=0)
    _last_token_ids_len: int = field(default=0)
    _last_logprobs_len: int = field(default=0)
    _incremental_states: Optional[dict] = field(default=None)
    _postprocess_result: Optional[Any] = field(default=None)

    text_diff: str = field(default="")
    length: int = field(default=0)

    def __post_init__(self):
        self.text_diff = self.text[self._last_text_len :]
        self.length = len(self.token_ids)


class TRTLLMWorkerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    request_id: str
    prompt: str | None = None
    prompt_token_ids: list[int]
    outputs: list[dict]
    finished: bool
    # TODO
    # prompt_logprobs: list[float]


class DisaggregatedTypeConverter:
    @staticmethod
    def to_llm_disaggregated_params(
        disaggregated_params: DisaggregatedParams,
    ) -> LlmDisaggregatedParams:
        if disaggregated_params is None:
            return None
        else:
            opaque_state = (
                base64.b64decode(disaggregated_params.encoded_opaque_state)
                if disaggregated_params.encoded_opaque_state is not None
                else None
            )

            return LlmDisaggregatedParams(
                request_type=disaggregated_params.request_type,
                first_gen_tokens=disaggregated_params.first_gen_tokens,
                ctx_request_id=disaggregated_params.ctx_request_id,
                opaque_state=opaque_state,
            )

    @staticmethod
    def to_oai_disaggregated_params(
        tllm_disagg_params: LlmDisaggregatedParams,
    ) -> DisaggregatedParams:
        if tllm_disagg_params is None:
            return None
        else:
            encoded_opaque_state = (
                base64.b64encode(tllm_disagg_params.opaque_state).decode("utf-8")
                if tllm_disagg_params is not None
                else None
            )
            return DisaggregatedParams(
                request_type=tllm_disagg_params.request_type,
                first_gen_tokens=tllm_disagg_params.first_gen_tokens,
                ctx_request_id=tllm_disagg_params.ctx_request_id,
                encoded_opaque_state=encoded_opaque_state,
            )


# Chat Completions


class DynamoTRTLLMChatCompletionResponseStreamChoice(
    ChatCompletionResponseStreamChoice
):
    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)


class DynamoTRTLLMChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[DynamoTRTLLMChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


## Completions


class DynamoTRTLLMCompletionResponseStreamChoice(CompletionResponseStreamChoice):
    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)


class DynamoTRTLLMCompletionStreamResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4().hex)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[DynamoTRTLLMCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)
