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
from typing import Any, List, Literal, Optional, TypeAlias, Union

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
    nvext: Optional[dict] = Field(default=None)


class DynamoTRTLLMChatCompletionRequest(ChatCompletionRequest):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)
    nvext: Optional[dict] = Field(default=None)


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


@dataclass(slots=True)
class Logprob:
    """Holds logprob and vocab rank for a token."""

    logprob: float
    rank: Optional[int] = None


# List of token_id_to_Logprob dict for prompt or generation texts
TokenLogprobs: TypeAlias = list[dict[int, Logprob]]


@dataclass
class TRTLLMWorkerResponseOutput:
    index: int
    text: str = ""
    token_ids: Optional[List[int]] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[TokenLogprobs] = field(default_factory=list)
    prompt_logprobs: Optional[TokenLogprobs] = field(default_factory=list)
    finish_reason: Optional[Literal["stop", "length", "timeout", "cancelled"]] = None
    stop_reason: Optional[Union[int, str]] = None
    generation_logits: Optional[torch.Tensor] = None
    disaggregated_params: Optional[DisaggregatedParams] = None

    # hidden fields for tracking the diffs
    _last_text_len: int = field(default=0, init=True, repr=False)
    _last_token_ids_len: int = field(default=0, init=True, repr=False)
    _last_logprobs_len: int = field(default=0, init=True, repr=False)
    _incremental_states: Optional[dict] = field(default=None, init=True, repr=False)
    # the result of result_handler passed to postprocess workers
    _postprocess_result: Any = None

    @property
    def length(self) -> int:
        return 0 if self.token_ids is None else len(self.token_ids)

    @property
    def text_diff(self) -> str:
        return self.text[self._last_text_len :]

    @property
    def token_ids_diff(self) -> List[int]:
        return (
            [] if self.token_ids is None else self.token_ids[self._last_token_ids_len :]
        )

    # Ignoring the mypy error here as this is copied from TensorRT-LLM project.
    # https://github.com/NVIDIA/TensorRT-LLM/blob/19c6e68bec891b66146a09647ee7b70230ef5f67/tensorrt_llm/executor/result.py#L68
    # TODO: Work with the TensorRT-LLM team to get this fixed.
    @property
    def logprobs_diff(self) -> List[float]:  # type: ignore
        return [] if self.logprobs is None else self.logprobs[self._last_logprobs_len :]  # type: ignore


class TRTLLMWorkerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    request_id: str
    prompt: str | None = None
    prompt_token_ids: list[int]
    outputs: list[dict]
    finished: bool


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
