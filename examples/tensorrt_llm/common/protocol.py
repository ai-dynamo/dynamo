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
from tensorrt_llm.llmapi import SamplingParams
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
    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)


class Tokens(BaseModel):
    tokens: list[int]


class Request(BaseModel):
    prompt: str
    sampling_params: dict
    streaming: bool


class TRTLLMWorkerRequest(BaseModel):
    id: str
    prompt: str | None = None
    sampling_params: dict
    streaming: bool = True
    conversation: Optional[List[ConversationMessage]] = Field(default=None)
    tokens: Optional[Tokens] = Field(default=None)
    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)

    def to_sampling_params(self) -> SamplingParams:
        sampling_params = SamplingParams(
            frequency_penalty=self.sampling_params.get("frequency_penalty", 0.0),
            return_log_probs=self.sampling_params.get("logprobs", False),
            max_tokens=self.sampling_params.get("max_tokens", 16),
            n=self.sampling_params.get("n", 1),
            presence_penalty=self.sampling_params.get("presence_penalty", 0.0),
            seed=self.sampling_params.get("seed", None),
            stop=self.sampling_params.get("stop", None),
            temperature=self.sampling_params.get("temperature", 0.7),
            # chat-completion-sampling-params
            best_of=self.sampling_params.get("best_of", None),
            use_beam_search=self.sampling_params.get("use_beam_search", False),
            top_k=self.sampling_params.get("top_k", 0),
            top_p=self.sampling_params.get("top_p", 1.0),
            top_p_min=self.sampling_params.get("top_p_min", None),
            min_p=self.sampling_params.get("min_p", 0.0),
            repetition_penalty=self.sampling_params.get("repetition_penalty", 1.0),
            length_penalty=self.sampling_params.get("length_penalty", 1.0),
            early_stopping=self.sampling_params.get("early_stopping", False),
            stop_token_ids=self.sampling_params.get("stop_token_ids", []),
            include_stop_str_in_output=self.sampling_params.get(
                "include_stop_str_in_output", False
            ),
            ignore_eos=self.sampling_params.get("ignore_eos", False),
            min_tokens=self.sampling_params.get("min_tokens", 0),
            skip_special_tokens=self.sampling_params.get("skip_special_tokens", False),
            spaces_between_special_tokens=self.sampling_params.get(
                "spaces_between_special_tokens", False
            ),
            truncate_prompt_tokens=self.sampling_params.get(
                "truncate_prompt_tokens", None
            ),
            # chat-completion-extra-params
            add_special_tokens=self.sampling_params.get("add_special_tokens", False),
        )
        return sampling_params


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
