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
from typing import Any, List, Literal, Optional, Union

import connect
import msgspec
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_core import core_schema
from typing_extensions import NotRequired
from vllm.inputs.data import TokensPrompt
from vllm.outputs import CompletionOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import PromptLogprobs, RequestMetrics


class Request(BaseModel):
    prompt: str
    sampling_params: dict


class Tokens(BaseModel):
    tokens: list[int]


class PrefillRequest(Request):
    request_id: str


class Response(BaseModel):
    text: str


class PrefillResponse(BaseModel):
    prefilled: bool


# Hack to override the type of multi_modal_data in TokensPrompt
# as pydantic doesn't understand generic types
# TokensPrompt is defined here: https://github.com/vllm-project/vllm/blob/a4c402a756fa3213caf9d2cde0e4ceb2d57727f2/vllm/inputs/data.py#L38
# multi_modal_data is defined here: https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L103
# ModalityData is defined here: https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L80
class PatchedTokensPrompt(TokensPrompt):
    multi_modal_data: NotRequired[Optional[Any]]  # type: ignore


# Monkey-patch the SamplingParams type to add a dummy core schema so pydantic can validate it
# Sampling params is a mspspec struct
# SamplingParams is defined here: https://github.com/vllm-project/vllm/blob/a4c402a756fa3213caf9d2cde0e4ceb2d57727f2/vllm/sampling_params.py#L88

SamplingParams.__get_pydantic_core_schema__ = classmethod(
    lambda cls, source, handler: core_schema.any_schema()
)


class vLLMGenerateRequest(BaseModel):
    """
    Serializable class of all the fields vLLM engine requires for inference
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    engine_prompt: PatchedTokensPrompt
    sampling_params: SamplingParams
    request_id: str
    prefix_hit_rate: Optional[float] = 0.0

    @field_validator("sampling_params", mode="before")
    @classmethod
    def parse_sampling_params(cls, v: Any) -> SamplingParams:
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, dict):
            return SamplingParams(**v)
        return v

    model_config = ConfigDict(
        json_encoders={SamplingParams: lambda v: msgspec.json.encode(v)}
    )


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageURLDetail(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURLDetail


MessageContent = Union[TextContent, ImageContent]


class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: List[MessageContent]


class MultiModalRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = True


class vLLMMultimodalRequest(vLLMGenerateRequest):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_url: str


class EncodeRequest(BaseModel):
    """
    Serializable class of all the fields vLLM engine requires for inference
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_url: str
    request_id: str
    serialized_request: Optional[connect.SerializedRequest] = None


class EncodeResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request_id: str


class MyRequestOutput(BaseModel):
    """
    RequestOutput from vLLM is not serializable by default
    https://github.com/vllm-project/vllm/blob/a4c402a756fa3213caf9d2cde0e4ceb2d57727f2/vllm/outputs.py#L85

    This class is used to serialize the RequestOutput and any recursively defined types
    We can do this because PromptLogprobs, RequestMetrics, and CompletionOutput are all serializable dataclasses
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    request_id: str
    prompt: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[PromptLogprobs] = None
    outputs: List[CompletionOutput]
    finished: bool
    metrics: Optional[RequestMetrics] = None
    # lora_request: Optional[LoRARequest] = None
    # encoder_prompt: Optional[str] = None
    # encoder_prompt_token_ids: Optional[List[int]] = None
    # num_cached_tokens: Optional[int] = None
    # multi_modal_placeholders: Optional[MultiModalPlaceholderDict] = None
