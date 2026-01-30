# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, List, Literal, Optional, Tuple, Union

import msgspec
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from pydantic_core import core_schema
from typing_extensions import NotRequired
from vllm.inputs.data import TokensPrompt
from vllm.logprobs import PromptLogprobs
from vllm.multimodal.inputs import MultiModalUUIDDict  # noqa: F401
from vllm.outputs import CompletionOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import RequestStateStats

import dynamo.nixl_connect as connect


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
            # Workaround for vLLM SamplingParams serialization/deserialization issue.
            #
            # Problem: When SamplingParams is serialized via msgspec.json.encode(),
            # Python sets are converted to JSON arrays (lists). The serialized dict
            # includes private fields like _all_stop_token_ids. Upon deserialization,
            # passing this dict to SamplingParams(**dict) causes __post_init__ to fail
            # because it expects _all_stop_token_ids to be a set (to call .update()),
            # but it's now a list.
            #
            # Solution: Filter out private fields (starting with '_') which are
            # internal state that should be computed by __post_init__, not passed
            # from serialized data. Public fields like stop_token_ids are preserved.
            v = {k: val for k, val in v.items() if not k.startswith("_")}
            return SamplingParams(**v)
        return v

    @field_serializer("sampling_params")
    def serialize_sampling_params(self, value: SamplingParams) -> dict[str, Any]:
        """Serialize SamplingParams, filtering out private fields.

        This is the primary fix for the set→list serialization issue.
        Private fields like _all_stop_token_ids are filtered out here
        so they never get sent over the wire.
        """
        serialized = json.loads(msgspec.json.encode(value))
        return {k: v for k, v in serialized.items() if not k.startswith("_")}

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageURLDetail(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURLDetail


class VideoURLDetail(BaseModel):
    url: str


class VideoContent(BaseModel):
    type: Literal["video_url"]
    video_url: VideoURLDetail


MessageContent = Union[TextContent, ImageContent, VideoContent]


class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: List[MessageContent]


class MultiModalRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = True


class MultiModalInput(BaseModel):
    image_url: Optional[str] = None
    video_url: Optional[str] = None


class MultiModalGroup(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    multimodal_input: Optional[MultiModalInput] = Field(default_factory=MultiModalInput)
    image_grid_thw: Optional[List[Any]] = None
    embeddings_shape: Optional[
        Union[Tuple[int, int, int], Tuple[int, int, int, int]]
    ] = None
    serialized_request: Optional[connect.RdmaMetadata | str] = None


class vLLMMultimodalRequest(vLLMGenerateRequest):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    multimodal_inputs: List[MultiModalGroup] = Field(default_factory=list)


class VLLMNativeEncoderRequest(BaseModel):
    """Request for vLLM-native encoder worker using ECConnector"""

    request_id: str
    token_ids: List[
        int
    ]  # Pre-tokenized prompt with placeholder tokens (for TokensPrompt)
    multimodal_inputs: List[MultiModalGroup] = Field(default_factory=list)
    modality: Optional[
        Literal["image", "video", "audio"]
    ] = None  # Can be inferred from inputs


class VLLMNativeEncoderResponse(BaseModel):
    """Response from vLLM-native encoder worker (ECConnector mode)"""

    request_id: str
    mm_hash: str  # vLLM's multimodal hash identifier
    modality: str  # "image", "video", "audio"
    connector_metadata: dict[str, Any]  # ECConnector config info for PD workers


class MyRequestOutput(BaseModel):
    """
    RequestOutput from vLLM is not serializable by default
    https://github.com/vllm-project/vllm/blob/a4c402a756fa3213caf9d2cde0e4ceb2d57727f2/vllm/outputs.py#L85

    This class is used to serialize the RequestOutput and any recursively defined types
    We can do this because PromptLogprobs, RequestStateStats, and CompletionOutput are all serializable dataclasses
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    prompt: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[PromptLogprobs] = None
    outputs: List[CompletionOutput]
    finished: bool
    metrics: Optional[RequestStateStats] = None
    kv_transfer_params: Optional[dict[str, Any]] = None
    # lora_request: Optional[LoRARequest] = None
    # encoder_prompt: Optional[str] = None
    # encoder_prompt_token_ids: Optional[List[int]] = None
    # num_cached_tokens: Optional[int] = None
    # multi_modal_placeholders: Optional[MultiModalPlaceholderDict] = None
