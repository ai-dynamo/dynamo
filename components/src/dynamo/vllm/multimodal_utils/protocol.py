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

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_serializer,
    field_validator,
)
from typing_extensions import NotRequired
from vllm.inputs import TokensPrompt  # noqa: F401
from vllm.logprobs import PromptLogprobs
from vllm.outputs import CompletionOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import RequestStateStats

from dynamo.common.multimodal.embedding_transfer import TransferRequest


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
# TokensPrompt is exported from vllm.inputs and implemented in vllm/inputs/llm.py.
# multi_modal_data is defined here: https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L103
# ModalityData is defined here: https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L80
class PatchedTokensPrompt(TokensPrompt):
    multi_modal_data: NotRequired[Optional[Any]]  # type: ignore


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
    def parse_sampling_params(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, dict):
            # vLLM's schema ignores extra keys. Keep rejecting misspelled
            # public options, while accepting private fields from older workers.
            for key in v:
                if (
                    not key.startswith("_")
                    and key not in SamplingParams.__struct_fields__
                ):
                    raise ValueError(f"Unknown sampling parameter: {key}")
        return v

    @field_serializer("sampling_params", mode="wrap")
    def serialize_sampling_params(
        self, value: SamplingParams, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        params = handler(value)
        # vLLM filters private SamplingParams fields at the top level only.
        # Backend selection is worker-local. Leave user JSON schemas intact.
        structured = params.get("structured_outputs")
        if isinstance(structured, dict):
            params["structured_outputs"] = {
                key: item for key, item in structured.items() if not key.startswith("_")
            }
        return params

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
    stream_options: Optional[dict] = None


class MultiModalInput(BaseModel):
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    # Frontend-decoded RGB descriptor, mutually exclusive with image_url.
    image_decoded: Optional[dict[str, Any]] = None


class MultiModalGroup(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    multimodal_input: Optional[MultiModalInput] = Field(default_factory=MultiModalInput)
    image_grid_thw: Optional[List[Any]] = None
    embeddings_shape: Optional[
        Union[Tuple[int, int, int], Tuple[int, int, int, int]]
    ] = None
    serialized_request: Optional[TransferRequest] = None
    loaded_embedding: Optional[torch.Tensor] = Field(default=None, exclude=True)


class vLLMMultimodalRequest(vLLMGenerateRequest):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # LoRA adapter name (matches the name used in load_lora)
    model: Optional[str] = None
    # Decode-only worker can have None for multimodal_inputs
    multimodal_inputs: Optional[List[MultiModalGroup]] = Field(default_factory=list)
    # Add these fields for Qwen VL (mRoPE) decode-only worker
    image_grid_thw: Optional[List[List[int]]] = None
    embeddings_shape: Optional[List[int]] = None
    # Opaque frontend-derived scope for encoded/decoded image caches.
    image_cache_scope: Optional[str] = None


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
