# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

import dynamo.nixl_connect as connect

TokenIdType = int


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
    stream: Optional[bool] = False


class MultiModalInput(BaseModel):
    image_url: Optional[str] = None
    video_url: Optional[str] = None


# TODO: move these to common for all LLMs once we adopt dynamo-run
# derived from lib/llm/src/protocols/common/preprocessor.rs
class StopConditions(BaseModel):
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stop_token_ids_hidden: Optional[List[TokenIdType]] = None
    min_tokens: Optional[int] = None
    ignore_eos: Optional[bool] = None


class SamplingOptions(BaseModel):
    n: Optional[int] = None
    best_of: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    use_beam_search: Optional[bool] = None
    length_penalty: Optional[float] = None
    seed: Optional[int] = None


class PreprocessedRequest(BaseModel):
    token_ids: List[TokenIdType]
    batch_token_ids: Optional[List[List[TokenIdType]]] = None
    stop_conditions: StopConditions
    sampling_options: SamplingOptions
    eos_token_ids: List[TokenIdType] = Field(default_factory=list)
    mdc_sum: Optional[str] = None
    annotations: List[str] = Field(default_factory=list)


class SglangMultimodalRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    request: PreprocessedRequest
    multimodal_input: Optional[MultiModalInput] = Field(default_factory=MultiModalInput)
    image_grid_thw: Optional[List[Any]] = None
    embeddings_shape: Optional[
        Union[Tuple[int, int, int], Tuple[int, int, int, int]]
    ] = None
    serialized_request: Optional[connect.RdmaMetadata] = None


class DisaggSglangMultimodalRequest(BaseModel):
    request: SglangMultimodalRequest
    sampling_params: dict
    data_parallel_rank: Optional[int] = None
