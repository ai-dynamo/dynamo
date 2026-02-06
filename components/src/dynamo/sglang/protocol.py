# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

import dynamo.nixl_connect as connect

TokenIdType = int


# ============================================================================
# Standard LLM Protocol Types
# ============================================================================
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
    stop_conditions: StopConditions
    sampling_options: SamplingOptions
    eos_token_ids: List[TokenIdType] = Field(default_factory=list)
    mdc_sum: Optional[str] = None
    annotations: List[str] = Field(default_factory=list)


EmbeddingInput = Union[str, List[str], List[int], List[List[int]]]


class EmbeddingRequest(BaseModel):
    model: str
    input: EmbeddingInput
    user: Optional[str] = None
    dimensions: Optional[
        int
    ] = None  # only supported in text-embedding-3 and later models from OpenAI


class DisaggPreprocessedRequest(BaseModel):
    request: Union[PreprocessedRequest, ChatCompletionRequest]
    sampling_params: dict
    data_parallel_rank: Optional[int] = None


# ============================================================================
# Multimodal Protocol Types
# ============================================================================


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


# ============================================================================
# Image diffusion Protocol Types
# ============================================================================


class NvExt(BaseModel):
    """NVIDIA extensions for image generation"""

    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    annotations: Optional[list[str]] = None


class CreateImageRequest(BaseModel):
    """OpenAI /v1/images/generations compatible request"""

    prompt: str
    model: str  # e.g. "stabilityai/stable-diffusion-3.5-medium"
    n: int = 1  # Number of images
    size: Optional[str] = "1024x1024"  # "WxH" format
    quality: Optional[str] = "standard"  # standard, hd
    response_format: Optional[str] = "url"  # url or b64_json
    user: Optional[str] = None

    # NVIDIA extensions nested under nvext
    nvext: Optional[NvExt] = None


class ImageData(BaseModel):
    url: Optional[str] = None  # S3 URL
    b64_json: Optional[str] = None  # Base64 encoded
    revised_prompt: Optional[str] = None


class ImagesResponse(BaseModel):
    """OpenAI-compatible response"""

    created: int  # Unix timestamp
    data: list[ImageData]
