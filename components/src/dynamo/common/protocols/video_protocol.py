# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generated from lib/llm/src/protocols/openai/media_schemas/video.json by
# scripts/generate_media_protocols.py. Do not edit.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, conint


class VideoData(BaseModel):
    """
    Video data in response
    """

    output_format: str
    """
    Actual container format of this video: "mp4", "webm", "gif"
    """
    url: str | None = None
    """
    URL of the generated video (if response_format is "url")
    """
    b64_json: str | None = None
    """
    Base64-encoded video (if response_format is "b64_json")
    """
    fps: int | None = None
    """
    Actual video frame rate when reported by the model
    """
    audio_sample_rate: int | None = None
    """
    Muxed audio sample rate when the generated video contains audio
    """


class VideoNvExt(BaseModel):
    """
    NVIDIA extensions to the OpenAI Videos API
    """

    annotations: list[str] | None = None
    """
    Annotations
    User requests triggers which result in the request issue back out-of-band information in the SSE
    stream using the `event:` field.
    """
    fps: conint(ge=1) | None = None
    """
    Frames per second, at least 1 (default: 24)
    """
    num_frames: conint(ge=1) | None = None
    """
    Number of frames to generate, at least 1 (overrides fps * seconds if set)
    """
    negative_prompt: str | None = None
    """
    A text description of the undesired video content.
    """
    num_inference_steps: int | None = None
    """
    The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference.
    """
    guidance_scale: float | None = None
    """
    The CFG scale. Higher values usually lead to more coherent output.
    """
    seed: int | None = None
    """
    The seed for the random number generator.
    """
    boundary_ratio: float | None = None
    """
    MoE expert switching boundary as a fraction of the denoising schedule (vLLM-Omni I2V).
    """
    guidance_scale_2: float | None = None
    """
    CFG scale for the low-noise expert (vLLM-Omni I2V dual-guidance).
    """


class NvCreateVideoRequest(BaseModel):
    """
    Request for video generation (/v1/videos endpoint)
    """

    prompt: str
    """
    The text prompt for video generation
    """
    model: str
    """
    The model to use for video generation
    """
    input_reference: str | None = None
    """
    Optional image reference that guides generation (for I2V)
    """
    seconds: conint(ge=1) | None = None
    """
    Clip duration in seconds, at least 1. The frontend rejects a value
    outside the range, so a worker never sees one.
    """
    size: str | None = None
    """
    Video size in WxH format (default: "832x480")
    """
    user: str | None = None
    """
    Optional user identifier
    """
    response_format: Literal['url', 'b64_json'] | None = None
    output_format: str | None = None
    """
    Output container format: "mp4", "webm", "gif", etc.
    This field is used as model hint and the model may not
    return the requested format, should check with output_format
    field in the response data.
    """
    stream: bool | None = None
    """
    Whether to stream the video generation (default: false)
    """
    nvext: VideoNvExt | None = None
    extra_args: dict[str, Any] | None = None
    """
    Worker-boundary passthrough. The frontend nests unknown top-level request fields (an OpenAI client's extra_body) under the "media_passthrough" key.
    """


class NvVideosResponse(BaseModel):
    """
    Response structure for video generation
    """

    id: str
    """
    Unique identifier for the response
    """
    object: str = 'video'
    """
    Object type (always "video")
    """
    model: str
    """
    Model used for generation
    """
    status: str = 'completed'
    """
    Status of the generation ("completed", "failed", etc.)
    """
    progress: int = 100
    """
    Progress percentage (0-100)
    """
    created: int
    """
    Unix timestamp of creation
    """
    data: list[VideoData] = Field([], validate_default=True)
    """
    Generated video data
    """
    error: str | None = None
    """
    Error message if generation failed
    """
    inference_time_s: float | None = None
    """
    Inference time in seconds
    """
