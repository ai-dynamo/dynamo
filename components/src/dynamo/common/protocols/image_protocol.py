# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generated from lib/llm/src/protocols/openai/media_schemas/image.json by
# scripts/generate_media_protocols.py. Do not edit.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, StrictBool, StrictFloat, StrictInt, StrictStr, conint


class ImageData(BaseModel):
    """
    One generated image. The worker sets one of `url` and `b64_json`.
    """

    url: StrictStr | None = None
    """
    URL of the generated image (if response_format is "url")
    """
    b64_json: StrictStr | None = None
    """
    Base64-encoded image (if response_format is "b64_json")
    """
    revised_prompt: StrictStr | None = None
    """
    The prompt the model used, when it rewrote the original prompt
    """


class ImageNvExt(BaseModel):
    """
    NVIDIA extensions to the OpenAI Images API
    """

    annotations: list[StrictStr] | None = None
    """
    Annotations
    User requests triggers which result in the request issue back out-of-band information in the SSE
    stream using the `event:` field.
    """
    negative_prompt: StrictStr | None = None
    """
    A text description of the undesired image(s).
    """
    num_inference_steps: conint(ge=1, strict=True) | None = None
    """
    The number of denoising steps, at least 1. More denoising steps usually lead to a higher quality image at the expense of slower inference.
    """
    guidance_scale: StrictFloat | None = None
    """
    The CFG scale. Higher values usually lead to more coherent images.
    """
    seed: StrictInt | None = None
    """
    The seed for the random number generator.
    i64 to match PyTorch's torch.manual_seed() accepted range.
    """


class ImageTokenDetails(BaseModel):
    """
    A token count split by kind.
    """

    text_tokens: conint(ge=0, strict=True)
    image_tokens: conint(ge=0, strict=True)


class ImageUsage(BaseModel):
    """
    Token usage of an image generation. Keeps the wire format of the OpenAI
    `ImageGenUsage`, including the singular `output_token_details`.
    """

    input_tokens: conint(ge=0, strict=True)
    """
    Tokens (image and text) in the input prompt
    """
    total_tokens: conint(ge=0, strict=True)
    """
    Tokens (image and text) used for the generation
    """
    output_tokens: conint(ge=0, strict=True)
    """
    Tokens the model generated
    """
    output_token_details: ImageTokenDetails | None = None
    input_tokens_details: ImageTokenDetails
    """
    The input tokens by kind
    """


class NvCreateImageRequest(BaseModel):
    """
    Request for image generation (/v1/images/generations and /v1/images/edits).

    The OpenAI fields keep the wire format of the OpenAI `CreateImageRequest`;
    the module documentation says why they are declared here and not wrapped.
    `model` and `size` are free text, because the OpenAI type accepts any
    string there.
    """

    prompt: StrictStr
    model: StrictStr | None = None
    n: conint(ge=1, le=10, strict=True) | None = None
    """
    Number of images to generate, 1 to 10 as the OpenAI API documents.
    The frontend rejects a value outside the range, so a worker never
    sees one and needs no range check of its own.
    """
    quality: Literal['standard', 'hd', 'high', 'medium', 'low', 'xhigh', 'max', 'auto'] | None = (
        None
    )
    response_format: Literal['url', 'b64_json'] | None = None
    output_format: Literal['png', 'jpeg', 'webp'] | None = None
    output_compression: conint(ge=0, le=100, strict=True) | None = None
    """
    Compression level of jpeg and webp output, 0 to 100 percent as the
    OpenAI API documents. The frontend rejects a value outside the range,
    so a worker never sees one.
    """
    stream: StrictBool | None = None
    partial_images: conint(ge=0, le=3, strict=True) | None = None
    """
    Number of partial images to stream before the final image, 0 to 3 as
    the OpenAI API documents. The frontend rejects a value outside the
    range, so a worker never sees one.
    """
    size: StrictStr | None = None
    """
    Image size in WxH format, or "auto"
    """
    moderation: Literal['auto', 'low'] | None = None
    background: Literal['auto', 'transparent', 'opaque'] | None = None
    style: Literal['vivid', 'natural'] | None = None
    user: StrictStr | None = None
    input_reference: StrictStr | None = None
    """
    Optional image reference that guides generation (for I2I/TI2I).
    """
    nvext: ImageNvExt | None = None
    extra_args: dict[str, Any] | None = None
    """
    Worker-boundary passthrough. The frontend nests unknown top-level request fields (an OpenAI client's extra_body) under the "media_passthrough" key.
    """


class NvImagesResponse(BaseModel):
    """
    Response for image generation.

    Keeps the wire format of the OpenAI `ImagesResponse`, which writes every
    absent optional field as null.
    """

    created: conint(ge=0, strict=True)
    """
    Unix timestamp of creation
    """
    data: list[ImageData]
    background: Literal['transparent', 'opaque'] | None = None
    output_format: Literal['png', 'jpeg', 'webp'] | None = None
    size: StrictStr | None = None
    """
    Image size in WxH format
    """
    quality: Literal['standard', 'hd', 'high', 'medium', 'low', 'xhigh', 'max', 'auto'] | None = (
        None
    )
    usage: ImageUsage | None = None
