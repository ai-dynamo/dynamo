# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image utilities for image diffusion.

Provides the image output options shared by the SGLang and TensorRT-LLM image
handlers: checking `output_format`, `quality`, and `background`, and encoding
images as PNG, JPEG, or WebP. The vLLM-Omni image handler has its own copy in
`dynamo.vllm.omni.output_formatter`, which the other workers cannot import.
"""

import io
from typing import Iterable, Literal, Optional, Union, cast

import numpy as np
from PIL import Image

from dynamo.llm.exceptions import InvalidArgument

# The values of `NvImagesResponse.output_format`.
ImageOutputFormat = Literal["png", "jpeg", "webp"]

# output_format -> (Pillow encoder, file extension, MIME type)
IMAGE_OUTPUT_FORMATS: dict[str, tuple[str, str, str]] = {
    "png": ("PNG", "png", "image/png"),
    "jpeg": ("JPEG", "jpg", "image/jpeg"),
    "webp": ("WEBP", "webp", "image/webp"),
}


def normalize_image_output_options(
    output_format: Optional[str],
    *,
    quality: Optional[str],
    background: Optional[str],
    handler_name: str,
) -> ImageOutputFormat:
    """Return the requested encoding, or raise `InvalidArgument` (HTTP 400).

    `output_format` defaults to `png` and is case-insensitive. The diffusion
    pipelines behind these handlers emit opaque RGB with no quality control, so
    any `quality` or `background` is rejected rather than silently ignored, as
    the vLLM-Omni image handler does.
    """
    if quality is not None:
        raise InvalidArgument(f"quality is not supported by the {handler_name}")
    if background is not None:
        raise InvalidArgument(f"background is not supported by the {handler_name}")

    if output_format is None:
        return "png"
    normalized = output_format.lower()
    if normalized not in IMAGE_OUTPUT_FORMATS:
        raise InvalidArgument(
            f"Unsupported output_format: {output_format!r}; "
            "expected 'png', 'jpeg', or 'webp'"
        )
    return cast(ImageOutputFormat, normalized)


def encode_image_bytes(
    image: Union[Image.Image, np.ndarray],
    output_format: ImageOutputFormat,
) -> bytes:
    """Encode a PIL image or a uint8 (H, W, C) array as `output_format`.

    JPEG has no alpha channel, so a transparent image is flattened onto white.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    encoder = IMAGE_OUTPUT_FORMATS[output_format][0]
    if encoder == "JPEG":
        has_transparency = "transparency" in image.info
        if image.mode not in ("L", "RGB") or has_transparency:
            if "A" in image.getbands() or has_transparency:
                rgba = image.convert("RGBA")
                flattened = Image.new("RGB", rgba.size, "white")
                flattened.paste(rgba, mask=rgba.getchannel("A"))
                image = flattened
            else:
                image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format=encoder)
    return buf.getvalue()


def uniform_image_size(sizes: Iterable[tuple[int, int]]) -> Optional[str]:
    """Return `WxH` when every (width, height) is the same positive size.

    Returns None for no images, mixed sizes, or a non-positive dimension, so
    the response reports a size only when it holds for every image.
    """
    distinct = set(sizes)
    if len(distinct) != 1:
        return None
    ((width, height),) = distinct
    if width <= 0 or height <= 0:
        return None
    return f"{width}x{height}"
