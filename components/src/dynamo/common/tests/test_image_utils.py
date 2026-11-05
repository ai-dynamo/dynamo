# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.image_utils."""

import io

import numpy as np
import pytest
from PIL import Image

from dynamo.common.utils.image_utils import (
    encode_image_bytes,
    normalize_image_output_options,
    uniform_image_size,
)
from dynamo.llm.exceptions import InvalidArgument

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

HANDLER = "Dynamo test image handler"


def _decode(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes))


@pytest.mark.parametrize(
    "output_format, expected",
    [(None, "png"), ("png", "png"), ("JPEG", "jpeg"), ("WebP", "webp")],
)
def test_normalize_output_format_is_case_insensitive(output_format, expected):
    assert (
        normalize_image_output_options(
            output_format, quality=None, background=None, handler_name=HANDLER
        )
        == expected
    )


def test_normalize_rejects_unknown_output_format():
    with pytest.raises(InvalidArgument, match="Unsupported output_format: 'gif'"):
        normalize_image_output_options(
            "gif", quality=None, background=None, handler_name=HANDLER
        )


@pytest.mark.parametrize(
    "quality, background, field",
    [
        ("high", None, "quality"),
        ("auto", None, "quality"),
        (None, "opaque", "background"),
    ],
)
def test_normalize_rejects_quality_and_background(quality, background, field):
    with pytest.raises(
        InvalidArgument, match=f"{field} is not supported by the {HANDLER}"
    ):
        normalize_image_output_options(
            "png", quality=quality, background=background, handler_name=HANDLER
        )


@pytest.mark.parametrize(
    "output_format, pil_format", [("png", "PNG"), ("jpeg", "JPEG"), ("webp", "WEBP")]
)
def test_encode_numpy_array(output_format, pil_format):
    array = np.zeros((16, 32, 3), dtype=np.uint8)
    decoded = _decode(encode_image_bytes(array, output_format))
    assert decoded.format == pil_format
    assert decoded.size == (32, 16)


def test_jpeg_flattens_rgba_onto_white():
    image = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    decoded = _decode(encode_image_bytes(image, "jpeg"))
    assert decoded.mode == "RGB"
    assert all(channel >= 250 for channel in decoded.getpixel((0, 0)))


def test_jpeg_flattens_palette_transparency_onto_white():
    image = Image.new("P", (8, 8), 0)
    image.putpalette([255, 0, 0] + [0, 0, 0] * 255)
    image.info["transparency"] = 0
    decoded = _decode(encode_image_bytes(image, "jpeg"))
    assert all(channel >= 250 for channel in decoded.getpixel((0, 0)))


def test_png_keeps_alpha():
    image = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    decoded = _decode(encode_image_bytes(image, "png"))
    assert decoded.mode == "RGBA"


@pytest.mark.parametrize(
    "sizes, expected",
    [
        ([(64, 32)], "64x32"),
        ([(64, 32), (64, 32)], "64x32"),
        ([(64, 32), (32, 64)], None),
        ([], None),
        ([(0, 32)], None),
    ],
)
def test_uniform_image_size(sizes, expected):
    assert uniform_image_size(sizes) == expected
