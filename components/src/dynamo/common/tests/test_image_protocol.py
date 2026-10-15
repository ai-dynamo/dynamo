# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.protocols.image_protocol module."""

import pytest

from dynamo.common.protocols.image_protocol import ImageData, NvImagesResponse
from dynamo.common.utils.output_modalities import RequestType, parse_request_type

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_images_response_wire_shape_without_optional_fields():
    response = NvImagesResponse(created=0, data=[ImageData(b64_json="xyz")])

    assert response.model_dump(exclude_none=True) == {
        "created": 0,
        "data": [{"b64_json": "xyz"}],
    }


def test_images_response_keeps_the_generation_parameters():
    # The Rust response carries these fields. A worker can set them. A client
    # can read them.
    response = NvImagesResponse(
        created=0,
        data=[],
        background="opaque",
        output_format="png",
        size="1024x1024",
        quality="high",
        usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )

    assert response.model_dump(exclude_none=True) == {
        "created": 0,
        "data": [],
        "background": "opaque",
        "output_format": "png",
        "size": "1024x1024",
        "quality": "high",
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }


def test_image_request_parsing_preserves_output_format_and_unsupported_controls():
    request, request_type = parse_request_type(
        {
            "prompt": "a cat",
            "output_format": "jpeg",
            "quality": "high",
            "background": "transparent",
        },
        ["image"],
    )

    assert request_type == RequestType.IMAGE_GENERATION
    assert request.output_format == "jpeg"
    assert request.quality == "high"
    assert request.background == "transparent"
