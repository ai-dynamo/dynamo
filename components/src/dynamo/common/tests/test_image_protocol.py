# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.protocols.image_protocol module."""

import pytest
from pydantic import ValidationError

from dynamo.common.protocols.image_protocol import (
    ImageData,
    ImageTokenDetails,
    ImageUsage,
    NvCreateImageRequest,
    NvImagesResponse,
)
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
        usage=ImageUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            input_tokens_details=ImageTokenDetails(text_tokens=10, image_tokens=0),
        ),
    )

    assert response.model_dump(exclude_none=True) == {
        "created": 0,
        "data": [],
        "background": "opaque",
        "output_format": "png",
        "size": "1024x1024",
        "quality": "high",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_tokens_details": {"text_tokens": 10, "image_tokens": 0},
        },
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


def test_images_response_rejects_usage_without_input_token_details():
    # The frontend requires the field, so the worker fails first.
    with pytest.raises(ValidationError):
        NvImagesResponse(
            created=0,
            data=[],
            usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        )


def test_image_request_accepts_a_value_outside_the_enum():
    # The frontend rejects the value. A worker accepts what a newer frontend
    # forwards for a field it does not read, so the request model takes any
    # string here.
    assert NvCreateImageRequest(prompt="a cat", quality="ultra").quality == "ultra"


def test_image_request_rejects_an_unknown_delivery_mode():
    # The worker reads response_format to build its response, so the request
    # model keeps the enum and an unknown mode fails before generation.
    with pytest.raises(ValidationError):
        NvCreateImageRequest(prompt="a cat", response_format="file")


def test_image_response_rejects_a_value_outside_the_enum():
    # A worker builds the response, so the response model keeps the enum.
    with pytest.raises(ValidationError):
        NvImagesResponse(created=0, data=[], quality="ultra")


@pytest.mark.parametrize("n", [0, 11])
def test_image_request_rejects_n_outside_supported_range(n):
    # The frontend rejects the value too. The model matches the frontend.
    with pytest.raises(ValidationError):
        NvCreateImageRequest(prompt="a cat", n=n)


@pytest.mark.parametrize(
    ("field", "value"),
    [("output_compression", 101), ("partial_images", 4)],
)
def test_image_request_rejects_a_value_above_the_api_maximum(field, value):
    # The frontend rejects the value too. The model matches the frontend.
    with pytest.raises(ValidationError):
        NvCreateImageRequest(prompt="a cat", **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [("output_compression", 100), ("partial_images", 3)],
)
def test_image_request_accepts_the_api_maximum(field, value):
    request = NvCreateImageRequest(prompt="a cat", **{field: value})

    assert getattr(request, field) == value
