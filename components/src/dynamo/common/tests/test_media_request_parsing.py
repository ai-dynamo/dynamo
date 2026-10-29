# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for parse_media_request.

A worker builds its request model from the body the frontend forwards. When
the model rejects the body, the client gets a 400 that names each rejected
field and the rule it broke, and never the rejected value.
"""

from typing import Any, cast

import pytest

from dynamo.common.protocols.image_protocol import NvCreateImageRequest
from dynamo.common.protocols.request_parsing import parse_media_request
from dynamo.common.protocols.video_protocol import NvCreateVideoRequest
from dynamo.llm.exceptions import InvalidArgument

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_a_valid_body_parses_as_the_given_model():
    request = parse_media_request(NvCreateImageRequest, {"prompt": "a cat", "n": 2})

    assert isinstance(request, NvCreateImageRequest)
    assert (request.prompt, request.n) == ("a cat", 2)


def test_a_rejected_field_is_named_with_its_rule():
    with pytest.raises(InvalidArgument) as excinfo:
        parse_media_request(
            NvCreateVideoRequest,
            {"prompt": "a cat", "model": "wan", "nvext": {"fps": 0}},
        )

    assert str(excinfo.value) == "nvext.fps: Input should be greater than or equal to 1"


def test_every_rejected_field_is_named():
    with pytest.raises(InvalidArgument) as excinfo:
        parse_media_request(
            NvCreateImageRequest,
            {"prompt": "a cat", "n": "two", "output_compression": 101},
        )

    assert str(excinfo.value) == (
        "n: Input should be a valid integer; "
        "output_compression: Input should be less than or equal to 100"
    )


def test_the_rejected_value_stays_out_of_the_message():
    """str(ValidationError) carries input_value; the client's 400 must not."""
    with pytest.raises(InvalidArgument) as excinfo:
        parse_media_request(NvCreateImageRequest, {"prompt": "a cat", "size": 1024})

    message = str(excinfo.value)
    assert message == "size: Input should be a valid string"
    assert "1024" not in message
    assert "input_value" not in message


def test_a_body_that_is_not_an_object_is_named_request():
    with pytest.raises(InvalidArgument) as excinfo:
        parse_media_request(NvCreateImageRequest, cast(Any, "not an object"))

    assert str(excinfo.value).startswith("request: Input should be a valid dictionary")
