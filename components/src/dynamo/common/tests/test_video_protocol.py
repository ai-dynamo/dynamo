# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.protocols.video_protocol module."""

import pytest

from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
    VideoNvExt,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_video_request_wire_shape():
    request = NvCreateVideoRequest(
        prompt="cat",
        model="wan",
        output_format="mp4",
        response_format="url",
        stream=True,
        nvext=VideoNvExt(boundary_ratio=0.3, guidance_scale_2=1.0),
    )

    assert request.model_dump(exclude_none=True) == {
        "prompt": "cat",
        "model": "wan",
        "response_format": "url",
        "output_format": "mp4",
        "stream": True,
        "nvext": {"boundary_ratio": 0.3, "guidance_scale_2": 1.0},
    }


def test_video_request_typed_references_preserve_wire_order():
    request = NvCreateVideoRequest(
        prompt="cat",
        model="video-model",
        input_references=[
            {"type": "image", "source": "https://example.com/cat.png"},
            {"type": "audio", "source": "data:audio/wav;base64,AA=="},
        ],
    )

    assert [reference.type for reference in request.input_references] == [
        "image",
        "audio",
    ]


@pytest.mark.parametrize(
    "input_references",
    [[], [{"type": "image", "source": "https://example.com/cat.png"}]],
)
def test_video_request_rejects_invalid_reference_combinations(input_references):
    kwargs: dict[str, object] = {"input_references": input_references}
    if input_references:
        kwargs["input_reference"] = "https://example.com/legacy.png"

    with pytest.raises(ValueError):
        NvCreateVideoRequest(prompt="cat", model="video-model", **kwargs)


def test_video_response_wire_shape():
    response = NvVideosResponse(
        id="r1",
        model="wan",
        created=0,
        data=[VideoData(output_format="mp4", url="http://example.com/v.mp4")],
    )

    assert response.model_dump(exclude_none=True) == {
        "id": "r1",
        "object": "video",
        "model": "wan",
        "status": "completed",
        "progress": 100,
        "created": 0,
        "data": [{"output_format": "mp4", "url": "http://example.com/v.mp4"}],
    }
