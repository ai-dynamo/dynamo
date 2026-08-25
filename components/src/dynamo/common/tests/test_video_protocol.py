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
        extra_params={
            "task": "t2va",
            "target": {"duration_seconds": 10.0, "aspect_ratio": "16:9"},
        },
        nvext=VideoNvExt(boundary_ratio=0.3, guidance_scale_2=1.0),
    )

    assert request.model_dump(exclude_none=True) == {
        "prompt": "cat",
        "model": "wan",
        "response_format": "url",
        "output_format": "mp4",
        "extra_params": {
            "task": "t2va",
            "target": {"duration_seconds": 10.0, "aspect_ratio": "16:9"},
        },
        "stream": True,
        "nvext": {"boundary_ratio": 0.3, "guidance_scale_2": 1.0},
    }


def test_video_request_extra_params_requires_object():
    with pytest.raises(ValueError):
        NvCreateVideoRequest.model_validate(
            {
                "prompt": "cat",
                "model": "video-model",
                "extra_params": ["not", "an", "object"],
            }
        )


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


def test_video_response_reports_media_metadata():
    response = VideoData(
        output_format="mp4",
        url="http://example.com/v.mp4",
        fps=24,
        audio_sample_rate=32000,
    )

    assert response.model_dump(exclude_none=True) == {
        "output_format": "mp4",
        "url": "http://example.com/v.mp4",
        "fps": 24,
        "audio_sample_rate": 32000,
    }
