# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.protocols.video_protocol module."""

import pytest
from pydantic import ValidationError

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

# ---------------------------------------------------------------------------
# NvCreateVideoRequest
# ---------------------------------------------------------------------------


class TestNvCreateVideoRequest:
    def test_required_fields(self):
        req = NvCreateVideoRequest(prompt="a cat", model="wan")
        assert req.prompt == "a cat"
        assert req.model == "wan"

    def test_missing_prompt_raises(self):
        with pytest.raises(ValidationError):
            NvCreateVideoRequest(model="wan")

    def test_missing_model_raises(self):
        with pytest.raises(ValidationError):
            NvCreateVideoRequest(prompt="a cat")

    def test_output_format_optional_defaults_none(self):
        req = NvCreateVideoRequest(prompt="p", model="m")
        assert req.output_format is None

    def test_output_format_mp4(self):
        req = NvCreateVideoRequest(prompt="p", model="m", output_format="mp4")
        assert req.output_format == "mp4"

    def test_output_format_mjpeg(self):
        req = NvCreateVideoRequest(prompt="p", model="m", output_format="mjpeg")
        assert req.output_format == "mjpeg"

    def test_output_format_arbitrary_string(self):
        req = NvCreateVideoRequest(prompt="p", model="m", output_format="webm")
        assert req.output_format == "webm"

    def test_response_format_optional(self):
        req = NvCreateVideoRequest(prompt="p", model="m")
        assert req.response_format is None

    def test_response_format_url(self):
        req = NvCreateVideoRequest(prompt="p", model="m", response_format="url")
        assert req.response_format == "url"

    def test_response_format_b64_json(self):
        req = NvCreateVideoRequest(prompt="p", model="m", response_format="b64_json")
        assert req.response_format == "b64_json"

    def test_nvext_optional(self):
        req = NvCreateVideoRequest(prompt="p", model="m")
        assert req.nvext is None

    def test_nvext_fields(self):
        req = NvCreateVideoRequest(
            prompt="p",
            model="m",
            nvext=VideoNvExt(fps=24, guidance_scale=7.5, seed=42),
        )
        assert req.nvext.fps == 24
        assert req.nvext.guidance_scale == 7.5
        assert req.nvext.seed == 42

    def test_stream_defaults_none(self):
        req = NvCreateVideoRequest(prompt="p", model="m")
        assert req.stream is None

    def test_stream_true(self):
        req = NvCreateVideoRequest(prompt="p", model="m", stream=True)
        assert req.stream is True

    def test_stream_false(self):
        req = NvCreateVideoRequest(prompt="p", model="m", stream=False)
        assert req.stream is False

    def test_json_round_trip(self):
        req = NvCreateVideoRequest(
            prompt="cat",
            model="wan",
            output_format="mp4",
            response_format="url",
            stream=True,
            nvext=VideoNvExt(boundary_ratio=0.3, guidance_scale_2=1.0),
        )
        restored = NvCreateVideoRequest.model_validate_json(req.model_dump_json())
        assert restored.output_format == "mp4"
        assert restored.stream is True
        assert restored.nvext.boundary_ratio == 0.3


# ---------------------------------------------------------------------------
# VideoData
# ---------------------------------------------------------------------------


class TestVideoData:
    def test_output_format_required(self):
        with pytest.raises(ValidationError):
            VideoData()

    def test_output_format_missing_with_url_raises(self):
        with pytest.raises(ValidationError):
            VideoData(url="http://example.com/v.mp4")

    def test_minimal_with_output_format(self):
        d = VideoData(output_format="mp4")
        assert d.output_format == "mp4"
        assert d.url is None
        assert d.b64_json is None

    def test_output_format_arbitrary_string(self):
        d = VideoData(output_format="webm")
        assert d.output_format == "webm"

    def test_with_url(self):
        d = VideoData(output_format="mp4", url="http://example.com/v.mp4")
        assert d.url == "http://example.com/v.mp4"

    def test_with_b64_json(self):
        d = VideoData(output_format="mp4", b64_json="abc123==")
        assert d.b64_json == "abc123=="
        assert d.output_format == "mp4"

    def test_json_round_trip(self):
        d = VideoData(output_format="mp4", url="http://example.com/v.mp4")
        restored = VideoData.model_validate_json(d.model_dump_json())
        assert restored.output_format == "mp4"
        assert restored.url == d.url


# ---------------------------------------------------------------------------
# NvVideosResponse
# ---------------------------------------------------------------------------


class TestNvVideosResponse:
    def test_data_list_carries_output_format(self):
        resp = NvVideosResponse(
            id="r1",
            model="wan",
            created=0,
            data=[VideoData(output_format="mp4", url="http://example.com/v.mp4")],
        )
        assert resp.data[0].output_format == "mp4"

    def test_empty_data_list_default(self):
        resp = NvVideosResponse(id="r2", model="m", created=0)
        assert resp.data == []
