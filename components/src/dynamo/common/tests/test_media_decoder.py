# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from dynamo.common.utils.media_decoder import (
    DEFAULT_FRONTEND_IMAGE_DECODER_MAX_ALLOC,
    DEFAULT_FRONTEND_VIDEO_NUM_FRAMES,
    DYN_MM_VIDEO_NUM_FRAMES,
    build_frontend_image_decoder_options,
    enable_frontend_video_decoding,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_frontend_image_decoder_options_only_set_limits():
    assert build_frontend_image_decoder_options() == {
        "limits": {"max_alloc": DEFAULT_FRONTEND_IMAGE_DECODER_MAX_ALLOC}
    }


def test_enable_frontend_video_decoding_uses_backend_default(monkeypatch):
    monkeypatch.delenv(DYN_MM_VIDEO_NUM_FRAMES, raising=False)
    decoder = Mock()

    enable_frontend_video_decoding(decoder)

    decoder.enable_video.assert_called_once_with(
        {"max_frames": DEFAULT_FRONTEND_VIDEO_NUM_FRAMES}
    )


def test_enable_frontend_video_decoding_uses_configured_frame_count(monkeypatch):
    monkeypatch.setenv(DYN_MM_VIDEO_NUM_FRAMES, "8")
    decoder = Mock()

    enable_frontend_video_decoding(decoder)

    decoder.enable_video.assert_called_once_with({"max_frames": 8})


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_enable_frontend_video_decoding_rejects_invalid_frame_count(
    monkeypatch, caplog, value
):
    monkeypatch.setenv(DYN_MM_VIDEO_NUM_FRAMES, value)
    decoder = Mock()

    enable_frontend_video_decoding(decoder)

    decoder.enable_video.assert_called_once_with(
        {"max_frames": DEFAULT_FRONTEND_VIDEO_NUM_FRAMES}
    )
    assert DYN_MM_VIDEO_NUM_FRAMES in caplog.text


def test_enable_frontend_video_decoding_warns_without_ffmpeg_binding(caplog):
    enable_frontend_video_decoding(object())

    assert "media-ffmpeg" in caplog.text
