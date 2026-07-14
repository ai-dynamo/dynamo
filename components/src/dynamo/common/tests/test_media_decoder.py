# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from dynamo.common.utils.media_decoder import (
    DEFAULT_FRONTEND_VIDEO_NUM_FRAMES,
    DYN_MM_VIDEO_NUM_FRAMES,
    enable_frontend_video_decoding,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_enable_frontend_video_decoding_uses_backend_default(monkeypatch):
    monkeypatch.delenv(DYN_MM_VIDEO_NUM_FRAMES, raising=False)
    decoder = Mock()

    enable_frontend_video_decoding(decoder)

    decoder.enable_video.assert_called_once_with(
        {"num_frames": DEFAULT_FRONTEND_VIDEO_NUM_FRAMES}
    )


def test_enable_frontend_video_decoding_uses_configured_frame_count(monkeypatch):
    monkeypatch.setenv(DYN_MM_VIDEO_NUM_FRAMES, "8")
    decoder = Mock()

    enable_frontend_video_decoding(decoder)

    decoder.enable_video.assert_called_once_with({"num_frames": 8})


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_enable_frontend_video_decoding_rejects_invalid_frame_count(
    monkeypatch, caplog, value
):
    monkeypatch.setenv(DYN_MM_VIDEO_NUM_FRAMES, value)
    decoder = Mock()

    enable_frontend_video_decoding(decoder)

    decoder.enable_video.assert_called_once_with(
        {"num_frames": DEFAULT_FRONTEND_VIDEO_NUM_FRAMES}
    )
    assert DYN_MM_VIDEO_NUM_FRAMES in caplog.text


def test_enable_frontend_video_decoding_warns_without_ffmpeg_binding(caplog):
    enable_frontend_video_decoding(object())

    assert "media-ffmpeg" in caplog.text
