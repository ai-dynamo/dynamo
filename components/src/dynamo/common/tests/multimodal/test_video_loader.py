# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from unittest.mock import AsyncMock

import numpy as np
import pytest

from dynamo.common.multimodal.video_loader import VideoLoader

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
async def test_load_video_content_supports_data_urls():
    loader = VideoLoader()
    payload = b"fake-video-payload"
    data_url = "data:video/mp4;base64," + base64.b64encode(payload).decode("utf-8")

    content = await loader._load_video_content(data_url)

    assert content.read() == payload


def test_calculate_frame_sampling_indices_caps_to_available_frames():
    indices = VideoLoader._calculate_frame_sampling_indices(
        total_frames=4,
        num_frames_to_sample=8,
        duration_sec=1.0,
        video_url="sample.mp4",
    )

    assert indices.tolist() == [0, 1, 2, 3]


def test_build_video_metadata_returns_qwen_compatible_fields():
    metadata = VideoLoader._build_video_metadata(
        total_frames=12,
        duration_sec=6.0,
        fps=2.0,
        indices=np.array([0, 3, 6, 9]),
        decoded_num_frames=4,
    )

    assert metadata == {
        "fps": 2.0,
        "duration": 6.0,
        "total_num_frames": 12,
        "frames_indices": [0, 3, 6, 9],
        "video_backend": "pyav",
        "do_sample_frames": False,
    }


@pytest.mark.asyncio
async def test_load_video_batch_uses_url_loader():
    loader = VideoLoader()
    first = (
        np.zeros((1, 2, 2, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0], "total_num_frames": 1},
    )
    second = (
        np.ones((1, 2, 2, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0], "total_num_frames": 1},
    )
    loader.load_video = AsyncMock(side_effect=[first, second])  # type: ignore[method-assign]

    videos = await loader.load_video_batch(
        [
            {"Url": "https://example.com/one.mp4"},
            {"Url": "https://example.com/two.mp4"},
        ]
    )

    np.testing.assert_array_equal(videos[0][0], first[0])
    np.testing.assert_array_equal(videos[1][0], second[0])
    assert videos[0][1] == first[1]
    assert videos[1][1] == second[1]


@pytest.mark.asyncio
async def test_load_video_batch_rejects_decoded_variant_without_frontend_decoding():
    loader = VideoLoader(enable_frontend_decoding=False)

    with pytest.raises(ValueError, match="enable_frontend_decoding=False"):
        await loader.load_video_batch([{"Decoded": {"shape": [1, 2, 2, 3]}}])
