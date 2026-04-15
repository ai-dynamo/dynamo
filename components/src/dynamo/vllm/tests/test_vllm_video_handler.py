# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from dynamo.vllm.handlers import BaseWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


def _make_handler(enable_multimodal: bool = True) -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.enable_multimodal = enable_multimodal
    handler.config = SimpleNamespace(model="Qwen/Qwen3-VL-2B-Instruct")
    handler.embedding_loader = None
    handler.image_loader = SimpleNamespace(load_image_batch=AsyncMock(return_value=[]))
    handler.video_loader = SimpleNamespace(load_video_batch=AsyncMock(return_value=[]))
    return handler


@pytest.mark.asyncio
async def test_extract_multimodal_data_loads_video_url_items():
    handler = _make_handler()
    video = (
        np.zeros((2, 4, 4, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0, 1], "total_num_frames": 2},
    )
    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])

    result = await handler._extract_multimodal_data(
        {"multi_modal_data": {"video_url": [{"Url": "https://example.com/video.mp4"}]}},
        "req-1",
        context=None,
    )

    assert result is not None
    assert result["video"] is video
    handler.image_loader.load_image_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_extract_multimodal_data_merges_image_embeddings_with_video():
    handler = _make_handler()
    image_mm_data = {"image": {"image_embeds": object()}}
    video = (
        np.ones((3, 4, 4, 3), dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0, 1, 2], "total_num_frames": 3},
    )
    handler.embedding_loader = SimpleNamespace(
        load_multimodal_embeddings=AsyncMock(return_value=image_mm_data)
    )
    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])

    result = await handler._extract_multimodal_data(
        {
            "multi_modal_data": {
                "image_url": [{"Url": "https://example.com/image.png"}],
                "video_url": [{"Url": "https://example.com/video.mp4"}],
            }
        },
        "req-2",
        context=None,
    )

    assert result is not None
    assert result["image"] is image_mm_data["image"]
    assert result["video"] is video
    handler.image_loader.load_image_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_extract_multimodal_data_falls_back_to_image_loader_for_decoded_images():
    handler = _make_handler()
    image = object()
    video = (
        np.full((1, 2, 2, 3), 7, dtype=np.uint8),
        {"fps": 2.0, "frames_indices": [0], "total_num_frames": 1},
    )
    handler.embedding_loader = SimpleNamespace(
        load_multimodal_embeddings=AsyncMock(return_value={"image": "unused"})
    )
    handler.image_loader.load_image_batch = AsyncMock(return_value=[image])
    handler.video_loader.load_video_batch = AsyncMock(return_value=[video])

    result = await handler._extract_multimodal_data(
        {
            "multi_modal_data": {
                "image_url": [{"Decoded": {"shape": [1, 1, 3]}}],
                "video_url": [{"Url": "https://example.com/video.mp4"}],
            }
        },
        "req-3",
        context=None,
    )

    assert result is not None
    assert result["image"] is image
    assert result["video"] is video
    handler.embedding_loader.load_multimodal_embeddings.assert_not_awaited()
    handler.image_loader.load_image_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_extract_multimodal_data_rejects_requests_when_disabled():
    handler = _make_handler(enable_multimodal=False)

    with pytest.raises(ValueError, match="multimodal processing is not enabled"):
        await handler._extract_multimodal_data(
            {
                "multi_modal_data": {
                    "video_url": [{"Url": "https://example.com/video.mp4"}]
                }
            },
            "req-4",
            context=None,
        )
