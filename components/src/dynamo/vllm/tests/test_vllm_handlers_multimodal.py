# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from dynamo.vllm.handlers import BaseWorkerHandler, _compute_mm_uuids

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
    pytest.mark.unit,
]


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


def _make_handler(model: str = "test-model") -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.enable_multimodal = True
    handler.config = SimpleNamespace(model=model)
    handler.image_loader = SimpleNamespace(load_image_batch=AsyncMock(return_value=[]))
    handler.video_loader = SimpleNamespace(load_video_batch=AsyncMock(return_value=[]))
    handler.audio_loader = SimpleNamespace(load_audio_batch=AsyncMock(return_value=[]))
    handler.embedding_loader = SimpleNamespace(load_multimodal_embeddings=AsyncMock())
    return handler


def test_compute_mm_uuids_supports_kimi_vision_chunk():
    image = Image.new("RGB", (2, 2), color="blue")

    mm_uuids = _compute_mm_uuids(
        {
            "vision_chunk": [
                {
                    "type": "image",
                    "image": image,
                    "uuid": None,
                }
            ]
        }
    )

    assert mm_uuids is not None
    assert "vision_chunk" in mm_uuids
    assert len(mm_uuids["vision_chunk"]) == 1
    assert re.fullmatch(r"[0-9a-f]{64}", mm_uuids["vision_chunk"][0])


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.6"])
async def test_extract_multimodal_data_kimi_uses_raw_vision_chunk_path(model):
    handler = _make_handler(model=model)
    fake_image = object()
    handler.image_loader.load_image_batch = AsyncMock(return_value=[fake_image])

    result = await handler._extract_multimodal_data(
        {
            "multi_modal_data": {
                "image_url": [{"Url": "http://img.png"}],
            }
        },
        "req-1",
        context=None,
    )

    handler.image_loader.load_image_batch.assert_awaited_once_with(
        [{"Url": "http://img.png"}]
    )
    handler.embedding_loader.load_multimodal_embeddings.assert_not_awaited()
    assert result == {
        "vision_chunk": {"type": "image", "image": fake_image, "uuid": None}
    }


@pytest.mark.asyncio
async def test_extract_multimodal_data_kimi_preserves_decoded_inputs():
    handler = _make_handler(model="moonshotai/Kimi-K2.6")
    decoded_metadata = {"handle": "nixl-1"}
    fake_image = object()
    handler.image_loader.load_image_batch = AsyncMock(return_value=[fake_image])

    result = await handler._extract_multimodal_data(
        {
            "multi_modal_data": {
                "image_url": [{"Decoded": decoded_metadata}],
            }
        },
        "req-1",
        context=None,
    )

    handler.image_loader.load_image_batch.assert_awaited_once_with(
        [{"Decoded": decoded_metadata}]
    )
    handler.embedding_loader.load_multimodal_embeddings.assert_not_awaited()
    assert result == {
        "vision_chunk": {"type": "image", "image": fake_image, "uuid": None}
    }
