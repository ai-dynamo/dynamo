# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

import dynamo.vllm.handlers as mod
from dynamo.vllm.handlers import _compute_mm_uuids

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _make_config(model: str = "test-model") -> MagicMock:
    config = MagicMock()
    config.model = model
    return config


def _make_handler(model: str = "test-model") -> mod.DecodeWorkerHandler:
    config = _make_config(model)
    with patch.object(mod.BaseWorkerHandler, "__init__", return_value=None):
        handler = mod.DecodeWorkerHandler(
            runtime=MagicMock(),
            config=config,
            engine=MagicMock(),
            default_sampling_params={},
        )
    handler.config = config
    handler.enable_multimodal = True
    handler.image_loader = MagicMock()
    handler.video_loader = MagicMock()
    handler.embedding_loader = MagicMock()
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


@pytest.mark.asyncio
async def test_extract_multimodal_data_kimi_uses_raw_vision_chunk_path():
    handler = _make_handler(model="moonshotai/Kimi-K2.5")
    fake_image = object()
    handler.image_loader.load_image_batch = AsyncMock(return_value=[fake_image])
    handler.video_loader.load_video_batch = AsyncMock(return_value=[])
    handler.embedding_loader.load_multimodal_embeddings = AsyncMock()

    result = await handler._extract_multimodal_data(
        {
            "multi_modal_data": {
                "image_url": [{"Url": "http://img.png"}],
            }
        },
        "req-1",
        None,
    )

    handler.image_loader.load_image_batch.assert_awaited_once_with(
        [{"Url": "http://img.png"}]
    )
    handler.embedding_loader.load_multimodal_embeddings.assert_not_awaited()
    assert result == {
        "vision_chunk": [
            {"type": "image", "image": fake_image, "uuid": None},
        ]
    }


@pytest.mark.asyncio
async def test_extract_multimodal_data_kimi_preserves_decoded_inputs():
    handler = _make_handler(model="moonshotai/Kimi-K2.5")
    decoded_metadata = {"handle": "nixl-1"}
    fake_image = object()
    handler.image_loader.load_image_batch = AsyncMock(return_value=[fake_image])
    handler.video_loader.load_video_batch = AsyncMock(return_value=[])
    handler.embedding_loader.load_multimodal_embeddings = AsyncMock()

    result = await handler._extract_multimodal_data(
        {
            "multi_modal_data": {
                "image_url": [{"Decoded": decoded_metadata}],
            }
        },
        "req-1",
        None,
    )

    handler.image_loader.load_image_batch.assert_awaited_once_with(
        [{"Decoded": decoded_metadata}]
    )
    handler.embedding_loader.load_multimodal_embeddings.assert_not_awaited()
    assert result == {
        "vision_chunk": [
            {"type": "image", "image": fake_image, "uuid": None},
        ]
    }
