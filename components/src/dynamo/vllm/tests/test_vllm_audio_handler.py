# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from PIL import Image

from dynamo.vllm.handlers import BaseWorkerHandler

pytestmark = [
    pytest.mark.asyncio,
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
    handler.config = SimpleNamespace(model="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    handler.embedding_loader = None
    handler.image_loader = SimpleNamespace(load_image_batch=AsyncMock(return_value=[]))
    handler.audio_loader = SimpleNamespace(load_audio_batch=AsyncMock(return_value=[]))
    return handler


async def test_extract_multimodal_data_loads_audio_url_items():
    handler = _make_handler()
    audio = (np.random.randn(16000).astype(np.float32), 16000.0)
    handler.audio_loader.load_audio_batch = AsyncMock(return_value=[audio])

    result = await handler._extract_multimodal_data(
        {"multi_modal_data": {"audio_url": [{"Url": "https://example.com/audio.wav"}]}},
        "req-1",
        context=None,
    )

    assert result is not None
    assert result["audio"] is audio
    handler.audio_loader.load_audio_batch.assert_awaited_once()


async def test_extract_multimodal_data_loads_image_and_audio_together():
    handler = _make_handler()
    image = Image.new("RGB", (2, 2))
    audio = (np.random.randn(16000).astype(np.float32), 16000.0)
    handler.image_loader.load_image_batch = AsyncMock(return_value=[image])
    handler.audio_loader.load_audio_batch = AsyncMock(return_value=[audio])

    result = await handler._extract_multimodal_data(
        {
            "multi_modal_data": {
                "image_url": [{"Url": "https://example.com/image.png"}],
                "audio_url": [{"Url": "https://example.com/audio.wav"}],
            }
        },
        "req-2",
        context=None,
    )

    assert result is not None
    assert result["image"] is image
    assert result["audio"] is audio
    handler.image_loader.load_image_batch.assert_awaited_once()
    handler.audio_loader.load_audio_batch.assert_awaited_once()


async def test_extract_multimodal_data_multiple_audio_items():
    handler = _make_handler()
    audio1 = (np.zeros(8000, dtype=np.float32), 16000.0)
    audio2 = (np.ones(8000, dtype=np.float32), 44100.0)
    handler.audio_loader.load_audio_batch = AsyncMock(return_value=[audio1, audio2])

    result = await handler._extract_multimodal_data(
        {
            "multi_modal_data": {
                "audio_url": [
                    {"Url": "https://example.com/a.wav"},
                    {"Url": "https://example.com/b.wav"},
                ],
            }
        },
        "req-3",
        context=None,
    )

    assert result is not None
    # Multiple items should be passed as a list, not unwrapped
    assert isinstance(result["audio"], list)
    assert len(result["audio"]) == 2


async def test_extract_multimodal_data_rejects_requests_when_disabled():
    handler = _make_handler(enable_multimodal=False)

    with pytest.raises(ValueError, match="multimodal processing is not enabled"):
        await handler._extract_multimodal_data(
            {
                "multi_modal_data": {
                    "audio_url": [{"Url": "https://example.com/audio.wav"}]
                }
            },
            "req-4",
            context=None,
        )
