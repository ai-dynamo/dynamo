# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from dynamo.sglang.protocol import (
    MultiModalGroup,
    MultiModalInput,
    PreprocessedRequest,
    SamplingOptions,
    SglangMultimodalRequest,
    StopConditions,
)
from dynamo.sglang.request_handlers.multimodal.encode_worker_handler import (
    Modality,
    MultimodalEncodeWorkerHandler,
)
from dynamo.sglang.request_handlers.multimodal.worker_handler import (
    EmbeddingsProcessor,
    _build_mm_items,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.skipif(Modality is None, reason="SGLang Modality is required"),
]


# Where the video-decoder preflight lives; patched to a no-op below to simulate a
# decord-enabled image (the runtime image ships without it).
_HANDLER_MOD = "dynamo.sglang.request_handlers.multimodal.encode_worker_handler"


def test_extract_media_urls_supports_video_urls():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)

    with patch(f"{_HANDLER_MOD}._ensure_video_decoder_available"):
        image_urls, video_urls = handler._extract_media_urls(
            {
                "multi_modal_data": {
                    "video_url": [
                        {"Url": "https://example.com/clip.mp4"},
                        "file:///tmp/local.mp4",
                    ]
                }
            }
        )

    assert image_urls == []
    assert video_urls == ["https://example.com/clip.mp4", "file:///tmp/local.mp4"]


def test_extract_media_urls_supports_mixed_image_and_video():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)

    with patch(f"{_HANDLER_MOD}._ensure_video_decoder_available"):
        image_urls, video_urls = handler._extract_media_urls(
            {
                "multi_modal_data": {
                    "image_url": [{"Url": "https://example.com/image.png"}],
                    "video_url": [{"Url": "https://example.com/clip.mp4"}],
                }
            }
        )

    assert image_urls == ["https://example.com/image.png"]
    assert video_urls == ["https://example.com/clip.mp4"]


def test_extract_media_urls_video_without_decord_raises_actionable_error():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)

    # Simulate an image built without the video decoder.
    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(RuntimeError, match="decord2"):
            handler._extract_media_urls(
                {"multi_modal_data": {"video_url": [{"Url": "https://x/c.mp4"}]}}
            )


@pytest.mark.asyncio
async def test_build_mm_items_routes_video_to_video_data():
    embeddings = torch.arange(24, dtype=torch.float16).reshape(6, 4)

    class _FakeEmbeddingsProcessor:
        async def process_embeddings(self, request):
            return embeddings, 17

        @staticmethod
        def create_multimodal_image_item(
            embeddings,
            image_grid_thw,
        ):
            return EmbeddingsProcessor.create_multimodal_image_item(
                embeddings,
                image_grid_thw,
            )

        @staticmethod
        def create_multimodal_video_item(
            embeddings,
            video_grid_thw,
            second_per_grid_ts=None,
            video_timestamps=None,
        ):
            return EmbeddingsProcessor.create_multimodal_video_item(
                embeddings,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                video_timestamps=video_timestamps,
            )

    request = SglangMultimodalRequest(
        request=PreprocessedRequest(
            token_ids=[151652, 151656, 151653],
            stop_conditions=StopConditions(max_tokens=32),
            sampling_options=SamplingOptions(temperature=0.0),
        ),
        multimodal_inputs=[
            MultiModalGroup(
                multimodal_input=MultiModalInput(),
                video_grid_thw=[2, 3, 4],
                second_per_grid_ts=0.5,
                video_timestamps=[0.25, 0.75],
                num_mm_tokens=6,
            )
        ],
    )

    image_items, video_items, combined_embeddings, tensor_id = await _build_mm_items(
        request, _FakeEmbeddingsProcessor()
    )

    assert tensor_id == 17
    assert torch.equal(combined_embeddings, embeddings)
    assert image_items == []
    assert len(video_items) == 1

    mm_item = video_items[0]
    assert mm_item["modality"] == "VIDEO"
    assert torch.equal(mm_item["video_grid_thw"], torch.tensor([[2, 3, 4]]))
    assert torch.equal(
        mm_item["second_per_grid_ts"], torch.tensor([0.5], dtype=torch.float32)
    )
    assert mm_item["video_timestamps"] == [[0.25, 0.75]]
