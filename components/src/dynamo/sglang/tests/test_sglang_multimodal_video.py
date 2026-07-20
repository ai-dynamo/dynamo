# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal import TransferRequest
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
    _mm_hashes_kwargs,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.skipif(Modality is None, reason="SGLang Modality is required"),
]


def test_extract_media_inputs_supports_video_urls():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)

    image_items, video_items = handler._extract_media_inputs(
        {
            "multi_modal_data": {
                "video_url": [
                    {"Url": "https://example.com/clip.mp4"},
                    "file:///tmp/local.mp4",
                ]
            }
        }
    )

    assert image_items == []
    assert video_items == [
        {"Url": "https://example.com/clip.mp4"},
        "file:///tmp/local.mp4",
    ]


def test_extract_media_inputs_supports_mixed_image_and_video():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)

    image_items, video_items = handler._extract_media_inputs(
        {
            "multi_modal_data": {
                "image_url": [{"Url": "https://example.com/image.png"}],
                "video_url": [{"Url": "https://example.com/clip.mp4"}],
            }
        }
    )

    assert image_items == [{"Url": "https://example.com/image.png"}]
    assert video_items == [{"Url": "https://example.com/clip.mp4"}]


@pytest.mark.asyncio
async def test_prepare_video_inputs_preserves_url_and_decoded_order():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler._embedding_cache = None
    frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    metadata = {
        "fps": 24.0,
        "duration": 10.0,
        "frames_indices": [0, 80, 160, 239],
    }
    handler._video_loader = SimpleNamespace(
        load_video_batch=AsyncMock(return_value=[(frames, metadata)])
    )
    decoded = {"shape": [4, 2, 2, 3], "dtype": "UINT8"}

    inputs, cache_keys, prechecked = await handler._prepare_video_inputs(
        [
            {"Url": "https://example.com/clip.mp4"},
            {"Decoded": decoded},
        ]
    )

    assert inputs[0] == "https://example.com/clip.mp4"
    assert isinstance(inputs[1], np.ndarray)
    assert inputs[1].avg_fps == pytest.approx(72 / 239)
    np.testing.assert_array_equal(inputs[1], frames)
    assert cache_keys == [None, None]
    assert prechecked == {}
    handler._video_loader.load_video_batch.assert_awaited_once_with(
        [{"Decoded": decoded}]
    )


@pytest.mark.asyncio
async def test_prepare_decoded_video_cache_hit_skips_pixel_transfer():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler._embedding_cache = MultimodalEmbeddingCacheManager(1024 * 1024)
    handler._decoded_content_hash_warning_emitted = False
    handler._video_loader = SimpleNamespace(
        load_video_batch=AsyncMock(
            side_effect=AssertionError("cached video pixels must not be transferred")
        )
    )
    content_hash = "0123456789abcdef"
    cached = CachedEmbedding(
        tensor=torch.ones((4, 3)),
        video_grid_thw=[2, 2, 4],
    )
    handler._embedding_cache.set(content_hash, cached)

    inputs, cache_keys, prechecked = await handler._prepare_video_inputs(
        [
            {
                "Decoded": {
                    "shape": [4, 2, 2, 3],
                    "dtype": "UINT8",
                    "content_hash": content_hash,
                }
            }
        ]
    )

    assert inputs == [None]
    assert cache_keys == [content_hash]
    assert prechecked == {0: cached}
    handler._video_loader.load_video_batch.assert_not_awaited()


def test_exact_video_replacement_preserves_interleaved_images():
    video_token_id = 20
    replacements = MultimodalEncodeWorkerHandler._extract_video_replacements(
        {
            "extra_args": {
                "mm_replacements_by_modality": {
                    "video": [
                        {
                            "placeholder_token_id": video_token_id,
                            "target_tokens": [30, video_token_id, 31],
                            "replacement_tokens": [
                                30,
                                40,
                                video_token_id,
                                video_token_id,
                                41,
                                31,
                            ],
                        }
                    ]
                }
            }
        }
    )
    assert replacements is not None

    expanded = MultimodalEncodeWorkerHandler._apply_video_replacements(
        [1, 10, 2, 30, video_token_id, 31, 3, 10, 4],
        replacements,
        token_counts=[2],
        video_token_id=video_token_id,
    )

    assert expanded == [
        1,
        10,
        2,
        30,
        40,
        video_token_id,
        video_token_id,
        41,
        31,
        3,
        10,
        4,
    ]


@pytest.mark.asyncio
async def test_encode_worker_forwards_exact_video_tokens_and_hashes():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler._embedding_cache = None
    handler._cache_publisher = None
    handler._decoded_content_hash_warning_emitted = False
    handler._missing_video_cache_key_config_warned = False
    handler._image_loader = None
    handler.image_token_id = 10
    handler.video_token_id = 20

    frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    metadata = {
        "fps": 24.0,
        "duration": 2.0,
        "frames_indices": [0, 12, 24, 36],
    }
    handler._video_loader = SimpleNamespace(
        load_video_batch=AsyncMock(return_value=[(frames, metadata)])
    )
    handler.encoder = SimpleNamespace(
        _encode=AsyncMock(
            return_value=(
                torch.tensor([2, 2, 4]),
                torch.arange(12, dtype=torch.float32).reshape(4, 3),
                {
                    "second_per_grid_ts": [0.5],
                    "video_timestamps": [[0.25, 0.75]],
                },
            )
        )
    )

    transfer_future = asyncio.get_running_loop().create_future()
    transfer_future.set_result(None)

    class _EmbeddingSender:
        async def send_embeddings(self, embeddings):
            return (
                TransferRequest(
                    embeddings_shape=list(embeddings.shape),
                    embedding_dtype_str=str(embeddings.dtype),
                    serialized_request={"kind": "test"},
                ),
                transfer_future,
            )

    class _PdWorkerClient:
        request = None

        async def round_robin(self, request_json, context=None):
            self.request = json.loads(request_json)

            async def responses():
                yield json.dumps({"token_ids": [7], "finished": True, "text": ""})

            return responses()

    handler.embedding_sender = _EmbeddingSender()
    handler.pd_worker_client = _PdWorkerClient()
    decoded = {
        "shape": [4, 2, 2, 3],
        "dtype": "UINT8",
        "content_hash": "0123456789abcdef",
    }
    replacement_tokens = [30, 40, 20, 20, 41, 20, 20, 31]
    raw_request = {
        "token_ids": [1, 30, 20, 31, 2],
        "stop_conditions": {"max_tokens": 8},
        "sampling_options": {"temperature": 0.0},
        "multi_modal_data": {"video_url": [{"Decoded": decoded}]},
        "extra_args": {
            "mm_hashes_by_modality": {"video": ["0123456789abcdef"]},
            "mm_replacements_by_modality": {
                "video": [
                    {
                        "placeholder_token_id": 20,
                        "target_tokens": [30, 20, 31],
                        "replacement_tokens": replacement_tokens,
                    }
                ]
            },
        },
    }

    outputs = [output async for output in handler.generate(raw_request, context=None)]

    assert outputs == [{"token_ids": [7]}]
    encoded_inputs, encoded_modality = handler.encoder._encode.await_args.args
    assert encoded_modality == Modality.VIDEO
    assert len(encoded_inputs) == 1
    np.testing.assert_array_equal(encoded_inputs[0], frames)
    pd_request = handler.pd_worker_client.request
    assert pd_request["request"]["token_ids"] == [1, *replacement_tokens, 2]
    assert pd_request["mm_hashes"] == ["0123456789abcdef"]
    assert pd_request["multimodal_inputs"][0]["multimodal_input"]["video_url"] is None
    assert pd_request["multimodal_inputs"][0]["num_mm_tokens"] == 4


def test_mm_hashes_are_forwarded_only_when_sglang_supports_them():
    request = SglangMultimodalRequest(
        request=PreprocessedRequest(
            token_ids=[1],
            stop_conditions=StopConditions(max_tokens=1),
            sampling_options=SamplingOptions(temperature=0.0),
        ),
        mm_hashes=["0123456789abcdef"],
    )

    class _SupportedEngine:
        async def async_generate(self, *, mm_hashes=None):
            pass

    class _LegacyEngine:
        async def async_generate(self, *, input_ids):
            pass

    assert _mm_hashes_kwargs(_SupportedEngine(), request) == {
        "mm_hashes": ["0123456789abcdef"]
    }
    assert _mm_hashes_kwargs(_LegacyEngine(), request) == {}


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
