# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang multimodal embedding cache behavior."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal import TransferRequest
from dynamo.sglang.request_handlers.multimodal.encode_worker_handler import (
    Modality,
    MultimodalEncodeWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.multimodal,
    pytest.mark.gpu_1,  # sglang tests run on GPU-enabled workers
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.skipif(Modality is None, reason="SGLang Modality is required"),
]


@pytest.fixture
def cache_handler() -> MultimodalEncodeWorkerHandler:
    """Create a lightweight handler instance for cache-path unit tests."""

    class _DummyEncoder:
        def __init__(self) -> None:
            self.encode_mock = AsyncMock()
            self.model_type = "test_video_model"
            self.vision_config = {
                "video": {"fps": 2.0, "max_frames": 768, "min_frames": 4}
            }
            self.video_processor = SimpleNamespace(temporal_patch_size=2)

        async def _encode(self, mm_items, modality):
            return await self.encode_mock(mm_items, modality)

    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)

    def _set_token_ids_for_test(image_token_id: int, video_token_id: int) -> None:
        handler.image_token_id = image_token_id
        handler.video_token_id = video_token_id

    handler.set_token_ids_for_test = _set_token_ids_for_test
    handler.set_token_ids_for_test(151655, 151656)
    handler._missing_video_cache_key_config_warned = False
    handler._decoded_content_hash_warning_emitted = False
    handler._image_loader = None
    handler._video_loader = None
    handler._embedding_cache = MultimodalEmbeddingCacheManager(
        capacity_bytes=32 * 1024 * 1024
    )
    handler._cache_publisher = None
    handler.encoder = _DummyEncoder()
    return handler


@pytest.mark.asyncio
async def test_encode_with_cache_partial_hit_and_reuse(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    """Partial-hit should encode only misses and preserve URL order in output."""

    urls = [
        "http://example.com/a.jpg",
        "http://example.com/b.jpg",
        "http://example.com/c.jpg",
    ]

    cached_tensor = torch.full((4, 3), fill_value=-1.0)
    cache_handler._embedding_cache.set(
        cache_handler._url_hash(urls[1]),
        CachedEmbedding(tensor=cached_tensor, image_grid_thw=[1, 2, 2]),
    )

    encoded = torch.arange(12 * 3, dtype=torch.float32).reshape(12, 3)
    cache_handler.encoder.encode_mock.return_value = (
        torch.tensor([[1, 2, 4], [1, 2, 2]]),
        encoded,
        None,
    )

    cache_keys = [cache_handler._url_hash(url) for url in urls]
    grid, full_embeddings, entries = await cache_handler._encode_with_cache(
        urls, cache_keys, Modality.IMAGE
    )

    cache_handler.encoder.encode_mock.assert_awaited_once_with(
        [urls[0], urls[2]], Modality.IMAGE
    )

    assert grid.tolist() == [[1, 2, 4], [1, 2, 2], [1, 2, 2]]
    assert [entry.image_grid_thw for entry in entries] == [
        [1, 2, 4],
        [1, 2, 2],
        [1, 2, 2],
    ]
    assert torch.equal(full_embeddings[:8], encoded[:8])
    assert torch.equal(full_embeddings[8:12], cached_tensor)
    assert torch.equal(full_embeddings[12:16], encoded[8:12])
    new_cached_entry = cache_handler._embedding_cache.get(
        cache_handler._url_hash(urls[0])
    )
    assert new_cached_entry is not None
    assert new_cached_entry.image_grid_thw == [1, 2, 4]
    assert new_cached_entry.video_grid_thw is None

    grid2, full_embeddings2, entries2 = await cache_handler._encode_with_cache(
        urls, cache_keys, Modality.IMAGE
    )
    assert cache_handler.encoder.encode_mock.await_count == 1
    assert grid2.tolist() == grid.tolist()
    assert [entry.image_grid_thw for entry in entries2] == [
        entry.image_grid_thw for entry in entries
    ]
    assert torch.equal(full_embeddings2, full_embeddings)


def test_publish_cache_delta_delegates_to_publisher(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    publisher = SimpleNamespace(publish_delta=AsyncMock())
    # publish_delta is sync in production bindings; AsyncMock still tracks calls.
    publisher.publish_delta = lambda added, removed: setattr(
        publisher, "last_call", (added, removed)
    )
    cache_handler._cache_publisher = publisher

    cache_handler._publish_cache_delta(["a"], ["b"])

    assert publisher.last_call == (["a"], ["b"])


@pytest.mark.asyncio
async def test_encode_with_cache_all_hit_no_remote_call(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    """All-cache-hit path should not call encoder at all."""
    urls = ["http://example.com/x.jpg", "http://example.com/y.jpg"]
    x = torch.ones(2, 3)
    y = torch.ones(1, 3) * 9

    cache_handler._embedding_cache.set(
        cache_handler._url_hash(urls[0]),
        CachedEmbedding(tensor=x, image_grid_thw=[1, 1, 2]),
    )
    cache_handler._embedding_cache.set(
        cache_handler._url_hash(urls[1]),
        CachedEmbedding(tensor=y, image_grid_thw=[1, 1, 1]),
    )

    cache_keys = [cache_handler._url_hash(url) for url in urls]
    grid, full_embeddings, entries = await cache_handler._encode_with_cache(
        urls, cache_keys, Modality.IMAGE
    )
    cache_handler.encoder.encode_mock.assert_not_called()
    assert grid.tolist() == [[1, 1, 2], [1, 1, 1]]
    assert [entry.image_grid_thw for entry in entries] == [
        [1, 1, 2],
        [1, 1, 1],
    ]
    assert torch.equal(full_embeddings, torch.cat([x, y], dim=0))


@pytest.mark.asyncio
async def test_encode_with_cache_keeps_prechecked_hit_after_eviction(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    """A decoded-image hit stays usable after it leaves the bounded LRU."""
    cache_key = "0123456789abcdef"
    cached_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    cached_entry = CachedEmbedding(
        tensor=cached_tensor,
        image_grid_thw=[1, 1, 2],
    )

    # _prepare_image_inputs holds the returned object while an asynchronous
    # NIXL read materializes other misses. Simulate eviction during that wait.
    cache_handler._embedding_cache = MultimodalEmbeddingCacheManager(
        capacity_bytes=cached_tensor.element_size() * cached_tensor.numel()
    )
    cache_handler._embedding_cache.set(cache_key, cached_entry)
    prechecked_entry = cache_handler._embedding_cache.get(cache_key)
    assert prechecked_entry is cached_entry
    cache_handler._embedding_cache.set(
        "fedcba9876543210",
        CachedEmbedding(
            tensor=torch.ones_like(cached_tensor),
            image_grid_thw=[1, 1, 2],
        ),
    )
    assert cache_handler._embedding_cache.get(cache_key) is None

    grid, embeddings, entries = await cache_handler._encode_with_cache(
        [None],
        [cache_key],
        Modality.IMAGE,
        prechecked_entries={0: prechecked_entry},
    )

    cache_handler.encoder.encode_mock.assert_not_called()
    assert grid.tolist() == [[1, 1, 2]]
    assert torch.equal(embeddings, cached_tensor)
    assert len(entries) == 1
    assert entries[0] is cached_entry


@pytest.mark.asyncio
async def test_encode_with_cache_reencodes_only_unkeyed_items(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    media_inputs = ["cacheable", "old-decoded-descriptor"]
    cache_keys = ["0123456789abcdef", None]
    first_embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    cache_handler.encoder.encode_mock.return_value = (
        torch.tensor([[1, 1, 1], [1, 1, 1]]),
        first_embeddings,
        None,
    )

    _, encoded, _ = await cache_handler._encode_with_cache(
        media_inputs, cache_keys, Modality.IMAGE
    )
    assert torch.equal(encoded, first_embeddings)
    assert cache_handler._embedding_cache.keys() == [cache_keys[0]]

    cache_handler.encoder.encode_mock.reset_mock()
    cache_handler.encoder.encode_mock.return_value = (
        torch.tensor([1, 1, 1]),
        torch.tensor([[5.0, 6.0]]),
        None,
    )

    _, encoded_again, _ = await cache_handler._encode_with_cache(
        media_inputs, cache_keys, Modality.IMAGE
    )

    cache_handler.encoder.encode_mock.assert_awaited_once_with(
        [media_inputs[1]], Modality.IMAGE
    )
    assert torch.equal(encoded_again, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))


@pytest.mark.asyncio
async def test_video_requests_reuse_cached_embeddings(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    """Second identical video request should reuse cached embeddings."""

    video_url = "https://example.com/clip.mp4"
    video_token_id = cache_handler.video_token_id

    cache_handler.encoder.encode_mock.return_value = (
        torch.tensor([2, 3, 4]),
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        {
            "second_per_grid_ts": [0.5],
            "video_timestamps": [[0.25, 0.75]],
        },
    )

    transfer_future = asyncio.get_running_loop().create_future()
    transfer_future.set_result(None)

    class _DummyEmbeddingSender:
        async def send_embeddings(self, embeddings):
            self.embeddings = embeddings
            return (
                TransferRequest(
                    embeddings_shape=list(embeddings.shape),
                    embedding_dtype_str=str(embeddings.dtype),
                    serialized_request={"kind": "mock-transfer"},
                ),
                transfer_future,
            )

    class _DummyPdWorkerClient:
        def __init__(self) -> None:
            self.requests = []

        async def round_robin(self, request_json, context=None):
            self.requests.append((json.loads(request_json), context))

            async def _responses():
                yield json.dumps({"token_ids": [7], "finished": True, "text": ""})

            return _responses()

    cache_handler.embedding_sender = _DummyEmbeddingSender()
    cache_handler.pd_worker_client = _DummyPdWorkerClient()

    raw_request = {
        "token_ids": [101, video_token_id, 102],
        "stop_conditions": {"max_tokens": 8},
        "sampling_options": {"temperature": 0.0},
        "multi_modal_data": {"video_url": [{"Url": video_url}]},
    }

    outputs = []
    async for item in cache_handler.generate(raw_request, context=None):
        outputs.append(item)

    outputs_second = []
    async for item in cache_handler.generate(raw_request, context=None):
        outputs_second.append(item)

    cache_handler.encoder.encode_mock.assert_awaited_once_with(
        [video_url], Modality.VIDEO
    )

    assert outputs == [{"token_ids": [7]}]
    assert outputs_second == [{"token_ids": [7]}]

    cached_entry = cache_handler._embedding_cache.get(
        cache_handler._media_cache_key(video_url, Modality.VIDEO, cache_handler.encoder)
    )
    assert cached_entry is not None
    assert cached_entry.video_grid_thw == [2, 3, 4]
    assert cached_entry.second_per_grid_ts == 0.5
    assert cached_entry.video_timestamps == [0.25, 0.75]

    assert len(cache_handler.pd_worker_client.requests) == 2
    for pd_request, request_context in cache_handler.pd_worker_client.requests:
        assert request_context is None
        assert pd_request["request"]["token_ids"] == [101] + [video_token_id] * 6 + [
            102
        ]
        group = pd_request["multimodal_inputs"][0]
        assert group["video_grid_thw"] == [2, 3, 4]
        assert group["second_per_grid_ts"] == 0.5
        assert group["video_timestamps"] == [0.25, 0.75]
        assert group["num_mm_tokens"] == 6


@pytest.mark.asyncio
async def test_video_request_skips_cache_key_when_cache_is_disabled(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    video_url = "https://example.com/clip.mp4"
    video_token_id = cache_handler.video_token_id
    cache_handler._embedding_cache = None
    cache_handler._image_loader = None
    cache_handler._media_cache_key = Mock(
        side_effect=AssertionError("video cache key must not be computed")
    )
    cache_handler.encoder.encode_mock.return_value = (
        torch.tensor([2, 3, 4]),
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        {
            "second_per_grid_ts": [0.5],
            "video_timestamps": [[0.25, 0.75]],
        },
    )

    transfer_future = asyncio.get_running_loop().create_future()
    transfer_future.set_result(None)

    class _DummyEmbeddingSender:
        async def send_embeddings(self, embeddings):
            return (
                TransferRequest(
                    embeddings_shape=list(embeddings.shape),
                    embedding_dtype_str=str(embeddings.dtype),
                    serialized_request={"kind": "mock-transfer"},
                ),
                transfer_future,
            )

    class _DummyPdWorkerClient:
        async def round_robin(self, request_json, context=None):
            async def _responses():
                yield json.dumps({"token_ids": [7], "finished": True, "text": ""})

            return _responses()

    cache_handler.embedding_sender = _DummyEmbeddingSender()
    cache_handler.pd_worker_client = _DummyPdWorkerClient()
    raw_request = {
        "token_ids": [101, video_token_id, 102],
        "stop_conditions": {"max_tokens": 8},
        "sampling_options": {"temperature": 0.0},
        "multi_modal_data": {"video_url": [{"Url": video_url}]},
    }

    outputs = [item async for item in cache_handler.generate(raw_request, context=None)]

    assert outputs == [{"token_ids": [7]}]
    cache_handler._media_cache_key.assert_not_called()
    cache_handler.encoder.encode_mock.assert_awaited_once_with(
        [video_url], Modality.VIDEO
    )


def test_aux_value_for_item_rejects_mismatched_batched_lists() -> None:
    with pytest.raises(ValueError, match="Auxiliary media metadata length mismatch"):
        MultimodalEncodeWorkerHandler._aux_value_for_item([0.5], 0, 2)

    assert MultimodalEncodeWorkerHandler._aux_value_for_item(0.5, 1, 2) == 0.5
    assert (
        MultimodalEncodeWorkerHandler._aux_value_for_item(torch.tensor([0.5]), 0, 1)
        == 0.5
    )
    assert MultimodalEncodeWorkerHandler._aux_value_for_item([0.25, 0.75], 0, 1) == [
        0.25,
        0.75,
    ]


@pytest.mark.asyncio
async def test_video_cache_key_includes_sampling_config(
    cache_handler: MultimodalEncodeWorkerHandler,
) -> None:
    """Changing video sampling config should force a cache miss for the same URL."""

    video_url = "https://example.com/clip.mp4"
    cache_handler.encoder.encode_mock.return_value = (
        torch.tensor([[2, 3, 4]]),
        torch.arange(24, dtype=torch.float32).reshape(6, 4),
        {
            "second_per_grid_ts": [0.5],
            "video_timestamps": [[0.25, 0.75]],
        },
    )

    first_key = cache_handler._media_cache_key(
        video_url, Modality.VIDEO, cache_handler.encoder
    )
    await cache_handler._encode_with_cache([video_url], [first_key], Modality.VIDEO)

    cache_handler.encoder.vision_config["video"]["fps"] = 4.0

    second_key = cache_handler._media_cache_key(
        video_url, Modality.VIDEO, cache_handler.encoder
    )
    await cache_handler._encode_with_cache([video_url], [second_key], Modality.VIDEO)
    await cache_handler._encode_with_cache([video_url], [second_key], Modality.VIDEO)

    assert first_key != second_key
    assert cache_handler.encoder.encode_mock.await_count == 2
    assert cache_handler._embedding_cache.get(first_key) is not None
    assert cache_handler._embedding_cache.get(second_key) is not None
