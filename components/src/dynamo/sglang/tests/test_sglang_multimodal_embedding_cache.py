# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang multimodal embedding cache behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.sglang.request_handlers.multimodal.encode_worker_handler import (
    MultimodalEncodeWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,  # sglang tests run on GPU-enabled workers
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


@pytest.fixture
def cache_handler() -> MultimodalEncodeWorkerHandler:
    """Create a lightweight handler instance for cache-path unit tests."""
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)
    handler._embedding_cache = MultimodalEmbeddingCacheManager(
        capacity_bytes=32 * 1024 * 1024
    )
    handler.encoder = SimpleNamespace(_encode=AsyncMock())
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

    # Pre-cache url[1] (4 tokens x 3 hidden)
    cached_tensor = torch.full((4, 3), fill_value=-1.0)
    cache_handler._embedding_cache.set(
        cache_handler._url_hash(urls[1]),
        CachedEmbedding(tensor=cached_tensor, image_grid_thw=[1, 2, 2]),
    )

    # Encode only misses url[0], url[2]: token counts [8, 4]
    encoded = torch.arange(12 * 3, dtype=torch.float32).reshape(12, 3)
    cache_handler.encoder._encode.return_value = (
        torch.tensor([[1, 2, 4], [1, 2, 2]]),
        encoded,
    )

    grid, full_embeddings = await cache_handler._encode_with_cache(urls)

    # Encoder called once for uncached URLs only
    cache_handler.encoder._encode.assert_awaited_once_with([urls[0], urls[2]])

    # Order should match original URL order: a(8), b(4 cached), c(4)
    assert grid.tolist() == [[1, 2, 4], [1, 2, 2], [1, 2, 2]]
    assert torch.equal(full_embeddings[:8], encoded[:8])
    assert torch.equal(full_embeddings[8:12], cached_tensor)
    assert torch.equal(full_embeddings[12:16], encoded[8:12])

    # Second call should be all-cache hit: no additional encoder calls
    grid2, full_embeddings2 = await cache_handler._encode_with_cache(urls)
    assert cache_handler.encoder._encode.await_count == 1
    assert grid2.tolist() == grid.tolist()
    assert torch.equal(full_embeddings2, full_embeddings)


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

    grid, full_embeddings = await cache_handler._encode_with_cache(urls)
    cache_handler.encoder._encode.assert_not_called()
    assert grid.tolist() == [[1, 1, 2], [1, 1, 1]]
    assert torch.equal(full_embeddings, torch.cat([x, y], dim=0))
