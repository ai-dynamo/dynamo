# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for load_multimodal_embeddings in prefill_worker_utils."""

from unittest.mock import AsyncMock, patch

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.vllm.multimodal_utils import prefill_worker_utils as mod
from dynamo.vllm.multimodal_utils.protocol import MultiModalGroup, MultiModalInput

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

MODEL = "test-model"
DTYPE = torch.float16


class TestLoadMultimodalEmbeddings:
    @pytest.mark.asyncio
    async def test_all_cached(self):
        """All URLs cached -> no encode worker call, returns accumulated mm_data."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        tensor = torch.randn(1, 10, dtype=DTYPE)
        grid = [[1, 2, 3]]
        url = "http://img1.png"
        key = mod.get_embedding_hash(url)
        cache.set(key, CachedEmbedding(tensor=tensor, image_grid_thw=grid))

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mm_data = await mod.load_multimodal_embeddings(
                AsyncMock(),
                [url],
                "req-1",
                None,
                model=MODEL,
                embeddings_dtype=DTYPE,
                cache=cache,
            )

        mock_fetch.assert_not_awaited()
        assert torch.equal(mm_data["image"], tensor)

    @pytest.mark.asyncio
    async def test_all_uncached_with_cache(self):
        """All URLs uncached with cache -> encode worker call, results cached."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        url = "http://img1.png"
        tensor = torch.randn(1, 10, dtype=DTYPE)
        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            image_grid_thw=[[1, 2, 3]],
            loaded_embedding=tensor,
        )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([fake_group], None),
        ) as mock_fetch:
            mm_data = await mod.load_multimodal_embeddings(
                AsyncMock(),
                [url],
                "req-1",
                None,
                model=MODEL,
                embeddings_dtype=DTYPE,
                cache=cache,
            )

        mock_fetch.assert_awaited_once()
        assert torch.equal(mm_data["image"], tensor)

        key = mod.get_embedding_hash(url)
        cached = cache.get(key)
        assert cached is not None
        assert torch.equal(cached.tensor, tensor)

    @pytest.mark.asyncio
    async def test_no_cache(self):
        """Without cache -> all URLs go to encode workers."""
        url = "http://img1.png"
        tensor = torch.randn(1, 10, dtype=DTYPE)
        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            loaded_embedding=tensor,
        )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([fake_group], None),
        ) as mock_fetch:
            mm_data = await mod.load_multimodal_embeddings(
                AsyncMock(),
                [url],
                "req-1",
                None,
                model=MODEL,
                embeddings_dtype=DTYPE,
                cache=None,
            )

        mock_fetch.assert_awaited_once()
        assert torch.equal(mm_data["image"], tensor)

    @pytest.mark.asyncio
    async def test_mixed_cache(self):
        """Mixed cache hits/misses -> only misses sent to encode workers."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)

        url_cached = "http://cached.png"
        url_miss = "http://miss.png"
        cached_tensor = torch.randn(1, 10, dtype=DTYPE)
        miss_tensor = torch.randn(1, 10, dtype=DTYPE)

        key = mod.get_embedding_hash(url_cached)
        cache.set(key, CachedEmbedding(tensor=cached_tensor, image_grid_thw=None))

        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            image_grid_thw=None,
            loaded_embedding=miss_tensor,
        )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([fake_group], None),
        ) as mock_fetch:
            mm_data = await mod.load_multimodal_embeddings(
                AsyncMock(),
                [url_cached, url_miss],
                "req-1",
                None,
                model=MODEL,
                embeddings_dtype=DTYPE,
                cache=cache,
            )

        mock_fetch.assert_awaited_once()
        call_args = mock_fetch.call_args
        assert call_args[0][1] == [url_miss]
        expected = torch.cat((cached_tensor, miss_tensor))
        assert torch.equal(mm_data["image"], expected)
