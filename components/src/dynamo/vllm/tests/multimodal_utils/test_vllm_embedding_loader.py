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
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

MODEL = "test-model"
DTYPE = torch.float16


class TestMultimodalEmbeddingLoader:
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
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, cache)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [url],
                "req-1",
                model=MODEL,
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
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, cache)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [url],
                "req-1",
                model=MODEL,
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
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, None)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [url],
                "req-1",
                model=MODEL,
            )

        mock_fetch.assert_awaited_once()
        assert torch.equal(mm_data["image"], tensor)

    @pytest.mark.asyncio
    async def test_decoded_item_cached_by_content_hash(self):
        """A frontend-decoded item reuses the canonical content hash as its
        cache key, so a second request skips the encode worker."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        content_hash = "0123456789abcdef"
        decoded_item = {"Decoded": {"shape": [4, 4, 3], "content_hash": content_hash}}
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
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, cache)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [decoded_item],
                "req-1",
                model=MODEL,
            )
            mm_data_again = await embedding_loader.load_multimodal_embeddings(
                [decoded_item],
                "req-2",
                model=MODEL,
            )

        mock_fetch.assert_awaited_once()
        assert mock_fetch.call_args[0][1] == [decoded_item]
        assert torch.equal(mm_data["image"], tensor)
        assert torch.equal(mm_data_again["image"], tensor)
        cached = cache.get(content_hash)
        assert cached is not None
        assert torch.equal(cached.tensor, tensor)

    @pytest.mark.asyncio
    async def test_decoded_item_without_content_hash_bypasses_cache(self):
        """A decoded descriptor lacking a valid content hash is fetched every
        time and never stored in the cache."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        decoded_item = {"Decoded": {"shape": [4, 4, 3]}}
        tensor = torch.randn(1, 10, dtype=DTYPE)

        def _fake_groups():
            return (
                [
                    MultiModalGroup(
                        multimodal_input=MultiModalInput(),
                        loaded_embedding=tensor,
                    )
                ],
                None,
            )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            side_effect=lambda *args, **kwargs: _fake_groups(),
        ) as mock_fetch:
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, cache)
            await embedding_loader.load_multimodal_embeddings(
                [decoded_item],
                "req-1",
                model=MODEL,
            )
            await embedding_loader.load_multimodal_embeddings(
                [decoded_item],
                "req-2",
                model=MODEL,
            )

        assert mock_fetch.await_count == 2
        assert cache.keys() == []

    def test_parse_image_item_variants(self):
        assert mod.parse_image_item("http://a.png") == ("http://a.png", None)
        assert mod.parse_image_item({"Url": "http://a.png"}) == (
            "http://a.png",
            None,
        )
        metadata = {"shape": [4, 4, 3], "content_hash": "0123456789abcdef"}
        assert mod.parse_image_item({"Decoded": metadata}) == (None, metadata)

        with pytest.raises(ValueError, match="Unsupported image item"):
            mod.parse_image_item({"Url": "http://a.png", "Decoded": metadata})
        with pytest.raises(ValueError, match="Unsupported image item"):
            mod.parse_image_item({"ignored": "value"})
        with pytest.raises(ValueError, match="Unsupported image item"):
            mod.parse_image_item(123)

    def test_image_item_cache_key_domains_are_disjoint(self):
        url_key = mod._image_item_cache_key({"Url": "http://a.png"})
        decoded_key = mod._image_item_cache_key(
            {"Decoded": {"content_hash": "0123456789abcdef"}}
        )
        assert url_key == mod.get_embedding_hash("http://a.png")
        assert len(url_key) == 64
        assert decoded_key == "0123456789abcdef"
        # Malformed hashes are unkeyed rather than mis-keyed.
        for bad in ("0x23456789abcdef", "0123456789ABCDEF", "", None):
            assert mod._image_item_cache_key({"Decoded": {"content_hash": bad}}) is None

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
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, cache)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [url_cached, url_miss],
                "req-1",
                model=MODEL,
            )

        mock_fetch.assert_awaited_once()
        call_args = mock_fetch.call_args
        assert call_args[0][1] == [url_miss]
        expected = torch.cat((cached_tensor, miss_tensor))
        assert torch.equal(mm_data["image"], expected)
