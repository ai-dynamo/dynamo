# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MultimodalEmbeddingCacheManager."""

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)


def _entry(tensor: torch.Tensor, grid: list | None = None) -> CachedEmbedding:
    return CachedEmbedding(tensor=tensor, image_grid_thw=grid)


class TestMultimodalEmbeddingCacheManagerBasicOperations:
    """Tests for basic get/set operations."""

    def test_set_and_get(self):
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        tensor = torch.randn(100, 100)

        result = cache.set("key1", _entry(tensor))
        assert result is True

        retrieved = cache.get("key1")
        assert retrieved is not None
        assert torch.equal(retrieved.tensor, tensor)
        assert retrieved.image_grid_thw is None

    def test_set_and_get_with_grid(self):
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        tensor = torch.randn(100, 100)
        grid = [[1, 2, 3]]

        cache.set("key1", _entry(tensor, grid))
        retrieved = cache.get("key1")

        assert retrieved is not None
        assert torch.equal(retrieved.tensor, tensor)
        assert retrieved.image_grid_thw == grid

    def test_get_nonexistent_key(self):
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        assert cache.get("nonexistent") is None

    def test_set_overwrites_existing_key(self):
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        tensor1 = torch.randn(10, 10)
        tensor2 = torch.randn(10, 10)

        cache.set("key1", _entry(tensor1))
        cache.set("key1", _entry(tensor2))

        retrieved = cache.get("key1")
        assert torch.equal(retrieved.tensor, tensor2)
        assert cache.stats["entries"] == 1


class TestMultimodalEmbeddingCacheManagerLRUEviction:
    """Tests for LRU eviction behavior."""

    def test_eviction_when_full(self):
        tensor_size = 10 * 10 * 4  # 400 bytes for float32
        capacity = tensor_size * 2 + 100
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=capacity)

        t1 = torch.randn(10, 10)
        t2 = torch.randn(10, 10)
        t3 = torch.randn(10, 10)

        cache.set("key1", _entry(t1))
        cache.set("key2", _entry(t2))
        cache.set("key3", _entry(t3))

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_get_updates_lru_order(self):
        tensor_size = 10 * 10 * 4
        capacity = tensor_size * 2 + 100
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=capacity)

        t1 = torch.randn(10, 10)
        t2 = torch.randn(10, 10)
        t3 = torch.randn(10, 10)

        cache.set("key1", _entry(t1))
        cache.set("key2", _entry(t2))

        # Access key1, making key2 the LRU
        cache.get("key1")

        cache.set("key3", _entry(t3))

        assert cache.get("key1") is not None
        assert cache.get("key2") is None  # Evicted (LRU)
        assert cache.get("key3") is not None

    def test_tensor_too_large_for_cache(self):
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=100)
        tensor = torch.randn(100, 100)

        result = cache.set("key1", _entry(tensor))

        assert result is False
        assert cache.get("key1") is None
        assert cache.stats["entries"] == 0


class TestMultimodalEmbeddingCacheManagerSizeTracking:
    """Tests for memory size tracking."""

    def test_current_bytes_tracking(self):
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)

        t1 = torch.randn(10, 10)
        t2 = torch.randn(20, 20)

        expected_size_1 = t1.element_size() * t1.numel()
        expected_size_2 = t2.element_size() * t2.numel()

        cache.set("key1", _entry(t1))
        assert cache.stats["current_bytes"] == expected_size_1

        cache.set("key2", _entry(t2))
        assert cache.stats["current_bytes"] == expected_size_1 + expected_size_2

    def test_size_updated_on_overwrite(self):
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)

        small_tensor = torch.randn(10, 10)
        large_tensor = torch.randn(20, 20)

        cache.set("key1", _entry(small_tensor))
        initial_size = cache.stats["current_bytes"]

        cache.set("key1", _entry(large_tensor))

        expected_size = large_tensor.element_size() * large_tensor.numel()
        assert cache.stats["current_bytes"] == expected_size
        assert cache.stats["current_bytes"] > initial_size


class TestMultimodalEmbeddingCacheManagerStats:
    """Tests for statistics tracking."""

    def test_hit_miss_tracking(self):
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        tensor = torch.randn(10, 10)

        cache.set("key1", _entry(tensor))

        cache.get("nonexistent1")
        cache.get("nonexistent2")

        cache.get("key1")
        cache.get("key1")
        cache.get("key1")

        stats = cache.stats
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 3 / 5

    def test_stats_content(self):
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        tensor = torch.randn(10, 10)
        cache.set("key1", _entry(tensor))

        stats = cache.stats

        assert "entries" in stats
        assert "current_bytes" in stats
        assert "capacity_bytes" in stats
        assert "utilization" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

        assert stats["entries"] == 1
        assert stats["capacity_bytes"] == 1024 * 1024

    def test_utilization_calculation(self):
        capacity = 1000
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=capacity)

        tensor = torch.zeros(25, dtype=torch.float32)
        cache.set("key1", _entry(tensor))

        stats = cache.stats
        expected_utilization = 100 / capacity
        assert abs(stats["utilization"] - expected_utilization) < 0.001


class TestMultimodalEmbeddingCacheManagerContiguousTensor:
    """Tests for contiguous tensor requirement."""

    def test_set_contiguous_tensor_succeeds(self):
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        tensor = torch.randn(10, 10)

        assert tensor.is_contiguous()
        result = cache.set("key1", _entry(tensor))
        assert result is True

    def test_set_non_contiguous_tensor_raises(self):
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)

        tensor = torch.randn(10, 20).t()
        assert not tensor.is_contiguous()

        with pytest.raises(AssertionError, match="Tensor must be contiguous"):
            cache.set("key1", _entry(tensor))


class TestCachedEmbeddingNamedTuple:
    """Tests for CachedEmbedding NamedTuple."""

    def test_fields(self):
        tensor = torch.randn(4, 4)
        grid = [[1, 2, 3]]
        entry = CachedEmbedding(tensor=tensor, image_grid_thw=grid)

        assert torch.equal(entry.tensor, tensor)
        assert entry.image_grid_thw == grid

    def test_none_grid(self):
        tensor = torch.randn(4, 4)
        entry = CachedEmbedding(tensor=tensor, image_grid_thw=None)
        assert entry.image_grid_thw is None

    def test_unpacking(self):
        tensor = torch.randn(4, 4)
        grid = [[1, 2, 3]]
        entry = CachedEmbedding(tensor=tensor, image_grid_thw=grid)
        t, g = entry
        assert torch.equal(t, tensor)
        assert g == grid
