# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for EncoderCacheManager."""

import pytest
import torch

from dynamo.common.memory.encoder_cache_manager import EncoderCacheManager


class TestEncoderCacheManagerInit:
    """Tests for initialization."""

    def test_init_valid_capacity(self):
        """Test initialization with valid capacity."""
        cache = EncoderCacheManager(capacity_bytes=1024)
        assert cache.stats["capacity_bytes"] == 1024
        assert cache.stats["current_bytes"] == 0
        assert cache.stats["entries"] == 0

    def test_init_invalid_capacity_zero(self):
        """Test initialization with zero capacity raises error."""
        with pytest.raises(ValueError, match="capacity_bytes must be positive"):
            EncoderCacheManager(capacity_bytes=0)

    def test_init_invalid_capacity_negative(self):
        """Test initialization with negative capacity raises error."""
        with pytest.raises(ValueError, match="capacity_bytes must be positive"):
            EncoderCacheManager(capacity_bytes=-100)


class TestEncoderCacheManagerBasicOperations:
    """Tests for basic get/set operations."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = EncoderCacheManager(capacity_bytes=1024 * 1024)  # 1MB
        tensor = torch.randn(100, 100)  # ~40KB for float32

        result = cache.set("key1", tensor)
        assert result is True

        retrieved = cache.get("key1")
        assert retrieved is not None
        assert torch.equal(retrieved, tensor)

    def test_get_nonexistent_key(self):
        """Test get returns None for nonexistent key."""
        cache = EncoderCacheManager(capacity_bytes=1024 * 1024)

        result = cache.get("nonexistent")
        assert result is None

    def test_set_overwrites_existing_key(self):
        """Test set overwrites existing key."""
        cache = EncoderCacheManager(capacity_bytes=1024 * 1024)
        tensor1 = torch.randn(10, 10)
        tensor2 = torch.randn(10, 10)

        cache.set("key1", tensor1)
        cache.set("key1", tensor2)

        retrieved = cache.get("key1")
        assert torch.equal(retrieved, tensor2)
        assert cache.stats["entries"] == 1


class TestEncoderCacheManagerLRUEviction:
    """Tests for LRU eviction behavior."""

    def test_eviction_when_full(self):
        """Test LRU eviction when cache is full."""
        # Small capacity to force eviction
        tensor_size = 10 * 10 * 4  # 400 bytes for float32
        capacity = tensor_size * 2 + 100  # Room for ~2 tensors
        cache = EncoderCacheManager(capacity_bytes=capacity)

        t1 = torch.randn(10, 10)
        t2 = torch.randn(10, 10)
        t3 = torch.randn(10, 10)

        cache.set("key1", t1)
        cache.set("key2", t2)

        # Adding third should evict first (LRU)
        cache.set("key3", t3)

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_get_updates_lru_order(self):
        """Test that get() updates LRU order."""
        tensor_size = 10 * 10 * 4  # 400 bytes
        capacity = tensor_size * 2 + 100  # Room for ~2 tensors
        cache = EncoderCacheManager(capacity_bytes=capacity)

        t1 = torch.randn(10, 10)
        t2 = torch.randn(10, 10)
        t3 = torch.randn(10, 10)

        cache.set("key1", t1)
        cache.set("key2", t2)

        # Access key1, making key2 the LRU
        cache.get("key1")

        # Adding third should evict key2 (now LRU)
        cache.set("key3", t3)

        assert cache.get("key1") is not None  # Not evicted (recently accessed)
        assert cache.get("key2") is None  # Evicted (LRU)
        assert cache.get("key3") is not None

    def test_tensor_too_large_for_cache(self):
        """Test that tensor larger than capacity is not cached."""
        cache = EncoderCacheManager(capacity_bytes=100)  # Very small
        tensor = torch.randn(100, 100)  # ~40KB, way larger than capacity

        result = cache.set("key1", tensor)

        assert result is False
        assert cache.get("key1") is None
        assert cache.stats["entries"] == 0


class TestEncoderCacheManagerSizeTracking:
    """Tests for memory size tracking."""

    def test_current_bytes_tracking(self):
        """Test that current_bytes is tracked correctly."""
        cache = EncoderCacheManager(capacity_bytes=1024 * 1024)

        t1 = torch.randn(10, 10)  # 400 bytes
        t2 = torch.randn(20, 20)  # 1600 bytes

        expected_size_1 = t1.element_size() * t1.numel()
        expected_size_2 = t2.element_size() * t2.numel()

        cache.set("key1", t1)
        assert cache.stats["current_bytes"] == expected_size_1

        cache.set("key2", t2)
        assert cache.stats["current_bytes"] == expected_size_1 + expected_size_2

    def test_size_updated_on_overwrite(self):
        """Test that size is updated correctly when overwriting."""
        cache = EncoderCacheManager(capacity_bytes=1024 * 1024)

        small_tensor = torch.randn(10, 10)  # 400 bytes
        large_tensor = torch.randn(20, 20)  # 1600 bytes

        cache.set("key1", small_tensor)
        initial_size = cache.stats["current_bytes"]

        cache.set("key1", large_tensor)

        expected_size = large_tensor.element_size() * large_tensor.numel()
        assert cache.stats["current_bytes"] == expected_size
        assert cache.stats["current_bytes"] > initial_size


class TestEncoderCacheManagerStats:
    """Tests for statistics tracking."""

    def test_hit_miss_tracking(self):
        """Test hit and miss counting."""
        cache = EncoderCacheManager(capacity_bytes=1024 * 1024)
        tensor = torch.randn(10, 10)

        cache.set("key1", tensor)

        # Misses
        cache.get("nonexistent1")
        cache.get("nonexistent2")

        # Hits
        cache.get("key1")
        cache.get("key1")
        cache.get("key1")

        stats = cache.stats
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 3 / 5

    def test_stats_content(self):
        """Test stats dictionary contains expected keys."""
        cache = EncoderCacheManager(capacity_bytes=1024 * 1024)
        tensor = torch.randn(10, 10)
        cache.set("key1", tensor)

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
        """Test utilization is calculated correctly."""
        capacity = 1000
        cache = EncoderCacheManager(capacity_bytes=capacity)

        # Create tensor of known size
        # float32 = 4 bytes, so 25 elements = 100 bytes
        tensor = torch.zeros(25, dtype=torch.float32)
        cache.set("key1", tensor)

        stats = cache.stats
        expected_utilization = 100 / capacity
        assert abs(stats["utilization"] - expected_utilization) < 0.001


class TestEncoderCacheManagerContiguousTensor:
    """Tests for contiguous tensor requirement."""

    def test_set_contiguous_tensor_succeeds(self):
        """Test that contiguous tensors can be cached."""
        cache = EncoderCacheManager(capacity_bytes=1024 * 1024)
        tensor = torch.randn(10, 10)

        assert tensor.is_contiguous()
        result = cache.set("key1", tensor)
        assert result is True

    def test_set_non_contiguous_tensor_raises(self):
        """Test that non-contiguous tensors raise AssertionError."""
        cache = EncoderCacheManager(capacity_bytes=1024 * 1024)

        # Create a non-contiguous tensor via transpose
        tensor = torch.randn(10, 20).t()
        assert not tensor.is_contiguous()

        with pytest.raises(AssertionError, match="Tensor must be contiguous"):
            cache.set("key1", tensor)
