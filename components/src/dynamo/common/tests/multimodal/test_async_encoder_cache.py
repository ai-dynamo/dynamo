# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AsyncEncoderCache."""

import asyncio

import pytest
import torch

from dynamo.common.memory.encoder_cache_manager import EncoderCacheManager
from dynamo.common.multimodal.async_encoder_cache import AsyncEncoderCache


class TestAsyncEncoderCacheBasicOperations:
    """Tests for basic operations."""

    @pytest.fixture
    def cache(self):
        """Create a cache for testing."""
        ecm = EncoderCacheManager(capacity_bytes=1024 * 1024)
        return AsyncEncoderCache(ecm)

    def test_sync_get_returns_none_for_missing_key(self, cache):
        """Test sync get returns None for nonexistent key."""
        assert cache.get("nonexistent") is None

    def test_sync_get_returns_cached_tensor(self, cache):
        """Test sync get returns tensor after it's cached."""
        tensor = torch.randn(10, 10)
        cache._cache.set("key1", tensor)

        result = cache.get("key1")
        assert torch.equal(result, tensor)

    @pytest.mark.asyncio
    async def test_get_or_compute_caches_result(self, cache):
        """Test get_or_compute caches the computed result."""
        tensor = torch.randn(10, 10)

        async def compute():
            return tensor

        result = await cache.get_or_compute("key1", compute)
        assert torch.equal(result, tensor)

        # Should be in cache now
        cached = cache.get("key1")
        assert cached is not None
        assert torch.equal(cached, tensor)

    @pytest.mark.asyncio
    async def test_get_or_compute_returns_cached(self, cache):
        """Test get_or_compute returns cached value without computing."""
        tensor = torch.randn(10, 10)
        cache._cache.set("key1", tensor)

        compute_called = False

        async def compute():
            nonlocal compute_called
            compute_called = True
            return torch.randn(10, 10)

        result = await cache.get_or_compute("key1", compute)

        assert torch.equal(result, tensor)
        assert not compute_called


class TestAsyncEncoderCacheRequestCoalescing:
    """Tests for request coalescing behavior."""

    @pytest.fixture
    def cache(self):
        """Create a cache for testing."""
        ecm = EncoderCacheManager(capacity_bytes=1024 * 1024)
        return AsyncEncoderCache(ecm)

    @pytest.mark.asyncio
    async def test_concurrent_requests_coalesce(self, cache):
        """Test that concurrent requests for same key only compute once."""
        compute_count = 0
        tensor = torch.randn(10, 10)
        compute_started = asyncio.Event()
        compute_proceed = asyncio.Event()

        async def compute():
            nonlocal compute_count
            compute_count += 1
            compute_started.set()  # Signal that compute has started
            await compute_proceed.wait()  # Wait for permission to proceed
            return tensor

        # Start concurrent requests as tasks
        task1 = asyncio.create_task(cache.get_or_compute("key1", compute))
        task2 = asyncio.create_task(cache.get_or_compute("key1", compute))
        task3 = asyncio.create_task(cache.get_or_compute("key1", compute))

        # Wait for compute to start (ensures requests are queued)
        await compute_started.wait()

        # Allow compute to complete
        compute_proceed.set()

        results = await asyncio.gather(task1, task2, task3)

        # All should get the same tensor
        for result in results:
            assert torch.equal(result, tensor)

        # But compute should only be called once
        assert compute_count == 1

    @pytest.mark.asyncio
    async def test_different_keys_compute_separately(self, cache):
        """Test that different keys compute independently."""
        compute_count = 0

        async def compute():
            nonlocal compute_count
            compute_count += 1
            return torch.randn(10, 10)

        await asyncio.gather(
            cache.get_or_compute("key1", compute),
            cache.get_or_compute("key2", compute),
            cache.get_or_compute("key3", compute),
        )

        assert compute_count == 3


class TestAsyncEncoderCacheExceptionHandling:
    """Tests for exception handling."""

    @pytest.fixture
    def cache(self):
        """Create a cache for testing."""
        ecm = EncoderCacheManager(capacity_bytes=1024 * 1024)
        return AsyncEncoderCache(ecm)

    @pytest.mark.asyncio
    async def test_exception_propagates_to_caller(self, cache):
        """Test that compute exceptions propagate to the caller."""

        async def compute():
            raise ValueError("compute failed")

        with pytest.raises(ValueError, match="compute failed"):
            await cache.get_or_compute("key1", compute)

    @pytest.mark.asyncio
    async def test_exception_propagates_to_all_waiters(self, cache):
        """Test that compute exceptions propagate to all waiting coroutines."""
        compute_started = asyncio.Event()
        compute_proceed = asyncio.Event()

        async def compute():
            compute_started.set()
            await compute_proceed.wait()
            raise ValueError("compute failed")

        # Start concurrent requests as tasks
        task1 = asyncio.create_task(cache.get_or_compute("key1", compute))
        task2 = asyncio.create_task(cache.get_or_compute("key1", compute))

        # Wait for compute to start
        await compute_started.wait()

        # Allow compute to proceed (and fail)
        compute_proceed.set()

        # Gather with return_exceptions=True to capture all results
        results = await asyncio.gather(task1, task2, return_exceptions=True)

        # Verify ALL tasks got the exception
        assert len(results) == 2
        for result in results:
            assert isinstance(result, ValueError)
            assert str(result) == "compute failed"

    @pytest.mark.asyncio
    async def test_in_flight_cleared_after_exception(self, cache):
        """Test that in_flight is cleared after an exception."""

        async def failing_compute():
            raise ValueError("compute failed")

        with pytest.raises(ValueError):
            await cache.get_or_compute("key1", failing_compute)

        # in_flight should be empty
        assert len(cache._in_flight) == 0

        # Should be able to retry
        tensor = torch.randn(10, 10)

        async def working_compute():
            return tensor

        result = await cache.get_or_compute("key1", working_compute)
        assert torch.equal(result, tensor)


class TestAsyncEncoderCacheStats:
    """Tests for statistics."""

    @pytest.fixture
    def cache(self):
        """Create a cache for testing."""
        ecm = EncoderCacheManager(capacity_bytes=1024 * 1024)
        return AsyncEncoderCache(ecm)

    def test_stats_includes_in_flight(self, cache):
        """Test that stats include in_flight count."""
        stats = cache.stats
        assert "in_flight" in stats
        assert stats["in_flight"] == 0

    @pytest.mark.asyncio
    async def test_stats_reflects_underlying_cache(self, cache):
        """Test that stats reflect underlying cache state."""
        tensor = torch.randn(10, 10)

        async def compute():
            return tensor

        await cache.get_or_compute("key1", compute)

        stats = cache.stats
        assert stats["entries"] == 1
        assert (
            stats["hits"] == 0
        )  # get_or_compute checks cache but we track differently
        assert stats["in_flight"] == 0
