# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Encoder Cache Manager

A simple LRU cache for encoder embeddings (tensors).
Maps content hash keys to tensors with capacity-based eviction.

Usage:
    cache = MultimodalEmbeddingCacheManager(capacity_bytes=4 * 1024**3)  # 4GB

    # Store embedding
    cache.set("abc123", embedding_tensor)

    # Retrieve embedding
    tensor = cache.get("abc123")  # Returns None if not found
"""

import logging
from collections import OrderedDict
from typing import NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)


class CachedEmbedding(NamedTuple):
    tensor: torch.Tensor
    image_grid_thw: list | None = None


class MultimodalEmbeddingCacheManager:
    """
    LRU cache for encoder embeddings.

    Stores tensors keyed by content hash with automatic eviction
    when capacity is exceeded.

    Thread Safety:
        This class is NOT thread-safe. It is designed to run within a single
        thread (e.g., an asyncio event loop). All access must be from the same
        thread to avoid race conditions. This is intentional to keep the
        implementation simple and avoid locking overhead.
    """

    def __init__(self, capacity_bytes: int):
        """
        Initialize the encoder cache.

        Args:
            capacity_bytes: Maximum cache capacity in bytes.
        """
        self._cache: OrderedDict[str, CachedEmbedding] = OrderedDict()
        self._capacity_bytes = capacity_bytes
        self._current_bytes = 0

        # Stats
        self._hits = 0
        self._misses = 0

        logger.info(
            f"MultimodalEmbeddingCacheManager initialized: capacity={capacity_bytes / 1024**3:.2f}GB"
        )

    @staticmethod
    def _tensor_size(tensor: torch.Tensor) -> int:
        """Calculate tensor size in bytes.

        Args:
            tensor: Must be a contiguous tensor.

        Returns:
            Size of the tensor in bytes.

        Raises:
            AssertionError: If tensor is not contiguous.
        """
        assert (
            tensor.is_contiguous()
        ), "Tensor must be contiguous for accurate size calculation"
        return tensor.element_size() * tensor.numel()

    def get(self, key: str) -> Optional[CachedEmbedding]:
        """
        Get a cached embedding from the cache.

        If found, the entry is moved to the end (most recently used).

        Args:
            key: Cache key (typically content hash).

        Returns:
            The cached embedding, or None if not found.
        """
        if key not in self._cache:
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return self._cache[key]

    def set(self, key: str, entry: CachedEmbedding) -> bool:
        """
        Store a cached embedding in the cache.

        If the key already exists, the old value is replaced.
        If adding the entry would exceed capacity, LRU entries are evicted.
        If the tensor itself is larger than capacity, it is not stored.

        Args:
            key: Cache key (typically content hash).
            entry: CachedEmbedding to cache.

        Returns:
            True if the entry was stored, False if it was too large.
        """
        size = self._tensor_size(entry.tensor)

        # Don't cache if single tensor exceeds capacity
        if size > self._capacity_bytes:
            logger.warning(
                f"Tensor too large to cache: {size / 1024**2:.1f}MB > "
                f"{self._capacity_bytes / 1024**3:.2f}GB capacity"
            )
            return False

        # If key exists, remove old entry first
        if key in self._cache:
            old_entry = self._cache.pop(key)
            self._current_bytes -= self._tensor_size(old_entry.tensor)

        # Evict LRU entries until we have space
        while self._current_bytes + size > self._capacity_bytes and self._cache:
            evicted_key, evicted_entry = self._cache.popitem(last=False)
            evicted_size = self._tensor_size(evicted_entry.tensor)
            self._current_bytes -= evicted_size
            logger.debug(
                f"Evicted key={evicted_key[:16]}..., size={evicted_size / 1024**2:.2f}MB"
            )

        # Store new entry
        self._cache[key] = entry
        self._current_bytes += size

        logger.debug(
            f"Cached key={key[:16] if len(key) > 16 else key}, "
            f"size={size / 1024**2:.2f}MB, "
            f"total={self._current_bytes / 1024**3:.3f}GB"
        )
        return True

    @property
    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats including entries, memory usage,
            hit/miss counts, and hit rate.
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "entries": len(self._cache),
            "current_bytes": self._current_bytes,
            "capacity_bytes": self._capacity_bytes,
            "utilization": self._current_bytes / self._capacity_bytes
            if self._capacity_bytes > 0
            else 0,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }
