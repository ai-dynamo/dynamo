# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Async Encoder Cache

Async wrapper over EncoderCacheManager with request coalescing.
Prevents duplicate encoding when multiple requests arrive for the same content.

Usage:
    cache = EncoderCacheManager(capacity_bytes=4 * 1024**3)
    async_cache = AsyncEncoderCache(cache)

    # Get from cache or compute with coalescing
    tensor = await async_cache.get_or_compute("hash123", encoder.encode)
"""

import asyncio
import logging
from typing import Awaitable, Callable, Dict, Optional

import torch

from dynamo.common.memory.encoder_cache_manager import EncoderCacheManager

logger = logging.getLogger(__name__)


def _suppress_unhandled_future_exception(future: asyncio.Future) -> None:
    """
    Callback to prevent 'Future exception was never retrieved' warning.

    When a Future has set_exception() called but no one awaits it (e.g., single
    caller that gets the exception via re-raise), asyncio warns. This callback
    retrieves the exception to suppress that warning.
    """
    if future.done() and not future.cancelled():
        try:
            future.exception()
        except asyncio.CancelledError:
            pass


class AsyncEncoderCache:
    """
    Async wrapper with request coalescing over EncoderCacheManager.

    Provides async get_or_compute that deduplicates concurrent requests
    for the same key, ensuring only one encoding runs at a time per key.

    Thread Safety:
        This class is NOT thread-safe. It is designed to run within a single
        asyncio event loop. All access must be from the same thread.
    """

    def __init__(self, cache: EncoderCacheManager):
        """
        Initialize the async encoder cache.

        Args:
            cache: Underlying EncoderCacheManager for storage.
        """
        self._cache = cache
        self._in_flight: Dict[str, asyncio.Future[torch.Tensor]] = {}

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Synchronous get from underlying cache.

        Args:
            key: Cache key.

        Returns:
            Cached tensor or None if not found.
        """
        return self._cache.get(key)

    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Awaitable[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Get from cache or compute with request coalescing.

        If the key is in cache, returns immediately.
        If another coroutine is already computing this key, waits for that result.
        Otherwise, computes and caches the result.

        Args:
            key: Cache key (typically content hash).
            compute_fn: Async function to compute the tensor if not cached.

        Returns:
            The cached or computed tensor.

        Raises:
            Exception: Re-raises any exception from compute_fn.
        """
        # Check cache first
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Wait if already in-flight
        if key in self._in_flight:
            logger.debug(f"Waiting for in-flight computation: key={key[:16]}...")
            return await self._in_flight[key]

        # Compute with coalescing
        future: asyncio.Future[torch.Tensor] = asyncio.Future()
        future.add_done_callback(_suppress_unhandled_future_exception)
        self._in_flight[key] = future
        try:
            tensor = await compute_fn()
            self._cache.set(key, tensor)
            future.set_result(tensor)
            return tensor
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            del self._in_flight[key]

    @property
    def stats(self) -> dict:
        """
        Get cache statistics from underlying cache.

        Returns:
            Dictionary with cache stats.
        """
        base_stats = self._cache.stats
        base_stats["in_flight"] = len(self._in_flight)
        return base_stats
