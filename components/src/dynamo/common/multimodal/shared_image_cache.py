# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Redis Cluster-backed cache for encoded multimodal image bytes."""

import hashlib
import logging
import os
import threading
import time
from collections import defaultdict
from typing import DefaultDict

from redis.asyncio.cluster import RedisCluster
from redis.exceptions import RedisClusterException, RedisError

logger = logging.getLogger(__name__)

_ENABLED_ENV = "DYN_MM_SHARED_IMAGE_CACHE_ENABLED"
_URL_ENV = "DYN_MM_SHARED_IMAGE_CACHE_URL"
_TTL_ENV = "DYN_MM_SHARED_IMAGE_CACHE_TTL_SECS"
_CONNECT_TIMEOUT_ENV = "DYN_MM_SHARED_IMAGE_CACHE_CONNECT_TIMEOUT_SECS"
_IO_TIMEOUT_ENV = "DYN_MM_SHARED_IMAGE_CACHE_IO_TIMEOUT_SECS"
_MAX_PENDING_DURATIONS = 10_000
# redis-py defines RedisClusterException outside the RedisError hierarchy.
_REDIS_OPERATION_ERRORS = (RedisError, RedisClusterException)


def _size_bucket(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    for upper_bound, label in (
        (64 * 1024, "le_64kib"),
        (256 * 1024, "le_256kib"),
        (1024 * 1024, "le_1mib"),
        (4 * 1024 * 1024, "le_4mib"),
        (16 * 1024 * 1024, "le_16mib"),
        (32 * 1024 * 1024, "le_32mib"),
    ):
        if size_bytes <= upper_bound:
            return label
    return "gt_32mib"


class SharedImageCacheStats:
    """Thread-safe buffer of Redis operation latency samples."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations: DefaultDict[tuple[str, str, str], list[float]] = defaultdict(
            list
        )
        self._pending_count = 0

    def record(
        self, operation: str, outcome: str, size_bucket: str, duration_s: float
    ) -> None:
        with self._lock:
            if self._pending_count >= _MAX_PENDING_DURATIONS:
                return
            self._durations[(operation, outcome, size_bucket)].append(duration_s)
            self._pending_count += 1

    def snapshot_and_drain(self) -> dict[tuple[str, str, str], list[float]]:
        """Atomically take all pending operation latency samples."""
        with self._lock:
            durations = dict(self._durations)
            self._durations.clear()
            self._pending_count = 0
            return durations


class SharedImageCache:
    """Fail-open Redis Cluster cache for compressed image payloads."""

    def __init__(self, client: RedisCluster, ttl_seconds: int) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds
        self.stats = SharedImageCacheStats()

    @classmethod
    def from_env(cls) -> "SharedImageCache | None":
        """Build the cache when explicitly enabled, otherwise return ``None``."""
        enabled = os.environ.get(_ENABLED_ENV, "0")
        if enabled not in ("0", "1"):
            raise ValueError(f"{_ENABLED_ENV} must be '0' or '1', got {enabled!r}")
        if enabled == "0":
            return None

        url = os.environ.get(_URL_ENV)
        if not url:
            raise ValueError(f"{_URL_ENV} must be set when {_ENABLED_ENV}=1")

        ttl_seconds = int(os.environ.get(_TTL_ENV, "3600"))
        if ttl_seconds <= 0:
            raise ValueError(f"{_TTL_ENV} must be greater than zero")

        connect_timeout_seconds = float(os.environ.get(_CONNECT_TIMEOUT_ENV, "0.1"))
        if connect_timeout_seconds <= 0:
            raise ValueError(f"{_CONNECT_TIMEOUT_ENV} must be greater than zero")

        io_timeout_seconds = float(os.environ.get(_IO_TIMEOUT_ENV, "2.0"))
        if io_timeout_seconds <= 0:
            raise ValueError(f"{_IO_TIMEOUT_ENV} must be greater than zero")

        client = RedisCluster.from_url(
            url,
            decode_responses=False,
            socket_connect_timeout=connect_timeout_seconds,
            socket_timeout=io_timeout_seconds,
        )
        # Do not log the URL: Redis URLs may contain credentials.
        logger.info("Shared image cache enabled")
        return cls(client, ttl_seconds)

    @staticmethod
    def _key(canonical_url: str) -> str:
        digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
        return f"dynamo:mm:image:{{{digest}}}:bytes"

    async def get(self, canonical_url: str) -> bytes | None:
        """Return cached encoded bytes, or ``None`` on a miss or cache error."""
        key = self._key(canonical_url)
        started = time.perf_counter()
        outcome = "error"
        size_bucket = "unknown"
        try:
            value = await self._client.get(key)
        except _REDIS_OPERATION_ERRORS as exc:
            logger.warning("Shared image cache read failed: %s", exc)
            return None
        else:
            outcome = "hit" if value is not None else "miss"
            size_bucket = _size_bucket(len(value) if value is not None else None)
            return value
        finally:
            self.stats.record(
                "get", outcome, size_bucket, time.perf_counter() - started
            )

    async def put(self, canonical_url: str, content: bytes) -> None:
        """Store validated encoded bytes, ignoring cache availability errors."""
        key = self._key(canonical_url)
        started = time.perf_counter()
        outcome = "error"
        size_bucket = _size_bucket(len(content))
        try:
            await self._client.set(key, content, ex=self._ttl_seconds)
        except _REDIS_OPERATION_ERRORS as exc:
            logger.warning("Shared image cache write failed: %s", exc)
        else:
            outcome = "success"
        finally:
            self.stats.record(
                "set", outcome, size_bucket, time.perf_counter() - started
            )

    async def delete(self, canonical_url: str) -> None:
        """Remove a corrupt entry, ignoring cache availability errors."""
        key = self._key(canonical_url)
        try:
            await self._client.delete(key)
        except _REDIS_OPERATION_ERRORS as exc:
            logger.warning("Shared image cache delete failed: %s", exc)
