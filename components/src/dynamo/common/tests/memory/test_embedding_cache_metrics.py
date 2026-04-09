# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for register_embedding_cache_metrics."""

from unittest.mock import MagicMock

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.utils.prometheus import register_embedding_cache_metrics

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


def _make_endpoint():
    """Create a mock Endpoint that captures the registered callback."""
    endpoint = MagicMock()
    endpoint.metrics.register_prometheus_expfmt_callback = MagicMock()
    return endpoint


def _get_callback(endpoint):
    """Extract the registered callback from the mock endpoint."""
    endpoint.metrics.register_prometheus_expfmt_callback.assert_called_once()
    return endpoint.metrics.register_prometheus_expfmt_callback.call_args[0][0]


def _parse_metric(text: str, name: str) -> float | None:
    """Parse a metric value from Prometheus expfmt text."""
    for line in text.split("\n"):
        if line.startswith(name + "{") or line.startswith(name + " "):
            # Extract the numeric value after the last space
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                return float(parts[1])
    return None


class TestEmbeddingCacheMetricsRegistration:
    """Tests for register_embedding_cache_metrics setup."""

    def test_registers_callback(self):
        """Callback is registered on the endpoint."""
        endpoint = _make_endpoint()
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024)
        register_embedding_cache_metrics(endpoint, cache, "test-model", "encoder")
        endpoint.metrics.register_prometheus_expfmt_callback.assert_called_once()

    def test_capacity_set_at_registration(self):
        """Capacity gauge is set immediately at registration time."""
        endpoint = _make_endpoint()
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=2 * 1024**3)
        register_embedding_cache_metrics(endpoint, cache, "test-model", "encoder")

        callback = _get_callback(endpoint)
        text = callback()
        cap = _parse_metric(text, "dynamo_component_embedding_cache_capacity_bytes")
        assert cap == 2 * 1024**3


class TestEmbeddingCacheMetricsDeltaCounters:
    """Tests for delta-based counter increments across scrapes."""

    def test_first_scrape_shows_misses(self):
        """First scrape after cache misses shows correct counter value."""
        endpoint = _make_endpoint()
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        register_embedding_cache_metrics(endpoint, cache, "m", "c")
        callback = _get_callback(endpoint)

        # Generate misses
        cache.get("miss1")
        cache.get("miss2")

        text = callback()
        misses = _parse_metric(text, "dynamo_component_embedding_cache_misses_total")
        assert misses == 2.0

    def test_second_scrape_accumulates(self):
        """Counter accumulates across multiple scrapes."""
        endpoint = _make_endpoint()
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        register_embedding_cache_metrics(endpoint, cache, "m", "c")
        callback = _get_callback(endpoint)

        # First batch of misses
        cache.get("miss1")
        text1 = callback()
        misses1 = _parse_metric(text1, "dynamo_component_embedding_cache_misses_total")
        assert misses1 == 1.0

        # Second batch
        cache.get("miss2")
        cache.get("miss3")
        text2 = callback()
        misses2 = _parse_metric(text2, "dynamo_component_embedding_cache_misses_total")
        assert misses2 == 3.0  # Accumulated, not reset

    def test_noop_scrape_no_change(self):
        """Scrape with no new activity returns same counter values."""
        endpoint = _make_endpoint()
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        register_embedding_cache_metrics(endpoint, cache, "m", "c")
        callback = _get_callback(endpoint)

        cache.get("miss1")
        text1 = callback()
        text2 = callback()  # No new activity

        misses1 = _parse_metric(text1, "dynamo_component_embedding_cache_misses_total")
        misses2 = _parse_metric(text2, "dynamo_component_embedding_cache_misses_total")
        assert misses1 == misses2 == 1.0

    def test_hits_after_cache_population(self):
        """Hits counter increments when cached items are re-accessed."""
        endpoint = _make_endpoint()
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        register_embedding_cache_metrics(endpoint, cache, "m", "c")
        callback = _get_callback(endpoint)

        tensor = torch.randn(10, 10)
        cache.set("key1", CachedEmbedding(tensor))

        # First access: miss (before set, we already missed above? No — set doesn't call get)
        # Access the cached item
        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("nonexistent")  # miss

        text = callback()
        hits = _parse_metric(text, "dynamo_component_embedding_cache_hits_total")
        misses = _parse_metric(text, "dynamo_component_embedding_cache_misses_total")
        assert hits == 2.0
        assert misses == 1.0

    def test_evictions_counter(self):
        """Evictions counter increments when LRU entries are evicted."""
        endpoint = _make_endpoint()
        # Tiny cache: ~400 bytes (one float32 100-element tensor)
        tensor_size = 100 * 4  # float32 = 4 bytes
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=tensor_size + 10)
        register_embedding_cache_metrics(endpoint, cache, "m", "c")
        callback = _get_callback(endpoint)

        t1 = torch.zeros(100, dtype=torch.float32)
        t2 = torch.zeros(100, dtype=torch.float32)

        cache.set("key1", CachedEmbedding(t1))
        cache.set("key2", CachedEmbedding(t2))  # Should evict key1

        text = callback()
        evictions = _parse_metric(
            text, "dynamo_component_embedding_cache_evictions_total"
        )
        assert evictions == 1.0


class TestEmbeddingCacheMetricsGauges:
    """Tests for gauge snapshot values."""

    def test_gauges_reflect_cache_state(self):
        """Gauges show current cache state at scrape time."""
        endpoint = _make_endpoint()
        capacity = 1024 * 1024
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=capacity)
        register_embedding_cache_metrics(endpoint, cache, "m", "c")
        callback = _get_callback(endpoint)

        tensor = torch.zeros(100, dtype=torch.float32)
        tensor_bytes = 100 * 4
        cache.set("key1", CachedEmbedding(tensor))

        text = callback()
        entries = _parse_metric(text, "dynamo_component_embedding_cache_entries")
        cur_bytes = _parse_metric(
            text, "dynamo_component_embedding_cache_current_bytes"
        )
        util = _parse_metric(text, "dynamo_component_embedding_cache_utilization")

        assert entries == 1.0
        assert cur_bytes == tensor_bytes
        assert abs(util - tensor_bytes / capacity) < 1e-6

    def test_gauges_update_on_subsequent_scrape(self):
        """Gauges reflect new state after more items are added."""
        endpoint = _make_endpoint()
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        register_embedding_cache_metrics(endpoint, cache, "m", "c")
        callback = _get_callback(endpoint)

        t1 = torch.zeros(50, dtype=torch.float32)
        cache.set("key1", CachedEmbedding(t1))
        text1 = callback()
        entries1 = _parse_metric(text1, "dynamo_component_embedding_cache_entries")
        assert entries1 == 1.0

        t2 = torch.zeros(50, dtype=torch.float32)
        cache.set("key2", CachedEmbedding(t2))
        text2 = callback()
        entries2 = _parse_metric(text2, "dynamo_component_embedding_cache_entries")
        assert entries2 == 2.0

    def test_empty_cache_gauges(self):
        """Empty cache returns zero gauges."""
        endpoint = _make_endpoint()
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024)
        register_embedding_cache_metrics(endpoint, cache, "m", "c")
        callback = _get_callback(endpoint)

        text = callback()
        assert _parse_metric(text, "dynamo_component_embedding_cache_entries") == 0.0
        assert (
            _parse_metric(text, "dynamo_component_embedding_cache_current_bytes") == 0.0
        )
        assert (
            _parse_metric(text, "dynamo_component_embedding_cache_utilization") == 0.0
        )


class TestEmbeddingCacheMetricsLabels:
    """Tests for label correctness."""

    def test_labels_present_in_output(self):
        """Model and component labels appear in metric output."""
        endpoint = _make_endpoint()
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024)
        register_embedding_cache_metrics(
            endpoint, cache, "Qwen/Qwen2.5-VL-3B", "encoder"
        )
        callback = _get_callback(endpoint)

        text = callback()
        assert 'model="Qwen/Qwen2.5-VL-3B"' in text
        assert 'dynamo_component="encoder"' in text
