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
from dynamo.common.utils.prometheus import (
    EmbeddingCacheMetrics,
    register_embedding_cache_metrics,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


def _parse_metric(text: str, name: str) -> float | None:
    """Parse a metric value from Prometheus expfmt text.

    TODO: Consolidate with _get_metric_value() in trtllm/tests/test_trtllm_additional_metrics.py
    into a shared test utility once more metric tests are added.
    """
    for line in text.split("\n"):
        if line.startswith(name + "{") or line.startswith(name + " "):
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                return float(parts[1])
    return None


@pytest.fixture()
def cache_env():
    """Set up endpoint mock, cache, register metrics, return (cache, callback)."""
    endpoint = MagicMock()
    endpoint.metrics.register_prometheus_expfmt_callback = MagicMock()
    cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
    register_embedding_cache_metrics(endpoint, cache, "test-model", "encoder")
    endpoint.metrics.register_prometheus_expfmt_callback.assert_called_once()
    callback = endpoint.metrics.register_prometheus_expfmt_callback.call_args[0][0]
    return cache, callback


class TestCounters:
    """Delta-based counter increments across scrapes."""

    def test_accumulation_and_noop(self, cache_env):
        """Counters accumulate across scrapes and stay flat with no activity."""
        cache, callback = cache_env

        cache.get("miss1")
        text1 = callback()
        assert (
            _parse_metric(text1, "dynamo_component_embedding_cache_misses_total") == 1.0
        )

        # No-op scrape: counter unchanged
        assert (
            _parse_metric(callback(), "dynamo_component_embedding_cache_misses_total")
            == 1.0
        )

        # More misses: counter accumulates
        cache.get("miss2")
        cache.get("miss3")
        text3 = callback()
        assert (
            _parse_metric(text3, "dynamo_component_embedding_cache_misses_total") == 3.0
        )

    def test_hits_and_misses(self, cache_env):
        """Hits and misses counted correctly after population."""
        cache, callback = cache_env

        cache.set("k", CachedEmbedding(torch.randn(10, 10)))
        cache.get("k")  # hit
        cache.get("k")  # hit
        cache.get("absent")  # miss

        text = callback()
        assert _parse_metric(text, "dynamo_component_embedding_cache_hits_total") == 2.0
        assert (
            _parse_metric(text, "dynamo_component_embedding_cache_misses_total") == 1.0
        )

    def test_evictions(self):
        """Eviction counter increments when LRU entry is displaced."""
        endpoint = MagicMock()
        endpoint.metrics.register_prometheus_expfmt_callback = MagicMock()
        tensor_bytes = 100 * 4
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=tensor_bytes + 10)
        register_embedding_cache_metrics(endpoint, cache, "m", "c")
        callback = endpoint.metrics.register_prometheus_expfmt_callback.call_args[0][0]

        cache.set("a", CachedEmbedding(torch.zeros(100, dtype=torch.float32)))
        cache.set("b", CachedEmbedding(torch.zeros(100, dtype=torch.float32)))

        text = callback()
        assert (
            _parse_metric(text, "dynamo_component_embedding_cache_evictions_total")
            == 1.0
        )


class TestGauges:
    """Snapshot gauge values."""

    def test_empty_then_populated(self, cache_env):
        """Gauges start at zero, then reflect state after insertions."""
        cache, callback = cache_env

        # Empty cache
        text0 = callback()
        assert _parse_metric(text0, "dynamo_component_embedding_cache_entries") == 0.0
        assert (
            _parse_metric(text0, "dynamo_component_embedding_cache_current_bytes")
            == 0.0
        )
        assert (
            _parse_metric(text0, "dynamo_component_embedding_cache_utilization") == 0.0
        )

        # Add one entry
        t = torch.zeros(100, dtype=torch.float32)
        t_bytes = 100 * 4
        cache.set("k1", CachedEmbedding(t))
        text1 = callback()
        assert _parse_metric(text1, "dynamo_component_embedding_cache_entries") == 1.0
        assert (
            _parse_metric(text1, "dynamo_component_embedding_cache_current_bytes")
            == t_bytes
        )
        assert (
            abs(
                _parse_metric(text1, "dynamo_component_embedding_cache_utilization")
                - t_bytes / (1024 * 1024)
            )
            < 1e-6
        )

        # Add a second entry
        cache.set("k2", CachedEmbedding(torch.zeros(50, dtype=torch.float32)))
        text2 = callback()
        assert _parse_metric(text2, "dynamo_component_embedding_cache_entries") == 2.0


class TestLabelsAndCompleteness:
    """Label correctness and metric name completeness."""

    def test_labels_present(self):
        """Model and component labels appear in output."""
        endpoint = MagicMock()
        endpoint.metrics.register_prometheus_expfmt_callback = MagicMock()
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024)
        register_embedding_cache_metrics(
            endpoint, cache, "Qwen/Qwen2.5-VL-3B", "encoder"
        )
        callback = endpoint.metrics.register_prometheus_expfmt_callback.call_args[0][0]

        text = callback()
        assert 'model="Qwen/Qwen2.5-VL-3B"' in text
        assert 'dynamo_component="encoder"' in text

    def test_all_metric_names_present(self, cache_env):
        """Every expected metric name appears in the scrape output."""
        cache, callback = cache_env
        # Generate at least one event so counters appear
        cache.set("k", CachedEmbedding(torch.zeros(10, dtype=torch.float32)))
        cache.get("k")
        cache.get("absent")
        text = callback()
        for name in EmbeddingCacheMetrics:
            assert (
                _parse_metric(text, name) is not None
            ), f"metric {name} missing from output"
