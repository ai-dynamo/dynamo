# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SGLang embedding worker's Prometheus metrics."""

import pytest
from prometheus_client import CollectorRegistry

from dynamo.sglang.request_handlers.embedding import embedding_handler as eh
from dynamo.sglang.request_handlers.embedding.metrics import (
    init_embedding_metrics,
    observe_embedding_batch_size,
    observe_embedding_input_tokens,
    reset_metrics_for_testing,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


@pytest.fixture
def fresh_registry() -> CollectorRegistry:
    """Reset the metrics-module singletons and hand back a fresh registry.

    Pairing the reset with the registry creation ensures that
    ``init_embedding_metrics`` actually creates new Histograms bound to
    THIS test's registry rather than reusing whichever one a previous
    test bound to.
    """
    reset_metrics_for_testing()
    registry = CollectorRegistry()
    init_embedding_metrics(registry)
    yield registry
    reset_metrics_for_testing()


def _collect_histogram(registry: CollectorRegistry, metric_name: str) -> dict:
    """Pull (count, sum, buckets) for the given histogram from the registry.

    Returns a dict keyed by label-tuple. Each value is a dict with keys
    ``count``, ``sum``, and ``buckets`` (a list of (le, cumulative_count)).
    """
    out: dict = {}
    for family in registry.collect():
        if family.name != metric_name:
            continue
        for sample in family.samples:
            labels = tuple(sorted(sample.labels.items()))
            entry = out.setdefault(labels, {"count": 0, "sum": 0.0, "buckets": []})
            if sample.name.endswith("_count"):
                entry["count"] = int(sample.value)
            elif sample.name.endswith("_sum"):
                entry["sum"] = float(sample.value)
            elif sample.name.endswith("_bucket"):
                le = sample.labels.get("le")
                entry["buckets"].append((le, int(sample.value)))
    return out


def test_observe_batch_size_increments_count(fresh_registry):
    observe_embedding_batch_size("Qwen/Qwen3-Embedding-4B", 3)
    observe_embedding_batch_size("Qwen/Qwen3-Embedding-4B", 1)

    data = _collect_histogram(fresh_registry, "dynamo_embedding_batch_size")
    key = (("model", "Qwen/Qwen3-Embedding-4B"),)
    assert key in data
    assert data[key]["count"] == 2
    assert data[key]["sum"] == 4.0  # 3 + 1


def test_observe_input_tokens_increments_count_and_sum(fresh_registry):
    observe_embedding_input_tokens("Qwen/Qwen3-Embedding-4B", 70)
    observe_embedding_input_tokens("Qwen/Qwen3-Embedding-4B", 200)
    observe_embedding_input_tokens("Qwen/Qwen3-Embedding-4B", 12)

    data = _collect_histogram(fresh_registry, "dynamo_embedding_input_tokens")
    key = (("model", "Qwen/Qwen3-Embedding-4B"),)
    assert key in data
    assert data[key]["count"] == 3
    assert data[key]["sum"] == 282.0  # 70 + 200 + 12


def test_observe_partitioned_by_model_label(fresh_registry):
    observe_embedding_batch_size("model-A", 5)
    observe_embedding_batch_size("model-B", 7)
    observe_embedding_batch_size("model-A", 2)

    data = _collect_histogram(fresh_registry, "dynamo_embedding_batch_size")
    key_a = (("model", "model-A"),)
    key_b = (("model", "model-B"),)
    assert data[key_a]["count"] == 2
    assert data[key_a]["sum"] == 7.0  # 5 + 2
    assert data[key_b]["count"] == 1
    assert data[key_b]["sum"] == 7.0


def test_negative_values_are_silently_dropped(fresh_registry):
    """Negative values are nonsensical for batch_size / input_tokens. The
    observation helper logs a warning and skips rather than raising or
    polluting the histogram."""
    observe_embedding_batch_size("model-A", 4)
    observe_embedding_batch_size("model-A", -1)
    observe_embedding_input_tokens("model-A", 100)
    observe_embedding_input_tokens("model-A", -5)

    bs = _collect_histogram(fresh_registry, "dynamo_embedding_batch_size")
    it = _collect_histogram(fresh_registry, "dynamo_embedding_input_tokens")
    key = (("model", "model-A"),)
    # Negative observations did NOT increment the count.
    assert bs[key]["count"] == 1
    assert bs[key]["sum"] == 4.0
    assert it[key]["count"] == 1
    assert it[key]["sum"] == 100.0


def test_buckets_are_in_expected_order(fresh_registry):
    """The histogram's buckets should be monotonically non-decreasing in
    cumulative count. Catches accidental bucket-reordering regressions."""
    for value in (1, 2, 5, 33, 200, 1000):
        observe_embedding_batch_size("m", value)

    data = _collect_histogram(fresh_registry, "dynamo_embedding_batch_size")
    key = (("model", "m"),)
    assert data[key]["count"] == 6
    buckets = data[key]["buckets"]
    # buckets are stored in registry order, which is the construction order;
    # cumulative counts must be monotonically non-decreasing.
    counts = [c for _le, c in buckets]
    assert counts == sorted(counts), f"Bucket counts not monotonic: {counts}"


def test_observers_noop_before_init_embedding_metrics():
    """Until ``init_embedding_metrics`` is called, ``observe_*`` is a no-op.

    Important: in test/CI configurations that don't wire the embedding
    metrics endpoint, handler call sites must NOT crash.
    """
    reset_metrics_for_testing()
    # Both helpers should silently do nothing — no exception.
    observe_embedding_batch_size("m", 3)
    observe_embedding_input_tokens("m", 100)


def test_handler_observes_metrics_via_transform_response(monkeypatch):
    """Integration-light test: spy on the observe functions to confirm
    the handler actually calls them from ``_transform_response``."""
    # Use a mock for the BaseWorkerHandler superclass init to avoid pulling in
    # the engine-level state.
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.handler_base.BaseWorkerHandler.__init__",
        lambda self, *a, **kw: None,
    )

    observed: list[tuple[str, str, int]] = []
    monkeypatch.setattr(
        eh,
        "observe_embedding_batch_size",
        lambda model, n: observed.append(("batch", model, n)),
    )
    monkeypatch.setattr(
        eh,
        "observe_embedding_input_tokens",
        lambda model, n: observed.append(("tokens", model, n)),
    )

    handler = eh.EmbeddingWorkerHandler.__new__(eh.EmbeddingWorkerHandler)

    # Simulate SGLang's response shape: list of {"embedding": [...], "meta_info":{"prompt_tokens": N}}.
    ret = [
        {"embedding": [0.1, 0.2, 0.3], "meta_info": {"prompt_tokens": 12}},
        {"embedding": [0.4, 0.5, 0.6], "meta_info": {"prompt_tokens": 8}},
    ]

    out = handler._transform_response(ret, "Qwen/Qwen3-Embedding-4B")

    # Sanity: response shape preserved.
    assert out["object"] == "list"
    assert len(out["data"]) == 2
    assert out["usage"]["prompt_tokens"] == 20

    # Metrics: both helpers called exactly once with the right model + values.
    assert ("batch", "Qwen/Qwen3-Embedding-4B", 2) in observed
    assert ("tokens", "Qwen/Qwen3-Embedding-4B", 20) in observed


def test_metric_failure_does_not_break_transform_response(monkeypatch):
    """If a metric observe call throws, the request must still succeed."""
    monkeypatch.setattr(
        "dynamo.sglang.request_handlers.handler_base.BaseWorkerHandler.__init__",
        lambda self, *a, **kw: None,
    )

    def _boom(*_a, **_kw):
        raise RuntimeError("prometheus_client exploded")

    monkeypatch.setattr(eh, "observe_embedding_batch_size", _boom)

    handler = eh.EmbeddingWorkerHandler.__new__(eh.EmbeddingWorkerHandler)
    floats = [0.1, 0.2]
    ret = [{"embedding": floats, "meta_info": {"prompt_tokens": 4}}]

    # Must not raise. The worker always emits base64 over the internal
    # wire format -- the Rust frontend decodes back to float at the HTTP
    # boundary if the client asked for float.
    out = handler._transform_response(ret, "model-A")
    assert out["data"][0]["embedding"] == eh._encode_floats_to_base64(floats)
    assert out["usage"]["prompt_tokens"] == 4
