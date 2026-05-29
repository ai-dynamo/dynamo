"""Real-vLLM 0.19.0 in-process E2E test for NixlConnectorWithPendingMetrics.

Doesn't need a GPU — exercises only the connector/stats/Prometheus layer.

The test simulates the "prefill is idle but blocks pinned" scenario by
directly populating ``NixlConnectorWorker._reqs_to_send`` after construction.
We bypass the worker's full __init__ (which would need a NIXL runtime) and
instead drive the metric pipeline manually with a mocked worker that exposes
exactly the attributes our connector reads (`_reqs_to_send`, `_reqs_to_process`,
`xfer_stats`).

This exercises the REAL NixlKVConnectorStats and the REAL NixlPromMetrics
(via super().observe()) from vLLM 0.19.0 — only the connector-worker
construction is mocked.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")

import sys
sys.path.insert(0, "/home/krish/repos/amz-ads/dynamo/components/src")

from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlKVConnectorStats,
)

from dynamo.vllm.custom_connectors.nixl_with_pending_metrics import (
    NixlConnectorWithPendingMetrics,
    NixlPromMetricsWithPending,
    _NUM_PENDING_SENDS_KEY,
    _NUM_IN_PROCESS_KEY,
)


def make_fake_worker(pending: int, in_process: int):
    """A minimal stand-in for NixlConnectorWorker exposing only what the
    custom connector reads. Uses the REAL NixlKVConnectorStats."""
    worker = MagicMock(spec=["_reqs_to_send", "_reqs_to_process", "xfer_stats"])
    worker._reqs_to_send = {f"r{i}": 0.0 for i in range(pending)}
    worker._reqs_to_process = set(f"p{i}" for i in range(in_process))
    worker.xfer_stats = NixlKVConnectorStats()
    return worker


def test_get_kv_connector_stats_with_pinned_blocks():
    """Critical case: prefill is idle but blocks pinned. Base class would
    return None; our override emits stats containing the gauge keys."""
    c = NixlConnectorWithPendingMetrics.__new__(NixlConnectorWithPendingMetrics)
    c.connector_worker = make_fake_worker(pending=42, in_process=42)

    stats = c.get_kv_connector_stats()

    assert stats is not None, "stats should be emitted when pending > 0"
    assert stats.data[_NUM_PENDING_SENDS_KEY] == 42
    assert stats.data[_NUM_IN_PROCESS_KEY] == 42
    print("PASS: idle-pinned prefill emits stats with pending=42, in_process=42")


def test_get_kv_connector_stats_fully_idle():
    """No transfers, no pinning -> stats not emitted (matches base behavior)."""
    c = NixlConnectorWithPendingMetrics.__new__(NixlConnectorWithPendingMetrics)
    c.connector_worker = make_fake_worker(pending=0, in_process=0)

    stats = c.get_kv_connector_stats()
    assert stats is None, "fully idle should return None"
    print("PASS: fully idle returns None")


def test_get_kv_connector_stats_no_worker():
    """Scheduler-side instance has no worker. No-op."""
    c = NixlConnectorWithPendingMetrics.__new__(NixlConnectorWithPendingMetrics)
    c.connector_worker = None
    assert c.get_kv_connector_stats() is None
    print("PASS: no worker returns None")


def test_prom_metrics_observe_sets_gauges():
    """The real Prometheus dispatch path: stats data dict -> observe() ->
    gauge.set(). Verifies BOTH our gauges record the right value, AND the
    parent's super().observe() doesn't crash with the new keys present."""

    # Build a minimal-but-real NixlPromMetricsWithPending. The parent
    # NixlPromMetrics expects: vllm_config, metric_types, labelnames,
    # per_engine_labelvalues. We provide enough to instantiate.
    from prometheus_client import Counter, Gauge, Histogram

    # Use REGISTRY=fresh so we don't conflict across test invocations
    from prometheus_client import CollectorRegistry, REGISTRY

    metric_types = {Gauge: Gauge, Counter: Counter, Histogram: Histogram}
    labelnames = ["model_name", "engine"]
    per_engine_labelvalues = {0: ["test-model", "0"]}

    # NixlPromMetrics needs a vllm_config — pass a MagicMock; it accesses
    # kv_transfer_config which the parent stores; we don't exercise it.
    metrics = NixlPromMetricsWithPending(
        vllm_config=MagicMock(),
        metric_types=metric_types,
        labelnames=labelnames,
        per_engine_labelvalues=per_engine_labelvalues,
    )

    # Build a representative stats payload as if it came from
    # NixlConnectorWithPendingMetrics.get_kv_connector_stats()
    stats_data = {
        "transfer_duration": [],
        "post_duration": [],
        "bytes_transferred": [],
        "num_descriptors": [],
        "num_failed_transfers": [],
        "num_failed_notifications": [],
        "num_kv_expired_reqs": [],
        _NUM_PENDING_SENDS_KEY: 17,
        _NUM_IN_PROCESS_KEY: 23,
    }

    metrics.observe(stats_data, engine_idx=0)

    # Read back the gauge child values from Prometheus directly
    pending_value = metrics.gauge_nixl_num_pending_sends[0]._value.get()
    in_process_value = metrics.gauge_nixl_num_in_process_reqs[0]._value.get()
    assert pending_value == 17.0, f"expected 17, got {pending_value}"
    assert in_process_value == 23.0, f"expected 23, got {in_process_value}"
    print(f"PASS: observe() set gauge_nixl_num_pending_sends={pending_value}, "
          f"gauge_nixl_num_in_process_reqs={in_process_value}")

    # Verify the gauges are registered with the expected names in the
    # default Prometheus REGISTRY
    names = {m.name for m in REGISTRY.collect()}
    assert "vllm:nixl_num_pending_sends" in names, f"missing gauge in registry: {names}"
    assert "vllm:nixl_num_in_process_reqs" in names, f"missing gauge in registry: {names}"
    print("PASS: gauges registered with names "
          "vllm:nixl_num_pending_sends, vllm:nixl_num_in_process_reqs")


def test_factory_can_resolve_via_module_path():
    """Confirm vLLM's KVConnectorFactory accepts kv_connector_module_path
    pointing at our module. This is the activation path that will fire at
    worker startup when the DGD specifies our connector."""
    import importlib

    mod = importlib.import_module(
        "dynamo.vllm.custom_connectors.nixl_with_pending_metrics"
    )
    cls = getattr(mod, "NixlConnectorWithPendingMetrics")
    assert cls is NixlConnectorWithPendingMetrics

    # Verify the base class is what vLLM's factory expects (a KVConnectorBase
    # subclass). The factory does `getattr(module, class_name)` -> uses it
    # directly. We confirm it's a subclass of NixlConnector which itself
    # inherits from KVConnectorBase_V1.
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
        NixlConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    assert issubclass(cls, NixlConnector)
    assert issubclass(cls, KVConnectorBase_V1)
    print(f"PASS: factory module-path resolution: {cls.__module__}.{cls.__name__}")
    print(f"PASS: MRO: {[c.__name__ for c in cls.__mro__]}")


if __name__ == "__main__":
    test_get_kv_connector_stats_with_pinned_blocks()
    test_get_kv_connector_stats_fully_idle()
    test_get_kv_connector_stats_no_worker()
    test_prom_metrics_observe_sets_gauges()
    test_factory_can_resolve_via_module_path()
    print()
    print("All real-vLLM tests passed.")
