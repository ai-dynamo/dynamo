"""Test vLLM's KVConnectorFactory resolution for our custom connector.

This is the exact path vLLM takes at worker startup when the DGD specifies
our connector via `kv_connector` + `kv_connector_module_path`. We test the
class-resolution step (without instantiating, which needs a NIXL runtime).
"""

import sys
sys.path.insert(0, "/home/krish/repos/amz-ads/dynamo/components/src")

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

# This is what the DGD's --kv-transfer-config will produce
kv_transfer_config = KVTransferConfig(
    kv_connector="NixlConnectorWithPendingMetrics",
    kv_role="kv_both",
    kv_connector_module_path=(
        "dynamo.vllm.custom_connectors.nixl_with_pending_metrics"
    ),
)

# The factory's class-resolution path (factory.py:_get_connector_class_with_compat)
cls = KVConnectorFactory.get_connector_class(kv_transfer_config)

print(f"Resolved class: {cls.__module__}.{cls.__name__}")
print(f"MRO: {[c.__name__ for c in cls.__mro__]}")

from dynamo.vllm.custom_connectors.nixl_with_pending_metrics import (
    NixlConnectorWithPendingMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import NixlConnector

assert cls is NixlConnectorWithPendingMetrics, (
    f"Factory returned {cls!r} but expected NixlConnectorWithPendingMetrics"
)
assert issubclass(cls, NixlConnector)

# Also confirm that the parent's build_prom_metrics is overridden
# (i.e., calling it via the resolved class returns our subclass)
from unittest.mock import MagicMock
from prometheus_client import Counter, Gauge, Histogram

prom = cls.build_prom_metrics(
    vllm_config=MagicMock(),
    metric_types={Gauge: Gauge, Counter: Counter, Histogram: Histogram},
    labelnames=["model_name", "engine"],
    per_engine_labelvalues={0: ["test-model", "0"]},
)
from dynamo.vllm.custom_connectors.nixl_with_pending_metrics import (
    NixlPromMetricsWithPending,
)
assert isinstance(prom, NixlPromMetricsWithPending), type(prom)

print()
print("PASS: KVConnectorFactory resolves NixlConnectorWithPendingMetrics")
print("PASS: build_prom_metrics returns NixlPromMetricsWithPending")
print()
print("Activation path confirmed. In a real worker startup vLLM would:")
print("  1. Read --kv-transfer-config (above)")
print("  2. Call KVConnectorFactory.create_connector(vllm_config, role, kv_cache_config)")
print("  3. This calls _get_connector_class_with_compat() -> our class")
print("  4. Instantiates via cls(vllm_config, role, kv_cache_config)")
