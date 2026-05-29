#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""NixlConnector subclass that exposes pending-send queue depth as Prometheus
gauges. Targets vLLM 0.19.0 (used by Dynamo 1.1.x).

Background
----------

In vLLM 0.19.0, a prefill request that completes (status
``FINISHED_LENGTH_CAPPED``) and is waiting for a decode worker to pull its KV
cache enters a side-channel map on the connector worker
(``NixlConnectorWorker._reqs_to_send``). The KV blocks remain allocated to
that request until either:

  1. The decode worker sends a notification on successful NIXL READ.
  2. The static ``VLLM_NIXL_ABORT_REQUEST_TIMEOUT`` (default 480 s) expires.

These requests are NOT counted in any standard scheduler-state metric —
``vllm:num_requests_running`` reports ``len(self.running)`` and
``vllm:num_requests_waiting`` reports ``len(self.waiting)``, both excluding
the pending-transfer side channel. So a prefill worker can be at 100 % KV
usage while both gauges read zero.

This connector exposes the missing visibility:

  * ``vllm:nixl_num_pending_sends`` — gauge, ``len(_reqs_to_send)``
  * ``vllm:nixl_num_in_process_reqs`` — gauge, ``len(_reqs_to_process)``

Activation
----------

Set the connector via the vLLM ``--kv-transfer-config`` flag::

    --kv-transfer-config '{
        "kv_connector": "NixlConnectorWithPendingMetrics",
        "kv_role": "kv_both",
        "kv_connector_module_path":
            "dynamo.vllm.custom_connectors.nixl_with_pending_metrics"
    }'
"""

from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    PromMetric,
    PromMetricT,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnector,
    NixlKVConnectorStats,
    NixlPromMetrics,
)
from vllm.v1.metrics.utils import create_metric_per_engine

# Data-dict keys (kept in one place so the snapshot site and the observe site
# can't drift).
_NUM_PENDING_SENDS_KEY = "num_pending_sends"
_NUM_IN_PROCESS_KEY = "num_in_process"


class NixlConnectorWithPendingMetrics(NixlConnector):
    """NixlConnector that snapshots pending-send queue depth into the stats
    container on every ``get_kv_connector_stats()`` call and exposes it as
    two Prometheus gauges via ``NixlPromMetricsWithPending``.
    """

    def get_kv_connector_stats(self):
        # Only the worker has _reqs_to_send / _reqs_to_process. On the
        # scheduler-process side this method is a no-op (matches base class).
        if self.connector_worker is None:
            return None
        worker = self.connector_worker
        pending = len(worker._reqs_to_send)
        in_process = len(worker._reqs_to_process)

        # Inject directly into the stats data dict. The existing
        # cross-process serialization treats `data` as a plain dict, so new
        # keys travel without further plumbing.
        worker.xfer_stats.data[_NUM_PENDING_SENDS_KEY] = pending
        worker.xfer_stats.data[_NUM_IN_PROCESS_KEY] = in_process

        # Always emit when there's any pending-side activity, even if the
        # base is_empty() would otherwise suppress the report. This catches
        # the "prefill is idle but KV is pinned" case, which is exactly the
        # diagnostic case we care about.
        if not worker.xfer_stats.is_empty() or pending > 0 or in_process > 0:
            return worker.xfer_stats.clone_and_reset()
        return None

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return NixlPromMetricsWithPending(
            vllm_config, metric_types, labelnames, per_engine_labelvalues,
        )

    # build_kv_connector_stats() and the rest of the surface are inherited
    # from NixlConnector unchanged.


class NixlPromMetricsWithPending(NixlPromMetrics):
    """``NixlPromMetrics`` plus two gauges for pending-send queue depth.

    v0.19.0 uses the free function ``create_metric_per_engine`` from
    ``vllm.v1.metrics.utils`` (replacing the ``self.make_per_engine`` method
    that was on the parent class in v0.16.0).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        gauge_pending = self._gauge_cls(
            name="vllm:nixl_num_pending_sends",
            documentation=(
                "Current number of finished prefill requests holding KV blocks "
                "pending decode-side pull (i.e., in _reqs_to_send). "
                "NOTE: This metric is tracked on the P instance."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_nixl_num_pending_sends = create_metric_per_engine(
            gauge_pending, self.per_engine_labelvalues
        )

        gauge_in_process = self._gauge_cls(
            name="vllm:nixl_num_in_process_reqs",
            documentation=(
                "Current number of NIXL-tracked requests on this worker "
                "(in-flight transfers + pending-send queue). "
                "NOTE: This metric is tracked on the P instance."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_nixl_num_in_process_reqs = create_metric_per_engine(
            gauge_in_process, self.per_engine_labelvalues
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        # Run the parent's histogram + counter dispatch first.
        super().observe(transfer_stats_data, engine_idx)
        # Then our two gauges, defensively — keys may be missing if a stats
        # object from a non-pending-aware source ever reaches us (e.g. during
        # a rolling upgrade).
        for gauge_obj, key in (
            (self.gauge_nixl_num_pending_sends, _NUM_PENDING_SENDS_KEY),
            (self.gauge_nixl_num_in_process_reqs, _NUM_IN_PROCESS_KEY),
        ):
            if key in transfer_stats_data:
                gauge_obj[engine_idx].set(transfer_stats_data[key])
