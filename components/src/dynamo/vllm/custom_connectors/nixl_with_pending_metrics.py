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

These requests are NOT counted in any standard scheduler-state metric.

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

Implementation notes
--------------------

Gauge values are stored as **single-element lists** in the stats data dict
(not bare scalars). This is because ``NixlKVConnectorStats.aggregate()``
iterates the data dict and calls ``list.extend()`` on every value — it
asserts each value is a list. Storing as ``[v]`` makes our gauge fields
aggregate-safe, and ``observe()`` reads the last element (gauge semantics:
latest wins).

We also subclass ``NixlKVConnectorStats`` so that ``reset()`` always
includes our gauge keys as empty lists. This avoids ``KeyError`` when
``aggregate()`` iterates ``other.data`` and looks up the same key on
``self.data`` after a reset. We lazily swap the worker's ``xfer_stats`` to
our subclass on the first ``get_kv_connector_stats()`` call.

We always return a stats object from ``get_kv_connector_stats()`` (even
when nothing else is happening), so ``observe()`` runs every cycle and the
Prometheus gauge gets updated — including dropping back to 0 once a pinned
request is released. Without this, the gauge would stay stuck at its last
non-zero reading after a successful round-trip.
"""

from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnector,
    NixlKVConnectorStats,
    NixlPromMetrics,
)
from vllm.v1.metrics.utils import create_metric_per_engine

# Data-dict keys (kept in one place so the snapshot site, the reset site,
# and the observe site can't drift).
_NUM_PENDING_SENDS_KEY = "num_pending_sends"
_NUM_IN_PROCESS_KEY = "num_in_process"


@dataclass
class NixlKVConnectorStatsWithPending(NixlKVConnectorStats):
    """``NixlKVConnectorStats`` extended with two gauge-style fields for
    pending-send queue depth.

    The fields are stored as single-element lists (``[scalar]``) so that
    ``NixlKVConnectorStats.aggregate()`` — which iterates the data dict and
    calls ``list.extend`` on every value — doesn't fail. ``observe()`` in
    ``NixlPromMetricsWithPending`` reads the LAST element of each list,
    which gives gauge semantics (latest value wins) when stats are
    aggregated across multiple steps or workers.
    """

    def reset(self):
        super().reset()
        # Always include our keys in the data dict, even on a fresh reset,
        # so aggregate() doesn't KeyError when other has these keys but
        # self was just reset.
        self.data[_NUM_PENDING_SENDS_KEY] = []
        self.data[_NUM_IN_PROCESS_KEY] = []

    def is_empty(self) -> bool:
        # The base class ``is_empty()`` only checks failure/transfer
        # counters. The scheduler-side aggregator does:
        #
        #     if not other.is_empty():
        #         for k, v in other.data.items(): self.data[k].extend(v)
        #
        # If we relied on the base is_empty(), our gauge keys would be
        # SILENTLY DROPPED whenever no transfers happened in the interval —
        # which is the common case for idle/normal-traffic workers. The
        # Prometheus gauge would then never update from its last value.
        # Treat our keys as "data" so aggregate() always extends them.
        return super().is_empty() and not (
            self.data.get(_NUM_PENDING_SENDS_KEY)
            or self.data.get(_NUM_IN_PROCESS_KEY)
        )

    def aggregate(self, other: KVConnectorStats) -> "NixlKVConnectorStatsWithPending":
        # Defensive override: if ``other.data`` contains our gauge keys but
        # ``self.data`` does not (e.g., self is a deserialized base-class
        # stats that didn't go through our ``build_kv_connector_stats``
        # factory), the base ``aggregate()`` would ``KeyError`` on
        # ``self.data[k]``. Pre-seed our keys before delegating.
        if not other.is_empty():
            for k in (_NUM_PENDING_SENDS_KEY, _NUM_IN_PROCESS_KEY):
                self.data.setdefault(k, [])
        return super().aggregate(other)


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

        # Lazily upgrade the worker's xfer_stats to our subclass. The base
        # NixlConnectorWorker.__init__ hardcodes
        # `self.xfer_stats = NixlKVConnectorStats()`, which means reset()
        # there wipes our keys. Swapping in our subclass ensures reset()
        # always re-initializes our gauge fields as empty lists.
        #
        # We merge the old container's data onto our subclass — but skip our
        # own gauge keys so we don't overwrite the empty lists that the
        # subclass's reset() just installed. (Today's base class doesn't
        # define our keys; this guard is forward-compat against a future
        # vLLM version that might pre-define them.)
        if not isinstance(worker.xfer_stats, NixlKVConnectorStatsWithPending):
            new_stats = NixlKVConnectorStatsWithPending()
            for k, v in worker.xfer_stats.data.items():
                if k in (_NUM_PENDING_SENDS_KEY, _NUM_IN_PROCESS_KEY):
                    continue
                new_stats.data[k] = v
            worker.xfer_stats = new_stats

        # Snapshot current queue depths as single-element lists.
        worker.xfer_stats.data[_NUM_PENDING_SENDS_KEY] = [
            len(worker._reqs_to_send)
        ]
        worker.xfer_stats.data[_NUM_IN_PROCESS_KEY] = [
            len(worker._reqs_to_process)
        ]

        # ALWAYS emit (don't gate on is_empty()). Prometheus gauges use
        # multiprocess_mode="mostrecent": if observe() doesn't run on a
        # given cycle, the gauge holds its previous value. Returning None
        # when pending drops back to 0 would leave the gauge stuck at the
        # last non-zero reading. Histograms/counters in the base class get
        # empty lists / no inc() calls here, which is a no-op.
        return worker.xfer_stats.clone_and_reset()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        # Returned to the scheduler side; needs to be our subclass so
        # aggregate() finds our keys in reset()-fresh instances.
        if data is not None:
            return NixlKVConnectorStatsWithPending(data=data)
        return NixlKVConnectorStatsWithPending()

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


class NixlPromMetricsWithPending(NixlPromMetrics):
    """``NixlPromMetrics`` plus two gauges for pending-send queue depth.

    The gauge values arrive in ``transfer_stats_data`` as single-element
    lists (so they survive ``aggregate()``). We read the last element —
    that's the most recent observation, which is the correct gauge value.
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
        # Then our two gauges. Values arrive as single-element lists (so
        # aggregate() doesn't break); take the last element for gauge
        # semantics (latest wins). Gauge.set() internally injects
        # time.time() as the timestamp when multiprocess_mode='mostrecent',
        # so we don't need to pass one.
        for gauge_obj, key in (
            (self.gauge_nixl_num_pending_sends, _NUM_PENDING_SENDS_KEY),
            (self.gauge_nixl_num_in_process_reqs, _NUM_IN_PROCESS_KEY),
        ):
            values = transfer_stats_data.get(key)
            if values is not None:
                gauge_obj[engine_idx].set(values[-1] if values else 0)
