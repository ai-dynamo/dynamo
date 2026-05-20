# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-facing Prometheus helpers for vendor-registry bridging.

Engines call :func:`register_global_registry` (for the default
``prometheus_client.REGISTRY``) or :func:`register_engine_registry` (for
a custom :class:`CollectorRegistry` they own — e.g. SGLang's
multiprocess registry) from inside
:meth:`LLMEngine.register_prometheus` to bridge their vendor-prefixed
registry (``vllm:``, ``sglang:``, ``trtllm_``, ``lmcache:``) into the
runtime's combined ``/metrics`` output.

The ``dynamo_component_*`` registry is owned by the framework Rust-side
(see ``dynamo_backend_common::metrics::{ComponentGauges, LifecycleGauges}``
and ``dynamo_backend_common::SnapshotPublisher``) — engines do not
construct it. Engines that want per-rank visibility implement
:meth:`LLMEngine.component_metrics_dp_ranks` +
:meth:`LLMEngine.attach_snapshot_publisher` and push snapshots via the
publisher's ``publish(rank, snap)``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING, Optional

from dynamo.common.utils.prometheus import gather_with_labels

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

    from dynamo._core.backend import EngineMetrics  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def ensure_prometheus_multiproc_dir(
    prefix: str,
) -> Optional["tempfile.TemporaryDirectory[str]"]:
    """Set ``PROMETHEUS_MULTIPROC_DIR`` for the engine's lifetime. Returns
    the :class:`TemporaryDirectory` to clean up, or ``None`` if the env
    var was operator-pre-set (we don't own cleanup). The env var must
    persist past ``start()`` because vLLM reads it on every registry
    touch."""
    existing = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if existing:
        if not os.path.isdir(existing):
            logger.warning(
                "PROMETHEUS_MULTIPROC_DIR=%s does not exist, recreating", existing
            )
            os.makedirs(existing, exist_ok=True)
        return None

    tmpdir = tempfile.TemporaryDirectory(prefix=prefix)
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = tmpdir.name
    logger.debug("Created PROMETHEUS_MULTIPROC_DIR at: %s", tmpdir.name)
    return tmpdir


def register_engine_registry(
    metrics: "EngineMetrics",
    registry: "CollectorRegistry",
    *,
    prefix_filters: Optional[list[str]] = None,
    exclude_prefixes: Optional[list[str]] = None,
) -> None:
    """Register a Python ``CollectorRegistry`` as a ``/metrics`` callback
    against the runtime's ``EngineMetrics`` handle. Auto-labels
    (namespace/component/endpoint/worker_id/model) are injected at
    scrape time.

    Engines that own a custom ``CollectorRegistry`` (e.g. SGLang's
    ``multiprocess.MultiProcessCollector`` over the scheduler ZMQ
    socket) call this directly. The simpler :func:`register_global_registry`
    handles the default ``prometheus_client.REGISTRY`` + K8s
    multiproc-conflict case.
    """
    labels = metrics.auto_labels
    metrics.register_prometheus_expfmt_callback(
        lambda: gather_with_labels(
            registry,
            labels,
            prefix_filters=prefix_filters,
            exclude_prefixes=exclude_prefixes,
        )
    )


def register_global_registry(
    metrics: "EngineMetrics",
    *,
    engine_prefix: str,
    multiproc_only_prefixes: Optional[list[str]] = None,
    exclude_prefixes: Optional[list[str]] = None,
) -> None:
    """Register the global ``prometheus_client.REGISTRY`` against
    ``/metrics``, handling the K8s ``MultiProcessCollector`` conflict.
    Use from inside :meth:`LLMEngine.register_prometheus` to bridge an
    engine's native metrics into the runtime's combined output.

    ``engine_prefix`` is the in-memory prefix (e.g. ``"vllm:"``);
    ``multiproc_only_prefixes`` is for prefixes that live only in
    ``.db`` files (e.g. ``["lmcache:"]``).
    """
    # Lazy import so engines (notably SGLang, which calls
    # set_prometheus_multiproc_dir during sgl.Engine init) can import
    # this module before prometheus_client touches the env var.
    from prometheus_client import REGISTRY, CollectorRegistry, multiprocess

    all_prefixes = [engine_prefix] + list(multiproc_only_prefixes or [])
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")

    if multiproc_dir and os.path.isdir(multiproc_dir):
        try:
            multiprocess.MultiProcessCollector(REGISTRY)
            register_engine_registry(
                metrics,
                REGISTRY,
                prefix_filters=all_prefixes,
                exclude_prefixes=exclude_prefixes,
            )
            return
        except ValueError as e:
            logger.debug(
                "MultiProcessCollector conflict with REGISTRY (%s); "
                "using a separate multiproc registry",
                e,
            )
            # REGISTRY already has the existing MultiProcessCollector,
            # so it covers engine_prefix metrics. The fresh mp_registry
            # is only needed to pick up multiproc-only prefixes (e.g.
            # `lmcache:`) that some forks scrape via a SECOND
            # MultiProcessCollector. Filtering the second callback to
            # `multiproc_only_prefixes` only avoids double-emitting
            # engine_prefix in one /metrics scrape.
            register_engine_registry(
                metrics,
                REGISTRY,
                prefix_filters=[engine_prefix],
                exclude_prefixes=exclude_prefixes,
            )
            if multiproc_only_prefixes:
                mp_registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(mp_registry)
                register_engine_registry(
                    metrics,
                    mp_registry,
                    prefix_filters=list(multiproc_only_prefixes),
                    exclude_prefixes=exclude_prefixes,
                )
            return

    if multiproc_dir:
        logger.warning(
            "PROMETHEUS_MULTIPROC_DIR=%s is not a valid directory; "
            "falling back to single-process metrics",
            multiproc_dir,
        )

    register_engine_registry(
        metrics,
        REGISTRY,
        prefix_filters=all_prefixes,
        exclude_prefixes=exclude_prefixes,
    )


__all__ = [
    "ensure_prometheus_multiproc_dir",
    "gather_with_labels",
    "register_engine_registry",
    "register_global_registry",
]
