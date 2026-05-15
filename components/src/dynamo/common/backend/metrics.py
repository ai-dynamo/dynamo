# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-facing Prometheus helpers.

Engines call :func:`register_global_registry` (for the default
``prometheus_client.REGISTRY``) or :func:`register_engine_registry` (for
a custom :class:`CollectorRegistry` they own — e.g. SGLang's
multiprocess registry) from inside
:meth:`LLMEngine.register_prometheus` to bridge their vendor-prefixed
registry (``vllm:``, ``sglang:``, ``trtllm_``, ``lmcache:``) into the
runtime's combined ``/metrics`` output.

The ``dynamo_component_*`` registry is owned by the framework and built
via the PyO3 bridge — engines do not construct it. See
:mod:`dynamo.common.backend._internal_metrics`.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

from dynamo.common.utils.prometheus import get_prometheus_expfmt

from ._internal_metrics import register_engine_registry

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

    from dynamo._core.backend import EngineMetrics  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def gather_with_labels(
    registry: "CollectorRegistry",
    auto_labels: Mapping[str, str],
    *,
    prefix_filters: Optional[list[str]] = None,
    exclude_prefixes: Optional[list[str]] = None,
) -> str:
    """Scrape ``registry`` into Prometheus exposition text with
    ``auto_labels`` injected at collection time. Existing labels on the
    source metrics win over auto-labels of the same name."""
    return get_prometheus_expfmt(
        registry,
        metric_prefix_filters=prefix_filters,
        exclude_prefixes=exclude_prefixes,
        inject_custom_labels=dict(auto_labels) if auto_labels else None,
    )


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
            register_engine_registry(
                metrics,
                REGISTRY,
                prefix_filters=[engine_prefix],
                exclude_prefixes=exclude_prefixes,
            )
            mp_registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(mp_registry)
            register_engine_registry(
                metrics,
                mp_registry,
                prefix_filters=all_prefixes,
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
