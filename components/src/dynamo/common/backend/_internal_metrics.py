# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-only Prometheus helpers.

The Rust ``Worker`` calls into this module via the PyO3 bridge in
``lib/bindings/python/rust/backend.rs`` — engines must **not** import
from here. Engine-facing helpers (``register_global_registry``,
``ensure_prometheus_multiproc_dir``, ``gather_with_labels``) live in
:mod:`dynamo.common.backend.metrics`.

If you're an engine author and you find yourself reading this module,
the answer is almost certainly "implement
:meth:`LLMEngine.component_metrics_sources` instead." The framework
handles gauge construction, label seeding, and ``/metrics`` callback
registration for you.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from dynamo.common.utils.prometheus import LLMBackendMetrics

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

    from dynamo._core.backend import EngineMetrics  # type: ignore[import-not-found]


def make_component_metrics(
    model_name: str,
    component_name: str,
) -> tuple[LLMBackendMetrics, "CollectorRegistry"]:
    """Build ``dynamo_component_*`` gauges on a dedicated registry.

    Returns ``(gauges, registry)``. Both label values must be non-empty
    — empty strings would mask the framework's auto-injected labels via
    the existing-label-wins rule in :func:`gather_with_labels`.
    """
    if not model_name:
        raise ValueError(
            "make_component_metrics requires a non-empty model_name; "
            "empty-string labels mask auto-injected values"
        )
    if not component_name:
        raise ValueError(
            "make_component_metrics requires a non-empty component_name; "
            "empty-string labels mask auto-injected values"
        )

    # Lazy import: see comment in dynamo.common.backend.metrics on the
    # SGLang set_prometheus_multiproc_dir ordering constraint.
    from prometheus_client import CollectorRegistry

    registry = CollectorRegistry()
    gauges = LLMBackendMetrics(
        registry=registry,
        model_name=model_name,
        component_name=component_name,
    )
    return gauges, registry


def register_engine_registry(
    metrics: "EngineMetrics",
    registry: "CollectorRegistry",
    *,
    prefix_filters: Optional[list[str]] = None,
    exclude_prefixes: Optional[list[str]] = None,
) -> None:
    """Register an engine-owned registry as a ``/metrics`` source.

    The framework calls this from the PyO3 bridge for the
    ``dynamo_component_*`` registry. Engines call it directly (e.g. SGLang
    bridging its multiprocess registry) or via
    :func:`dynamo.common.backend.metrics.register_global_registry`.
    """
    # Local import to break the metrics ↔ _internal_metrics cycle.
    # `metrics` re-exports `register_engine_registry` for engine
    # consumers, and `_internal_metrics` consumes `gather_with_labels`
    # from `metrics` — pull the helper in at call time, not at module
    # load.
    from .metrics import gather_with_labels

    labels = metrics.auto_labels
    metrics.register_prometheus_expfmt_callback(
        lambda: gather_with_labels(
            registry,
            labels,
            prefix_filters=prefix_filters,
            exclude_prefixes=exclude_prefixes,
        )
    )


__all__ = [
    "make_component_metrics",
    "register_engine_registry",
]
