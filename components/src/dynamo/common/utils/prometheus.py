# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prometheus metrics utilities for Dynamo components.

This module provides shared functionality for collecting and exposing Prometheus metrics
from backend engines (SGLang, vLLM, etc.) via Dynamo's metrics endpoint.

Note: Engine metrics take time to appear after engine initialization,
while Dynamo runtime metrics are available immediately after component creation.
"""

import enum
import logging
import re
import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Pattern

from dynamo._core import Endpoint
from dynamo.prometheus_names import kvstats, labels, model_info, name_prefix

# Import CollectorRegistry and Gauge only for type hints to avoid importing prometheus_client at module load time.
# prometheus_client must be imported AFTER set_prometheus_multiproc_dir() is called.
# See main.py worker() function for detailed explanation.
if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

    from dynamo.common.memory import MultimodalEmbeddingCacheManager

# Auto-label injection: always injects dynamo_namespace, dynamo_component, dynamo_endpoint labels
# into engine metrics based on the endpoint hierarchy.
#
# Rust counterpart: lib/runtime/src/metrics.rs create_metric() function
# Label constants defined in: lib/runtime/src/metrics/prometheus_names.rs labels module


# Single source of truth for embedding cache metric names.
class EmbeddingCacheMetrics(str, enum.Enum):
    """Prometheus metric names for the multimodal embedding cache."""

    HITS_TOTAL = f"{name_prefix.COMPONENT}_embedding_cache_hits_total"
    MISSES_TOTAL = f"{name_prefix.COMPONENT}_embedding_cache_misses_total"
    EVICTIONS_TOTAL = f"{name_prefix.COMPONENT}_embedding_cache_evictions_total"
    UTILIZATION = f"{name_prefix.COMPONENT}_embedding_cache_utilization"
    CURRENT_BYTES = f"{name_prefix.COMPONENT}_embedding_cache_current_bytes"
    ENTRIES = f"{name_prefix.COMPONENT}_embedding_cache_entries"


def register_engine_metrics_callback(
    endpoint: Endpoint,
    registry: "CollectorRegistry",
    metric_prefix_filters: Optional[list[str]] = None,
    exclude_prefixes: Optional[list[str]] = None,
    inject_custom_labels: Optional[dict[str, str]] = None,
    namespace_name: Optional[str] = None,
    component_name: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None:
    """
    Register a callback to expose engine Prometheus metrics via Dynamo's metrics endpoint.

    This registers a callback that is invoked when /metrics is scraped, passing through
    engine-specific metrics alongside Dynamo runtime metrics.

    Automatically injects dynamo_namespace, dynamo_component, dynamo_endpoint, model,
    and model_name labels when namespace_name and component_name are provided.

    Label Precedence (highest to lowest):
    1. Existing labels from source metrics - never changed, never overwritten
    2. Auto-injected labels (dynamo_*, model*) - added by Dynamo automatically
    3. Custom labels (inject_custom_labels) - user-provided, lowest precedence

    If inject_custom_labels contains keys that conflict with auto-injected labels,
    a warning is logged and the auto-injected value takes precedence.

    Args:
        endpoint: Dynamo endpoint object with metrics.register_prometheus_expfmt_callback()
        registry: Prometheus registry to collect from (e.g., REGISTRY or CollectorRegistry)
        metric_prefix_filters: List of prefixes to filter metrics (e.g., ["vllm:"], ["vllm:", "lmcache:"], or None for no filtering)
        exclude_prefixes: List of metric name prefixes to exclude (e.g., ["python_", "process_"])
        inject_custom_labels: Optional dict of custom labels to inject (e.g. {"lora_adapter": "my-lora"}).
                      Injected at collection time without modifying source metrics.
                      Reserved labels (le, quantile) will raise ValueError.
                      Auto-labels (dynamo_namespace, dynamo_component, dynamo_endpoint, model,
                      model_name) are added automatically and should not be in inject_custom_labels.
        namespace_name: Explicit namespace name for auto-labels (from config.namespace)
        component_name: Explicit component name for auto-labels (from config.component)
        endpoint_name: Explicit endpoint name for auto-labels (from config.endpoint, defaults to "generate")
        model_name: Model name/path for auto-labels (from config.model, injected as both 'model' and 'model_name')

    Example:
        from prometheus_client import REGISTRY
        # Auto-labels: automatically adds hierarchy labels
        register_engine_metrics_callback(
            generate_endpoint, REGISTRY,
            metric_prefix_filters=["vllm:"],
            namespace_name="prod", component_name="vllm-worker", endpoint_name="generate"
        )

        # Include multiple metric prefixes
        register_engine_metrics_callback(
            generate_endpoint, REGISTRY, metric_prefix_filters=["vllm:", "lmcache:"]
        )

        # With filtering and prefixing for TensorRT-LLM
        register_engine_metrics_callback(
            generate_endpoint, REGISTRY,
            exclude_prefixes=["python_", "process_"],
            metric_prefix_filters=["trtllm_"],
        )

        # Inject additional labels (auto-labels are added automatically)
        register_engine_metrics_callback(
            generate_endpoint, REGISTRY,
            metric_prefix_filters=["vllm:"],
            inject_custom_labels={"lora_adapter": "my-lora"}
        )
    """

    # Auto-inject hierarchy labels
    final_inject_labels = inject_custom_labels.copy() if inject_custom_labels else {}

    if namespace_name and component_name:
        # Extract hierarchy information
        # Mirrors Rust auto-label injection in lib/runtime/src/metrics.rs create_metric()
        endpoint_name_final = endpoint_name or "generate"

        # Add auto-labels using constants from prometheus_names.labels
        # These align with Rust auto-labels defined in lib/runtime/src/metrics/prometheus_names.rs
        auto_labels = {
            labels.NAMESPACE: namespace_name,  # "dynamo_namespace"
            labels.COMPONENT: component_name,  # "dynamo_component"
            labels.ENDPOINT: endpoint_name_final,  # "dynamo_endpoint"
        }

        # Add model labels if model_name is provided
        if model_name:
            auto_labels[labels.MODEL] = model_name  # "model" (OpenAI standard)
            auto_labels[
                labels.MODEL_NAME
            ] = model_name  # "model_name" (engine-native compatibility)

        # Validate that user didn't provide conflicting auto-labels
        # Warn but don't error - custom labels have lower precedence than auto-labels
        if inject_custom_labels:
            for key in auto_labels:
                if key in inject_custom_labels:
                    logging.warning(
                        f"Custom label '{key}' conflicts with auto-injected label. "
                        f"Auto-injected value takes precedence. Custom value '{inject_custom_labels[key]}' ignored."
                    )

        # Merge labels with correct precedence:
        # 1. Existing labels (from source metrics) - never overwritten
        # 2. Auto-labels (dynamo_*, model*) - injected by Dynamo
        # 3. Custom labels (inject_custom_labels) - user-provided, lowest precedence
        # Put custom labels first, then overwrite with auto-labels (higher precedence)
        final_inject_labels = {**final_inject_labels, **auto_labels}
        logging.debug(
            f"Auto-injecting labels: "
            f"namespace={namespace_name}, component={component_name}, endpoint={endpoint_name_final}, model={model_name}"
        )

    def get_expfmt() -> str:
        """Callback to return engine Prometheus metrics in exposition format"""
        result = get_prometheus_expfmt(
            registry,
            metric_prefix_filters=metric_prefix_filters,
            exclude_prefixes=exclude_prefixes,
            inject_custom_labels=final_inject_labels if final_inject_labels else None,
        )
        return result

    endpoint.metrics.register_prometheus_expfmt_callback(get_expfmt)


@lru_cache(maxsize=64)
def _compile_exclude_pattern(exclude_prefixes: tuple[str, ...]) -> Pattern:
    """Compile and cache regex for excluding metric prefixes.

    Args take tuple not list - lru_cache requires hashable args (tuples are hashable, lists are not).
    """
    escaped_prefixes = [re.escape(prefix) for prefix in exclude_prefixes]
    prefixes_regex = "|".join(escaped_prefixes)
    return re.compile(rf"^(# (HELP|TYPE) )?({prefixes_regex})")


@lru_cache(maxsize=64)
def _compile_include_pattern(metric_prefixes: tuple[str, ...]) -> Pattern:
    """Compile and cache regex for including metrics by prefix.

    Args take tuple not list - lru_cache requires hashable args (tuples are hashable, lists are not).
    Supports multiple prefixes with OR logic (e.g., ("vllm:", "lmcache:")).
    """
    escaped_prefixes = [re.escape(prefix) for prefix in metric_prefixes]
    prefixes_regex = "|".join(escaped_prefixes)
    return re.compile(rf"^(# (HELP|TYPE) )?({prefixes_regex})")


def get_prometheus_expfmt(
    registry: "CollectorRegistry",
    metric_prefix_filters: Optional[list[str]] = None,
    exclude_prefixes: Optional[list[str]] = None,
    inject_custom_labels: Optional[dict[str, str]] = None,
) -> str:
    """
    Get Prometheus metrics from a registry formatted as text using the standard text encoder.

    Collects all metrics from the registry and returns them in Prometheus text exposition format.
    Optionally filters metrics by prefix, excludes certain prefixes, adds a prefix, and injects labels.

    IMPORTANT: prometheus_client is imported lazily here because it must be imported AFTER
    set_prometheus_multiproc_dir() is called by SGLang's engine initialization. Importing
    at module level causes prometheus_client to initialize in single-process mode before
    PROMETHEUS_MULTIPROC_DIR is set, which breaks TokenizerMetricsCollector metrics.

    Args:
        registry: Prometheus registry to collect from.
                 Pass CollectorRegistry with MultiProcessCollector for SGLang.
                 Pass REGISTRY for vLLM single-process mode.
        metric_prefix_filters: Optional list of prefixes to filter displayed metrics (e.g., ["vllm:"] or ["vllm:", "lmcache:"]).
                             If None, returns all metrics. Supports single string or list of strings. (default: None)
        exclude_prefixes: List of metric name prefixes to exclude (e.g., ["python_", "process_"])
        inject_custom_labels: Optional dict of custom labels to inject at collection time.
                      Example: {"lora_adapter": "my-lora"}
                      Reserved labels (le, quantile) will raise ValueError.

                      Label Precedence (highest to lowest):
                      1. Existing labels from source metrics - never changed
                      2. Auto-injected labels (via register_engine_metrics_callback)
                      3. Custom labels (inject_custom_labels) - lowest precedence

    Returns:
        Formatted metrics text in Prometheus exposition format. Returns empty string on error.

    Example:
        # Filter to include only vllm and lmcache metrics
        get_prometheus_expfmt(registry, metric_prefix_filters=["vllm:", "lmcache:"])

        # Filter out python_/process_ metrics (TRT-LLM natively outputs trtllm_* prefix)
        get_prometheus_expfmt(registry, metric_prefix_filters=["trtllm_"])

        # Inject labels (custom labels, not auto-injected ones)
        get_prometheus_expfmt(
            registry, metric_prefix_filters=["vllm:"],
            inject_custom_labels={"lora_adapter": "my-lora"}
        )
    """
    from prometheus_client import CollectorRegistry, generate_latest

    try:
        # If label injection requested, wrap registry with custom collector
        if inject_custom_labels:
            # Delayed import: LabelInjectingCollector imports prometheus_client.registry.Collector
            # at module level. This import must happen AFTER set_prometheus_multiproc_dir() is
            # called by SGLang's engine initialization. Importing at the top of this file would
            # trigger prometheus_client initialization too early (before PROMETHEUS_MULTIPROC_DIR
            # is set), breaking multiprocess metrics collection.
            from dynamo.common.utils.label_injecting_collector import (
                LabelInjectingCollector,
            )

            # Create temporary registry with label-injecting collector
            temp_registry = CollectorRegistry()
            temp_registry.register(
                LabelInjectingCollector(registry, inject_custom_labels)
            )
            registry = temp_registry

        # Generate metrics in Prometheus text format
        metrics_text = generate_latest(registry).decode("utf-8")

        if metric_prefix_filters or exclude_prefixes:
            lines = []

            # Get cached compiled patterns
            exclude_line_pattern = None
            if exclude_prefixes:
                exclude_line_pattern = _compile_exclude_pattern(tuple(exclude_prefixes))

            # Build include pattern if needed
            include_pattern = None
            if metric_prefix_filters:
                filter_tuple: tuple[str, ...] = tuple(metric_prefix_filters)
                include_pattern = _compile_include_pattern(filter_tuple)

            for line in metrics_text.split("\n"):
                if not line.strip():
                    continue

                # Skip excluded lines entirely
                if exclude_line_pattern and exclude_line_pattern.match(line):
                    continue

                # Apply include filter if specified
                if include_pattern and not include_pattern.match(line):
                    continue

                lines.append(line)

            result = "\n".join(lines)
            if result and not result.endswith("\n"):
                result += "\n"
            return result
        else:
            # Ensure metrics_text ends with newline
            if metrics_text and not metrics_text.endswith("\n"):
                metrics_text += "\n"
            return metrics_text

    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
        return ""


class LLMBackendMetrics:
    """Prometheus metrics for LLM backends with `dynamo_component_` prefix.

    Usage:
        metrics = LLMBackendMetrics(registry, model_name="Qwen/Qwen3-0.6B", component_name="backend")
        metrics.set_total_blocks("0", 1000)
        metrics.set_gpu_cache_usage("0", 0.75)
        metrics.set_model_load_time(5.2)
    """

    def __init__(
        self,
        registry: Optional["CollectorRegistry"] = None,
        model_name: str = "",
        component_name: str = "",
    ) -> None:
        """Create all Dynamo component gauges."""
        from prometheus_client import Gauge

        self.total_blocks = Gauge(
            f"{name_prefix.COMPONENT}_{kvstats.TOTAL_BLOCKS}",
            "Total number of KV cache blocks available on the worker.",
            labelnames=[labels.MODEL, labels.COMPONENT, labels.DP_RANK],
            registry=registry,
            multiprocess_mode="max",
        )
        self.gpu_cache_usage_percent = Gauge(
            f"{name_prefix.COMPONENT}_{kvstats.GPU_CACHE_USAGE_PERCENT}",
            "GPU cache usage as a percentage (0.0-1.0).",
            labelnames=[labels.MODEL, labels.COMPONENT, labels.DP_RANK],
            registry=registry,
            multiprocess_mode="max",
        )
        self.model_load_time = Gauge(
            f"{name_prefix.COMPONENT}_{model_info.LOAD_TIME_SECONDS}",
            "Model load time in seconds.",
            labelnames=[labels.MODEL, labels.COMPONENT],
            registry=registry,
            multiprocess_mode="max",
        )
        self.model_name = model_name
        self.component_name = component_name

    def set_total_blocks(self, dp_rank: str, value: int) -> None:
        self.total_blocks.labels(
            **{
                labels.MODEL: self.model_name,
                labels.COMPONENT: self.component_name,
                labels.DP_RANK: dp_rank,
            }
        ).set(value)

    def set_gpu_cache_usage(self, dp_rank: str, value: float) -> None:
        self.gpu_cache_usage_percent.labels(
            **{
                labels.MODEL: self.model_name,
                labels.COMPONENT: self.component_name,
                labels.DP_RANK: dp_rank,
            }
        ).set(value)

    def set_model_load_time(self, value: float) -> None:
        self.model_load_time.labels(
            **{labels.MODEL: self.model_name, labels.COMPONENT: self.component_name}
        ).set(value)


def register_embedding_cache_metrics(
    endpoint: "Endpoint",
    cache: "MultimodalEmbeddingCacheManager",
    model_name: str = "",
    component_name: str = "",
) -> None:
    """Register Prometheus metrics for a MultimodalEmbeddingCacheManager instance.

    Creates a dedicated CollectorRegistry to avoid prometheus_client import-ordering
    issues with SGLang's multiprocess mode. Counters use delta-based increments
    derived from the cache's monotonic stats. A threading.Lock protects against
    concurrent scrape races (axum may serve /metrics from multiple threads).

    Must be called AFTER engine initialization to ensure prometheus_client is safe
    to import.

    Thread safety note:
        The callback reads cache.stats from the axum HTTP thread while the asyncio
        event loop thread mutates the cache. Under CPython, individual attribute reads
        (ints, len()) are GIL-protected, so the worst case is a slightly inconsistent
        snapshot within a single scrape (e.g., hits incremented but misses not yet).
        This is acceptable for monitoring metrics — values are eventually consistent.
        The threading.Lock only serializes concurrent scrapes against each other.

    Args:
        endpoint: Dynamo Endpoint with metrics.register_prometheus_expfmt_callback().
        cache: The MultimodalEmbeddingCacheManager instance to observe.
        model_name: Model name for the 'model' label.
        component_name: Component name for the 'dynamo_component' label.
    """
    # Lazy import: prometheus_client must be imported AFTER set_prometheus_multiproc_dir()
    # in SGLang's multiprocess mode. This matches the existing pattern used by
    # get_prometheus_expfmt() and LLMBackendMetrics.__init__() in this file.
    from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest

    registry = CollectorRegistry()
    label_names = [labels.MODEL, labels.COMPONENT]
    label_values = {labels.MODEL: model_name, labels.COMPONENT: component_name}

    ECM = EmbeddingCacheMetrics

    # Counters (delta-incremented from cache's monotonic stats on each scrape)
    hits_counter = Counter(
        ECM.HITS_TOTAL,
        "Total embedding cache hits.",
        labelnames=label_names,
        registry=registry,
    )
    misses_counter = Counter(
        ECM.MISSES_TOTAL,
        "Total embedding cache misses.",
        labelnames=label_names,
        registry=registry,
    )
    evictions_counter = Counter(
        ECM.EVICTIONS_TOTAL,
        "Total embedding cache evictions.",
        labelnames=label_names,
        registry=registry,
    )

    # Gauges (snapshot values set on each scrape)
    utilization_gauge = Gauge(
        ECM.UTILIZATION,
        "Cache memory utilization ratio (0.0-1.0).",
        labelnames=label_names,
        registry=registry,
    )
    current_bytes_gauge = Gauge(
        ECM.CURRENT_BYTES,
        "Current cache memory usage in bytes.",
        labelnames=label_names,
        registry=registry,
    )
    entries_gauge = Gauge(
        ECM.ENTRIES,
        "Number of entries in the cache.",
        labelnames=label_names,
        registry=registry,
    )
    # Initialize all labeled metrics so they appear in output from the first scrape,
    # even before any cache activity (Prometheus best practice: export zeros, not absent).
    hits_counter.labels(**label_values)
    misses_counter.labels(**label_values)
    evictions_counter.labels(**label_values)

    lock = threading.Lock()
    prev_state = {"hits": 0, "misses": 0, "evictions": 0}

    def _collect_embedding_cache_metrics() -> str:
        """Callback invoked on each /metrics scrape."""
        with lock:
            stats = cache.stats

            # Delta-based counter increments from monotonic source values
            delta_hits = stats["hits"] - prev_state["hits"]
            delta_misses = stats["misses"] - prev_state["misses"]
            delta_evictions = stats["evictions"] - prev_state["evictions"]

            if delta_hits > 0:
                hits_counter.labels(**label_values).inc(delta_hits)
            if delta_misses > 0:
                misses_counter.labels(**label_values).inc(delta_misses)
            if delta_evictions > 0:
                evictions_counter.labels(**label_values).inc(delta_evictions)

            prev_state["hits"] = stats["hits"]
            prev_state["misses"] = stats["misses"]
            prev_state["evictions"] = stats["evictions"]

            # Set gauge snapshots
            utilization_gauge.labels(**label_values).set(stats["utilization"])
            current_bytes_gauge.labels(**label_values).set(stats["current_bytes"])
            entries_gauge.labels(**label_values).set(stats["entries"])

            return generate_latest(registry).decode("utf-8")

    endpoint.metrics.register_prometheus_expfmt_callback(
        _collect_embedding_cache_metrics
    )
    logging.info(
        "Registered embedding cache metrics (model=%s, component=%s)",
        model_name,
        component_name,
    )
