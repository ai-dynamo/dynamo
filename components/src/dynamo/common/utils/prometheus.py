# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prometheus metrics utilities for Dynamo components.

This module provides shared functionality for collecting and exposing Prometheus metrics
from backend engines (SGLang, vLLM, etc.) via Dynamo's metrics endpoint.

Note: Engine metrics take time to appear after engine initialization,
while Dynamo runtime metrics are available immediately after component creation.
"""

import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Pattern

from dynamo._core import Endpoint
from dynamo.prometheus_names import kvstats, labels, model_info, name_prefix

# Import CollectorRegistry and Gauge only for type hints to avoid importing prometheus_client at module load time.
# prometheus_client must be imported AFTER set_prometheus_multiproc_dir() is called.
# See main.py worker() function for detailed explanation.
if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry, Gauge


def register_engine_metrics_callback(
    endpoint: Endpoint,
    registry: "CollectorRegistry",
    metric_prefix_filters: Optional[list[str]] = None,
    exclude_prefixes: Optional[list[str]] = None,
    add_prefix: Optional[str] = None,
) -> None:
    """
    Register a callback to expose engine Prometheus metrics via Dynamo's metrics endpoint.

    This registers a callback that is invoked when /metrics is scraped, passing through
    engine-specific metrics alongside Dynamo runtime metrics.

    Args:
        endpoint: Dynamo endpoint object with metrics.register_prometheus_expfmt_callback()
        registry: Prometheus registry to collect from (e.g., REGISTRY or CollectorRegistry)
        metric_prefix_filters: List of prefixes to filter metrics (e.g., ["vllm:"], ["vllm:", "lmcache:"], or None for no filtering)
        exclude_prefixes: List of metric name prefixes to exclude (e.g., ["python_", "process_"])
        add_prefix: Prefix to add to remaining metrics (e.g., "trtllm_")

    Example:
        from prometheus_client import REGISTRY
        register_engine_metrics_callback(
            generate_endpoint, REGISTRY, metric_prefix_filters=["vllm:"]
        )

        # Include multiple metric prefixes
        register_engine_metrics_callback(
            generate_endpoint, REGISTRY, metric_prefix_filters=["vllm:", "lmcache:"]
        )

        # With filtering and prefixing for TensorRT-LLM
        register_engine_metrics_callback(
            generate_endpoint, REGISTRY,
            exclude_prefixes=["python_", "process_"],
            add_prefix="trtllm_"
        )
    """

    def get_expfmt() -> str:
        """Callback to return engine Prometheus metrics in exposition format"""
        result = get_prometheus_expfmt(
            registry,
            metric_prefix_filters=metric_prefix_filters,
            exclude_prefixes=exclude_prefixes,
            add_prefix=add_prefix,
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


@lru_cache(maxsize=128)
def _compile_help_type_pattern() -> Pattern:
    """Compile and cache regex for extracting metric names from HELP/TYPE comment lines."""
    return re.compile(r"^# (HELP|TYPE) (\S+)(.*)$")


def get_prometheus_expfmt(
    registry,
    metric_prefix_filters: Optional[list[str]] = None,
    exclude_prefixes: Optional[list[str]] = None,
    add_prefix: Optional[str] = None,
) -> str:
    """
    Get Prometheus metrics from a registry formatted as text using the standard text encoder.

    Collects all metrics from the registry and returns them in Prometheus text exposition format.
    Optionally filters metrics by prefix, excludes certain prefixes, and adds a prefix.

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
        add_prefix: Prefix to add to remaining metrics (e.g., "trtllm_")

    Returns:
        Formatted metrics text in Prometheus exposition format. Returns empty string on error.

    Example:
        # Filter to include only vllm and lmcache metrics
        get_prometheus_expfmt(registry, metric_prefix_filters=["vllm:", "lmcache:"])

        # Filter out python_/process_ metrics and add trtllm_ prefix
        get_prometheus_expfmt(registry, exclude_prefixes=["python_", "process_"], add_prefix="trtllm_")
    """
    from prometheus_client import generate_latest

    try:
        # Generate metrics in Prometheus text format
        metrics_text = generate_latest(registry).decode("utf-8")

        if metric_prefix_filters or exclude_prefixes or add_prefix:
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

            # Get cached HELP/TYPE pattern
            help_type_pattern = _compile_help_type_pattern()

            for line in metrics_text.split("\n"):
                if not line.strip():
                    continue

                # Skip excluded lines entirely
                if exclude_line_pattern and exclude_line_pattern.match(line):
                    continue

                # Apply include filter if specified
                if include_pattern and not include_pattern.match(line):
                    continue

                # Apply prefix transformation if needed
                if add_prefix:
                    # Handle HELP/TYPE comments
                    if line.startswith("# HELP ") or line.startswith("# TYPE "):
                        match = help_type_pattern.match(line)
                        if match:
                            comment_type, metric_name, rest = match.groups()
                            # Remove existing prefix if present
                            if metric_prefix_filters:
                                for prefix in metric_prefix_filters:
                                    if metric_name.startswith(prefix):
                                        metric_name = metric_name.removeprefix(prefix)
                                        break
                            # Only add prefix if it doesn't already exist
                            if not metric_name.startswith(add_prefix):
                                metric_name = add_prefix + metric_name
                            line = f"# {comment_type} {metric_name}{rest}"
                    # Handle metric lines
                    elif line and not line.startswith("#"):
                        # Extract metric name (first token)
                        parts = line.split(None, 1)
                        if parts:
                            metric_name_part = parts[0]
                            rest_of_line = parts[1] if len(parts) > 1 else ""

                            # Remove existing prefix if present
                            if metric_prefix_filters:
                                for prefix in metric_prefix_filters:
                                    if metric_name_part.startswith(prefix):
                                        metric_name_part = (
                                            metric_name_part.removeprefix(prefix)
                                        )
                                        break

                            # Only add prefix if it doesn't already exist
                            if not metric_name_part.startswith(add_prefix):
                                metric_name_part = add_prefix + metric_name_part

                            # Reconstruct line
                            line = metric_name_part + (
                                " " + rest_of_line if rest_of_line else ""
                            )
                        else:
                            # Empty line or just whitespace, skip prefix addition
                            pass

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


class DynamoComponentGauges:
    """Bundle of Dynamo component Prometheus gauges.

    This class holds references to all Dynamo component gauges for a specific registry.
    Use DynamoComponentMetrics.create_all() to create an instance.

    Attributes:
        total_blocks: Gauge for tracking total KV cache blocks available.
        gpu_cache_usage_percent: Gauge for tracking GPU cache usage percentage (0.0-1.0).
        model_load_time: Gauge for tracking model load time in seconds.
    """

    def __init__(
        self,
        total_blocks: "Gauge",
        gpu_cache_usage_percent: "Gauge",
        model_load_time: "Gauge",
    ):
        self.total_blocks = total_blocks
        self.gpu_cache_usage_percent = gpu_cache_usage_percent
        self.model_load_time = model_load_time


class DynamoComponentMetrics:
    """Factory for creating Dynamo-scoped Prometheus metrics.

    This class provides strongly-typed factory methods for creating Dynamo component metrics
    that are stable across backend engine changes. All metrics use the `dynamo_component_`
    prefix and are designed to be exposed via the metrics passthrough callback.

    For metrics to appear via the callback, the metrics registry must include
    `dynamo_component_` prefix in `metric_prefix_filters`.

    Example - Create all metrics at once:
        from prometheus_client import CollectorRegistry
        from dynamo.common.utils.prometheus import DynamoComponentMetrics

        registry = CollectorRegistry()
        gauges = DynamoComponentMetrics.create_all(registry=registry)
        gauges.total_blocks.labels(dp_rank="0").set(1000)
        gauges.gpu_cache_usage_percent.labels(dp_rank="0").set(0.75)
        gauges.model_load_time.labels(model="my-model", dynamo_component="backend").set(5.2)

    Example - Create individual metrics:
        registry = CollectorRegistry()
        gauge = DynamoComponentMetrics.total_blocks(registry=registry)
        gauge.labels(dp_rank="0").set(1000)
    """

    @staticmethod
    def create_all(registry=None) -> DynamoComponentGauges:
        """Create all Dynamo component gauges in a single call.

        This is the recommended way to create metrics - it returns a bundle containing
        all available gauges. Adding new metrics in the future won't require changing
        any signatures in calling code.

        Args:
            registry: Optional Prometheus CollectorRegistry. If None, uses default registry.

        Returns:
            DynamoComponentGauges instance containing all gauges.

        Example:
            registry = CollectorRegistry()
            gauges = DynamoComponentMetrics.create_all(registry=registry)
            gauges.total_blocks.labels(dp_rank="0").set(1000)
        """
        return DynamoComponentGauges(
            total_blocks=DynamoComponentMetrics.total_blocks(registry),
            gpu_cache_usage_percent=DynamoComponentMetrics.gpu_cache_usage_percent(
                registry
            ),
            model_load_time=DynamoComponentMetrics.model_load_time(registry),
        )

    @staticmethod
    def total_blocks(registry=None) -> "Gauge":
        """Create a gauge for tracking total KV cache blocks available.

        Args:
            registry: Optional Prometheus CollectorRegistry. If None, uses default registry.

        Returns:
            Gauge with metric name `dynamo_component_kvstats_total_blocks` and label `dp_rank`.
            Initialized to 0 for dp_rank="0" so it appears in metrics immediately.

        Labels:
            dp_rank: Data-parallel rank ID (string representation of integer).
        """
        # Import deferred: prometheus_client must be imported AFTER set_prometheus_multiproc_dir()
        # is called (see main.py worker() function for details)
        from prometheus_client import Gauge

        gauge = Gauge(
            f"{name_prefix.COMPONENT}_{kvstats.TOTAL_BLOCKS}",
            "Total number of KV cache blocks available on the worker.",
            labelnames=[labels.DP_RANK],
            registry=registry,
            multiprocess_mode="max",
        )
        # Initialize to 0 so metric appears immediately in /metrics output
        gauge.labels(**{labels.DP_RANK: "0"}).set(0)
        return gauge

    @staticmethod
    def gpu_cache_usage_percent(registry=None) -> "Gauge":
        """Create a gauge for tracking GPU cache usage percentage.

        Args:
            registry: Optional Prometheus CollectorRegistry. If None, uses default registry.

        Returns:
            Gauge with metric name `dynamo_component_kvstats_gpu_cache_usage_percent` and label `dp_rank`.
            Initialized to 0.0 for dp_rank="0" so it appears in metrics immediately.

        Labels:
            dp_rank: Data-parallel rank ID (string representation of integer).
        """
        # Import deferred: prometheus_client must be imported AFTER set_prometheus_multiproc_dir()
        # is called (see main.py worker() function for details)
        from prometheus_client import Gauge

        gauge = Gauge(
            f"{name_prefix.COMPONENT}_{kvstats.GPU_CACHE_USAGE_PERCENT}",
            "GPU cache usage as a percentage (0.0-1.0).",
            labelnames=[labels.DP_RANK],
            registry=registry,
            multiprocess_mode="max",
        )
        # Initialize to 0.0 so metric appears immediately in /metrics output
        gauge.labels(**{labels.DP_RANK: "0"}).set(0.0)
        return gauge

    @staticmethod
    def model_load_time(registry=None) -> "Gauge":
        """Create a gauge for tracking model load time in seconds.

        Args:
            registry: Optional Prometheus CollectorRegistry. If None, uses default registry.

        Returns:
            Gauge with metric name `dynamo_component_model_load_time` with labels `model` and `dynamo_component`.
            Not initialized (no default labels) - caller must provide model name and component.

        Labels:
            model: Model name/identifier (string).
            dynamo_component: Component name (e.g., "backend", "prefill", "decode", "decoder", etc.).
        """
        # Import deferred: prometheus_client must be imported AFTER set_prometheus_multiproc_dir()
        # is called (see main.py worker() function for details)
        from prometheus_client import Gauge

        gauge = Gauge(
            f"{name_prefix.COMPONENT}_{model_info.LOAD_TIME_SECONDS}",
            "Model load time in seconds.",
            labelnames=[labels.MODEL, labels.COMPONENT],
            registry=registry,
            multiprocess_mode="max",
        )
        # Note: Not initialized with default labels - caller must provide model name and component
        return gauge
