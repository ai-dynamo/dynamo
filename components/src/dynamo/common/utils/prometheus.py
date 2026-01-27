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

# Import CollectorRegistry only for type hints to avoid importing prometheus_client at module load time.
# prometheus_client must be imported AFTER set_prometheus_multiproc_dir() is called.
# See main.py worker() function for detailed explanation.
if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry


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
        return get_prometheus_expfmt(
            registry,
            metric_prefix_filters=metric_prefix_filters,
            exclude_prefixes=exclude_prefixes,
            add_prefix=add_prefix,
        )

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
