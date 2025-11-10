# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from prometheus_client import generate_latest

from dynamo._core import Endpoint

# Import CollectorRegistry only for type hints to avoid importing prometheus_client at module load time.
# prometheus_client must be imported AFTER set_prometheus_multiproc_dir() is called.
# See main.py worker() function for detailed explanation.
if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry


def register_engine_metrics_callback(
    endpoint: Endpoint,
    registry: "CollectorRegistry",
    metric_prefix_filter: Optional[str] = None,
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
        metric_prefix_filter: Prefix to filter metrics (e.g., "vllm:" or "sglang:", None for no filtering)
        exclude_prefixes: List of metric name prefixes to exclude (e.g., ["python_", "process_"])
        add_prefix: Prefix to add to remaining metrics (e.g., "trtllm:")

    Example:
        from prometheus_client import REGISTRY
        register_engine_metrics_callback(
            generate_endpoint, REGISTRY, metric_prefix_filter="vllm:"
        )

        # With filtering and prefixing for TensorRT-LLM
        register_engine_metrics_callback(
            generate_endpoint, REGISTRY,
            exclude_prefixes=["python_", "process_"],
            add_prefix="trtllm:"
        )
    """

    def get_expfmt() -> str:
        """Callback to return engine Prometheus metrics in exposition format"""
        return get_prometheus_expfmt(
            registry,
            metric_prefix_filter=metric_prefix_filter,
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
def _compile_include_pattern(metric_prefix: str) -> Pattern:
    """Compile and cache regex for including metrics by prefix."""
    escaped_prefix = re.escape(metric_prefix)
    return re.compile(rf"^(# (HELP|TYPE) )?{escaped_prefix}")


@lru_cache(maxsize=128)
def _compile_help_type_pattern() -> Pattern:
    """Compile and cache regex for extracting metric names from HELP/TYPE comment lines."""
    return re.compile(r"^# (HELP|TYPE) (\S+)(.*)$")


def get_prometheus_expfmt(
    registry,
    metric_prefix_filter: Optional[str] = None,
    exclude_prefixes: Optional[list[str]] = None,
    add_prefix: Optional[str] = None,
) -> str:
    """
    Get Prometheus metrics from a registry formatted as text using the standard text encoder.

    Collects all metrics from the registry and returns them in Prometheus text exposition format.
    Optionally filters metrics by prefix, excludes certain prefixes, and adds a prefix.

    Args:
        registry: Prometheus registry to collect from.
                 Pass CollectorRegistry with MultiProcessCollector for SGLang.
                 Pass REGISTRY for vLLM single-process mode.
        metric_prefix_filter: Optional prefix to filter displayed metrics (e.g., "vllm:").
                             If None, returns all metrics. (default: None)
        exclude_prefixes: List of metric name prefixes to exclude (e.g., ["python_", "process_"])
        add_prefix: Prefix to add to remaining metrics (e.g., "trtllm:")

    Returns:
        Formatted metrics text in Prometheus exposition format. Returns empty string on error.

    Example:
        # Filter out python_/process_ metrics and add trtllm: prefix
        get_prometheus_expfmt(registry, exclude_prefixes=["python_", "process_"], add_prefix="trtllm:")
    """
    try:
        # Generate metrics in Prometheus text format
        metrics_text = generate_latest(registry).decode("utf-8")

        if metric_prefix_filter or exclude_prefixes or add_prefix:
            lines = []

            # Get cached compiled patterns
            exclude_line_pattern = None
            if exclude_prefixes:
                exclude_line_pattern = _compile_exclude_pattern(tuple(exclude_prefixes))

            # Build include pattern if needed
            include_pattern = None
            if metric_prefix_filter:
                include_pattern = _compile_include_pattern(metric_prefix_filter)

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
                            if metric_prefix_filter and metric_name.startswith(
                                metric_prefix_filter
                            ):
                                metric_name = metric_name[len(metric_prefix_filter) :]
                            new_metric_name = add_prefix + metric_name
                            line = f"# {comment_type} {new_metric_name}{rest}"
                    # Handle metric lines
                    elif line and not line.startswith("#"):
                        # Remove existing prefix if present
                        if metric_prefix_filter and line.startswith(
                            metric_prefix_filter
                        ):
                            line = line[len(metric_prefix_filter) :]
                        line = add_prefix + line

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
