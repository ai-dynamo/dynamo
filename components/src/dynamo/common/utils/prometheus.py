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
from typing import TYPE_CHECKING, Optional

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
    metric_prefix: str,
    engine_name: str,
) -> None:
    """
    Register a callback to expose engine Prometheus metrics via Dynamo's metrics endpoint.

    This registers a callback that is invoked when /metrics is scraped, passing through
    engine-specific metrics alongside Dynamo runtime metrics.

    Args:
        endpoint: Dynamo endpoint object with metrics.register_prometheus_expfmt_callback()
        registry: Prometheus registry to collect from (e.g., REGISTRY or CollectorRegistry)
        metric_prefix: Prefix to filter metrics (e.g., "vllm:" or "sglang:")
        engine_name: Name of the engine for logging (e.g., "vLLM" or "SGLang")

    Example:
        from prometheus_client import REGISTRY
        register_engine_metrics_callback(
            generate_endpoint, REGISTRY, "vllm:", "vLLM"
        )
    """

    def get_expfmt() -> str:
        """Callback to return engine Prometheus metrics in exposition format"""
        return get_prometheus_expfmt(registry, metric_prefix_filter=metric_prefix)

    endpoint.metrics.register_prometheus_expfmt_callback(get_expfmt)


def get_prometheus_expfmt(
    registry,
    metric_prefix_filter: Optional[str] = None,
) -> str:
    """
    Get Prometheus metrics from a registry formatted as text using the standard text encoder.

    Collects all metrics from the registry and returns them in Prometheus text exposition format.
    Optionally filters metrics by prefix.

    Prometheus exposition format consists of:
    - Comment lines starting with # (HELP and TYPE declarations)
    - Metric lines with format: metric_name{label="value"} metric_value timestamp

    Example output format:
        # HELP vllm:request_success_total Number of successful requests
        # TYPE vllm:request_success_total counter
        vllm:request_success_total{model="llama2",endpoint="generate"} 150.0
        # HELP vllm:time_to_first_token_seconds Time to first token
        # TYPE vllm:time_to_first_token_seconds histogram
        vllm:time_to_first_token_seconds_bucket{model="llama2",le="0.01"} 10.0
        vllm:time_to_first_token_seconds_bucket{model="llama2",le="0.1"} 45.0
        vllm:time_to_first_token_seconds_count{model="llama2"} 50.0
        vllm:time_to_first_token_seconds_sum{model="llama2"} 2.5

    Args:
        registry: Prometheus registry to collect from.
                 Pass CollectorRegistry with MultiProcessCollector for SGLang.
                 Pass REGISTRY for vLLM single-process mode.
        metric_prefix_filter: Optional prefix to filter displayed metrics (e.g., "vllm:").
                             If None, returns all metrics. (default: None)

    Returns:
        Formatted metrics text in Prometheus exposition format. Returns empty string on error.

    Example:
        from prometheus_client import REGISTRY
        metrics_text = get_prometheus_expfmt(REGISTRY)
        print(metrics_text)

        # With filter
        vllm_metrics = get_prometheus_expfmt(REGISTRY, metric_prefix_filter="vllm:")
    """
    try:
        # Generate metrics in Prometheus text format
        metrics_text = generate_latest(registry).decode("utf-8")

        if metric_prefix_filter:
            # Filter lines: keep metric lines starting with prefix and their HELP/TYPE comments
            escaped_prefix = re.escape(metric_prefix_filter)
            pattern = rf"^(?:{escaped_prefix}|# (?:HELP|TYPE) {escaped_prefix})"
            filtered_lines = [
                line for line in metrics_text.split("\n") if re.match(pattern, line)
            ]
            result = "\n".join(filtered_lines)
            if result:
                # Ensure result ends with newline
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
