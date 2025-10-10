# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prometheus metrics utilities for Dynamo components.

This module provides shared functionality for collecting and logging Prometheus metrics
across different backend components (SGLang, vLLM, etc.).

Note: Engine metrics (vLLM, SGLang, TRT-LLM) take time to appear after engine initialization,
while Dynamo runtime metrics are available immediately after component creation.
"""

import logging
from typing import TYPE_CHECKING, Optional

from dynamo._core import Endpoint
from dynamo.common.utils.env import env_is_truthy

# Import CollectorRegistry only for type hints to avoid importing prometheus_client at module load time.
# prometheus_client must be imported AFTER set_prometheus_multiproc_dir() is called.
# See main.py worker() function for detailed explanation.
if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry


def is_engine_metrics_callback_enabled() -> bool:
    """
    Check if engine metrics callback passthrough is enabled.

    Returns:
        True if callback-based metrics passthrough is enabled when DYN_ENGINE_METRICS_ENABLED is set to a truthy value.
        False otherwise (including when not set).

    Note: To enable, explicitly set DYN_ENGINE_METRICS_ENABLED=1 or =true

    Example:
        export DYN_ENGINE_METRICS_ENABLED=1     # Enable callback
        export DYN_ENGINE_METRICS_ENABLED=true  # Enable callback
        export DYN_ENGINE_METRICS_ENABLED=0     # Disable callback (or just unset)
    """
    return env_is_truthy("DYN_ENGINE_METRICS_ENABLED")


def register_engine_metrics_callback(
    endpoint: Endpoint,
    registry: "CollectorRegistry",
    metric_prefix: str,
    engine_name: str,
) -> None:
    """
    Register a callback to expose engine Prometheus metrics via endpoint.

    This function registers a callback that will be invoked when the /metrics endpoint
    is scraped, allowing engine metrics to be included in the endpoint output.

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
    if not is_engine_metrics_callback_enabled():
        logging.info(f"{engine_name} metrics passthrough disabled")
        return

    def get_expfmt() -> str:
        """Callback to return engine Prometheus metrics in exposition format"""
        return get_prometheus_expfmt(registry, metric_prefix_filter=metric_prefix)

    endpoint.metrics.register_prometheus_expfmt_callback(get_expfmt)
    logging.info(f"Registered {engine_name} metrics exposition text callback")


def get_prometheus_expfmt(
    registry,
    metric_prefix_filter: Optional[str] = None,
) -> str:
    """
    Get Prometheus metrics from a registry formatted as text using the standard text encoder.

    Collects all metrics from the registry and returns them in Prometheus text exposition format.
    Optionally filters metrics by prefix.

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
        from prometheus_client import generate_latest

        # Generate metrics in Prometheus text format
        metrics_text = generate_latest(registry).decode("utf-8")

        if metric_prefix_filter:
            # Filter lines that contain the prefix
            filtered_lines = []
            for line in metrics_text.split("\n"):
                # Keep comment lines (TYPE, HELP) and metric lines that match prefix
                if line.startswith("#") or line.startswith(metric_prefix_filter):
                    filtered_lines.append(line)

            result = "\n".join(filtered_lines)
            if result:
                logging.debug(f"=== {metric_prefix_filter} Prometheus Metrics ===")
                logging.debug("\n" + result)
                logging.debug("=" * 50)
                # Ensure result ends with newline
                if result and not result.endswith("\n"):
                    result += "\n"
            else:
                logging.debug(
                    f"No {metric_prefix_filter} metrics collected yet. "
                    f"Metrics will appear after engine initialization completes."
                )
            return result
        else:
            if metrics_text.strip():
                logging.debug("=== Prometheus Metrics ===")
                logging.debug("\n" + metrics_text)
                logging.debug("=" * 50)
            else:
                logging.debug("No metrics collected yet")
            # Ensure metrics_text ends with newline
            if metrics_text and not metrics_text.endswith("\n"):
                metrics_text += "\n"
            return metrics_text

    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
        return ""
