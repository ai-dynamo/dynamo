# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge Triton's native metrics into Dynamo's /metrics endpoint and OTLP export."""

import logging
import threading
from collections.abc import Iterable

from prometheus_client.metrics_core import Metric
from prometheus_client.parser import text_string_to_metric_families
from tritonserver import Server as TritonServer

from dynamo.common.utils.prometheus import get_prometheus_typed
from dynamo.runtime import DistributedRuntime
from dynamo.triton.args import Config

logger = logging.getLogger(__name__)


class TritonMetricsBridge:
    """Guards calls to triton_runtime.Server.metrics() to ensure that
    none are executed concurrently with, or after, server.stop()."""

    def __init__(self, server: TritonServer) -> None:
        self._server = server
        self._lock = threading.Lock()
        self._active = True

    def collect(self) -> str:
        """Return native Triton exposition text, unless close() has already run."""
        with self._lock:
            if not self._active:
                return ""

            # Dynamo's callback wrapper isolates exceptions from the root scrape.
            return self._server.metrics()

    def collect_typed(self) -> list:
        """Return Triton's metrics as typed families for the OTLP export,
        unless close() has already run.

        Triton only renders exposition text, so the text is parsed here. The
        lock is held across the parse as well as the native call, which keeps
        close() waiting for an in-flight typed collect exactly as it does for
        collect().
        """
        with self._lock:
            if not self._active:
                return []

            # Native and parse errors propagate; the runtime logs them and
            # skips this callback for the export.
            families = list(text_string_to_metric_families(self._server.metrics()))
            return get_prometheus_typed(_ParsedFamilies(families))

    def close(self) -> None:
        """Block any future collect() or collect_typed() call from reaching the
        native server, and wait for an in-flight one to finish.
        """
        with self._lock:
            self._active = False


class _ParsedFamilies:
    """The collect() view of already parsed families that
    get_prometheus_typed() reads."""

    def __init__(self, families: Iterable[Metric]) -> None:
        self._families = families

    def collect(self) -> Iterable[Metric]:
        return self._families


def create_metrics_endpoint_url(config: Config) -> str:
    return f"{config.namespace}.{config.server_id}._metrics"


def _register_triton_metrics_bridge(
    runtime: DistributedRuntime,
    config: Config,
    server: TritonServer,
) -> TritonMetricsBridge:
    """Register the process-scoped Triton metrics callbacks with Dynamo: the
    exposition text for /metrics and the typed families for the OTLP export."""
    # The endpoint owns the process-wide callbacks and is never served
    # or registered in discovery. Model endpoints use a hash suffix from
    # endpoint_slug, ensuring this reserved name does not conflict with them.
    metrics_endpoint = runtime.endpoint(create_metrics_endpoint_url(config))
    metrics_bridge = TritonMetricsBridge(server)
    metrics_endpoint.metrics.register_prometheus_expfmt_callback(metrics_bridge.collect)
    metrics_endpoint.metrics.register_prometheus_typed_callback(
        metrics_bridge.collect_typed
    )

    logger.info(
        "Registered Triton metrics with Dynamo endpoint",
    )
    return metrics_bridge


def _stop_triton_server(
    server: TritonServer, metrics_bridge: TritonMetricsBridge
) -> None:
    """Close the metrics bridge before stopping Triton, ensuring that
    server.stop() is never executed while a metrics collection is still
    accessing the native server object."""
    metrics_bridge.close()
    server.stop()
