# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge Triton's native metrics into Dynamo's /metrics endpoint."""

import logging
import threading

from tritonserver import Server as TritonServer

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

    def close(self) -> None:
        """Block any future collect() call from reaching the native server, and
        wait for an in-flight collect() to finish.
        """
        with self._lock:
            self._active = False


def create_metrics_endpoint_url(runtime: DistributedRuntime, config: Config) -> str:
    return f"{config.namespace}.{config.server_id}._metrics"


def _register_triton_metrics_bridge(
    runtime: DistributedRuntime,
    config: Config,
    server: TritonServer,
) -> TritonMetricsBridge:
    """Register one process-scoped Triton exposition callback with Dynamo."""
    # The endpoint owns a single process-wide callback and is never served
    # or registered in discovery. Model endpoints use a hash suffix from
    # endpoint_slug, ensuring this reserved name does not conflict with them.
    metrics_endpoint = runtime.endpoint(create_metrics_endpoint_url(runtime, config))
    metrics_bridge = TritonMetricsBridge(server)
    metrics_endpoint.metrics.register_prometheus_expfmt_callback(metrics_bridge.collect)

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
