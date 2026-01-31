# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import sglang as sgl
import zmq
import zmq.asyncio
from prometheus_client import CollectorRegistry
from sglang.srt.utils import get_local_ip_auto, get_zmq_socket, maybe_wrap_ipv6_address

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

from dynamo.common.utils.prometheus import (
    DynamoComponentGauges,
    DynamoComponentMetrics,
    register_engine_metrics_callback,
)
from dynamo.llm import (
    WorkerMetricsPublisher,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
)
from dynamo.prometheus_names import labels
from dynamo.runtime import Component, Endpoint
from dynamo.sglang.args import Config

# Create a dedicated registry for dynamo_component metrics
# This ensures these metrics are isolated and can be exposed via their own callback
DYNAMO_COMPONENT_REGISTRY = CollectorRegistry()


def format_zmq_endpoint(endpoint_template: str, ip_address: str) -> str:
    """Format ZMQ endpoint by replacing wildcard with IP address.

    Properly handles IPv6 addresses by wrapping them in square brackets.
    Uses SGLang's maybe_wrap_ipv6_address for consistent formatting.

    Args:
        endpoint_template: ZMQ endpoint template with wildcard (e.g., "tcp://*:5557")
        ip_address: IP address to use (can be IPv4 or IPv6)

    Returns:
        Formatted ZMQ endpoint string

    Example:
        >>> format_zmq_endpoint("tcp://*:5557", "192.168.1.1")
        'tcp://192.168.1.1:5557'
        >>> format_zmq_endpoint("tcp://*:5557", "2a02:6b8:c46:2b4:0:74c1:75b0:0")
        'tcp://[2a02:6b8:c46:2b4:0:74c1:75b0:0]:5557'
    """
    # Use SGLang's utility to wrap IPv6 addresses in brackets
    formatted_ip = maybe_wrap_ipv6_address(ip_address)
    return endpoint_template.replace("*", formatted_ip)


class DynamoSglangPublisher:
    """
    Handles SGLang kv events and metrics reception and publishing.
    """

    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
        component: Component,
        generate_endpoint: Endpoint,
        metrics_labels: Optional[List[Tuple[str, str]]] = None,
        component_gauges: Optional[DynamoComponentGauges] = None,
    ) -> None:
        """Initialize the SGLang publisher for metrics and KV events.

        Args:
            engine: The SGLang engine instance.
            config: SGLang configuration including server args.
            component: The Dynamo runtime component.
            generate_endpoint: The Dynamo endpoint for generation requests.
            metrics_labels: Optional list of label key-value pairs for metrics.
            component_gauges: Bundle of Dynamo component gauges (created via DynamoComponentMetrics.create_all()).
        """
        self.engine = engine
        self.server_args = config.server_args
        self.dynamo_args = config.dynamo_args
        self.generate_endpoint = generate_endpoint
        self.component = component
        self.metrics_publisher = WorkerMetricsPublisher()
        self.component_gauges = component_gauges
        # Endpoint creation is deferred to async context in setup_sgl_metrics

        # Set default values (can be overridden later if needed)
        self.dp_rank = 0

        # ZMQ setup for receiving scheduler metrics
        self._ctx = zmq.asyncio.Context()  # type: ignore
        self._sock = get_zmq_socket(
            self._ctx, zmq.PULL, self.engine.port_args.metrics_ipc_name, True  # type: ignore
        )

    async def run(self) -> None:
        """Continuously receive scheduler metrics from ZMQ socket and publish them."""
        while True:
            try:
                kv_metrics = await self._sock.recv_pyobj()  # type: ignore
                dp_rank = (
                    kv_metrics.data_parallel_rank
                    if kv_metrics.data_parallel_rank is not None
                    else self.dp_rank
                )
                active_decode_blocks = kv_metrics.kv_active_blocks
                self.metrics_publisher.publish(dp_rank, active_decode_blocks)
                if self.component_gauges:
                    dp_rank_str = str(dp_rank)
                    # Publish total blocks if available
                    if hasattr(kv_metrics, "kv_total_blocks"):
                        self.component_gauges.total_blocks.labels(
                            **{labels.DP_RANK: dp_rank_str}
                        ).set(kv_metrics.kv_total_blocks)
                    # Publish GPU cache usage if available
                    if hasattr(kv_metrics, "gpu_cache_usage"):
                        self.component_gauges.gpu_cache_usage_percent.labels(
                            **{labels.DP_RANK: dp_rank_str}
                        ).set(kv_metrics.gpu_cache_usage)
            except Exception:
                logging.exception(
                    "Failed to receive or publish SGLang scheduler metrics"
                )

    def init_engine_metrics_publish(self) -> None:
        """Publish initial dummy metrics to bootstrap the metrics endpoint."""
        logging.info("Sending dummy metrics to initialize")
        self.metrics_publisher.publish(self.dp_rank, 0)
        if self.component_gauges:
            dp_rank_str = str(self.dp_rank)
            self.component_gauges.total_blocks.labels(
                **{labels.DP_RANK: dp_rank_str}
            ).set(0)
            self.component_gauges.gpu_cache_usage_percent.labels(
                **{labels.DP_RANK: dp_rank_str}
            ).set(0.0)

    def init_kv_event_publish(self) -> Optional[ZmqKvEventPublisher]:
        """Initialize KV event publisher if configured.

        Returns:
            ZmqKvEventPublisher instance if kv_events_config is set, None otherwise.
        """
        self.kv_publisher = None
        if self.server_args.kv_events_config:
            kv_events = json.loads(self.server_args.kv_events_config)
            ep = kv_events.get("endpoint")
            if not ep:
                raise ValueError(
                    "sglang kv_events_config is set but missing 'endpoint'"
                )
            zmq_ep = format_zmq_endpoint(ep, get_local_ip_auto())

            zmq_config = ZmqKvEventPublisherConfig(
                worker_id=self.generate_endpoint.connection_id(),
                kv_block_size=self.server_args.page_size,
                zmq_endpoint=zmq_ep,
                enable_local_indexer=self.dynamo_args.enable_local_indexer,
            )
            logging.info(f"Setting up ZMQ kv event publisher at {zmq_ep}")
            self.kv_publisher = ZmqKvEventPublisher(
                component=self.component, config=zmq_config
            )
        return self.kv_publisher


def setup_prometheus_registry(
    engine: sgl.Engine, generate_endpoint: Endpoint
) -> "CollectorRegistry":
    """Set up Prometheus registry for SGLang metrics collection.

    SGLang uses multiprocess architecture where metrics are stored in shared memory.
    MultiProcessCollector aggregates metrics from all worker processes. The Prometheus
    registry collects sglang:* metrics which are exposed via the metrics server endpoint
    (set DYN_SYSTEM_PORT to a positive value to enable, e.g., DYN_SYSTEM_PORT=8081).

    IMPORTANT: prometheus_client must be imported AFTER sgl.Engine() has called
    set_prometheus_multiproc_dir(). Importing at module level causes prometheus_client
    to initialize in single-process mode before PROMETHEUS_MULTIPROC_DIR is set,
    which breaks TokenizerMetricsCollector metrics (TTFT, ITL, e2e latency, etc.).

    Args:
        engine: The SGLang engine instance.
        generate_endpoint: The Dynamo endpoint for generation requests.

    Returns:
        Configured CollectorRegistry with multiprocess support.
    """
    from prometheus_client import CollectorRegistry, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)

    # Register callback for SGLang metrics
    register_engine_metrics_callback(
        endpoint=generate_endpoint,
        registry=registry,
        metric_prefix_filters=["sglang:"],
    )

    # Register callback for Dynamo component metrics using dedicated registry
    register_engine_metrics_callback(
        endpoint=generate_endpoint,
        registry=DYNAMO_COMPONENT_REGISTRY,
    )

    return registry


async def setup_sgl_metrics(
    engine: sgl.Engine,
    config: Config,
    component: Component,
    generate_endpoint: Endpoint,
) -> tuple[DynamoSglangPublisher, asyncio.Task, list[tuple[str, str]]]:
    """Create publisher, initialize metrics, and start the metrics publishing loop.

    Args:
        engine: The SGLang engine instance.
        config: SGLang configuration including server args.
        component: The Dynamo runtime component.
        generate_endpoint: The Dynamo endpoint for generation requests.

    Returns:
        Tuple of (publisher instance, running asyncio task, metrics labels).
    """
    # Create registry and register callbacks
    setup_prometheus_registry(engine, generate_endpoint)

    # Create all Dynamo component gauges using the dedicated registry
    component_gauges = DynamoComponentMetrics.create_all(
        registry=DYNAMO_COMPONENT_REGISTRY
    )

    metrics_labels = [("model", engine.server_args.served_model_name)]
    publisher = DynamoSglangPublisher(
        engine,
        config,
        component,
        generate_endpoint,
        metrics_labels,
        component_gauges,
    )
    # Create endpoint in async context (must await before publishing)
    await publisher.metrics_publisher.create_endpoint(component)
    logging.debug("SGLang metrics publisher endpoint created")

    publisher.init_engine_metrics_publish()
    publisher.init_kv_event_publish()

    task = asyncio.create_task(publisher.run())
    logging.info("SGLang metrics loop started")
    return publisher, task, metrics_labels
