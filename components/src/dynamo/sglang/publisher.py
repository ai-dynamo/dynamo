# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import sglang as sgl
import zmq
import zmq.asyncio
from sglang.srt.disaggregation.kv_events import ZmqEventPublisher
from sglang.srt.utils import get_local_ip_auto, get_zmq_socket, maybe_wrap_ipv6_address

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

from dynamo.common.utils.prometheus import register_engine_metrics_callback
from dynamo.llm import (
    WorkerMetricsPublisher,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
)
from dynamo.runtime import Component, Endpoint
from dynamo.sglang.args import Config


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


# Note: We use SGLang's ZmqEventPublisher.offset_endpoint_port() directly
# to ensure perfect alignment between publisher (SGLang) and subscriber (dynamo).
# This is the same pattern used by dynamo+vLLM.


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
    ) -> None:
        """Initialize the SGLang publisher for metrics and KV events.

        Args:
            engine: The SGLang engine instance.
            config: SGLang configuration including server args.
            component: The Dynamo runtime component.
            generate_endpoint: The Dynamo endpoint for generation requests.
            metrics_labels: Optional list of label key-value pairs for metrics.
        """
        self.engine = engine
        self.server_args = config.server_args
        self.dynamo_args = config.dynamo_args
        self.generate_endpoint = generate_endpoint
        self.component = component
        self.metrics_publisher = WorkerMetricsPublisher()
        # Endpoint creation is deferred to async context in setup_sgl_metrics

        # Set default values (can be overridden later if needed)
        self.dp_rank = 0

        self._running = True
        self.kv_publishers: List[ZmqKvEventPublisher] = []

        # ZMQ setup for receiving scheduler metrics (leader node only)
        # Non-leader nodes don't receive scheduler metrics via this socket - they only
        # need KV event publishing which is set up separately in init_kv_event_publish()
        node_rank = getattr(self.server_args, "node_rank", 0) or 0
        if node_rank == 0:
            self._ctx = zmq.asyncio.Context()  # type: ignore
            self._sock = get_zmq_socket(
                self._ctx,
                zmq.PULL,
                self.engine.port_args.metrics_ipc_name,
                True,  # type: ignore
            )
        else:
            self._ctx = None
            self._sock = None
            logging.info(
                f"Non-leader node (node_rank={node_rank}): skipping scheduler metrics "
                "ZMQ socket setup. KV event publishing will still be configured."
            )

    async def run(self) -> None:
        """Continuously receive scheduler metrics from ZMQ socket and publish them.

        On non-leader nodes (node_rank >= 1), this is a no-op since they don't have
        a scheduler metrics socket. They only publish KV events via init_kv_event_publish().
        """
        if self._sock is None:
            # Non-leader node: no scheduler metrics to receive
            # Just wait until stopped (KV events are handled by separate publishers)
            while self._running:
                await asyncio.sleep(1)
            return

        while self._running:
            try:
                kv_metrics = await self._sock.recv_pyobj()  # type: ignore
                dp_rank = (
                    kv_metrics.data_parallel_rank
                    if kv_metrics.data_parallel_rank is not None
                    else self.dp_rank
                )
                self.metrics_publisher.publish(dp_rank, kv_metrics.kv_active_blocks)
            except Exception:
                if self._running:
                    logging.exception(
                        "Failed to receive or publish SGLang scheduler metrics"
                    )

    def cleanup(self) -> None:
        """Clean up ZMQ resources."""
        self._running = False

        # Close ZMQ socket and context
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception as e:
                logging.warning(f"Failed to close ZMQ socket: {e}")

        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception as e:
                logging.warning(f"Failed to terminate ZMQ context: {e}")

        # Shutdown kv publishers
        for publisher in self.kv_publishers:
            try:
                publisher.shutdown()
            except Exception as e:
                logging.warning(f"Failed to shutdown kv publisher: {e}")

        logging.info("DynamoSglangPublisher cleanup complete")

    def init_engine_metrics_publish(self) -> None:
        """Publish initial dummy metrics to bootstrap the metrics endpoint."""
        logging.info("Sending dummy metrics to initialize")
        self.metrics_publisher.publish(self.dp_rank, 0)

    def init_kv_event_publish(self) -> List[ZmqKvEventPublisher]:
        """Initialize KV event publisher(s) if configured.

        For DP attention mode, creates one subscriber per LOCAL DP rank port.
        Each SGLang scheduler in DP attention mode publishes to a unique port
        (base_port + attn_dp_rank). In multi-node setups, each node's dynamo.sglang
        instance subscribes only to the DP ranks running on that node.

        Multi-node handling:
        - Each node runs dynamo.sglang alongside its local SGLang DP ranks
        - Each dynamo.sglang subscribes only to LOCAL DP ranks (same node)
        - SGLang binds locally (wildcard), Dynamo connects locally
        - NATS handles cross-node event distribution

        Returns:
            List of ZmqKvEventPublisher instances if kv_events_config is set,
            empty list otherwise.
        """
        if self.server_args.kv_events_config:
            kv_events = json.loads(self.server_args.kv_events_config)
            base_ep = kv_events.get("endpoint")
            local_ip = get_local_ip_auto()

            # Determine DP attention configuration
            dp_size = getattr(self.server_args, "dp_size", 1) or 1
            enable_dp_attention = getattr(
                self.server_args, "enable_dp_attention", False
            )
            nnodes = getattr(self.server_args, "nnodes", 1) or 1
            node_rank = getattr(self.server_args, "node_rank", 0) or 0

            if enable_dp_attention and dp_size > 1:
                # Calculate which DP ranks are local to this node
                # DP ranks are distributed evenly across nodes
                local_dp_size = dp_size // nnodes if nnodes > 0 else dp_size
                start_dp_rank = node_rank * local_dp_size
                end_dp_rank = start_dp_rank + local_dp_size

                logging.info(
                    f"DP attention mode: node_rank={node_rank}, dp_size={dp_size}, "
                    f"nnodes={nnodes}. Subscribing to local DP ranks [{start_dp_rank}, {end_dp_rank})"
                )
            else:
                # Standard mode: single subscriber for rank 0
                start_dp_rank = 0
                end_dp_rank = 1

            for dp_rank in range(start_dp_rank, end_dp_rank):
                # Use SGLang's offset_endpoint_port to ensure alignment with publishers
                # This is the same function SGLang schedulers use to determine their bind ports
                zmq_ep = ZmqEventPublisher.offset_endpoint_port(base_ep, dp_rank)
                if not zmq_ep:
                    logging.warning(
                        f"Skipping ZMQ subscriber for dp_rank={dp_rank}: "
                        f"offset_endpoint_port returned None for base_ep={base_ep}"
                    )
                    continue

                zmq_ep = format_zmq_endpoint(zmq_ep, local_ip)

                zmq_config = ZmqKvEventPublisherConfig(
                    worker_id=self.generate_endpoint.connection_id(),
                    kv_block_size=self.server_args.page_size,
                    zmq_endpoint=zmq_ep,
                    enable_local_indexer=self.dynamo_args.enable_local_indexer,
                )
                logging.info(
                    f"Setting up ZMQ kv event subscriber for dp_rank={dp_rank} "
                    f"(connecting to {zmq_ep})"
                )
                publisher = ZmqKvEventPublisher(
                    component=self.component, config=zmq_config
                )
                self.kv_publishers.append(publisher)

        # Maintain backward compatibility: set kv_publisher to first publisher if any
        self.kv_publisher = self.kv_publishers[0] if self.kv_publishers else None

        return self.kv_publishers


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
    register_engine_metrics_callback(
        endpoint=generate_endpoint,
        registry=registry,
        metric_prefix_filters=["sglang:"],
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
    metrics_labels = [("model", engine.server_args.served_model_name)]
    publisher = DynamoSglangPublisher(
        engine, config, component, generate_endpoint, metrics_labels
    )
    # Create endpoint in async context (must await before publishing)
    await publisher.metrics_publisher.create_endpoint(component)
    logging.debug("SGLang metrics publisher endpoint created")

    publisher.init_engine_metrics_publish()
    publisher.init_kv_event_publish()

    task = asyncio.create_task(publisher.run())
    logging.info("SGLang metrics loop started")
    return publisher, task, metrics_labels
