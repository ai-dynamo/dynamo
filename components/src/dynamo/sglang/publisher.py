# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from typing import List, Optional, Tuple

import sglang as sgl
import zmq
import zmq.asyncio
from prometheus_client import CollectorRegistry, multiprocess
from sglang.srt.utils import get_local_ip_auto, get_zmq_socket

from dynamo.common.utils.prometheus import register_engine_metrics_callback
from dynamo.llm import (
    ForwardPassMetrics,
    KvStats,
    SpecDecodeStats,
    WorkerMetricsPublisher,
    WorkerStats,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
)
from dynamo.runtime import Component, Endpoint
from dynamo.sglang.args import Config


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
        self.generate_endpoint = generate_endpoint
        self.component = component
        self.metrics_publisher = WorkerMetricsPublisher()
        self.metrics_publisher.create_endpoint(component, metrics_labels)

        # Set default values (can be overridden later if needed)
        self.request_total_slots = 1024
        self.dp_rank = 0
        # TODO: Get actual GPU blocks from SGLang engine instead of hardcoded value
        # This hardcoded value causes dynamo_component_kvstats_total_blocks to be incorrect.
        self.num_gpu_block = 1024

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
                self._record_values(
                    request_active_slots=kv_metrics.request_active_slots,
                    request_total_slots=kv_metrics.request_total_slots,
                    kv_active_blocks=kv_metrics.kv_active_blocks,
                    kv_total_blocks=kv_metrics.kv_total_blocks,
                    num_requests_waiting=kv_metrics.num_requests_waiting,
                    gpu_cache_usage_perc=kv_metrics.gpu_cache_usage_perc,
                    gpu_prefix_cache_hit_rate=kv_metrics.gpu_prefix_cache_hit_rate,
                    data_parallel_rank=kv_metrics.data_parallel_rank,
                )
            except Exception:
                logging.exception(
                    "Failed to receive or publish SGLang scheduler metrics"
                )

    def init_engine_metrics_publish(self) -> None:
        """Publish initial dummy metrics to bootstrap the metrics endpoint."""
        worker_stats = WorkerStats(
            request_active_slots=0,
            request_total_slots=self.request_total_slots,
            num_requests_waiting=0,
            data_parallel_rank=self.dp_rank,
        )
        kv_stats = KvStats(
            kv_active_blocks=0,
            # TODO: num_gpu_block to get actual GPU blocks from SGLang engine instead of hardcoded value
            kv_total_blocks=self.num_gpu_block,
            gpu_cache_usage_perc=0.0,
            gpu_prefix_cache_hit_rate=0.0,
        )
        metrics = ForwardPassMetrics(
            worker_stats=worker_stats,
            kv_stats=kv_stats,
            spec_decode_stats=None,
        )
        logging.info("Sending dummy metrics to initialize")
        self.metrics_publisher.publish(metrics)

    def init_kv_event_publish(self) -> Optional[ZmqKvEventPublisher]:
        """Initialize KV event publisher if configured.

        Returns:
            ZmqKvEventPublisher instance if kv_events_config is set, None otherwise.
        """
        self.kv_publisher = None
        if self.server_args.kv_events_config:
            kv_events = json.loads(self.server_args.kv_events_config)
            ep = kv_events.get("endpoint")
            zmq_ep = ep.replace("*", get_local_ip_auto()) if ep else None

            zmq_config = ZmqKvEventPublisherConfig(
                worker_id=self.generate_endpoint.connection_id(),
                kv_block_size=self.server_args.page_size,
                zmq_endpoint=zmq_ep,
            )
            logging.info(f"Setting up ZMQ kv event publisher at {zmq_ep}")
            self.kv_publisher = ZmqKvEventPublisher(
                component=self.component, config=zmq_config
            )
        return self.kv_publisher

    def _record(
        self,
        worker_stats: WorkerStats,
        kv_stats: KvStats,
        spec_decode_stats: Optional[SpecDecodeStats] = None,
    ) -> None:
        """Package and publish metrics.

        Args:
            worker_stats: Worker-level statistics.
            kv_stats: KV cache statistics.
            spec_decode_stats: Optional speculative decoding statistics.
        """
        metrics = ForwardPassMetrics(
            worker_stats=worker_stats,
            kv_stats=kv_stats,
            spec_decode_stats=spec_decode_stats,
        )
        self.metrics_publisher.publish(metrics)

    def _record_values(
        self,
        request_active_slots: int,
        request_total_slots: int,
        kv_active_blocks: int,
        kv_total_blocks: int,
        num_requests_waiting: int,
        gpu_cache_usage_perc: float,
        gpu_prefix_cache_hit_rate: float,
        data_parallel_rank: Optional[int] = None,
        spec_decode_stats: Optional[SpecDecodeStats] = None,
    ) -> None:
        """Create stats objects from raw values and publish.

        Args:
            request_active_slots: Number of active request slots.
            request_total_slots: Total number of request slots.
            kv_active_blocks: Number of active KV cache blocks.
            kv_total_blocks: Total number of KV cache blocks.
            num_requests_waiting: Number of queued requests.
            gpu_cache_usage_perc: GPU cache utilization percentage.
            gpu_prefix_cache_hit_rate: Prefix cache hit rate.
            data_parallel_rank: Optional data parallel rank.
            spec_decode_stats: Optional speculative decoding statistics.
        """
        worker_stats = WorkerStats(
            request_active_slots=request_active_slots,
            request_total_slots=request_total_slots,
            num_requests_waiting=num_requests_waiting,
            data_parallel_rank=data_parallel_rank
            if data_parallel_rank is not None
            else self.dp_rank,
        )
        kv_stats = KvStats(
            kv_active_blocks=kv_active_blocks,
            kv_total_blocks=kv_total_blocks,
            gpu_cache_usage_perc=gpu_cache_usage_perc,
            gpu_prefix_cache_hit_rate=gpu_prefix_cache_hit_rate,
        )
        self._record(worker_stats, kv_stats, spec_decode_stats)


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
    publisher.init_engine_metrics_publish()
    publisher.init_kv_event_publish()

    # Register Prometheus metrics callback if enabled
    if engine.server_args.enable_metrics:
        # SGLang uses multiprocess architecture where metrics are stored in shared memory.
        # MultiProcessCollector aggregates metrics from all worker processes.
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        register_engine_metrics_callback(
            endpoint=generate_endpoint,
            registry=registry,
            metric_prefix_filter="sglang:",
        )

    task = asyncio.create_task(publisher.run())
    logging.info("SGLang metrics loop started")
    return publisher, task, metrics_labels
