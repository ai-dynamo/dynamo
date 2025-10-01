# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import List, Optional, Tuple

import sglang as sgl
import zmq
import zmq.asyncio
from sglang.srt.utils import get_zmq_socket

from dynamo.llm import (
    ForwardPassMetrics,
    KvStats,
    SpecDecodeStats,
    WorkerMetricsPublisher,
    WorkerStats,
)
from dynamo.runtime import Component


class DynamoSglangStatPublisher:
    """
    Handles SGLang metrics reception and publishing.
    """

    def __init__(
        self,
        engine: sgl.Engine,
        component: Component,
        metrics_labels: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.engine = engine
        self.inner = WorkerMetricsPublisher()
        self.inner.create_endpoint(component, metrics_labels)

        # Set default values (can be overridden later if needed)
        self.request_total_slots = 1024
        self.dp_rank = 0
        self.num_gpu_block = 1024

        # ZMQ setup for receiving scheduler metrics
        self._ctx = zmq.asyncio.Context()  # type: ignore
        self._sock = get_zmq_socket(
            self._ctx, zmq.PULL, self.engine.port_args.metrics_ipc_name, True  # type: ignore
        )

    async def run(self) -> None:
        """Main loop to receive scheduler metrics and publish them"""
        while True:
            try:
                kv_metrics = await self._sock.recv_pyobj()  # type: ignore
                self.record_values(
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

    def init_publish(self) -> None:
        worker_stats = WorkerStats(
            request_active_slots=0,
            request_total_slots=self.request_total_slots,
            num_requests_waiting=0,
            data_parallel_rank=self.dp_rank,
        )
        kv_stats = KvStats(
            kv_active_blocks=0,
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
        self.inner.publish(metrics)

    def record(
        self,
        worker_stats: WorkerStats,
        kv_stats: KvStats,
        spec_decode_stats: Optional[SpecDecodeStats] = None,
    ) -> None:
        metrics = ForwardPassMetrics(
            worker_stats=worker_stats,
            kv_stats=kv_stats,
            spec_decode_stats=spec_decode_stats,
        )
        self.inner.publish(metrics)

    def record_values(
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
        self.record(worker_stats, kv_stats, spec_decode_stats)


async def setup_sgl_metrics(
    engine: sgl.Engine,
    component: Component,
) -> tuple[DynamoSglangStatPublisher, asyncio.Task, list[tuple[str, str]]]:
    """
    Convenience bootstrap: create endpoint, publish an initial update, and start the metrics loop.
    """
    metrics_labels = [("model", engine.server_args.served_model_name)]
    publisher = DynamoSglangStatPublisher(engine, component, metrics_labels)
    publisher.init_publish()

    task = asyncio.create_task(publisher.run())
    logging.info("SGLang metrics loop started")
    return publisher, task, metrics_labels
