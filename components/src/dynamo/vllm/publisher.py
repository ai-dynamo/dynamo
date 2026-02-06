# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import List, Optional, Tuple

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

from dynamo.llm import WorkerMetricsPublisher
from dynamo.runtime import Component


class NullStatLogger(StatLoggerBase):
    def __init__(self):
        pass

    def record(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        engine_idx: int = 0,
        *args,
        **kwargs,
    ):
        pass

    def log_engine_initialized(self):
        pass


class DynamoStatLoggerPublisher(StatLoggerBase):
    """Stat logger publisher. Wrapper for the WorkerMetricsPublisher to match the StatLoggerBase interface."""

    def __init__(
        self,
        component: Component,
        dp_rank: int,
        metrics_labels: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.inner = WorkerMetricsPublisher()
        self._component = component
        self.dp_rank = dp_rank
        self.num_gpu_block = 1
        # Schedule async endpoint creation
        self._endpoint_task = asyncio.create_task(self._create_endpoint())

    async def _create_endpoint(self) -> None:
        """Create the NATS endpoint asynchronously."""
        try:
            await self.inner.create_endpoint(self._component)
            logging.debug("vLLM metrics publisher endpoint created")
        except Exception:
            logging.exception("Failed to create vLLM metrics publisher endpoint")
            raise

    # TODO: Remove this and pass as metadata through shared storage
    def set_num_gpu_block(self, num_blocks):
        self.num_gpu_block = num_blocks

    def record(
        self,
        scheduler_stats: SchedulerStats,
        iteration_stats: Optional[IterationStats],
        engine_idx: int = 0,
        *args,
        **kwargs,
    ):
        active_decode_blocks = int(self.num_gpu_block * scheduler_stats.kv_cache_usage)
        self.inner.publish(self.dp_rank, active_decode_blocks)

    def init_publish(self):
        self.inner.publish(self.dp_rank, 0)

    def log_engine_initialized(self) -> None:
        pass


class StatLoggerFactory:
    """Factory for creating stat logger publishers. Required by vLLM."""

    def __init__(
        self,
        component: Component,
        dp_rank: int = 0,
        metrics_labels: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.component = component
        self.created_logger: Optional[DynamoStatLoggerPublisher] = None
        self.dp_rank = dp_rank
        self.metrics_labels = metrics_labels or []

    def create_stat_logger(self, dp_rank: int) -> StatLoggerBase:
        if self.dp_rank != dp_rank:
            return NullStatLogger()
        logger = DynamoStatLoggerPublisher(
            self.component, dp_rank, metrics_labels=self.metrics_labels
        )
        self.created_logger = logger

        return logger

    def __call__(self, vllm_config: VllmConfig, dp_rank: int) -> StatLoggerBase:
        return self.create_stat_logger(dp_rank=dp_rank)

    # TODO Remove once we publish metadata to shared storage
    def set_num_gpu_blocks_all(self, num_blocks):
        if self.created_logger:
            self.created_logger.set_num_gpu_block(num_blocks)

    def init_publish(self):
        if self.created_logger:
            self.created_logger.init_publish()
