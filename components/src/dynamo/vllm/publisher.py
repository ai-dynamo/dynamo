# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Optional

from prometheus_client import CollectorRegistry
from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

from dynamo.common.utils.prometheus import LLMBackendMetrics
from dynamo.llm import WorkerMetricsPublisher
from dynamo.runtime import Endpoint

# Create a dedicated registry for dynamo_component metrics
# This ensures these metrics are isolated and can be exposed via their own callback
DYNAMO_COMPONENT_REGISTRY = CollectorRegistry()


class DynamoStatLoggerPublisher(StatLoggerBase):
    """Stat logger publisher. Wrapper for the WorkerMetricsPublisher to match the StatLoggerBase interface."""

    def __init__(
        self,
        endpoint: Endpoint,
        dp_rank: int = 0,
        component_gauges: Optional[LLMBackendMetrics] = None,
    ) -> None:
        self.inner = WorkerMetricsPublisher()
        self._endpoint = endpoint
        self.dp_rank = dp_rank
        self.component_gauges = component_gauges or LLMBackendMetrics()
        self.num_gpu_block = 1
        # Schedule async endpoint creation
        self._endpoint_task = asyncio.create_task(self._create_endpoint())

    async def _create_endpoint(self) -> None:
        """Create the NATS endpoint asynchronously."""
        try:
            await self.inner.create_endpoint(self._endpoint)
            logging.debug("vLLM metrics publisher endpoint created")
        except Exception:
            logging.exception("Failed to create vLLM metrics publisher endpoint")
            raise

    # TODO: Remove this and pass as metadata through shared storage
    def set_num_gpu_block(self, num_blocks: int) -> None:
        self.num_gpu_block = num_blocks

    def record(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        mm_cache_stats: object = None,
        engine_idx: int = 0,
        *args: object,
        **kwargs: object,
    ) -> None:
        if scheduler_stats is None:
            return

        active_decode_blocks = int(self.num_gpu_block * scheduler_stats.kv_cache_usage)
        self.inner.publish(self.dp_rank, kv_used_blocks=active_decode_blocks)

        dp_rank_str = str(self.dp_rank)
        self.component_gauges.set_total_blocks(dp_rank_str, self.num_gpu_block)

        self.component_gauges.set_gpu_cache_usage(
            dp_rank_str, scheduler_stats.kv_cache_usage
        )

    def init_publish(self) -> None:
        self.inner.publish(self.dp_rank, kv_used_blocks=0)
        dp_rank_str = str(self.dp_rank)
        self.component_gauges.set_total_blocks(dp_rank_str, self.num_gpu_block)
        self.component_gauges.set_gpu_cache_usage(dp_rank_str, 0.0)

    def log_engine_initialized(self) -> None:
        pass


class NoopStatLogger(StatLoggerBase):
    """Drops stats for workers without scheduler metrics."""

    def __init__(
        self,
        vllm_config: Optional[VllmConfig] = None,
        engine_index: int = 0,
    ) -> None:
        # StatLoggerBase requires this vLLM-compatible constructor.
        del vllm_config, engine_index

    def record(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        mm_cache_stats: object = None,
        engine_idx: int = 0,
        *args: object,
        **kwargs: object,
    ) -> None:
        return

    def log_engine_initialized(self) -> None:
        pass


class StatLoggerFactory:
    """Factory for creating stat logger publishers. Required by vLLM."""

    def __init__(
        self,
        endpoint: Endpoint,
        component_gauges: Optional[LLMBackendMetrics] = None,
        embedding_worker: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.component_gauges = component_gauges
        self.embedding_worker = embedding_worker
        self.created_loggers: list[DynamoStatLoggerPublisher] = []

    def create_stat_logger(self, dp_rank: int) -> StatLoggerBase:
        if self.embedding_worker:
            return NoopStatLogger()
        # Engine setup installs gauges before vLLM creates its loggers.
        assert (
            self.component_gauges is not None
        ), "component_gauges must be set before creating stat loggers"
        logger = DynamoStatLoggerPublisher(
            endpoint=self.endpoint,
            dp_rank=dp_rank,
            component_gauges=self.component_gauges,
        )
        self.created_loggers.append(logger)

        return logger

    def __call__(self, vllm_config: VllmConfig, dp_rank: int) -> StatLoggerBase:
        return self.create_stat_logger(dp_rank=dp_rank)

    # TODO Remove once we publish metadata to shared storage
    def set_num_gpu_blocks_all(self, num_blocks: int) -> None:
        for logger in self.created_loggers:
            logger.set_num_gpu_block(num_blocks)

    def init_publish(self) -> None:
        for logger in self.created_loggers:
            logger.init_publish()
