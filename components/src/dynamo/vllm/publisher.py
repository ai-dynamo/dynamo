# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import List, Optional, Tuple

from prometheus_client import CollectorRegistry
from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

from dynamo.common.utils.prometheus import DynamoComponentGauges, DynamoComponentMetrics
from dynamo.llm import WorkerMetricsPublisher
from dynamo.prometheus_names import labels
from dynamo.runtime import Component

# Create a dedicated registry for dynamo_component metrics
# This ensures these metrics are isolated and can be exposed via their own callback
DYNAMO_COMPONENT_REGISTRY = CollectorRegistry()

# Gauges will be lazily initialized after PROMETHEUS_MULTIPROC_DIR is set
DYNAMO_COMPONENT_GAUGES: DynamoComponentGauges | None = None


def _ensure_gauges_initialized():
    """Lazy initialization of gauges after PROMETHEUS_MULTIPROC_DIR is set."""
    global DYNAMO_COMPONENT_GAUGES
    if DYNAMO_COMPONENT_GAUGES is None:
        DYNAMO_COMPONENT_GAUGES = DynamoComponentMetrics.create_all(
            registry=DYNAMO_COMPONENT_REGISTRY
        )  # pyright: ignore[reportConstantRedefinition]


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
        # Ensure gauges are initialized (deferred until after PROMETHEUS_MULTIPROC_DIR is set)
        _ensure_gauges_initialized()

        active_decode_blocks = int(self.num_gpu_block * scheduler_stats.kv_cache_usage)
        self.inner.publish(self.dp_rank, active_decode_blocks)

        dp_rank_str = str(self.dp_rank)
        DYNAMO_COMPONENT_GAUGES.total_blocks.labels(
            **{labels.DP_RANK: dp_rank_str}
        ).set(self.num_gpu_block)

        # Set GPU cache usage percentage directly from scheduler_stats
        # Note: vLLM's scheduler_stats.kv_cache_usage returns very small values
        # (e.g., 0.0000834 for ~0.08% usage), which Prometheus outputs in scientific
        # notation (8.34e-05). This is the correct value and will be properly parsed.
        DYNAMO_COMPONENT_GAUGES.gpu_cache_usage_percent.labels(
            **{labels.DP_RANK: dp_rank_str}
        ).set(scheduler_stats.kv_cache_usage)

    def init_publish(self):
        # Ensure gauges are initialized (deferred until after PROMETHEUS_MULTIPROC_DIR is set)
        _ensure_gauges_initialized()

        self.inner.publish(self.dp_rank, 0)
        dp_rank_str = str(self.dp_rank)
        DYNAMO_COMPONENT_GAUGES.total_blocks.labels(
            **{labels.DP_RANK: dp_rank_str}
        ).set(0)
        DYNAMO_COMPONENT_GAUGES.gpu_cache_usage_percent.labels(
            **{labels.DP_RANK: dp_rank_str}
        ).set(0.0)

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
