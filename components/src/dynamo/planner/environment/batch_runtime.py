# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle owner for native single-pool batch scheduling I/O."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Optional

import aiohttp

from dynamo.planner.config.planner_config import BatchSchedulingConfig
from dynamo.planner.core.types import (
    BatchDrainLimitDecision,
    BatchSchedulingObservation,
)
from dynamo.planner.environment.batch import (
    BatchGatewayJobSource,
    BatchSchedulingCollector,
    LlmdAsyncOpenMetricsSource,
    OpenMetricsOnlineTrafficSource,
    RedisLeasedDrainLimitActuator,
)

logger = logging.getLogger(__name__)


class NativeBatchSchedulingProvider:
    """Own HTTP/Redis clients for the native Planner batch tick path.

    Construction is side-effect free so ``construct_environment`` remains a
    synchronous dependency-composition root. Network clients are created in
    ``initialize`` and released idempotently in ``shutdown``. A best-effort
    zero-rate lease on shutdown prevents planned termination from leaving a
    positive admission decision alive until TTL expiry.
    """

    def __init__(
        self,
        config: BatchSchedulingConfig,
        *,
        actuation_enabled: bool = True,
        writer_fence: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("NativeBatchSchedulingProvider requires enabled config")
        self._config = config
        self._actuation_enabled = actuation_enabled
        if actuation_enabled and writer_fence is None:
            raise ValueError(
                "non-advisory native batch scheduling requires a writer fence"
            )
        self._writer_fence = writer_fence
        self._gateway_session: Optional[aiohttp.ClientSession] = None
        self._metrics_session: Optional[aiohttp.ClientSession] = None
        self._collector: Optional[BatchSchedulingCollector] = None
        self._actuator: Optional[RedisLeasedDrainLimitActuator] = None
        self._initialized = False
        self._shutdown = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self._shutdown:
            raise RuntimeError("batch scheduling provider was already shut down")

        cfg = self._config
        gateway = cfg.gateway
        metrics = cfg.metrics
        redis = cfg.redis
        pool = cfg.pool
        if gateway is None or metrics is None or pool is None:
            raise RuntimeError("enabled batch scheduling config is incomplete")

        try:
            self._gateway_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=gateway.request_timeout_seconds)
            )
            self._metrics_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=metrics.request_timeout_seconds)
            )
            jobs = BatchGatewayJobSource(
                base_url=gateway.base_url,
                session=self._gateway_session,
                pool_resolver=lambda _job: pool.pool_id,
                work_class_resolver=lambda _job: pool.work_class,
                headers={"X-MaaS-Username": gateway.tenant},
                page_size=gateway.page_size,
                max_pages=gateway.max_pages,
                max_jobs=gateway.max_jobs,
                collection_timeout_seconds=gateway.collection_timeout_seconds,
                detail_concurrency=gateway.detail_concurrency,
            )
            online = OpenMetricsOnlineTrafficSource(
                pool_id=pool.pool_id,
                metrics_url=metrics.frontend_metrics_url,
                session=self._metrics_session,
                match_labels=metrics.online_match_labels,
                request_timeout_seconds=metrics.request_timeout_seconds,
            )
            feedback = LlmdAsyncOpenMetricsSource(
                pools=[pool.pool_id],
                metrics_url=metrics.dispatcher_metrics_url,
                session=self._metrics_session,
                request_timeout_seconds=metrics.request_timeout_seconds,
            )
            self._collector = BatchSchedulingCollector(
                batch_jobs=jobs,
                online_traffic=online,
                dispatcher_feedback=feedback,
                optional_source_timeout_seconds=metrics.request_timeout_seconds,
            )

            if self._actuation_enabled:
                if redis is None or redis.url is None:
                    raise RuntimeError(
                        "non-advisory batch scheduling config has no Redis URL"
                    )
                redis_url = redis.url.get_secret_value()
                writer_id = uuid.uuid4().hex
                self._actuator = RedisLeasedDrainLimitActuator.from_url(
                    redis_url,
                    control_key_resolver=self._control_key_for_pool,
                    writer_id=writer_id,
                    decode_responses=True,
                    socket_connect_timeout=redis.connect_timeout_seconds,
                    socket_timeout=redis.socket_timeout_seconds,
                )
                now_s = time.time()
                await self._actuator.initialize_writer(
                    BatchDrainLimitDecision(
                        pool_id=pool.pool_id,
                        max_admission_rps=0.0,
                        valid_until_s=now_s + pool.drain_lease_duration_seconds,
                        decision_id=f"planner-startup-{time.time_ns()}",
                    )
                )
                writer_fence = self._writer_fence
                if writer_fence is None:
                    raise RuntimeError("batch writer fence is unavailable")
                await writer_fence(writer_id)
            self._initialized = True
        except BaseException as error:
            await self._close_resources(publish_pause=False, primary_error=error)
            raise

    async def collect(self) -> BatchSchedulingObservation:
        if not self._initialized or self._collector is None:
            raise RuntimeError("batch scheduling provider is not initialized")
        return await self._collector.collect()

    async def apply_drain_limits(
        self, decisions: list[BatchDrainLimitDecision]
    ) -> None:
        if not self._actuation_enabled:
            raise RuntimeError("batch drain-limit actuation is disabled")
        if not self._initialized or self._actuator is None:
            raise RuntimeError("batch scheduling provider is not initialized")
        if len(decisions) != 1:
            raise ValueError(
                "single-pool batch scheduling requires exactly one drain decision"
            )
        await self._actuator.apply_drain_limit(decisions[0])

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        await self._close_resources(
            publish_pause=self._initialized and self._actuation_enabled
        )

    def _control_key_for_pool(self, pool_id: str) -> str:
        pool = self._config.pool
        redis = self._config.redis
        if pool is None or redis is None:
            raise RuntimeError("enabled batch scheduling config is incomplete")
        if pool_id != pool.pool_id:
            raise ValueError(
                f"drain decision targeted unexpected pool {pool_id!r}; "
                f"configured pool is {pool.pool_id!r}"
            )
        return redis.control_key

    async def _close_resources(
        self,
        *,
        publish_pause: bool,
        primary_error: BaseException | None = None,
    ) -> None:
        actuator = self._actuator
        self._actuator = None
        self._collector = None
        self._initialized = False

        gateway_session = self._gateway_session
        metrics_session = self._metrics_session
        self._gateway_session = None
        self._metrics_session = None
        cancellation: Optional[asyncio.CancelledError] = None

        async def cleanup(label: str, operation: Awaitable[None]) -> None:
            nonlocal cancellation
            try:
                await operation
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except Exception:
                logger.exception("Failed to %s during batch-provider cleanup", label)

        if publish_pause and actuator is not None:
            pool = self._config.pool
            redis = self._config.redis
            if pool is None or redis is None:
                logger.error(
                    "Cannot publish shutdown batch pause: enabled config is incomplete"
                )
            else:
                now_s = time.time()
                pause = BatchDrainLimitDecision(
                    pool_id=pool.pool_id,
                    max_admission_rps=0.0,
                    valid_until_s=now_s + pool.drain_lease_duration_seconds,
                    decision_id=f"planner-shutdown-{time.time_ns()}",
                )
                await cleanup(
                    "publish zero batch-drain lease",
                    asyncio.wait_for(
                        actuator.apply_drain_limit(pause),
                        timeout=max(
                            1.0,
                            redis.connect_timeout_seconds
                            + redis.socket_timeout_seconds,
                        ),
                    ),
                )
        if actuator is not None:
            await cleanup("close Redis actuator", actuator.aclose())
        if gateway_session is not None:
            await cleanup("close Batch Gateway HTTP session", gateway_session.close())
        if metrics_session is not None:
            await cleanup("close metrics HTTP session", metrics_session.close())
        if primary_error is None and cancellation is not None:
            raise cancellation


__all__ = ["NativeBatchSchedulingProvider"]
