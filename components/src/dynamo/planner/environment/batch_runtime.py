# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle owner for native single-pool batch scheduling I/O."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

import aiohttp
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

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import BatchSchedulingConfig

logger = logging.getLogger(__name__)


class NativeBatchSchedulingProvider:
    """Own HTTP/Redis clients for the native Planner batch tick path.

    Construction is side-effect free so ``construct_environment`` remains a
    synchronous dependency-composition root. Network clients are created in
    ``initialize`` and released idempotently in ``shutdown``. A best-effort
    zero-rate lease on shutdown prevents planned termination from leaving a
    positive admission decision alive until TTL expiry.
    """

    def __init__(self, config: BatchSchedulingConfig) -> None:
        if not config.enabled:
            raise ValueError("NativeBatchSchedulingProvider requires enabled config")
        self._config = config
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
        assert cfg.gateway is not None
        assert cfg.metrics is not None
        assert cfg.redis is not None
        assert cfg.pool is not None

        try:
            self._gateway_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=cfg.gateway.request_timeout_seconds)
            )
            self._metrics_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=cfg.metrics.request_timeout_seconds)
            )
            jobs = BatchGatewayJobSource(
                base_url=cfg.gateway.base_url,
                session=self._gateway_session,
                pool_resolver=lambda _job: cfg.pool.pool_id,
                work_class_resolver=lambda _job: cfg.pool.work_class,
                headers={"X-MaaS-Username": cfg.gateway.tenant},
                page_size=cfg.gateway.page_size,
                max_pages=cfg.gateway.max_pages,
            )
            online = OpenMetricsOnlineTrafficSource(
                pool_id=cfg.pool.pool_id,
                metrics_url=cfg.metrics.frontend_metrics_url,
                session=self._metrics_session,
                match_labels=cfg.metrics.online_match_labels,
            )
            feedback = LlmdAsyncOpenMetricsSource(
                pools=[cfg.pool.pool_id],
                metrics_url=cfg.metrics.dispatcher_metrics_url,
                session=self._metrics_session,
            )
            self._collector = BatchSchedulingCollector(
                batch_jobs=jobs,
                online_traffic=online,
                dispatcher_feedback=feedback,
            )

            redis_url = cfg.redis.url
            get_secret_value = getattr(redis_url, "get_secret_value", None)
            if callable(get_secret_value):
                redis_url = get_secret_value()
            self._actuator = RedisLeasedDrainLimitActuator.from_url(
                str(redis_url),
                control_key_resolver=self._control_key_for_pool,
                decode_responses=True,
                socket_connect_timeout=cfg.redis.connect_timeout_seconds,
                socket_timeout=cfg.redis.socket_timeout_seconds,
            )
            self._initialized = True
        except BaseException:
            await self._close_resources(publish_pause=False)
            raise

    async def collect(self) -> BatchSchedulingObservation:
        if not self._initialized or self._collector is None:
            raise RuntimeError("batch scheduling provider is not initialized")
        return await self._collector.collect()

    async def apply_drain_limits(
        self, decisions: list[BatchDrainLimitDecision]
    ) -> None:
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
        await self._close_resources(publish_pause=self._initialized)

    def _control_key_for_pool(self, pool_id: str) -> str:
        pool = self._config.pool
        redis = self._config.redis
        assert pool is not None and redis is not None
        if pool_id != pool.pool_id:
            raise ValueError(
                f"drain decision targeted unexpected pool {pool_id!r}; "
                f"configured pool is {pool.pool_id!r}"
            )
        return redis.control_key

    async def _close_resources(self, *, publish_pause: bool) -> None:
        actuator = self._actuator
        self._actuator = None
        self._collector = None
        self._initialized = False

        cancellation: Optional[asyncio.CancelledError] = None
        try:
            if publish_pause and actuator is not None:
                pool = self._config.pool
                assert pool is not None
                now_s = time.time()
                pause = BatchDrainLimitDecision(
                    pool_id=pool.pool_id,
                    max_admission_rps=0.0,
                    valid_until_s=now_s + pool.drain_lease_duration_seconds,
                    decision_id=f"planner-shutdown-{time.time_ns()}",
                )
                await asyncio.wait_for(
                    actuator.apply_drain_limit(pause),
                    timeout=max(
                        1.0,
                        self._config.redis.connect_timeout_seconds
                        + self._config.redis.socket_timeout_seconds,
                    ),
                )
        except asyncio.CancelledError as exc:
            cancellation = exc
        except Exception:
            if publish_pause:
                logger.exception(
                    "Failed to publish zero batch-drain lease during shutdown; "
                    "the existing Redis lease will expire by TTL"
                )
        try:
            if actuator is not None:
                await actuator.aclose()
        finally:
            gateway_session = self._gateway_session
            metrics_session = self._metrics_session
            self._gateway_session = None
            self._metrics_session = None
            try:
                if gateway_session is not None:
                    await gateway_session.close()
            finally:
                if metrics_session is not None:
                    await metrics_session.close()
        if cancellation is not None:
            raise cancellation


__all__ = ["NativeBatchSchedulingProvider"]
