# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle tests for the native single-pool batch scheduling provider."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.planner.config.planner_config import BatchSchedulingConfig
from dynamo.planner.core.types import (
    BatchDrainLimitDecision,
    BatchSchedulingObservation,
)
from dynamo.planner.environment import batch_runtime
from dynamo.planner.environment.batch_runtime import NativeBatchSchedulingProvider

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _config() -> BatchSchedulingConfig:
    return BatchSchedulingConfig.model_validate(
        {
            "enabled": True,
            "gateway": {
                "base_url": "http://batch-gateway:8000/",
                "tenant": "planner-poc",
                "request_timeout_seconds": 7.0,
                "page_size": 50,
                "max_pages": 12,
                "max_jobs": 600,
                "collection_timeout_seconds": 11.0,
                "detail_concurrency": 7,
            },
            "metrics": {
                "frontend_metrics_url": "http://frontend:8000/metrics",
                "dispatcher_metrics_url": "http://llm-d-async:9090/metrics",
                "online_match_labels": {
                    "model": "test-model",
                    "request_type": "chat",
                    "streaming": "true",
                },
                "request_timeout_seconds": 3.0,
            },
            "redis": {
                "url": "redis://batch-gateway-valkey:6379/0",
                "control_key": "llm-d-async:drain-limit:dynamo-batch",
                "connect_timeout_seconds": 1.5,
                "socket_timeout_seconds": 2.5,
            },
            "pool": {
                "pool_id": "dynamo-batch",
                "work_class": "gsm8k-128",
                "safe_rps_per_ready_replica": 10.0,
                "drain_lease_duration_seconds": 60.0,
                "max_replicas": 8,
            },
        }
    )


def _session() -> MagicMock:
    session = MagicMock()
    session.close = AsyncMock()
    return session


def _actuator() -> MagicMock:
    actuator = MagicMock()
    actuator.initialize_writer = AsyncMock()
    actuator.apply_drain_limit = AsyncMock()
    actuator.aclose = AsyncMock()
    return actuator


@dataclass
class _InitializedProvider:
    provider: NativeBatchSchedulingProvider
    gateway_session: MagicMock
    metrics_session: MagicMock
    session_constructor: MagicMock
    actuator: MagicMock
    actuator_factory: MagicMock
    writer_fence: AsyncMock


async def _initialize_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    actuation_enabled: bool = True,
) -> _InitializedProvider:
    gateway_session = _session()
    metrics_session = _session()
    session_constructor = MagicMock(side_effect=[gateway_session, metrics_session])
    actuator = _actuator()
    actuator_factory = MagicMock(return_value=actuator)
    writer_fence = AsyncMock()
    monkeypatch.setattr(
        batch_runtime.aiohttp,
        "ClientSession",
        session_constructor,
    )
    monkeypatch.setattr(
        batch_runtime.RedisLeasedDrainLimitActuator,
        "from_url",
        actuator_factory,
    )

    provider = NativeBatchSchedulingProvider(
        _config(),
        actuation_enabled=actuation_enabled,
        writer_fence=writer_fence if actuation_enabled else None,
    )
    await provider.initialize()
    return _InitializedProvider(
        provider=provider,
        gateway_session=gateway_session,
        metrics_session=metrics_session,
        session_constructor=session_constructor,
        actuator=actuator,
        actuator_factory=actuator_factory,
        writer_fence=writer_fence,
    )


@pytest.mark.asyncio
async def test_construction_is_side_effect_free_and_uninitialized_calls_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_constructor = MagicMock()
    actuator_factory = MagicMock()
    monkeypatch.setattr(
        batch_runtime.aiohttp,
        "ClientSession",
        session_constructor,
    )
    monkeypatch.setattr(
        batch_runtime.RedisLeasedDrainLimitActuator,
        "from_url",
        actuator_factory,
    )

    provider = NativeBatchSchedulingProvider(_config(), writer_fence=AsyncMock())

    session_constructor.assert_not_called()
    actuator_factory.assert_not_called()
    assert provider._gateway_session is None
    assert provider._metrics_session is None
    assert provider._collector is None
    assert provider._actuator is None
    with pytest.raises(RuntimeError, match="not initialized"):
        await provider.collect()
    with pytest.raises(RuntimeError, match="not initialized"):
        await provider.apply_drain_limits([])


def test_construction_rejects_disabled_config() -> None:
    with pytest.raises(ValueError, match="requires enabled config"):
        NativeBatchSchedulingProvider(BatchSchedulingConfig())


@pytest.mark.asyncio
async def test_initialize_composes_real_sources_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _initialize_provider(monkeypatch)
    provider = runtime.provider
    config = provider._config
    assert config.gateway is not None
    assert config.metrics is not None
    assert config.redis is not None
    assert config.pool is not None

    await provider.initialize()

    assert runtime.session_constructor.call_count == 2
    gateway_timeout = runtime.session_constructor.call_args_list[0].kwargs["timeout"]
    metrics_timeout = runtime.session_constructor.call_args_list[1].kwargs["timeout"]
    assert gateway_timeout.total == config.gateway.request_timeout_seconds
    assert metrics_timeout.total == config.metrics.request_timeout_seconds

    collector = provider._collector
    assert collector is not None
    jobs = collector._batch_jobs
    assert jobs._base_url == config.gateway.base_url.rstrip("/")
    assert jobs._session is runtime.gateway_session
    assert jobs._headers == {"X-MaaS-Username": config.gateway.tenant}
    assert jobs._page_size == config.gateway.page_size
    assert jobs._max_pages == config.gateway.max_pages
    assert jobs._max_jobs == config.gateway.max_jobs
    assert jobs._collection_timeout_seconds == config.gateway.collection_timeout_seconds
    assert jobs._detail_concurrency == config.gateway.detail_concurrency
    assert jobs._pool_resolver({}) == config.pool.pool_id
    assert jobs._work_class_resolver({}) == config.pool.work_class

    online = collector._online_traffic
    assert online._pool_id == config.pool.pool_id
    assert online._metrics_url == config.metrics.frontend_metrics_url
    assert online._session is runtime.metrics_session
    assert online._match_labels == config.metrics.online_match_labels
    assert online._request_timeout_seconds == config.metrics.request_timeout_seconds

    feedback = collector._dispatcher_feedback
    assert feedback._pools == (config.pool.pool_id,)
    assert feedback._metrics_url == config.metrics.dispatcher_metrics_url
    assert feedback._session is runtime.metrics_session
    assert feedback._request_timeout_seconds == config.metrics.request_timeout_seconds
    assert (
        collector._optional_source_timeout_seconds
        == config.metrics.request_timeout_seconds
    )

    runtime.actuator_factory.assert_called_once()
    call = runtime.actuator_factory.call_args
    assert call.args == (config.redis.url.get_secret_value(),)
    assert call.kwargs["decode_responses"] is True
    assert call.kwargs["socket_connect_timeout"] == config.redis.connect_timeout_seconds
    assert call.kwargs["socket_timeout"] == config.redis.socket_timeout_seconds
    writer_id = call.kwargs["writer_id"]
    assert isinstance(writer_id, str) and writer_id
    assert provider._actuator is runtime.actuator
    runtime.actuator.initialize_writer.assert_awaited_once()
    startup_pause = runtime.actuator.initialize_writer.await_args.args[0]
    assert startup_pause.pool_id == config.pool.pool_id
    assert startup_pause.max_admission_rps == 0.0
    assert startup_pause.valid_until_s > 0
    assert startup_pause.decision_id.startswith("planner-startup-")
    runtime.writer_fence.assert_awaited_once_with(writer_id)


@pytest.mark.asyncio
async def test_writer_fence_blocks_actuation_until_bootstrap_and_refresh_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = [_session(), _session()]
    monkeypatch.setattr(
        batch_runtime.aiohttp,
        "ClientSession",
        MagicMock(side_effect=sessions),
    )
    actuator = _actuator()
    events: list[str] = []
    actuator.initialize_writer.side_effect = lambda _pause: events.append("redis-zero")
    monkeypatch.setattr(
        batch_runtime.RedisLeasedDrainLimitActuator,
        "from_url",
        MagicMock(return_value=actuator),
    )
    provider: NativeBatchSchedulingProvider

    async def fence(_writer_id: str) -> None:
        events.append("kubernetes-fence-refresh")
        assert provider._initialized is False
        with pytest.raises(RuntimeError, match="not initialized"):
            await provider.apply_drain_limits(
                [BatchDrainLimitDecision("dynamo-batch", 1.0, 100.0, "early")]
            )

    provider = NativeBatchSchedulingProvider(_config(), writer_fence=fence)

    await provider.initialize()

    assert events == ["redis-zero", "kubernetes-fence-refresh"]
    assert provider._initialized is True


@pytest.mark.asyncio
async def test_writer_fence_failure_preserves_bootstrap_zero_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = [_session(), _session()]
    monkeypatch.setattr(
        batch_runtime.aiohttp,
        "ClientSession",
        MagicMock(side_effect=sessions),
    )
    actuator = _actuator()
    monkeypatch.setattr(
        batch_runtime.RedisLeasedDrainLimitActuator,
        "from_url",
        MagicMock(return_value=actuator),
    )
    fence = AsyncMock(side_effect=RuntimeError("fence failed"))
    provider = NativeBatchSchedulingProvider(_config(), writer_fence=fence)

    with pytest.raises(RuntimeError, match="fence failed"):
        await provider.initialize()

    actuator.initialize_writer.assert_awaited_once()
    actuator.apply_drain_limit.assert_not_awaited()
    actuator.aclose.assert_awaited_once_with()
    sessions[0].close.assert_awaited_once_with()
    sessions[1].close.assert_awaited_once_with()
    assert provider._initialized is False


@pytest.mark.asyncio
async def test_advisory_provider_observes_without_redis_actuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _initialize_provider(monkeypatch, actuation_enabled=False)
    observation = BatchSchedulingObservation()
    collector = MagicMock()
    collector.collect = AsyncMock(return_value=observation)
    runtime.provider._collector = collector

    assert await runtime.provider.collect() is observation
    runtime.actuator_factory.assert_not_called()
    with pytest.raises(RuntimeError, match="actuation is disabled"):
        await runtime.provider.apply_drain_limits([])

    await runtime.provider.shutdown()

    runtime.actuator.apply_drain_limit.assert_not_awaited()
    runtime.actuator.aclose.assert_not_awaited()
    runtime.gateway_session.close.assert_awaited_once_with()
    runtime.metrics_session.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_advisory_provider_initializes_without_redis_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_payload = _config().model_dump(mode="python")
    config_payload["redis"] = None
    config = BatchSchedulingConfig.model_validate(config_payload)
    sessions = [_session(), _session()]
    session_constructor = MagicMock(side_effect=sessions)
    actuator_factory = MagicMock()
    monkeypatch.setattr(batch_runtime.aiohttp, "ClientSession", session_constructor)
    monkeypatch.setattr(
        batch_runtime.RedisLeasedDrainLimitActuator,
        "from_url",
        actuator_factory,
    )
    provider = NativeBatchSchedulingProvider(config, actuation_enabled=False)

    await provider.initialize()

    assert provider._initialized is True
    actuator_factory.assert_not_called()
    await provider.shutdown()


@pytest.mark.asyncio
async def test_collect_and_actuation_require_exactly_one_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _initialize_provider(monkeypatch)
    observation = BatchSchedulingObservation()
    collector = MagicMock()
    collector.collect = AsyncMock(return_value=observation)
    runtime.provider._collector = collector
    decision = BatchDrainLimitDecision(
        pool_id="dynamo-batch",
        max_admission_rps=4.0,
        valid_until_s=200.0,
        decision_id="decision-1",
    )

    assert await runtime.provider.collect() is observation
    await runtime.provider.apply_drain_limits([decision])

    collector.collect.assert_awaited_once_with()
    runtime.actuator.apply_drain_limit.assert_awaited_once_with(decision)
    with pytest.raises(ValueError, match="exactly one"):
        await runtime.provider.apply_drain_limits([])
    with pytest.raises(ValueError, match="exactly one"):
        await runtime.provider.apply_drain_limits([decision, decision])
    runtime.actuator.apply_drain_limit.assert_awaited_once_with(decision)


@pytest.mark.asyncio
async def test_redis_control_key_resolver_rejects_wrong_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _initialize_provider(monkeypatch)
    config = runtime.provider._config
    assert config.redis is not None
    resolver = runtime.actuator_factory.call_args.kwargs["control_key_resolver"]

    assert resolver("dynamo-batch") == config.redis.control_key
    with pytest.raises(ValueError, match="unexpected pool"):
        resolver("another-pool")


@pytest.mark.asyncio
async def test_shutdown_publishes_zero_lease_closes_resources_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _initialize_provider(monkeypatch)
    clock = MagicMock()
    clock.time.return_value = 100.0
    clock.time_ns.return_value = 123456789
    monkeypatch.setattr(batch_runtime, "time", clock)

    await runtime.provider.shutdown()
    await runtime.provider.shutdown()

    runtime.actuator.apply_drain_limit.assert_awaited_once()
    pause = runtime.actuator.apply_drain_limit.await_args.args[0]
    assert pause == BatchDrainLimitDecision(
        pool_id="dynamo-batch",
        max_admission_rps=0.0,
        valid_until_s=160.0,
        decision_id="planner-shutdown-123456789",
    )
    runtime.actuator.aclose.assert_awaited_once_with()
    runtime.gateway_session.close.assert_awaited_once_with()
    runtime.metrics_session.close.assert_awaited_once_with()
    assert runtime.provider._collector is None
    assert runtime.provider._actuator is None
    assert runtime.provider._gateway_session is None
    assert runtime.provider._metrics_session is None


@pytest.mark.asyncio
async def test_shutdown_pause_is_best_effort_but_resources_still_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _initialize_provider(monkeypatch)
    runtime.actuator.apply_drain_limit.side_effect = RuntimeError("redis unavailable")

    await runtime.provider.shutdown()

    runtime.actuator.apply_drain_limit.assert_awaited_once()
    runtime.actuator.aclose.assert_awaited_once_with()
    runtime.gateway_session.close.assert_awaited_once_with()
    runtime.metrics_session.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_metrics_session_initialization_failure_closes_gateway_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway_session = _session()
    session_constructor = MagicMock(
        side_effect=[gateway_session, RuntimeError("metrics session failed")]
    )
    actuator_factory = MagicMock()
    monkeypatch.setattr(
        batch_runtime.aiohttp,
        "ClientSession",
        session_constructor,
    )
    monkeypatch.setattr(
        batch_runtime.RedisLeasedDrainLimitActuator,
        "from_url",
        actuator_factory,
    )
    provider = NativeBatchSchedulingProvider(_config(), writer_fence=AsyncMock())

    with pytest.raises(RuntimeError, match="metrics session failed"):
        await provider.initialize()

    gateway_session.close.assert_awaited_once_with()
    actuator_factory.assert_not_called()
    assert provider._gateway_session is None
    assert provider._metrics_session is None
    assert provider._collector is None
    assert provider._actuator is None
    assert provider._initialized is False


@pytest.mark.asyncio
async def test_redis_initialization_failure_closes_both_http_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway_session = _session()
    metrics_session = _session()
    session_constructor = MagicMock(side_effect=[gateway_session, metrics_session])
    actuator_factory = MagicMock(side_effect=RuntimeError("redis init failed"))
    monkeypatch.setattr(
        batch_runtime.aiohttp,
        "ClientSession",
        session_constructor,
    )
    monkeypatch.setattr(
        batch_runtime.RedisLeasedDrainLimitActuator,
        "from_url",
        actuator_factory,
    )
    provider = NativeBatchSchedulingProvider(_config(), writer_fence=AsyncMock())

    with pytest.raises(RuntimeError, match="redis init failed"):
        await provider.initialize()

    gateway_session.close.assert_awaited_once_with()
    metrics_session.close.assert_awaited_once_with()
    assert provider._gateway_session is None
    assert provider._metrics_session is None
    assert provider._collector is None
    assert provider._actuator is None
    assert provider._initialized is False


@pytest.mark.asyncio
async def test_writer_bootstrap_failure_preserves_primary_and_best_effort_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway_session = _session()
    metrics_session = _session()
    metrics_session.close.side_effect = RuntimeError("secondary close failure")
    session_constructor = MagicMock(side_effect=[gateway_session, metrics_session])
    actuator = _actuator()
    actuator.initialize_writer.side_effect = RuntimeError("bootstrap failed")
    actuator.aclose.side_effect = RuntimeError("redis close failure")
    monkeypatch.setattr(batch_runtime.aiohttp, "ClientSession", session_constructor)
    monkeypatch.setattr(
        batch_runtime.RedisLeasedDrainLimitActuator,
        "from_url",
        MagicMock(return_value=actuator),
    )
    provider = NativeBatchSchedulingProvider(_config(), writer_fence=AsyncMock())

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        await provider.initialize()

    actuator.aclose.assert_awaited_once_with()
    gateway_session.close.assert_awaited_once_with()
    metrics_session.close.assert_awaited_once_with()
    assert provider._initialized is False


@pytest.mark.asyncio
async def test_shutdown_cancellation_still_closes_every_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _initialize_provider(monkeypatch)
    runtime.actuator.apply_drain_limit.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await runtime.provider.shutdown()

    runtime.actuator.aclose.assert_awaited_once_with()
    runtime.gateway_session.close.assert_awaited_once_with()
    runtime.metrics_session.close.assert_awaited_once_with()
