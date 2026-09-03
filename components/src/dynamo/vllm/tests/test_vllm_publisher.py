# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vLLM stat-logger factory.

These tests focus on the embedding-worker gating path: vLLM workers run
either a chat/decode engine or a pooling (embedding) engine, and the
chat-shaped Prometheus collectors are only meaningful on the former.
The factory is the single seam where vLLM calls into dynamo per dp_rank,
so it is also the seam where the embedding worker must short-circuit
the chat-shaped pipeline.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import dynamo.vllm.publisher as publisher_mod
from dynamo.vllm.publisher import (
    DynamoStatLoggerPublisher,
    NoopStatLogger,
    StatLoggerFactory,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_factory_returns_noop_logger_for_embedding_worker(monkeypatch):
    """``create_stat_logger`` returns a ``NoopStatLogger`` on the
    embedding path -- no ``DynamoStatLoggerPublisher`` /
    ``WorkerMetricsPublisher`` / NATS endpoint construction.

    Why this matters: ``DynamoStatLoggerPublisher.__init__`` schedules a
    ``create_endpoint`` task on the runtime and registers chat-shaped
    publish callbacks. On a pooling engine there is no kv_cache_usage to
    publish (vLLM never emits ``SchedulerStats``) and the endpoint is
    never queried -- so the factory must not construct it at all.
    """

    def _explode(*_a, **_kw):
        raise AssertionError(
            "embedding-worker path must not construct DynamoStatLoggerPublisher"
        )

    monkeypatch.setattr(publisher_mod, "DynamoStatLoggerPublisher", _explode)

    factory = StatLoggerFactory(
        endpoint=SimpleNamespace(),
        embedding_worker=True,
    )

    logger = factory.create_stat_logger(dp_rank=0)

    assert isinstance(logger, NoopStatLogger)


def test_noop_stat_logger_record_is_safe_with_none_stats():
    """vLLM calls ``record`` every iteration even when the engine has
    nothing useful to report. The chat-path publisher returns early on
    ``scheduler_stats is None``; the embedding noop must accept the same
    shape (and the variadic mm/engine_idx args vLLM passes) without
    raising."""
    logger = NoopStatLogger()

    # Mirrors the call shape from vllm/v1/metrics/loggers.py.
    logger.record(None, None)
    logger.record(None, None, None, 0)
    logger.record(
        scheduler_stats=None,
        iteration_stats=None,
        mm_cache_stats=None,
        engine_idx=0,
    )
    logger.log_engine_initialized()


def test_factory_embedding_flag_skips_component_gauges_assert():
    """On the chat path the factory asserts
    ``component_gauges is not None`` because ``setup_vllm_engine`` is
    responsible for setting it before vLLM invokes the factory. The
    embedding path skips that step entirely (no chat-shaped gauges to
    register), so the factory must not blow up when it stays None."""
    factory = StatLoggerFactory(
        endpoint=SimpleNamespace(),
        embedding_worker=True,
    )
    assert factory.component_gauges is None

    # Would AssertionError on the chat path; must succeed here.
    logger = factory.create_stat_logger(dp_rank=0)
    assert isinstance(logger, NoopStatLogger)


def test_factory_default_is_chat_path(monkeypatch):
    """Sibling check: the default (``embedding_worker=False``) still
    constructs ``DynamoStatLoggerPublisher`` so the gating doesn't
    accidentally swallow the chat path."""
    constructed = []

    def _fake_publisher(*args, **kwargs):
        constructed.append(kwargs)
        return Mock(spec=DynamoStatLoggerPublisher)

    monkeypatch.setattr(publisher_mod, "DynamoStatLoggerPublisher", _fake_publisher)

    endpoint = SimpleNamespace()
    component_gauges = SimpleNamespace()
    factory = StatLoggerFactory(endpoint=endpoint, component_gauges=component_gauges)

    factory.create_stat_logger(dp_rank=3)

    assert len(constructed) == 1
    assert constructed[0]["endpoint"] is endpoint
    assert constructed[0]["dp_rank"] == 3
    assert constructed[0]["component_gauges"] is component_gauges


def test_factory_initializes_every_dp_rank_logger(monkeypatch):
    loggers = []

    def _fake_publisher(*args, **kwargs):
        logger = Mock(spec=DynamoStatLoggerPublisher)
        loggers.append(logger)
        return logger

    monkeypatch.setattr(publisher_mod, "DynamoStatLoggerPublisher", _fake_publisher)

    factory = StatLoggerFactory(
        endpoint=SimpleNamespace(), component_gauges=SimpleNamespace()
    )
    for dp_rank in range(3):
        factory.create_stat_logger(dp_rank=dp_rank)

    factory.set_num_gpu_blocks_all(4096)
    factory.init_publish()

    assert factory.created_loggers == dict(enumerate(loggers))
    for logger in loggers:
        logger.set_num_gpu_block.assert_called_once_with(4096)
        logger.init_publish.assert_called_once_with()


def test_factory_binds_deferred_endpoint_to_every_dp_rank_logger(monkeypatch):
    loggers = []

    def _fake_publisher(*args, **kwargs):
        assert kwargs["endpoint"] is None
        logger = Mock(spec=DynamoStatLoggerPublisher)
        loggers.append(logger)
        return logger

    monkeypatch.setattr(publisher_mod, "DynamoStatLoggerPublisher", _fake_publisher)

    factory = StatLoggerFactory(endpoint=None, component_gauges=SimpleNamespace())
    factory.create_stat_logger(dp_rank=0)
    factory.create_stat_logger(dp_rank=1)

    endpoint = SimpleNamespace()
    factory.bind_endpoint(endpoint)

    assert factory.endpoint is endpoint
    for logger in loggers:
        logger.bind_endpoint.assert_called_once_with(endpoint)


@pytest.mark.asyncio
async def test_deferred_logger_starts_with_fresh_metrics_state(monkeypatch):
    publishers = [Mock(create_endpoint=AsyncMock()), Mock(create_endpoint=AsyncMock())]
    monkeypatch.setattr(
        publisher_mod, "WorkerMetricsPublisher", Mock(side_effect=publishers)
    )

    logger = DynamoStatLoggerPublisher(
        endpoint=None,
        component_gauges=SimpleNamespace(),
    )
    logger.inner.publish(dp_rank=0, kv_used_blocks=7)

    endpoint = SimpleNamespace()
    logger.bind_endpoint(endpoint)
    assert logger.inner is publishers[1]

    assert logger._endpoint_task is not None
    await logger._endpoint_task
    publishers[0].publish.assert_called_once_with(dp_rank=0, kv_used_blocks=7)
    publishers[0].create_endpoint.assert_not_called()
    publishers[1].create_endpoint.assert_awaited_once_with(endpoint)
