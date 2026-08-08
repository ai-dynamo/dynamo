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
from unittest.mock import AsyncMock, Mock, call

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
    # Embedding factory never tracks a created chat logger, so the
    # downstream ``init_publish`` / ``set_num_gpu_blocks_all`` calls in
    # the chat path are safe no-ops if anyone ever wires them on the
    # embedding branch by mistake.
    assert not factory.created_loggers


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


@pytest.mark.asyncio
async def test_snapshot_factory_defers_endpoint_and_publishes_gauges(monkeypatch):
    inners = [Mock(create_endpoint=AsyncMock()), Mock(create_endpoint=AsyncMock())]
    monkeypatch.setattr(
        publisher_mod,
        "WorkerMetricsPublisher",
        Mock(side_effect=inners),
    )
    component_gauges = Mock()
    factory = StatLoggerFactory(component_gauges=component_gauges)

    loggers = [
        factory.create_stat_logger(dp_rank=0),
        factory.create_stat_logger(dp_rank=1),
    ]
    assert all(isinstance(logger, DynamoStatLoggerPublisher) for logger in loggers)
    assert factory.created_loggers == loggers
    assert all(logger._endpoint_task is None for logger in loggers)

    factory.set_num_gpu_blocks_all(48)
    factory.init_publish()
    loggers[0].record(Mock(kv_cache_usage=0.25), None)
    loggers[1].record(Mock(kv_cache_usage=0.5), None)

    inners[0].publish.assert_has_calls(
        [call(0, kv_used_blocks=0), call(0, kv_used_blocks=12)]
    )
    inners[1].publish.assert_has_calls(
        [call(1, kv_used_blocks=0), call(1, kv_used_blocks=24)]
    )
    component_gauges.set_total_blocks.assert_has_calls(
        [call("0", 48), call("1", 48), call("0", 48), call("1", 48)]
    )
    component_gauges.set_gpu_cache_usage.assert_has_calls(
        [call("0", 0.0), call("1", 0.0), call("0", 0.25), call("1", 0.5)]
    )

    endpoint = SimpleNamespace()
    factory.bind_endpoint(endpoint)
    assert all(logger._endpoint_task is not None for logger in loggers)
    await loggers[0]._endpoint_task
    await loggers[1]._endpoint_task
    for inner in inners:
        inner.create_endpoint.assert_awaited_once_with(endpoint)
