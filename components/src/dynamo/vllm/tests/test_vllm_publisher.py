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
from unittest.mock import Mock

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
    assert factory.created_loggers == []


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


def test_seed_broadcasts_to_all_dp_ranks(monkeypatch):
    """Regression for #12052: the pre-first-record seed must reach every
    per-rank publisher, not just the last one created.

    vLLM calls the factory once per data-parallel rank, and each returned
    ``DynamoStatLoggerPublisher`` is a distinct object vLLM retains. If the
    factory only remembers the last one, ``set_num_gpu_blocks_all`` /
    ``init_publish`` seed the ``total_blocks`` and ``gpu_cache_usage_percent``
    gauges for a single rank, so non-last ranks lack their zero-valued labels
    until the first scheduler record arrives.
    """
    created = []

    def _fake_publisher(*_args, **kwargs):
        m = Mock(spec=DynamoStatLoggerPublisher)
        m.dp_rank = kwargs.get("dp_rank")
        created.append(m)
        return m

    monkeypatch.setattr(publisher_mod, "DynamoStatLoggerPublisher", _fake_publisher)

    factory = StatLoggerFactory(
        endpoint=SimpleNamespace(), component_gauges=SimpleNamespace()
    )
    factory.create_stat_logger(dp_rank=0)
    factory.create_stat_logger(dp_rank=1)

    factory.set_num_gpu_blocks_all(123)
    factory.init_publish()

    assert len(created) == 2
    for publisher in created:
        publisher.set_num_gpu_block.assert_called_once_with(123)
        publisher.init_publish.assert_called_once_with()
