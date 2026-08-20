# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vLLM stat-logger factory."""

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
    assert factory.created_loggers == []


def test_noop_stat_logger_record_is_safe_with_none_stats():
    logger = NoopStatLogger()

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
    factory = StatLoggerFactory(
        endpoint=SimpleNamespace(),
        embedding_worker=True,
    )
    assert factory.component_gauges is None

    logger = factory.create_stat_logger(dp_rank=0)
    assert isinstance(logger, NoopStatLogger)


def test_factory_default_is_chat_path(monkeypatch):
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


def test_factory_initializes_every_data_parallel_logger(monkeypatch):
    loggers = [Mock(spec=DynamoStatLoggerPublisher) for _ in range(2)]
    monkeypatch.setattr(
        publisher_mod,
        "DynamoStatLoggerPublisher",
        Mock(side_effect=loggers),
    )
    factory = StatLoggerFactory(
        endpoint=SimpleNamespace(),
        component_gauges=SimpleNamespace(),
    )

    factory.create_stat_logger(dp_rank=0)
    factory.create_stat_logger(dp_rank=1)
    factory.set_num_gpu_blocks_all(9)
    factory.init_publish()

    for logger in loggers:
        logger.set_num_gpu_block.assert_called_once_with(9)
        logger.init_publish.assert_called_once_with()


def test_initial_publish_uses_configured_gpu_block_count():
    logger = DynamoStatLoggerPublisher.__new__(DynamoStatLoggerPublisher)
    logger.inner = Mock()
    logger.dp_rank = 1
    logger.component_gauges = Mock()
    logger.num_gpu_block = 9

    logger.init_publish()

    logger.component_gauges.set_total_blocks.assert_called_once_with("1", 9)
