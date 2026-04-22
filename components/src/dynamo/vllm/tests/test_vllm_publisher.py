# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dynamo.vllm.publisher."""

from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_logger(monkeypatch):
    from dynamo.vllm.publisher import DynamoStatLoggerPublisher

    inner = MagicMock()
    component_gauges = MagicMock()

    def create_task(coro):
        coro.close()
        return MagicMock()

    monkeypatch.setattr("dynamo.vllm.publisher.WorkerMetricsPublisher", lambda: inner)
    monkeypatch.setattr("dynamo.vllm.publisher.asyncio.create_task", create_task)

    logger = DynamoStatLoggerPublisher(
        endpoint=MagicMock(),
        dp_rank=0,
        component_gauges=component_gauges,
    )
    logger.num_gpu_block = 100
    return logger


def test_record_is_noop_when_scheduler_stats_is_none(monkeypatch):
    # vLLM passes scheduler_stats=None during sleep-mode transitions.
    # record() must not raise AttributeError, which would kill EngineCore.
    logger = _make_logger(monkeypatch)
    logger.record(scheduler_stats=None, iteration_stats=None)
    logger.inner.publish.assert_not_called()
    logger.component_gauges.set_total_blocks.assert_not_called()
    logger.component_gauges.set_gpu_cache_usage.assert_not_called()


def test_record_publishes_when_scheduler_stats_present(monkeypatch):
    logger = _make_logger(monkeypatch)
    stats = MagicMock(kv_cache_usage=0.5)
    logger.record(scheduler_stats=stats, iteration_stats=None)
    logger.inner.publish.assert_called_once_with(0, kv_used_blocks=50)
    logger.component_gauges.set_gpu_cache_usage.assert_called_once_with("0", 0.5)
