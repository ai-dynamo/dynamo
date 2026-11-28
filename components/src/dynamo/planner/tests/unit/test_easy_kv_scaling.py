# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Easy-mode KV scaling for decode-only and aggregated modes."""

import pytest

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core import load_scaling
from dynamo.planner.core.state_machine import PlannerScalingState
from dynamo.planner.core.types import EngineCapabilities, WorkerCapabilities

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

MAX_KV = 1000
LOAD_UP_RATE = 80
LOAD_DOWN_RATE = 20


def _state(mode, target="load", max_kv_tokens=MAX_KV):
    config = PlannerConfig(
        mode=mode,
        optimization_target="load",
        served_model_name="test",
        decode_scale_up_kv_rate=LOAD_UP_RATE,
        decode_scale_down_kv_rate=LOAD_DOWN_RATE,
        min_gpu_budget=-1,
        max_gpu_budget=-1,
    )
    config.optimization_target = target
    caps = WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1, max_num_batched_tokens=1024, max_kv_tokens=max_kv_tokens
        ),
    )
    return PlannerScalingState(config, caps)


def _decide(state, mode, fpm_stats, num_workers):
    if mode == "agg":
        return state._agg_easy_decision(fpm_stats, num_workers)
    return state._decode_easy_decision(fpm_stats, num_workers)


def _fpms(n, sched_kv=0, queued_kv=0, queued_prefill=0):
    return {
        (f"w{i}", 0): ForwardPassMetrics(
            worker_id=f"w{i}",
            scheduled_requests=ScheduledRequestMetrics(sum_decode_kv_tokens=sched_kv),
            queued_requests=QueuedRequestMetrics(
                sum_decode_kv_tokens=queued_kv, sum_prefill_tokens=queued_prefill
            ),
        )
        for i in range(n)
    }


def test_agg_counts_queued_prefill_and_decode_does_not():
    # 300 + 100 = 0.4 of max_kv; adding 400 queued prefill tokens lands exactly
    # on the inclusive load scale-up threshold.
    fpms = _fpms(1, sched_kv=300, queued_kv=100, queued_prefill=400)
    assert (300 + 100 + 400) / MAX_KV == LOAD_UP_RATE / 100.0

    agg = _state("agg")
    assert _decide(agg, "agg", fpms, 1) == 2

    decode = _state("decode")
    assert _decide(decode, "decode", fpms, 1) is None
    assert decode._diag_load_reason == "no_change"


@pytest.mark.parametrize("mode", ["decode", "agg"])
@pytest.mark.parametrize(
    ("target", "up", "down", "inclusive"),
    [
        ("load", LOAD_UP_RATE / 100.0, LOAD_DOWN_RATE / 100.0, True),
        (
            "latency",
            load_scaling._DECODE_LATENCY_SCALE_UP,
            load_scaling._DECODE_LATENCY_SCALE_DOWN,
            False,
        ),
        (
            "throughput",
            load_scaling._DECODE_THROUGHPUT_SCALE_UP,
            load_scaling._DECODE_THROUGHPUT_SCALE_DOWN,
            False,
        ),
    ],
)
def test_threshold_equality(mode, target, up, down, inclusive):
    up_tokens = round(up * MAX_KV)
    down_tokens = round(down * MAX_KV)
    # The ratio must equal the threshold exactly, or this tests nothing.
    assert up_tokens / MAX_KV == up
    assert down_tokens / MAX_KV == down

    state = _state(mode, target)
    result = _decide(state, mode, _fpms(1, sched_kv=up_tokens), 1)
    if inclusive:
        assert result == 2
    else:
        assert result is None
        assert state._diag_load_reason == "no_change"

    state = _state(mode, target)
    result = _decide(state, mode, _fpms(2, sched_kv=down_tokens), 2)
    if inclusive:
        assert result == 1
    else:
        assert result is None
        assert state._diag_load_reason == "no_change"


@pytest.mark.parametrize("mode", ["decode", "agg"])
@pytest.mark.parametrize("max_kv_tokens", [None, 0])
def test_missing_engine_metadata(mode, max_kv_tokens):
    state = _state(mode, max_kv_tokens=max_kv_tokens)
    assert _decide(state, mode, _fpms(1, sched_kv=900), 1) is None
    assert state._diag_load_reason == "missing_engine_metadata"
    assert state._diag_load_metadata_available == {"decode": False}


@pytest.mark.parametrize("mode", ["decode", "agg"])
@pytest.mark.parametrize(
    "missing", ["decode_scale_up_kv_rate", "decode_scale_down_kv_rate"]
)
def test_missing_load_thresholds(mode, missing):
    state = _state(mode)
    # PlannerConfig validation rejects these, so clear one after construction.
    setattr(state._config, missing, None)
    assert _decide(state, mode, _fpms(1, sched_kv=900), 1) is None
    assert state._diag_load_reason == "insufficient_data"
    assert state._diag_load_metadata_available == {"decode": True}


@pytest.mark.parametrize("mode", ["decode", "agg"])
@pytest.mark.parametrize(
    ("fpm_stats", "num_workers"), [(_fpms(1, sched_kv=900), 0), ({}, 1)]
)
def test_no_workers_or_stats_is_insufficient_data(mode, fpm_stats, num_workers):
    state = _state(mode)
    assert _decide(state, mode, fpm_stats, num_workers) is None
    assert state._diag_load_reason == "insufficient_data"
    assert state._diag_load_metadata_available == {"decode": True}
