# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import math
import os
from unittest.mock import Mock, patch

import pytest

from dynamo.planner.utils.planner_core import (
    DecodePlanner,
    PlannerSharedState,
    PrefillPlanner,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture(autouse=True)
def mock_prometheus_metrics():
    with patch("dynamo.planner.utils.planner_core.Gauge") as mock_gauge:
        mock_gauge.return_value = Mock()
        yield


def _build_args():
    args = argparse.Namespace()
    args.adjustment_interval = 60
    args.prefill_engine_num_gpu = 1
    args.decode_engine_num_gpu = 1
    args.min_endpoint = 1
    args.max_gpu_budget = -1
    args.ttft = 500.0
    args.itl = 50.0
    args.backend = "vllm"
    args.no_operation = True
    args.no_correction = True
    args.metric_pulling_prometheus_endpoint = "http://localhost:9090"
    args.metric_reporting_prometheus_port = 0
    args.load_predictor = "constant"
    args.load_predictor_warmup_trace = None
    args.profile_results_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "profiling_results",
        "H200_TP1P_TP1D",
    )
    args.environment = "kubernetes"
    args.namespace = "test-namespace"
    args.mode = "disagg"
    return args


def _build_prometheus_client(samples):
    client = Mock()
    client.get_avg_time_to_first_token.side_effect = [
        s["ttft_ms"] / 1000 for s in samples
    ]
    client.get_avg_inter_token_latency.side_effect = [
        s["itl_ms"] / 1000 for s in samples
    ]
    client.get_avg_request_count.side_effect = [s["num_req"] for s in samples]
    client.get_avg_request_duration.side_effect = [
        s["request_duration"] for s in samples
    ]
    client.get_avg_input_sequence_tokens.side_effect = [s["isl"] for s in samples]
    client.get_avg_output_sequence_tokens.side_effect = [s["osl"] for s in samples]
    return client


def _build_planners(args, prometheus_client):
    shared_state = PlannerSharedState()
    prefill_planner = PrefillPlanner(None, args, shared_state=shared_state)
    decode_planner = DecodePlanner(None, args, shared_state=shared_state)
    prefill_planner.prometheus_api_client = prometheus_client
    decode_planner.prometheus_api_client = prometheus_client
    prefill_planner.model_name = "test-model"
    decode_planner.model_name = "test-model"

    async def mock_get_workers_info(require_prefill=True, require_decode=True):
        return (
            ["prefill-0"] if require_prefill else [],
            ["decode-0"] if require_decode else [],
        )

    prefill_planner.get_workers_info = mock_get_workers_info
    decode_planner.get_workers_info = mock_get_workers_info
    return prefill_planner, decode_planner, shared_state


def _expected_prefill(args, prefill_planner, sample):
    pred_prefill_throughput = (
        sample["num_req"] * sample["isl"] / args.adjustment_interval
    )
    thpt_per_gpu = prefill_planner.prefill_interpolator.interpolate_thpt_per_gpu(
        sample["isl"]
    )
    expected = math.ceil(
        pred_prefill_throughput / thpt_per_gpu / args.prefill_engine_num_gpu
    )
    return max(expected, args.min_endpoint)


def _expected_decode(args, decode_planner, sample):
    (
        pred_decode_thpt_per_gpu,
        _,
        _,
    ) = decode_planner.decode_interpolator.find_best_throughput_per_gpu(
        itl=args.itl, context_length=sample["isl"] + sample["osl"] / 2
    )
    pred_decode_throughput = (
        sample["num_req"] * sample["osl"] / args.adjustment_interval
    )
    expected = math.ceil(
        pred_decode_throughput / pred_decode_thpt_per_gpu / args.decode_engine_num_gpu
    )
    return max(expected, args.min_endpoint)


def _run_interval(prefill_planner, decode_planner, shared_state):
    asyncio.run(
        prefill_planner.observe_metrics(require_prefill=True, require_decode=True)
    )
    decode_planner.update_predictors_from_metrics(shared_state.last_metrics)
    next_num_p = prefill_planner.plan_adjustment()
    next_num_d = decode_planner.plan_adjustment()
    return next_num_p, next_num_d


def test_disagg_scale_up():
    args = _build_args()
    samples = [
        {
            "num_req": 10,
            "isl": 3000,
            "osl": 150,
            "ttft_ms": 400.0,
            "itl_ms": 30.0,
            "request_duration": 20.0,
        },
        {
            "num_req": 5000,
            "isl": 3000,
            "osl": 150,
            "ttft_ms": 400.0,
            "itl_ms": 30.0,
            "request_duration": 20.0,
        },
    ]
    client = _build_prometheus_client(samples)
    prefill_planner, decode_planner, shared_state = _build_planners(args, client)

    low_p, low_d = _run_interval(prefill_planner, decode_planner, shared_state)
    high_p, high_d = _run_interval(prefill_planner, decode_planner, shared_state)

    assert low_p == _expected_prefill(args, prefill_planner, samples[0])
    assert low_d == _expected_decode(args, decode_planner, samples[0])
    assert high_p == _expected_prefill(args, prefill_planner, samples[1])
    assert high_d == _expected_decode(args, decode_planner, samples[1])
    assert high_p > low_p
    assert high_d > low_d


def test_disagg_scale_down():
    args = _build_args()
    samples = [
        {
            "num_req": 5000,
            "isl": 3000,
            "osl": 150,
            "ttft_ms": 400.0,
            "itl_ms": 30.0,
            "request_duration": 20.0,
        },
        {
            "num_req": 10,
            "isl": 3000,
            "osl": 150,
            "ttft_ms": 400.0,
            "itl_ms": 30.0,
            "request_duration": 20.0,
        },
    ]
    client = _build_prometheus_client(samples)
    prefill_planner, decode_planner, shared_state = _build_planners(args, client)

    high_p, high_d = _run_interval(prefill_planner, decode_planner, shared_state)
    low_p, low_d = _run_interval(prefill_planner, decode_planner, shared_state)

    assert high_p == _expected_prefill(args, prefill_planner, samples[0])
    assert high_d == _expected_decode(args, decode_planner, samples[0])
    assert low_p == _expected_prefill(args, prefill_planner, samples[1])
    assert low_d == _expected_decode(args, decode_planner, samples[1])
    assert low_p < high_p
    assert low_d < high_d
