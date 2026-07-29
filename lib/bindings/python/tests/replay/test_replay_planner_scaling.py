# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo.mocker import MockEngineArgs
from dynamo.replay import run_trace_replay

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(120),
]


def _write_burst_idle_trace(
    tmp_path,
    *,
    input_tokens,
    output_tokens,
    sentinel_ms=12_000.0,
    request_count=32,
):
    trace_path = tmp_path / "planner_burst_idle.jsonl"
    records = [
        {
            "timestamp": 0.0,
            "input_length": input_tokens,
            "output_length": output_tokens,
            "hash_ids": [request_id],
        }
        for request_id in range(request_count)
    ]
    records.append(
        {
            "timestamp": sentinel_ms,
            "input_length": 1,
            "output_length": 1,
            "hash_ids": [10_000],
        }
    )
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return trace_path


def _planner_config(mode, report_output_dir, scale_component=None):
    config = {
        "mode": mode,
        "optimization_target": "load",
        "enable_load_scaling": True,
        "enable_throughput_scaling": False,
        "load_adjustment_interval_seconds": 1,
        "throughput_adjustment_interval_seconds": 60,
        "load_scaling_down_sensitivity": 0,
        "min_endpoint": 1,
        "max_gpu_budget": 3 if mode == "agg" else 4,
        "metric_reporting_prometheus_port": 0,
        "report_interval_hours": None,
        "report_output_dir": str(report_output_dir),
        "scheduling": {"scale_interval_seconds": 1.0},
    }
    if mode == "agg":
        config.update(
            decode_scale_up_kv_rate=1.0,
            decode_scale_down_kv_rate=0.0,
        )
    elif scale_component == "prefill":
        config.update(
            prefill_scale_up_queue_tokens=1,
            prefill_scale_down_queue_tokens=0,
            decode_scale_up_kv_rate=100.0,
            decode_scale_down_kv_rate=0.0,
        )
    else:
        config.update(
            prefill_scale_up_queue_tokens=1_000_000,
            prefill_scale_down_queue_tokens=0,
            decode_scale_up_kv_rate=1.0,
            decode_scale_down_kv_rate=0.0,
        )
    return config


def test_actual_aggregated_planner_scales_up_then_down(tmp_path):
    trace_path = _write_burst_idle_trace(
        tmp_path,
        input_tokens=128,
        output_tokens=512,
    )
    report = run_trace_replay(
        trace_path,
        extra_engine_args=MockEngineArgs(
            block_size=64,
            num_gpu_blocks=16,
            max_num_seqs=32,
            speedup_ratio=50.0,
        ),
        num_workers=1,
        planner_config=_planner_config("agg", tmp_path),
    )

    events = [
        (event.component, event.from_count, event.to_count)
        for event in report.scaling_events
    ]
    assert report.trace_report["completed_requests"] == 33
    assert events == [
        ("agg", 1, 2),
        ("agg", 2, 3),
        ("agg", 3, 2),
        ("agg", 2, 1),
    ]
    assert report.trace_report["decode_worker_seconds"] > (
        report.trace_report["duration_ms"] / 1000.0
    )


@pytest.mark.parametrize(
    ("component", "input_tokens", "output_tokens"),
    [
        pytest.param("prefill", 512, 1, id="prefill"),
        pytest.param("decode", 64, 128, id="decode"),
    ],
)
def test_actual_disaggregated_planner_scales_each_pool_up_then_down(
    tmp_path,
    component,
    input_tokens,
    output_tokens,
):
    trace_path = _write_burst_idle_trace(
        tmp_path,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        sentinel_ms=12_000.0 if component == "prefill" else 500_000.0,
        request_count=2 if component == "prefill" else 32,
    )
    prefill_args = MockEngineArgs(
        block_size=64,
        num_gpu_blocks=16 if component == "prefill" else 64,
        max_num_seqs=1 if component == "prefill" else 32,
        speedup_ratio=0.01 if component == "prefill" else 1000.0,
        startup_time=2.0 if component == "prefill" else None,
        worker_type="prefill",
    )
    decode_args = MockEngineArgs(
        block_size=64,
        num_gpu_blocks=64 if component == "prefill" else 16,
        max_num_seqs=32,
        speedup_ratio=1000.0 if component == "prefill" else 0.1,
        startup_time=2.0 if component == "decode" else None,
        worker_type="decode",
    )
    report = run_trace_replay(
        trace_path,
        prefill_engine_args=prefill_args,
        decode_engine_args=decode_args,
        num_prefill_workers=1,
        num_decode_workers=1,
        planner_config=_planner_config("disagg", tmp_path, component),
    )

    events = [
        (event.from_count, event.to_count)
        for event in report.scaling_events
        if event.component == component
    ]
    assert report.trace_report["completed_requests"] == (
        3 if component == "prefill" else 33
    )
    assert events[0] == (1, 2)
    assert events[-1] == (2, 1)
    assert max(to_count for _, to_count in events) > 1
    worker_seconds_key = f"{component}_worker_seconds"
    assert report.trace_report[worker_seconds_key] > (
        report.trace_report["duration_ms"] / 1000.0
    )
