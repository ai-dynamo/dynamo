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


def _assert_lifecycle_operations_are_consistent(operations):
    assert [operation["operation_ordinal"] for operation in operations] == list(
        range(len(operations))
    )
    for operation in operations:
        state = operation["state_after_batch"]
        assert state["active"] == sorted(state["active"])
        assert state["starting"] == sorted(state["starting"])
        assert state["draining"] == sorted(state["draining"])
        assert not (set(state["active"]) & set(state["starting"]))
        assert not (set(state["active"]) & set(state["draining"]))
        assert not (set(state["starting"]) & set(state["draining"]))
        releases = operation["topology_released_request_uuids"]
        assert len(releases) == len(set(releases))
        for transition in operation["transitions"]:
            assert (
                transition["origin_operation_ordinal"] <= operation["operation_ordinal"]
            )
        if operation["cause"] == "planner_scale":
            assert operation["planner_tick_ordinal"] is not None
        else:
            assert operation["planner_tick_ordinal"] is None
            assert operation["origin_operation_ordinal"] is not None
            assert (
                operation["origin_operation_ordinal"] < operation["operation_ordinal"]
            )


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

    assert report.per_request is None
    assert report.coverage["capture_per_request"] is False
    assert report.planner.total_ticks == len(report.planner.ticks)
    assert report.planner.total_ticks > 0
    events = [
        (event.component, event.from_count, event.to_count)
        for event in report.planner.scaling_events
    ]
    assert report.summary["completed_requests"] == 33
    assert events == [
        ("agg", 1, 2),
        ("agg", 2, 3),
        ("agg", 3, 2),
        ("agg", 2, 1),
    ]
    lifecycle = report.planner.lifecycle_operations
    assert lifecycle
    _assert_lifecycle_operations_are_consistent(lifecycle)
    transitions = {
        transition["transition"]
        for operation in lifecycle
        for transition in operation["transitions"]
    }
    assert transitions == {
        "worker_ready",
        "worker_draining",
        "worker_removed",
    }
    assert report.summary["decode_worker_seconds"] > (
        report.summary["duration_ms"] / 1000.0
    )


def test_summary_only_planner_replay_keeps_metrics_decisions_and_tick_count(tmp_path):
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
        capture_per_request=False,
        capture_planner_details=False,
    )

    assert report.summary["completed_requests"] == 33
    assert report.per_request is None
    assert report.coverage["capture_per_request"] is False
    assert report.coverage["capture_planner_details"] is False
    assert report.planner.total_ticks > 0
    assert report.planner.scaling_events
    assert report.planner.ticks == []
    assert report.planner.lifecycle_operations == []


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
        for event in report.planner.scaling_events
        if event.component == component
    ]
    assert report.summary["completed_requests"] == (3 if component == "prefill" else 33)
    assert events[0] == (1, 2)
    assert events[-1] == (2, 1)
    assert max(to_count for _, to_count in events) > 1
    lifecycle = [
        operation
        for operation in report.planner.lifecycle_operations
        if operation["pool"] == component
    ]
    _assert_lifecycle_operations_are_consistent(lifecycle)
    starting = next(
        operation
        for operation in lifecycle
        if any(
            transition["transition"] == "worker_starting"
            for transition in operation["transitions"]
        )
    )
    ready = next(
        operation
        for operation in lifecycle
        if operation["cause"] == "worker_ready_event"
    )
    assert ready["origin_operation_ordinal"] == starting["operation_ordinal"]
    assert ready["at_ms"] > starting["at_ms"]
    assert any(
        transition["transition"] == "worker_removed"
        for operation in lifecycle
        for transition in operation["transitions"]
    )
    assert any(
        transition["transition"] == "worker_draining"
        for operation in lifecycle
        for transition in operation["transitions"]
    )
    worker_seconds_key = f"{component}_worker_seconds"
    assert report.summary[worker_seconds_key] > (report.summary["duration_ms"] / 1000.0)
