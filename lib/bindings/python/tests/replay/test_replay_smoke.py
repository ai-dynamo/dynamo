# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from dynamo.llm import KvRouterConfig
from dynamo.mocker import MockEngineArgs, run_mocker_trace_replay
from dynamo.replay import ReplayReport, run_synthetic_trace_replay, run_trace_replay
from dynamo.replay.reporting import format_report_table, write_report_json

from .replay_utils import (
    _assert_basic_report_counts,
    _assert_basic_report_metrics,
    _decode_args,
    _partial_router_config,
    _prefill_args,
    _report_summary,
    _router_config,
    _sglang_args,
    _vllm_args,
    _write_applied_compute_agentic_trace,
    _write_multiturn_trace,
    _write_trace_and_args,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(120),
]


@pytest.mark.parametrize("engine_type", ["vllm", "sglang"])
@pytest.mark.parametrize("replay_mode", ["offline", "online"])
@pytest.mark.parametrize("router_mode", ["round_robin", "kv_router"])
@pytest.mark.parametrize("serving_mode", ["agg", "disagg"])
def test_run_trace_replay_smoke_matrix(
    tmp_path, engine_type, replay_mode, router_mode, serving_mode
):
    trace_path = _write_trace_and_args(tmp_path)
    if serving_mode == "disagg":
        if replay_mode != "offline":
            pytest.skip("disagg replay only supports offline mode")
        report = run_trace_replay(
            trace_path,
            prefill_engine_args=_prefill_args(),
            decode_engine_args=_decode_args(),
            router_config=_router_config(),
            num_prefill_workers=2,
            num_decode_workers=2,
            replay_mode=replay_mode,
            router_mode=router_mode,
        )
    else:
        args_path = _vllm_args() if engine_type == "vllm" else _sglang_args()
        num_workers = 1 if router_mode == "round_robin" else 2
        report = run_trace_replay(
            trace_path,
            extra_engine_args=args_path,
            num_workers=num_workers,
            replay_mode=replay_mode,
            router_mode=router_mode,
        )

    _assert_basic_report_counts(
        report,
        num_requests=2,
        input_tokens=64,
        output_tokens=2,
    )


@pytest.mark.parametrize("engine_type", ["vllm", "sglang"])
@pytest.mark.parametrize("replay_mode", ["offline", "online"])
def test_run_trace_replay_invariant_counts_match(tmp_path, engine_type, replay_mode):
    trace_path = _write_trace_and_args(tmp_path)
    args_path = _vllm_args() if engine_type == "vllm" else _sglang_args()

    single = run_trace_replay(
        trace_path,
        extra_engine_args=args_path,
        num_workers=1,
        replay_mode=replay_mode,
    )
    multi_round_robin = run_trace_replay(
        trace_path,
        extra_engine_args=args_path,
        num_workers=4,
        replay_mode=replay_mode,
        router_mode="round_robin",
    )
    multi_kv_router = run_trace_replay(
        trace_path,
        extra_engine_args=args_path,
        num_workers=4,
        replay_mode=replay_mode,
        router_mode="kv_router",
    )

    for field in (
        "num_requests",
        "completed_requests",
        "total_input_tokens",
        "total_output_tokens",
    ):
        assert (
            _report_summary(single)[field] == _report_summary(multi_round_robin)[field]
        )
        assert _report_summary(single)[field] == _report_summary(multi_kv_router)[field]


@pytest.mark.parametrize("replay_mode", ["offline", "online"])
def test_run_trace_replay_supports_multiturn_sessions(tmp_path, replay_mode):
    trace_path = _write_multiturn_trace(tmp_path)

    report = run_trace_replay(
        trace_path,
        extra_engine_args=_vllm_args(),
        num_workers=2,
        replay_mode=replay_mode,
        router_mode="kv_router",
    )

    _assert_basic_report_counts(
        report,
        num_requests=4,
        input_tokens=64,
        output_tokens=2,
    )


def test_offline_replay_per_request_capture_is_explicit(tmp_path):
    trace_path = _write_multiturn_trace(tmp_path)

    summary_only = run_trace_replay(
        trace_path,
        extra_engine_args=_vllm_args(),
        replay_mode="offline",
    )
    captured = run_trace_replay(
        trace_path,
        extra_engine_args=_vllm_args(),
        replay_mode="offline",
        capture_per_request=True,
    )

    assert summary_only.per_request is None
    assert summary_only.coverage["capture_per_request"] is False
    assert captured.per_request is not None
    assert len(captured.per_request) == 4
    assert captured.coverage["per_request_records"] == 4


def test_dynamo_mocker_wrapper_returns_public_replay_report(tmp_path):
    trace_path = _write_multiturn_trace(tmp_path)

    report = run_mocker_trace_replay(
        trace_path,
        extra_engine_args=_vllm_args(),
    )

    assert isinstance(report, ReplayReport)
    assert report.summary["completed_requests"] == 4


def test_online_replay_keeps_summary_dictionary_result(tmp_path):
    trace_path = _write_multiturn_trace(tmp_path)

    report = run_trace_replay(
        trace_path,
        extra_engine_args=_vllm_args(),
        replay_mode="online",
    )

    assert isinstance(report, dict)
    assert report["completed_requests"] == 4


def test_offline_replay_report_serializes_complete_result(tmp_path):
    trace_path = _write_multiturn_trace(tmp_path)
    report = run_trace_replay(
        trace_path,
        extra_engine_args=_vllm_args(),
        replay_mode="offline",
    )

    assert set(report.to_dict()) == {
        "summary",
        "per_request",
        "coverage",
        "planner",
    }
    assert report.to_dict()["per_request"] is None


def test_per_request_capture_records_queued_routes_and_dp_identity():
    dp_report = run_synthetic_trace_replay(
        8,
        2,
        4,
        extra_engine_args=MockEngineArgs(
            block_size=4,
            num_gpu_blocks=64,
            max_num_seqs=4,
            speedup_ratio=1000.0,
            dp_size=2,
        ),
        num_workers=1,
        replay_mode="offline",
        router_mode="kv_router",
        replay_concurrency=2,
        capture_per_request=True,
    )
    dp_routes = [record["routing_history"][0] for record in dp_report.per_request]
    assert {(route["logical_worker_id"], route["dp_rank"]) for route in dp_routes} == {
        (0, 0),
        (0, 1),
    }
    assert {(route["scheduler_id"], route["dp_rank"]) for route in dp_routes} == {
        (0, 0),
        (1, 1),
    }
    assert {route["outcome"] for route in dp_routes} == {"immediate"}

    queued_report = run_synthetic_trace_replay(
        8,
        8,
        4,
        extra_engine_args=MockEngineArgs(
            block_size=4,
            num_gpu_blocks=6,
            max_num_seqs=1,
            speedup_ratio=1000.0,
        ),
        num_workers=2,
        replay_mode="offline",
        router_mode="kv_router",
        router_config=KvRouterConfig(
            router_queue_threshold=0.0,
            router_event_threads=1,
            router_temperature=0.0,
        ),
        replay_concurrency=4,
        capture_per_request=True,
    )
    queued_routes = [
        record["routing_history"][0]
        for record in queued_report.per_request
        if record["routing_history"][0]["outcome"] == "queued"
    ]
    assert len(queued_routes) == 2
    assert {route["logical_worker_id"] for route in queued_routes} == {0, 1}
    for route in queued_routes:
        assert route["queue_entered_at_ms"] == 0.0
        assert route["released_at_ms"] > route["queue_entered_at_ms"]
        assert route["queue_wait_ms"] == pytest.approx(
            route["released_at_ms"] - route["queue_entered_at_ms"]
        )


def test_online_replay_rejects_in_memory_per_request_capture(tmp_path):
    trace_path = _write_multiturn_trace(tmp_path)
    with pytest.raises(ValueError, match="capture_per_request only supports"):
        run_trace_replay(
            trace_path,
            extra_engine_args=_vllm_args(),
            replay_mode="online",
            capture_per_request=True,
        )
    with pytest.raises(ValueError, match="capture_per_request only supports"):
        run_synthetic_trace_replay(
            8,
            2,
            1,
            extra_engine_args=_vllm_args(),
            replay_mode="online",
            replay_concurrency=1,
            capture_per_request=True,
        )


def test_online_trace_replay_emits_per_request_goodput_and_capacity(tmp_path):
    trace_path = _write_multiturn_trace(tmp_path)
    jsonl_path = tmp_path / "online_requests.jsonl"

    report = run_trace_replay(
        trace_path,
        extra_engine_args=_vllm_args(),
        num_workers=2,
        replay_mode="online",
        router_mode="kv_router",
        report_jsonl_path=jsonl_path,
        sla_e2e_ms=1_000_000.0,
    )

    records = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 4
    assert {record["session_id"] for record in records} == {
        "session-a",
        "session-b",
    }
    assert all(record["decode_worker_idx"] is not None for record in records)
    assert report["goodput_completed_requests"] == 4
    assert report["decode_worker_seconds"] > 0.0
    assert report["decode_gpus_per_worker"] == 1
    assert report["gpu_hours"] > 0.0


def test_online_trace_replay_supports_agentic_mooncake(tmp_path):
    trace_path = tmp_path / "agentic.jsonl"
    records = [
        {
            "request_id": "root",
            "session_id": "root",
            "timestamp": 0.0,
            "input_length": 64,
            "output_length": 2,
            "hash_ids": [1],
        },
        {
            "request_id": "dependent",
            "session_id": "dependent",
            "timestamp": 0.0,
            "delay": 5.0,
            "wait_for": ["root"],
            "input_length": 64,
            "output_length": 2,
            "hash_ids": [2],
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    report = run_trace_replay(
        trace_path,
        extra_engine_args=_vllm_args(),
        num_workers=2,
        replay_mode="online",
        router_mode="kv_router",
        trace_format="agentic_mooncake",
    )

    _assert_basic_report_counts(
        report,
        num_requests=2,
        input_tokens=64,
        output_tokens=2,
    )


def test_online_synthetic_replay_supports_goodput_sla():
    report = run_synthetic_trace_replay(
        64,
        2,
        2,
        extra_engine_args=_vllm_args(),
        num_workers=2,
        replay_mode="online",
        arrival_interval_ms=1.0,
        sla_e2e_ms=1_000_000.0,
    )

    assert report["goodput_completed_requests"] == 2


@pytest.mark.parametrize("replay_mode", ["offline", "online"])
def test_run_trace_replay_supports_applied_compute_agentic_format_with_concurrency(
    tmp_path, replay_mode
):
    trace_path = _write_applied_compute_agentic_trace(tmp_path)

    report = run_trace_replay(
        trace_path,
        extra_engine_args=_vllm_args(),
        num_workers=2,
        replay_concurrency=2,
        replay_mode=replay_mode,
        router_mode="kv_router",
        trace_format="applied_compute_agentic",
        trace_shared_prefix_ratio=0.5,
        trace_num_prefix_groups=1,
    )

    report = _report_summary(report)
    assert report["num_requests"] == 5
    assert report["completed_requests"] == 5
    assert report["total_input_tokens"] == 64 + 68 + 72 + 64 + 68
    assert report["total_output_tokens"] == 10


def test_run_trace_replay_rejects_applied_compute_agentic_format_without_concurrency(
    tmp_path,
):
    trace_path = _write_applied_compute_agentic_trace(tmp_path)

    with pytest.raises(Exception, match="replay_concurrency"):
        run_trace_replay(
            trace_path,
            extra_engine_args=_vllm_args(),
            num_workers=2,
            replay_mode="offline",
            trace_format="applied_compute_agentic",
        )


def test_direct_agentic_dynamo_trace_rejects_replay_concurrency():
    trace_path = (
        Path(__file__).resolve().parents[5]
        / "lib"
        / "bench"
        / "testdata"
        / "pi_request_trace.jsonl.gz"
    )

    with pytest.raises(Exception, match="not supported with replay_concurrency"):
        run_trace_replay(
            trace_path,
            extra_engine_args=_vllm_args(),
            replay_concurrency=2,
            trace_format="dynamo",
        )


def test_direct_agentic_dynamo_trace_honors_per_request_capture():
    trace_path = (
        Path(__file__).resolve().parents[5]
        / "lib"
        / "bench"
        / "testdata"
        / "pi_request_trace.jsonl.gz"
    )

    report = run_trace_replay(
        trace_path,
        extra_engine_args=_vllm_args(),
        replay_mode="offline",
        trace_format="dynamo",
        capture_per_request=True,
    )

    assert report.per_request
    assert report.coverage["capture_per_request"] is True
    assert report.coverage["per_request_records"] == len(report.per_request)
    assert report.summary["completed_requests"] == len(report.per_request)


def test_planner_replay_accepts_multi_shard_dynamo_trace(tmp_path):
    trace_paths = []
    for index in range(2):
        trace_path = tmp_path / f"trace-{index}.jsonl"
        trace_path.write_text(
            json.dumps(
                {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "request_end",
                    "event_time_unix_ms": 1_010 + index * 10,
                    "request": {
                        "request_id": f"request-{index}",
                        "request_received_ms": 1_000 + index * 10,
                        "output_tokens": 2,
                        "replay": {
                            "trace_block_size": 64,
                            "input_length": 64,
                            "input_sequence_hashes": [101 + index],
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        trace_paths.append(trace_path)

    planner_report = run_trace_replay(
        trace_paths,
        extra_engine_args=_vllm_args(),
        num_workers=1,
        trace_format="dynamo",
        planner_config={
            "mode": "agg",
            "optimization_target": "throughput",
            "report_interval_hours": None,
            "live_dashboard_port": 0,
        },
    )

    _assert_basic_report_counts(
        planner_report.summary,
        num_requests=2,
        input_tokens=64,
        output_tokens=2,
    )


@pytest.mark.parametrize("replay_mode", ["offline", "online"])
def test_run_trace_replay_supports_distinct_trace_and_engine_block_sizes(
    tmp_path, replay_mode
):
    trace_path = tmp_path / "trace_block_size_split.jsonl"
    trace_path.write_text(
        '{"timestamp":1000.0,"input_length":128,"output_length":2,"hash_ids":[101]}\n',
        encoding="utf-8",
    )

    report = run_trace_replay(
        trace_path,
        extra_engine_args=_vllm_args(),
        num_workers=1,
        replay_mode=replay_mode,
        trace_block_size=512,
    )

    _assert_basic_report_counts(
        report,
        num_requests=1,
        input_tokens=128,
        output_tokens=2,
    )


@pytest.mark.parametrize("engine_type", ["vllm", "sglang"])
@pytest.mark.parametrize("replay_mode", ["offline", "online"])
@pytest.mark.parametrize("router_mode", ["round_robin", "kv_router"])
@pytest.mark.parametrize("serving_mode", ["agg", "disagg"])
def test_run_synthetic_trace_replay_smoke_matrix(
    tmp_path, engine_type, replay_mode, router_mode, serving_mode
):
    if serving_mode == "disagg":
        if replay_mode != "offline":
            pytest.skip("disagg replay only supports offline mode")
        report = run_synthetic_trace_replay(
            64,
            2,
            2,
            prefill_engine_args=_prefill_args(),
            decode_engine_args=_decode_args(),
            router_config=_router_config(),
            num_prefill_workers=2,
            num_decode_workers=2,
            replay_mode=replay_mode,
            router_mode=router_mode,
            arrival_interval_ms=5.0,
        )
    else:
        args_path = _vllm_args() if engine_type == "vllm" else _sglang_args()
        num_workers = 1 if router_mode == "round_robin" else 2
        report = run_synthetic_trace_replay(
            64,
            2,
            2,
            extra_engine_args=args_path,
            num_workers=num_workers,
            replay_mode=replay_mode,
            router_mode=router_mode,
            arrival_interval_ms=5.0,
        )

    _assert_basic_report_counts(
        report,
        num_requests=2,
        input_tokens=64,
        output_tokens=2,
    )


@pytest.mark.parametrize("engine_type", ["vllm", "sglang"])
@pytest.mark.parametrize("replay_mode", ["offline", "online"])
def test_run_synthetic_trace_replay_invariant_counts_match(
    tmp_path, engine_type, replay_mode
):
    args_path = _vllm_args() if engine_type == "vllm" else _sglang_args()

    single = run_synthetic_trace_replay(
        64,
        2,
        2,
        extra_engine_args=args_path,
        num_workers=1,
        replay_mode=replay_mode,
        arrival_interval_ms=5.0,
    )
    multi_round_robin = run_synthetic_trace_replay(
        64,
        2,
        2,
        extra_engine_args=args_path,
        num_workers=4,
        replay_mode=replay_mode,
        router_mode="round_robin",
        arrival_interval_ms=5.0,
    )
    multi_kv_router = run_synthetic_trace_replay(
        64,
        2,
        2,
        extra_engine_args=args_path,
        num_workers=4,
        replay_mode=replay_mode,
        router_mode="kv_router",
        arrival_interval_ms=5.0,
    )

    for field in (
        "num_requests",
        "completed_requests",
        "total_input_tokens",
        "total_output_tokens",
    ):
        assert (
            _report_summary(single)[field] == _report_summary(multi_round_robin)[field]
        )
        assert _report_summary(single)[field] == _report_summary(multi_kv_router)[field]


@pytest.mark.parametrize("replay_mode", ["offline", "online"])
def test_run_synthetic_trace_replay_supports_multiturn_workloads(tmp_path, replay_mode):
    report = run_synthetic_trace_replay(
        64,
        2,
        3,
        extra_engine_args=_vllm_args(),
        num_workers=2,
        replay_mode=replay_mode,
        router_mode="kv_router",
        arrival_interval_ms=1.0,
        turns_per_session=2,
        inter_turn_delay_ms=5.0,
        shared_prefix_ratio=0.5,
        num_prefix_groups=2,
    )

    _assert_basic_report_counts(
        report,
        num_requests=6,
        input_tokens=64,
        output_tokens=2,
    )


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "expected_message"),
    [
        (0, 2, "input_tokens must be at least 1"),
        (2, 0, "output_tokens must be at least 1"),
    ],
)
def test_run_synthetic_trace_replay_workload_validates_zero_token_lengths(
    input_tokens, output_tokens, expected_message
):
    with pytest.raises(Exception, match=expected_message):
        run_synthetic_trace_replay(
            input_tokens,
            output_tokens,
            2,
            extra_engine_args=_vllm_args(),
            num_workers=2,
            replay_mode="offline",
            router_mode="kv_router",
            arrival_interval_ms=1.0,
            turns_per_session=2,
        )


@pytest.mark.parametrize("engine_type", ["vllm", "sglang"])
@pytest.mark.parametrize("replay_mode", ["offline", "online"])
def test_run_synthetic_concurrency_replay_counts_match(
    tmp_path, engine_type, replay_mode
):
    args_path = _vllm_args() if engine_type == "vllm" else _sglang_args()

    report = run_synthetic_trace_replay(
        64,
        2,
        3,
        extra_engine_args=args_path,
        num_workers=2,
        replay_mode=replay_mode,
        replay_concurrency=2,
    )

    _assert_basic_report_counts(
        report,
        num_requests=3,
        input_tokens=64,
        output_tokens=2,
    )


@pytest.mark.parametrize("replay_mode", ["offline", "online"])
def test_run_trace_replay_accepts_router_config(tmp_path, replay_mode):
    trace_path = _write_trace_and_args(tmp_path)
    args_path = _vllm_args()
    router_config_path = _router_config()

    report = run_trace_replay(
        trace_path,
        extra_engine_args=args_path,
        router_config=router_config_path,
        num_workers=2,
        replay_mode=replay_mode,
        router_mode="kv_router",
    )

    _assert_basic_report_counts(
        report,
        num_requests=2,
        input_tokens=64,
        output_tokens=2,
    )


@pytest.mark.parametrize("replay_mode", ["offline", "online"])
def test_run_trace_replay_accepts_partial_router_config_json(tmp_path, replay_mode):
    trace_path = _write_trace_and_args(tmp_path)
    args_path = _vllm_args()

    report = run_trace_replay(
        trace_path,
        extra_engine_args=args_path,
        router_config=_partial_router_config(),
        num_workers=2,
        replay_mode=replay_mode,
        router_mode="kv_router",
    )

    _assert_basic_report_counts(
        report,
        num_requests=2,
        input_tokens=64,
        output_tokens=2,
    )


@pytest.mark.parametrize("replay_mode", ["offline", "online"])
def test_run_trace_replay_accepts_partial_extra_engine_args_json(tmp_path, replay_mode):
    trace_path = _write_trace_and_args(tmp_path)

    report = run_trace_replay(
        trace_path,
        extra_engine_args=MockEngineArgs(block_size=64, speedup_ratio=1000.0),
        num_workers=1,
        replay_mode=replay_mode,
    )

    _assert_basic_report_counts(
        report,
        num_requests=2,
        input_tokens=64,
        output_tokens=2,
    )


@pytest.mark.parametrize("router_mode", ["round_robin", "kv_router"])
def test_run_trace_replay_supports_disagg_offline(tmp_path, router_mode):
    trace_path = _write_trace_and_args(tmp_path)

    report = run_trace_replay(
        trace_path,
        prefill_engine_args=_prefill_args(),
        decode_engine_args=_decode_args(),
        router_config=_router_config(),
        num_prefill_workers=2,
        num_decode_workers=2,
        replay_mode="offline",
        router_mode=router_mode,
    )

    _assert_basic_report_counts(
        report,
        num_requests=2,
        input_tokens=64,
        output_tokens=2,
    )
    _assert_basic_report_metrics(report)


@pytest.mark.parametrize("router_mode", ["round_robin", "kv_router"])
def test_run_synthetic_trace_replay_disagg_preserves_expected_output_tokens(
    router_mode,
):
    report = run_synthetic_trace_replay(
        128,
        7,
        6,
        prefill_engine_args=_prefill_args(),
        decode_engine_args=_decode_args(),
        router_config=_router_config(),
        num_prefill_workers=2,
        num_decode_workers=2,
        replay_mode="offline",
        router_mode=router_mode,
        arrival_interval_ms=1.0,
    )

    _assert_basic_report_counts(
        report,
        num_requests=6,
        input_tokens=128,
        output_tokens=7,
    )
    _assert_basic_report_metrics(report)


def test_run_trace_replay_rejects_partial_disagg_args(tmp_path):
    trace_path = _write_trace_and_args(tmp_path)

    with pytest.raises(Exception, match="must be provided together"):
        run_trace_replay(
            trace_path,
            prefill_engine_args=_prefill_args(),
            replay_mode="offline",
            router_mode="kv_router",
        )


def test_run_trace_replay_rejects_online_disagg(tmp_path):
    trace_path = _write_trace_and_args(tmp_path)

    with pytest.raises(
        Exception, match="disagg replay only supports replay_mode='offline'"
    ):
        run_trace_replay(
            trace_path,
            prefill_engine_args=_prefill_args(),
            decode_engine_args=_decode_args(),
            router_config=_router_config(),
            num_prefill_workers=2,
            num_decode_workers=2,
            replay_mode="online",
            router_mode="kv_router",
        )


def test_run_trace_replay_rejects_disagg_worker_counts_for_aggregated_mode(tmp_path):
    trace_path = _write_trace_and_args(tmp_path)

    with pytest.raises(
        Exception,
        match="num_prefill_workers and num_decode_workers are only used for disagg replay",
    ):
        run_trace_replay(
            trace_path,
            extra_engine_args=MockEngineArgs(block_size=64, speedup_ratio=1000.0),
            num_workers=1,
            num_prefill_workers=2,
            num_decode_workers=2,
            replay_mode="offline",
        )


def test_format_report_table_matches_aiperf_shape():
    report = {
        "mean_ttft_ms": 18.26,
        "min_ttft_ms": 11.22,
        "max_ttft_ms": 106.32,
        "p99_ttft_ms": 68.82,
        "p90_ttft_ms": 27.76,
        "p75_ttft_ms": 16.62,
        "std_ttft_ms": 12.07,
        "mean_ttst_ms": 11.40,
        "min_ttst_ms": 0.02,
        "max_ttst_ms": 85.91,
        "p99_ttst_ms": 34.54,
        "p90_ttst_ms": 12.59,
        "p75_ttst_ms": 11.65,
        "std_ttst_ms": 7.01,
        "mean_e2e_latency_ms": 487.30,
        "min_e2e_latency_ms": 267.07,
        "max_e2e_latency_ms": 769.57,
        "p99_e2e_latency_ms": 715.99,
        "p90_e2e_latency_ms": 580.83,
        "p75_e2e_latency_ms": 536.17,
        "std_e2e_latency_ms": 79.60,
        "mean_itl_ms": 11.23,
        "min_itl_ms": 8.80,
        "max_itl_ms": 13.17,
        "p99_itl_ms": 12.48,
        "p90_itl_ms": 11.73,
        "p75_itl_ms": 11.37,
        "std_itl_ms": 0.45,
        "mean_output_token_throughput_per_user": 89.23,
        "min_output_token_throughput_per_user": 75.93,
        "max_output_token_throughput_per_user": 113.60,
        "p99_output_token_throughput_per_user": 102.28,
        "p90_output_token_throughput_per_user": 90.91,
        "p75_output_token_throughput_per_user": 90.29,
        "std_output_token_throughput_per_user": 3.70,
        "output_throughput_tok_s": 10944.03,
        "request_throughput_rps": 255.54,
        "completed_requests": 711,
        "wall_time_ms": 4046.31,
        "prefix_cache_reused_ratio": 0.3587,
        "first_admission_prefix_cache_reused_ratio": 0.1234,
    }

    rendered = format_report_table(report)

    assert "NVIDIA AIPerf | LLM Metrics" in rendered
    assert "Time to First Token (ms)" in rendered
    assert "Output Token Throughput (tokens/sec)" in rendered
    assert "Request Throughput (requests/sec)" in rendered
    assert "Prefix Cache Reused Ratio: 0.36" in rendered
    assert "First Admission Prefix Cache Reused Ratio: 0.12" in rendered
    assert "10,944.03" in rendered
    assert "255.54" in rendered
    assert "N/A" in rendered


def test_write_report_json_creates_file(tmp_path):
    report_path = write_report_json({"completed_requests": 2}, tmp_path / "report.json")
    assert (
        report_path.read_text(encoding="utf-8") == '{\n  "completed_requests": 2\n}\n'
    )
