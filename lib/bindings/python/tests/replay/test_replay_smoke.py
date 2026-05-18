# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import MockEngineArgs
from dynamo.replay import run_synthetic_trace_replay, run_trace_replay
from dynamo.replay.reporting import format_report_table, write_report_json

from .replay_utils import (
    _assert_basic_report_counts,
    _assert_basic_report_metrics,
    _decode_args,
    _partial_router_config,
    _prefill_args,
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


@pytest.fixture(autouse=True)
def _skip_online(request):
    callspec = getattr(request.node, "callspec", None)
    if callspec and callspec.params.get("replay_mode") == "online":
        pytest.skip("intermittent hang in online replay mode (#9548)")


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
        assert single[field] == multi_round_robin[field]
        assert single[field] == multi_kv_router[field]


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
        assert single[field] == multi_round_robin[field]
        assert single[field] == multi_kv_router[field]


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
    }

    rendered = format_report_table(report)

    assert "NVIDIA AIPerf | LLM Metrics" in rendered
    assert "Time to First Token (ms)" in rendered
    assert "Output Token Throughput (tokens/sec)" in rendered
    assert "Request Throughput (requests/sec)" in rendered
    assert "Prefix Cache Reused Ratio: 0.36" in rendered
    assert "10,944.03" in rendered
    assert "255.54" in rendered
    assert "N/A" in rendered


def test_write_report_json_creates_file(tmp_path):
    report_path = write_report_json({"completed_requests": 2}, tmp_path / "report.json")
    assert (
        report_path.read_text(encoding="utf-8") == '{\n  "completed_requests": 2\n}\n'
    )
