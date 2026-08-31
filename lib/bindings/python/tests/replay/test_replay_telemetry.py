# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo._core import run_mocker_synthetic_trace_replay
from dynamo.mocker import MockEngineArgs

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(120),
]


def _replay_kwargs():
    return {
        "input_tokens": 32,
        "output_tokens": 8,
        "request_count": 6,
        "extra_engine_args": MockEngineArgs(
            block_size=4,
            num_gpu_blocks=32,
            max_num_seqs=1,
            speedup_ratio=100.0,
        ),
        "num_workers": 2,
        "replay_concurrency": 6,
        "router_mode": "kv_router",
    }


def test_telemetry_capture_callback_and_jsonl_share_one_sample_stream(tmp_path):
    callback_samples = []
    jsonl_path = tmp_path / "telemetry.jsonl"

    result = run_mocker_synthetic_trace_replay(
        **_replay_kwargs(),
        capture_telemetry=True,
        telemetry_sample_interval_ms=0.1,
        telemetry_callback=callback_samples.append,
        telemetry_jsonl_path=jsonl_path,
    )

    telemetry = result.telemetry
    assert telemetry["sample_interval_ms"] == 0.1
    samples = telemetry["samples"]
    assert samples
    assert callback_samples == samples
    assert [json.loads(line) for line in jsonl_path.read_text().splitlines()] == samples

    assert samples[0]["kind"] == "baseline"
    assert [sample["sample_ordinal"] for sample in samples] == list(range(len(samples)))
    assert {sample["kind"] for sample in samples} <= {
        "baseline",
        "periodic",
        "final",
    }
    assert sum(sample["traffic"]["arriving_requests"] for sample in samples) == 6
    assert sum(sample["traffic"]["completed_requests"] for sample in samples) == 6

    expected_sample_keys = {
        "sample_ordinal",
        "kind",
        "interval_start_ms",
        "sampled_at_ms",
        "traffic",
        "prefill_scheduler_metrics",
        "decode_scheduler_metrics",
        "prefill_interval_metrics",
        "decode_interval_metrics",
        "router_pending_prefill_requests",
        "router_pending_decode_requests",
        "active_prefill_ids",
        "active_decode_ids",
        "starting_prefill_ids",
        "starting_decode_ids",
        "draining_prefill_ids",
        "draining_decode_ids",
    }
    assert set(samples[0]) == expected_sample_keys
    assert set(samples[0]["traffic"]) == {
        "duration_s",
        "arriving_requests",
        "completed_requests",
        "avg_isl",
        "avg_osl",
        "avg_ttft_ms",
        "avg_itl_ms",
        "ttft_count",
        "itl_count",
        "avg_router_kv_hit_rate",
        "router_kv_hit_rate_count",
        "avg_accept_length",
        "accept_length_forward_count",
    }
    assert set(samples[0]["decode_interval_metrics"]) == {
        "cache_hit_tokens",
        "cache_total_tokens",
        "preemptions",
    }
    assert set(samples[0]["decode_scheduler_metrics"][0]) == {
        "worker_id",
        "dp_rank",
        "active_blocks",
        "inactive_blocks",
        "total_blocks",
        "active_cache_usage",
        "physical_cache_usage",
        "running_requests",
        "waiting_requests",
    }


def test_callback_only_does_not_retain_telemetry():
    samples = []
    result = run_mocker_synthetic_trace_replay(
        **_replay_kwargs(),
        telemetry_callback=samples.append,
        telemetry_sample_interval_ms=1.0,
    )

    assert samples
    assert result.telemetry is None


def test_jsonl_only_does_not_retain_telemetry(tmp_path):
    jsonl_path = tmp_path / "telemetry.jsonl"

    result = run_mocker_synthetic_trace_replay(
        **_replay_kwargs(),
        telemetry_jsonl_path=jsonl_path,
        telemetry_sample_interval_ms=1.0,
    )

    assert jsonl_path.read_text().splitlines()
    assert result.telemetry is None


def test_disabled_telemetry_preserves_default_native_result():
    result = run_mocker_synthetic_trace_replay(**_replay_kwargs())

    assert result.telemetry is None


def test_telemetry_callback_preserves_python_exception_type():
    def raise_from_telemetry(sample):
        raise ValueError(f"boom from telemetry sample {sample['sample_ordinal']}")

    with pytest.raises(ValueError, match="boom from telemetry sample 0"):
        run_mocker_synthetic_trace_replay(
            **_replay_kwargs(),
            telemetry_callback=raise_from_telemetry,
        )


@pytest.mark.parametrize("interval", [0.0, -1.0, float("inf"), float("nan")])
def test_enabled_telemetry_requires_positive_finite_interval(interval):
    with pytest.raises(ValueError, match="positive finite"):
        run_mocker_synthetic_trace_replay(
            **_replay_kwargs(),
            capture_telemetry=True,
            telemetry_sample_interval_ms=interval,
        )


def test_native_telemetry_rejects_online_mode():
    with pytest.raises(ValueError, match="offline"):
        run_mocker_synthetic_trace_replay(
            **_replay_kwargs(),
            replay_mode="online",
            capture_telemetry=True,
        )


def test_native_telemetry_rejects_non_callable_callback():
    with pytest.raises(TypeError, match="callable"):
        run_mocker_synthetic_trace_replay(
            **_replay_kwargs(),
            telemetry_callback=object(),
        )
