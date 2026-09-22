# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading

import pytest

from dynamo._core import run_mocker_synthetic_trace_replay, run_mocker_trace_replay
from dynamo.mocker import MockEngineArgs

from .replay_utils import _write_multiturn_trace

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(120),
]


def _replay_kwargs(topology="aggregated"):
    kwargs = {
        "input_tokens": 32,
        "output_tokens": 8,
        "request_count": 6,
        "replay_concurrency": 6,
        "router_mode": "kv_router",
    }
    if topology == "aggregated":
        kwargs.update(
            extra_engine_args=MockEngineArgs(
                block_size=4,
                num_gpu_blocks=32,
                max_num_seqs=1,
                speedup_ratio=100.0,
            ),
            num_workers=2,
        )
    elif topology == "disaggregated":
        kwargs.update(
            prefill_engine_args=MockEngineArgs(
                block_size=4,
                num_gpu_blocks=32,
                max_num_seqs=1,
                speedup_ratio=100.0,
                worker_type="prefill",
            ),
            decode_engine_args=MockEngineArgs(
                block_size=4,
                num_gpu_blocks=32,
                max_num_seqs=1,
                speedup_ratio=100.0,
                worker_type="decode",
            ),
            num_prefill_workers=2,
            num_decode_workers=2,
        )
    else:
        raise AssertionError(f"unsupported topology: {topology}")
    return kwargs


@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
def test_telemetry_capture_callback_and_jsonl_share_one_sample_stream(
    tmp_path, topology
):
    callback_samples = []
    jsonl_path = tmp_path / "telemetry.jsonl"

    result = run_mocker_synthetic_trace_replay(
        **_replay_kwargs(topology),
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
    assert [sample["sample_ordinal"] for sample in samples] == list(
        range(len(samples))
    )
    assert sum(sample["traffic"]["arriving_requests"] for sample in samples) == 6
    assert sum(sample["traffic"]["completed_requests"] for sample in samples) == 6

    assert set(samples[0]) == {
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


class _DisabledScalingPolicy:
    def initial_tick_ms(self):
        return float("inf")

    def on_tick(self, _metrics):
        raise AssertionError("disabled scaling policy must not receive ticks")


def test_telemetry_and_scaling_policy_can_coexist():
    result = run_mocker_synthetic_trace_replay(
        **_replay_kwargs(),
        scaling_policy=_DisabledScalingPolicy(),
        capture_telemetry=True,
        telemetry_sample_interval_ms=1.0,
    )

    assert result.summary["completed_requests"] == 6
    assert result.telemetry["samples"]
    assert result.lifecycle_operations == []


def test_callback_and_jsonl_only_do_not_retain_telemetry(tmp_path):
    callback_samples = []
    callback_result = run_mocker_synthetic_trace_replay(
        **_replay_kwargs(),
        telemetry_callback=callback_samples.append,
        telemetry_sample_interval_ms=1.0,
    )
    assert callback_samples
    assert callback_result.telemetry is None

    jsonl_path = tmp_path / "telemetry.jsonl"
    jsonl_result = run_mocker_synthetic_trace_replay(
        **_replay_kwargs(),
        telemetry_jsonl_path=jsonl_path,
        telemetry_sample_interval_ms=1.0,
    )
    lines = jsonl_path.read_text().splitlines()
    assert lines
    samples = [json.loads(line) for line in lines]
    assert [sample["sample_ordinal"] for sample in samples] == list(
        range(len(samples))
    )
    assert jsonl_result.telemetry is None


def test_jsonl_only_replay_releases_gil_for_background_python_thread(tmp_path):
    started = threading.Event()
    stop = threading.Event()
    progress = 0

    def run_background():
        nonlocal progress
        started.set()
        while not stop.is_set():
            progress += 1

    thread = threading.Thread(target=run_background)
    thread.start()
    assert started.wait(timeout=1.0)
    progress_before_replay = progress
    kwargs = _replay_kwargs()
    kwargs.update(request_count=20_000, replay_concurrency=16)
    jsonl_path = tmp_path / "telemetry.jsonl"

    try:
        result = run_mocker_synthetic_trace_replay(
            **kwargs,
            telemetry_jsonl_path=jsonl_path,
            telemetry_sample_interval_ms=1_000_000_000.0,
        )
        progress_during_replay = progress - progress_before_replay
    finally:
        stop.set()
        thread.join(timeout=1.0)

    assert progress_during_replay > 0
    assert not thread.is_alive()
    assert jsonl_path.read_text().splitlines()
    assert result.telemetry is None


def test_jsonl_open_failure_fails_replay(tmp_path):
    with pytest.raises(Exception):
        run_mocker_synthetic_trace_replay(
            **_replay_kwargs(),
            telemetry_jsonl_path=tmp_path,
        )


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


def test_native_telemetry_rejects_online_mode_and_non_callable_callback():
    with pytest.raises(ValueError, match="offline"):
        run_mocker_synthetic_trace_replay(
            **_replay_kwargs(),
            replay_mode="online",
            capture_telemetry=True,
        )

    with pytest.raises(TypeError, match="callable"):
        run_mocker_synthetic_trace_replay(
            **_replay_kwargs(),
            telemetry_callback=object(),
        )


def test_native_rejects_colliding_output_paths_before_truncation(tmp_path):
    trace_path = _write_multiturn_trace(tmp_path)
    output = tmp_path / "samples.jsonl"
    output.write_text("sentinel")
    alias = tmp_path / "missing" / ".." / "samples.jsonl"

    with pytest.raises(ValueError, match="must refer to different files"):
        run_mocker_trace_replay(
            [trace_path],
            report_jsonl_path=output,
            telemetry_jsonl_path=alias,
        )

    assert output.read_text() == "sentinel"
    assert not (tmp_path / "missing").exists()


def test_native_rejects_prospective_case_insensitive_output_aliases(tmp_path):
    probe = tmp_path / "CaseSensitivityProbe"
    probe.write_text("probe")
    if not (tmp_path / "casesensitivityprobe").exists():
        pytest.skip("filesystem is case-sensitive")

    trace_path = _write_multiturn_trace(tmp_path)
    report = tmp_path / "Samples.jsonl"
    telemetry = tmp_path / "samples.jsonl"

    with pytest.raises(ValueError, match="must refer to different files"):
        run_mocker_trace_replay(
            [trace_path],
            report_jsonl_path=report,
            telemetry_jsonl_path=telemetry,
        )

    assert not report.exists()
    assert not telemetry.exists()


def test_native_rechecks_output_alias_before_report_write(tmp_path):
    trace_path = _write_multiturn_trace(tmp_path)
    report = tmp_path / "requests.jsonl"
    telemetry = tmp_path / "telemetry.jsonl"

    def swap_report_target(_sample):
        if not report.is_symlink():
            report.symlink_to(telemetry.name)

    with pytest.raises(ValueError, match="must refer to different files"):
        run_mocker_trace_replay(
            [trace_path],
            report_jsonl_path=report,
            telemetry_callback=swap_report_target,
            telemetry_jsonl_path=telemetry,
        )

    samples = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert samples
    assert report.is_symlink()


def test_native_rejects_hard_linked_output_paths_before_truncation(tmp_path):
    trace_path = _write_multiturn_trace(tmp_path)
    output = tmp_path / "requests.jsonl"
    output.write_text("sentinel")
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.hardlink_to(output)

    with pytest.raises(ValueError, match="must refer to different files"):
        run_mocker_trace_replay(
            [trace_path],
            report_jsonl_path=output,
            telemetry_jsonl_path=telemetry,
        )

    assert output.read_text() == "sentinel"
    assert telemetry.read_text() == "sentinel"


def test_native_rejects_telemetry_alias_of_input_before_truncation(tmp_path):
    trace_path = _write_multiturn_trace(tmp_path)
    original = trace_path.read_text()

    with pytest.raises(ValueError, match="must not refer to replay input trace"):
        run_mocker_trace_replay([trace_path], telemetry_jsonl_path=trace_path)

    assert trace_path.read_text() == original


def test_late_option_validation_does_not_truncate_telemetry_output(tmp_path):
    output = tmp_path / "telemetry.jsonl"
    output.write_text("sentinel")

    with pytest.raises(Exception, match="replay_concurrency"):
        run_mocker_synthetic_trace_replay(
            **{**_replay_kwargs(), "replay_concurrency": 0},
            telemetry_jsonl_path=output,
        )

    assert output.read_text() == "sentinel"


def test_trace_parse_failure_does_not_truncate_telemetry_output(tmp_path):
    trace_path = tmp_path / "invalid.jsonl"
    trace_path.write_text("not-json\n")
    output = tmp_path / "telemetry.jsonl"
    output.write_text("sentinel")

    with pytest.raises(Exception):
        run_mocker_trace_replay(
            [trace_path],
            extra_engine_args=MockEngineArgs(
                block_size=4,
                num_gpu_blocks=32,
                max_num_seqs=1,
                speedup_ratio=100.0,
            ),
            telemetry_jsonl_path=output,
        )

    assert output.read_text() == "sentinel"
