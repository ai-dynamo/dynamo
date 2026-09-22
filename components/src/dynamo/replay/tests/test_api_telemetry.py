# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.replay import api

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


def _native_result(*, telemetry=None):
    return SimpleNamespace(
        summary={"request_count": 1},
        per_request=None,
        coverage={
            "capture_per_request": False,
            "capture_planner_details": False,
            "per_request_records": 0,
        },
        lifecycle_operations=[],
        telemetry=telemetry,
    )


def test_synthetic_replay_forwards_and_materializes_telemetry(monkeypatch) -> None:
    seen = {}
    sample = {
        "sample_ordinal": 0,
        "kind": "baseline",
        "interval_start_ms": 0.0,
        "sampled_at_ms": 0.0,
    }

    def run_native(*args, **kwargs):
        seen.update(kwargs)
        return _native_result(
            telemetry={"sample_interval_ms": 2_500.0, "samples": [sample]}
        )

    def callback(_snapshot):
        return None

    monkeypatch.setattr(api, "_run_mocker_synthetic_trace_replay", run_native)

    report = api.run_synthetic_trace_replay(
        16,
        4,
        1,
        capture_telemetry=True,
        telemetry_sample_interval_ms=2_500.0,
        telemetry_callback=callback,
        telemetry_jsonl_path="samples.jsonl",
    )

    assert seen["capture_telemetry"] is True
    assert seen["telemetry_sample_interval_ms"] == 2_500.0
    assert seen["telemetry_callback"] is callback
    assert seen["telemetry_jsonl_path"] == "samples.jsonl"
    assert report.telemetry is not None
    assert report.telemetry.sample_interval_ms == 2_500.0
    assert report.telemetry.samples == [sample]


def test_default_replay_report_shape_omits_disabled_telemetry(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "_run_mocker_synthetic_trace_replay",
        lambda *args, **kwargs: _native_result(),
    )

    report = api.run_synthetic_trace_replay(16, 4, 1)

    assert report.telemetry is None
    assert "telemetry" not in report.to_dict()


@pytest.mark.parametrize("interval", [0.0, -1.0, float("inf"), float("nan"), True])
def test_enabled_telemetry_rejects_invalid_interval(interval) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        api.run_synthetic_trace_replay(
            16,
            4,
            1,
            capture_telemetry=True,
            telemetry_sample_interval_ms=interval,
        )


def test_telemetry_rejects_online_mode_and_non_callable_callback() -> None:
    with pytest.raises(ValueError, match="offline"):
        api.run_synthetic_trace_replay(
            16,
            4,
            1,
            replay_mode="online",
            capture_telemetry=True,
        )

    with pytest.raises(TypeError, match="callable"):
        api.run_synthetic_trace_replay(
            16,
            4,
            1,
            telemetry_callback=object(),
        )


def test_trace_replay_rejects_colliding_output_paths_before_native_call(
    monkeypatch, tmp_path
) -> None:
    called = False

    def run_native(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("native replay must not run for colliding outputs")

    monkeypatch.setattr(api, "_run_mocker_trace_replay", run_native)
    output = tmp_path / "samples.jsonl"
    output.write_text("sentinel")
    alias = tmp_path / "missing" / ".." / "samples.jsonl"

    with pytest.raises(ValueError, match="must refer to different files"):
        api.run_trace_replay(
            tmp_path / "trace.jsonl",
            report_jsonl_path=output,
            telemetry_jsonl_path=alias,
        )

    assert called is False
    assert output.read_text() == "sentinel"
    assert not (tmp_path / "missing").exists()


def test_trace_replay_rejects_prospective_case_insensitive_output_aliases(
    monkeypatch, tmp_path
) -> None:
    probe = tmp_path / "CaseSensitivityProbe"
    probe.write_text("probe")
    if not (tmp_path / "casesensitivityprobe").exists():
        pytest.skip("filesystem is case-sensitive")

    called = False

    def run_native(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("native replay must not run for colliding outputs")

    monkeypatch.setattr(api, "_run_mocker_trace_replay", run_native)
    report = tmp_path / "Samples.jsonl"
    telemetry = tmp_path / "samples.jsonl"

    with pytest.raises(ValueError, match="must refer to different files"):
        api.run_trace_replay(
            tmp_path / "trace.jsonl",
            report_jsonl_path=report,
            telemetry_jsonl_path=telemetry,
        )

    assert called is False
    assert not report.exists()
    assert not telemetry.exists()


def test_trace_replay_rejects_telemetry_alias_of_input_before_native_call(
    monkeypatch, tmp_path
) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("sentinel")
    called = False

    def run_native(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("native replay must not run for colliding input")

    monkeypatch.setattr(api, "_run_mocker_trace_replay", run_native)

    with pytest.raises(ValueError, match="must not refer to replay input trace"):
        api.run_trace_replay(trace, telemetry_jsonl_path=trace)

    assert called is False
    assert trace.read_text() == "sentinel"


def test_trace_replay_rejects_hard_linked_output_paths_before_native_call(
    monkeypatch, tmp_path
) -> None:
    called = False

    def run_native(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("native replay must not run for colliding outputs")

    monkeypatch.setattr(api, "_run_mocker_trace_replay", run_native)
    output = tmp_path / "requests.jsonl"
    output.write_text("sentinel")
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.hardlink_to(output)

    with pytest.raises(ValueError, match="must refer to different files"):
        api.run_trace_replay(
            tmp_path / "trace.jsonl",
            report_jsonl_path=output,
            telemetry_jsonl_path=telemetry,
        )

    assert called is False
    assert output.read_text() == "sentinel"
    assert telemetry.read_text() == "sentinel"
