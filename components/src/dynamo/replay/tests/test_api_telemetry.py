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
        telemetry_options=api.TelemetryOptions(
            sample_interval_ms=2_500.0,
            capture_in_memory=True,
            callback=callback,
            jsonl_path="samples.jsonl",
        ),
    )

    assert seen["capture_telemetry"] is True
    assert seen["telemetry_sample_interval_ms"] == 2_500.0
    assert seen["telemetry_callback"] is callback
    assert seen["telemetry_jsonl_path"] == "samples.jsonl"
    assert report.telemetry is not None
    assert report.telemetry.sample_interval_ms == 2_500.0
    assert report.telemetry.samples == [sample]


def test_default_replay_omits_native_telemetry_options(monkeypatch) -> None:
    seen = {}

    def run_native(*args, **kwargs):
        seen.update(kwargs)
        return _native_result()

    monkeypatch.setattr(api, "_run_mocker_synthetic_trace_replay", run_native)

    report = api.run_synthetic_trace_replay(16, 4, 1)

    assert "capture_telemetry" not in seen
    assert "telemetry_sample_interval_ms" not in seen
    assert "telemetry_callback" not in seen
    assert "telemetry_jsonl_path" not in seen
    assert report.telemetry is None
    assert "telemetry" not in report.to_dict()


def test_telemetry_options_use_in_memory_as_the_default_sink(monkeypatch) -> None:
    seen = {}

    def run_native(*args, **kwargs):
        seen.update(kwargs)
        return _native_result(telemetry={"sample_interval_ms": 1_000.0, "samples": []})

    monkeypatch.setattr(api, "_run_mocker_synthetic_trace_replay", run_native)

    api.run_synthetic_trace_replay(
        16,
        4,
        1,
        telemetry_options=api.TelemetryOptions(),
    )

    assert seen["capture_telemetry"] is True
    assert seen["telemetry_sample_interval_ms"] == 1_000.0


@pytest.mark.parametrize(
    ("options", "expected_callback", "expected_path"),
    [
        (api.TelemetryOptions(callback=lambda _sample: None), True, None),
        (api.TelemetryOptions(jsonl_path="samples.jsonl"), False, "samples.jsonl"),
    ],
)
def test_external_sink_does_not_retain_samples_by_default(
    monkeypatch,
    options,
    expected_callback,
    expected_path,
) -> None:
    seen = {}

    def run_native(*args, **kwargs):
        seen.update(kwargs)
        return _native_result()

    monkeypatch.setattr(api, "_run_mocker_synthetic_trace_replay", run_native)

    api.run_synthetic_trace_replay(16, 4, 1, telemetry_options=options)

    assert seen["capture_telemetry"] is False
    assert (seen["telemetry_callback"] is not None) is expected_callback
    assert seen["telemetry_jsonl_path"] == expected_path


def test_telemetry_options_reject_no_sink() -> None:
    with pytest.raises(ValueError, match="at least one sink"):
        api.run_synthetic_trace_replay(
            16,
            4,
            1,
            telemetry_options=api.TelemetryOptions(capture_in_memory=False),
        )
