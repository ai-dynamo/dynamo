# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the scorecard's AIPerf-exact goodput + stability metrics.

Pure-Python — no Dynamo runtime. Validates the per-request good/bad rule
against AIPerf semantics (joint AND, inclusive <=, ITL undefined for osl<2).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from autoscaling_arena.measurement import AllocationConfig, MeasurementWindow
from autoscaling_arena.scorecard import (
    SCORECARD_SCHEMA_VERSION,
    SLOProfile,
    average_gpu_count,
    goodput_for_profile,
    oscillation_count,
    request_is_good,
    request_itl_ms,
    scorecard,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _rec(ttft, e2e, osl):
    return {"ttft_ms": ttft, "e2e_latency_ms": e2e, "output_length": osl}


# --- ITL formula (AIPerf: (e2e - ttft)/(osl-1)) ---------------------------


def test_itl_formula():
    assert request_itl_ms(_rec(100.0, 300.0, 5)) == pytest.approx(50.0)  # (300-100)/4


def test_itl_undefined_when_osl_below_2():
    assert request_itl_ms(_rec(100.0, 120.0, 1)) is None


def test_itl_undefined_when_ttft_missing():
    assert request_itl_ms(_rec(None, 300.0, 5)) is None


def test_itl_prefers_mocker_field_over_recompute():
    # When the record carries itl_ms (the mocker's AIPerf-aligned value), use it
    # verbatim — even if it differs from the (e2e-ttft)/(osl-1) recomputation
    # (they diverge under output clamping).
    rec = {
        "ttft_ms": 100.0,
        "e2e_latency_ms": 300.0,
        "output_length": 10,
        "itl_ms": 100.0,
    }
    assert request_itl_ms(rec) == 100.0  # field, not (300-100)/9 = 22.2


def test_itl_field_none_means_undefined():
    rec = {
        "ttft_ms": 100.0,
        "e2e_latency_ms": 120.0,
        "output_length": 1,
        "itl_ms": None,
    }
    assert request_itl_ms(rec) is None


# --- per-request good/bad (joint AND, inclusive) --------------------------

INTERACTIVE = SLOProfile(name="interactive", ttft_ms=300.0, itl_ms=50.0)
AGENTIC = SLOProfile(name="agentic", e2e_ms=3000.0, itl_ms=200.0)


def test_good_when_all_constraints_met():
    # ttft 250<=300; itl=(410-250)/4=40<=50
    assert request_is_good(_rec(250.0, 410.0, 5), INTERACTIVE) is True


def test_bad_when_ttft_exceeds():
    assert request_is_good(_rec(350.0, 510.0, 5), INTERACTIVE) is False


def test_bad_when_itl_exceeds():
    # itl=(490-250)/4=60>50
    assert request_is_good(_rec(250.0, 490.0, 5), INTERACTIVE) is False


def test_bad_when_itl_undefined_and_constrained():
    # osl=1 -> itl None -> cannot satisfy itl constraint
    assert request_is_good(_rec(250.0, 250.0, 1), INTERACTIVE) is False


def test_inclusive_boundary_passes():
    # ttft exactly 300, itl exactly 50: (300 + 50*4)=500 e2e, osl=5 -> itl=50
    assert request_is_good(_rec(300.0, 500.0, 5), INTERACTIVE) is True


def test_e2e_profile_boundary():
    # osl=30 keeps itl=(e2e-ttft)/29 ~100 <= 200, so e2e is the deciding constraint.
    assert request_is_good(_rec(100.0, 3000.0, 30), AGENTIC) is True  # e2e==3000 ok
    assert request_is_good(_rec(100.0, 3001.0, 30), AGENTIC) is False  # e2e>3000


def test_no_constraints_profile_never_good():
    empty = SLOProfile(name="empty")
    assert request_is_good(_rec(1.0, 2.0, 5), empty) is False


@pytest.mark.parametrize("terminal_status", ["rejected", "failed", "canceled"])
def test_non_completed_terminal_status_is_never_good(terminal_status):
    rec = _rec(100.0, 180.0, 5)
    rec["terminal_status"] = terminal_status

    assert request_is_good(rec, INTERACTIVE) is False


def test_completed_terminal_status_can_be_good():
    rec = _rec(100.0, 180.0, 5)
    rec["terminal_status"] = "completed"

    assert request_is_good(rec, INTERACTIVE) is True


# --- goodput aggregation --------------------------------------------------


def test_goodput_rps_and_rate():
    recs = [
        _rec(250.0, 410.0, 5),  # good
        _rec(350.0, 510.0, 5),  # bad ttft
        _rec(250.0, 410.0, 5),  # good
        _rec(250.0, 250.0, 1),  # bad (itl undefined)
    ]
    g = goodput_for_profile(
        recs, duration_s=2.0, attempted_requests=4, profile=INTERACTIVE
    )
    assert g["good_count"] == 2.0
    assert g["goodput_rps"] == pytest.approx(1.0)  # 2 good / 2 s
    assert g["good_rate"] == pytest.approx(0.5)  # 2 / 4


def test_average_gpu_count_uses_benchmark_duration():
    # 16 GPU-hours over a two-hour benchmark means an eight-GPU average fleet.
    assert average_gpu_count(16.0, duration_s=7200.0) == pytest.approx(8.0)
    assert average_gpu_count(16.0, duration_s=0.0) is None
    assert average_gpu_count(0.0, duration_s=7200.0) is None


def test_scorecard_efficiency_is_goodput_per_average_gpu():
    report = SimpleNamespace(
        trace_report={
            "duration_ms": 7200_000.0,
            "completed_requests": 2,
            "num_requests": 2,
        },
        gpu_hours=16.0,
        per_request=[
            _rec(250.0, 410.0, 5),
            _rec(250.0, 410.0, 5),
        ],
        scaling_events=[],
    )
    result = scorecard(report, profiles=(INTERACTIVE,))

    assert result["scorecard_schema_version"] == SCORECARD_SCHEMA_VERSION
    assert result["benchmark_hours"] == pytest.approx(2.0)
    assert result["average_gpus"] == pytest.approx(8.0)
    assert result["profiles"]["interactive"]["goodput_rps"] == pytest.approx(2 / 7200)
    assert result["profiles"]["interactive"]["goodput_per_gpu"] == pytest.approx(
        2 / 7200 / 8
    )
    assert "goodput_per_gpu_hour" not in result["profiles"]["interactive"]


def test_scorecard_prefers_exact_trace_gpu_hours_even_when_zero():
    report = SimpleNamespace(
        trace_report={
            "duration_ms": 2_000.0,
            "completed_requests": 1,
            "num_requests": 1,
            "gpu_hours": 0.0,
        },
        # Compatibility-only fallback must not replace an exact zero reported
        # by the Rust replay runtime.
        gpu_hours=99.0,
        per_request=[
            {
                "terminal_status": "completed",
                "ttft_ms": 100.0,
                "itl_ms": 10.0,
            }
        ],
        scaling_events=[],
    )

    result = scorecard(report, profiles=(INTERACTIVE,))

    assert result["gpu_hours"] == 0.0
    assert result["profiles"]["interactive"]["good_count"] == 1.0
    assert result["profiles"]["interactive"]["goodput_per_gpu"] is None


# --- oscillation (stability axis) -----------------------------------------


def _ev(component, reason):
    return SimpleNamespace(component=component, reason=reason)


def test_oscillation_counts_direction_reversals():
    events = [
        _ev("prefill", "scale_up"),
        _ev("prefill", "scale_up"),  # no reversal (up->up)
        _ev("prefill", "scale_down"),  # reversal 1 (up->down)
        _ev("prefill", "scale_up"),  # reversal 2 (down->up)
    ]
    osc = oscillation_count(events)
    assert osc["prefill"] == 2
    assert osc["total"] == 2


def test_oscillation_separates_components():
    events = [
        _ev("prefill", "scale_up"),
        _ev("decode", "scale_up"),
        _ev("decode", "scale_down"),  # decode reversal
        _ev("prefill", "scale_down"),  # prefill reversal
    ]
    osc = oscillation_count(events)
    assert osc["prefill"] == 1
    assert osc["decode"] == 1
    assert osc["total"] == 2


def test_summary_goodput_requires_the_same_constraints_not_only_the_same_name():
    report = SimpleNamespace(
        trace_report={
            "duration_ms": 1000.0,
            "completed_requests": 1,
            "num_requests": 1,
            "gpu_hours": 1.0 / 3600,
            "goodput_request_throughput_rps": 1.0,
            "goodput_completed_requests": 1,
        },
        per_request=None,
        scaling_events=[],
    )
    replay_profile = SLOProfile(name="interactive", ttft_ms=1000.0)
    stricter_profile = SLOProfile(name="interactive", ttft_ms=10.0)
    scored = scorecard(report, profiles=(stricter_profile,), sla_profile=replay_profile)
    assert scored["profiles"]["interactive"]["goodput_rps"] is None
    assert scored["goodput_available"] is False


@pytest.mark.parametrize("policy, expected", [("fail", False), ("skip", True)])
def test_short_output_policy_uses_emitted_not_requested_tokens(policy, expected):
    rec = {
        "terminal_status": "completed",
        "output_length": 1,
        "requested_output_length": 100,
        "ttft_ms": 100.0,
        "e2e_latency_ms": 100.0,
        "itl_ms": None,
    }
    assert request_is_good(rec, INTERACTIVE, short_output_itl=policy) is expected


@pytest.mark.parametrize("output_length", [None, 0, 2])
def test_skip_policy_does_not_waive_other_missing_itl(output_length):
    rec = {
        "terminal_status": "completed",
        "ttft_ms": 100.0,
        "itl_ms": None,
    }
    if output_length is not None:
        rec["output_length"] = output_length
    assert not request_is_good(rec, INTERACTIVE, short_output_itl="skip")


def test_skip_policy_does_not_waive_ttft_or_terminal_status():
    rec = _rec(301.0, 301.0, 1)
    assert not request_is_good(rec, INTERACTIVE, short_output_itl="skip")
    rec.update(ttft_ms=100.0, terminal_status="canceled")
    assert not request_is_good(rec, INTERACTIVE, short_output_itl="skip")


@pytest.mark.parametrize("latency", [float("nan"), float("inf"), -1.0])
@pytest.mark.parametrize("field", ["ttft_ms", "itl_ms", "e2e_latency_ms"])
def test_invalid_latency_cannot_satisfy_constraint(latency, field):
    rec = {"ttft_ms": 1.0, "itl_ms": 1.0, "e2e_latency_ms": 2.0}
    rec[field] = latency
    profile = SLOProfile("all", ttft_ms=10.0, itl_ms=10.0, e2e_ms=10.0)
    assert not request_is_good(rec, profile)


def _summary_report():
    return SimpleNamespace(
        trace_report={
            "duration_ms": 2000.0,
            "completed_requests": 2,
            "num_requests": 3,
            "gpu_hours": 4.0 / 3600,
            "goodput_request_throughput_rps": 1.0,
            "goodput_completed_requests": 2,
        },
        per_request=None,
        scaling_events=[],
    )


def test_strict_short_output_policy_cannot_use_native_itl_summary():
    result = scorecard(
        _summary_report(), profiles=(INTERACTIVE,), sla_profile=INTERACTIVE
    )
    assert result["short_output_itl"] == "fail"
    assert result["goodput_available"] is False
    profile = result["profiles"]["interactive"]
    assert profile["goodput_rps"] is None
    assert profile["unavailable_reason"] == "native_summary_skips_short_output_itl"


@pytest.mark.parametrize("policy", ["fail", "skip"])
def test_summary_without_itl_constraint_supports_both_policies(policy):
    profile = SLOProfile("ttft", ttft_ms=100.0)
    result = scorecard(
        _summary_report(),
        profiles=(profile,),
        sla_profile=profile,
        short_output_itl=policy,
    )
    assert result["goodput_available"] is True
    assert result["profiles"]["ttft"]["goodput_rps"] == 1.0


def test_compatible_summary_and_capture_have_identical_goodput():
    report = _summary_report()
    summarized = scorecard(
        report,
        profiles=(INTERACTIVE,),
        sla_profile=INTERACTIVE,
        short_output_itl="skip",
    )
    report.per_request = [
        _rec(100.0, 100.0, 1),
        _rec(100.0, 120.0, 3),
        dict(_rec(100.0, 120.0, 3), terminal_status="rejected"),
    ]
    captured = scorecard(
        report,
        profiles=(INTERACTIVE,),
        sla_profile=INTERACTIVE,
        short_output_itl="skip",
    )
    for metric in ("good_count", "goodput_rps", "good_rate", "goodput_per_gpu"):
        assert captured["profiles"]["interactive"][metric] == pytest.approx(
            summarized["profiles"]["interactive"][metric]
        )


def test_summary_missing_good_count_is_unavailable():
    report = _summary_report()
    del report.trace_report["goodput_completed_requests"]
    result = scorecard(
        report,
        profiles=(INTERACTIVE,),
        sla_profile=INTERACTIVE,
        short_output_itl="skip",
    )
    assert result["goodput_available"] is False


def test_invalid_short_output_policy_is_rejected():
    with pytest.raises(ValueError, match="short_output_itl"):
        scorecard(_summary_report(), short_output_itl="ignore")


def _window_report():
    return SimpleNamespace(
        trace_report={
            "duration_ms": 8000.0,
            "completed_requests": 3,
            "num_requests": 4,
            "gpu_hours": 16.0 / 3600,
        },
        per_request=None,
        planner=SimpleNamespace(lifecycle_operations=[]),
        scaling_events=[],
    )


def _window_records():
    # Arrival at the left edge is included; at the right edge is excluded.
    return [
        dict(_rec(100.0, 120.0, 3), arrival_time_ms=1000, terminal_status="completed"),
        dict(
            _rec(100.0, 120.0, 3),
            arrival_time_ms=5000,
            terminal_time_ms=7000,
            terminal_status="completed",
        ),
        dict(_rec(100.0, 120.0, 3), arrival_time_ms=5500, terminal_status="rejected"),
        dict(_rec(100.0, 120.0, 3), arrival_time_ms=6000, terminal_status="completed"),
    ]


def test_window_scores_arrival_cohort_including_drain_completions():
    result = scorecard(
        _window_report(),
        profiles=(INTERACTIVE,),
        measurement_window=MeasurementWindow(1, 6),
        allocation=AllocationConfig({"agg": 1}, {"agg": 2}),
        request_records=iter(_window_records()),
    )
    assert result["attempted_requests"] == 3
    assert result["completed_requests"] == 2
    assert result["duration_s"] == 5
    assert result["replay_duration_s"] == 8
    assert result["gpu_hours"] == pytest.approx(10 / 3600)
    assert result["average_gpus"] == 2
    assert result["profiles"]["interactive"]["goodput_rps"] == pytest.approx(2 / 5)
    assert result["profiles"]["interactive"]["goodput_per_gpu"] == pytest.approx(1 / 5)
    assert result["profiles"]["interactive"]["good_rate"] == pytest.approx(2 / 3)
    assert result["measurement"]["completion_policy"] == "include_drain"
    assert result["context_metrics_basis"] == "full_replay"


def test_streamed_records_override_capture_and_are_consumed_once_for_all_profiles():
    report = _window_report()
    report.per_request = []
    rows = iter(_window_records())
    profiles = (INTERACTIVE, SLOProfile("strict", ttft_ms=1.0))
    streamed = scorecard(report, profiles=profiles, request_records=rows)
    report.per_request = _window_records()
    captured = scorecard(report, profiles=profiles)
    assert streamed == captured
    assert list(rows) == []
    assert streamed["profiles"]["interactive"]["good_count"] == 3
    assert streamed["profiles"]["strict"]["good_count"] == 0


@pytest.mark.parametrize("windowed", [False, True])
def test_scoring_rejects_incomplete_capture(windowed):
    with pytest.raises(ValueError, match="one record per attempted request"):
        scorecard(
            _window_report(),
            profiles=(INTERACTIVE,),
            measurement_window=MeasurementWindow(1, 6) if windowed else None,
            allocation=AllocationConfig({"agg": 1}, {"agg": 2}) if windowed else None,
            request_records=iter(_window_records()[:2]),
        )


def test_window_cannot_be_scored_from_native_summary():
    with pytest.raises(ValueError, match="requires request records"):
        scorecard(
            _window_report(),
            measurement_window=MeasurementWindow(1, 6),
            allocation=AllocationConfig({"agg": 1}, {"agg": 2}),
        )


def test_static_window_idle_tail_uses_same_duration_for_rate_and_gpu_average():
    result = scorecard(
        _window_report(),
        profiles=(INTERACTIVE,),
        measurement_window=MeasurementWindow(0, 10),
        allocation=AllocationConfig({"agg": 1}, {"agg": 2}, allow_idle_tail=True),
        request_records=iter(_window_records()),
    )
    assert result["duration_s"] == 10
    assert result["gpu_hours"] == pytest.approx(20 / 3600)
    assert result["average_gpus"] == 2
    assert result["profiles"]["interactive"]["goodput_rps"] == pytest.approx(3 / 10)
    assert result["profiles"]["interactive"]["goodput_per_gpu"] == pytest.approx(3 / 20)


def test_zero_output_e2e_only_capture_matches_native_summary():
    report = _summary_report()
    report.trace_report.update(
        completed_requests=1,
        num_requests=1,
        goodput_completed_requests=1,
        goodput_request_throughput_rps=0.5,
    )
    profile = SLOProfile("e2e", e2e_ms=100.0)
    summarized = scorecard(report, profiles=(profile,), sla_profile=profile)
    report.per_request = [
        {
            "terminal_status": "completed",
            "first_admit_ms": 1.0,
            "arrival_time_ms": 0.0,
            "terminal_time_ms": 50.0,
            "output_length": 0,
            "e2e_latency_ms": None,
            "ttft_ms": None,
            "itl_ms": None,
        }
    ]
    captured = scorecard(report, profiles=(profile,))
    for metric in ("good_count", "goodput_rps", "good_rate", "goodput_per_gpu"):
        assert captured["profiles"]["e2e"][metric] == pytest.approx(
            summarized["profiles"]["e2e"][metric]
        )
    assert not request_is_good(
        report.per_request[0], INTERACTIVE, short_output_itl="skip"
    )


def test_canonical_request_never_admitted_is_not_good():
    rec = dict(_rec(100.0, 120.0, 3), first_admit_ms=None)
    assert not request_is_good(rec, INTERACTIVE)
