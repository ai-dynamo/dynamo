# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the scorecard's AIPerf-exact goodput + stability metrics.

Pure-Python — no Dynamo runtime. Validates the per-request good/bad rule
against AIPerf semantics (joint AND, inclusive <=, ITL undefined for osl<2).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
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
