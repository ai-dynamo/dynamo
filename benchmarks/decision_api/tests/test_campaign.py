# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from dynamo_decision_perf.campaign import can_continue, planned_runs, screen_capacity

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def result(rate=2, errors=0):
    return {
        "audit": {"status": "valid"},
        "summary": {
            "rates": {"valid_http_requests_per_second": rate},
            "counts": {
                "schema_errors": 0,
                "http_errors": errors,
                "timeouts": 0,
                "incomplete_requests": 0,
                "rejections": 0,
                "transport_errors": 0,
                "cancelled": 0,
                "pipeline_errors": 0,
            },
        },
    }


def test_screening_capacity_excludes_invalid_or_failed_runs():
    assert screen_capacity([result(2), result(4), result(100, errors=1)]) == 4
    bad = result()
    bad["audit"]["status"] = "invalid"
    with pytest.raises(ValueError):
        screen_capacity([bad])


@pytest.mark.parametrize(
    "bucket", ["rejections", "transport_errors", "cancelled", "pipeline_errors"]
)
def test_screening_capacity_rejects_every_error_bucket(bucket):
    bad = result()
    bad["summary"]["counts"][bucket] = 1
    with pytest.raises(ValueError, match="error-free"):
        screen_capacity([bad])


def test_safety_gate_stops_correctness_oom_and_unsettled_backlog():
    assert can_continue(
        result(),
        {
            "oom": False,
            "backlog_bounded": True,
            "outstanding_drained": True,
            "client_saturated": False,
        },
    )
    assert not can_continue(result(), {})
    assert not can_continue(
        result(),
        {
            "oom": True,
            "backlog_bounded": True,
            "outstanding_drained": True,
            "client_saturated": False,
        },
    )
    bad = result()
    bad["summary"]["counts"]["schema_errors"] = 1
    assert not can_continue(
        bad,
        {
            "oom": False,
            "backlog_bounded": True,
            "outstanding_drained": True,
            "client_saturated": False,
        },
    )


def test_plan_has_sequential_shapes_load_and_recovery():
    runs = planned_runs(["base", "tokens1024"], 2)
    assert runs[0]["campaign"] == "serial"
    assert runs[-1]["campaign"] == "recovery"
    assert all(p["duration"] <= 1800 for p in runs)
    assert [p["rate"] for p in runs if p["campaign"] == "sustained"] == [
        0.5,
        1,
        1.5,
        2,
        2.5,
    ]
    assert not any(p["campaign"] == "sustained" for p in planned_runs(["base"]))
