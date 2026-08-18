# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum

import pytest

from dynamo.replay.report import PlannerReplayDetails, ReplayReport

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


class _Outcome(Enum):
    APPLIED = "applied"


@dataclass
class _ScalingEvent:
    at_s: float
    outcome: _Outcome


def test_replay_report_to_dict_materializes_nested_planner_details() -> None:
    report = ReplayReport(
        summary={"completed_requests": 3},
        per_request=[{"request_id": "request-1"}],
        coverage={"captured_request_count": 1},
        planner=PlannerReplayDetails(
            metadata={"details_captured": True},
            ticks=[{"ordinal": 0}],
            scaling_events=[_ScalingEvent(at_s=1.0, outcome=_Outcome.APPLIED)],
            lifecycle_operations=[{"operation": "start"}],
            total_ticks=1,
            html_report_path="planner.html",
        ),
    )

    assert report.to_dict() == {
        "summary": {"completed_requests": 3},
        "per_request": [{"request_id": "request-1"}],
        "coverage": {"captured_request_count": 1},
        "planner": {
            "metadata": {"details_captured": True},
            "ticks": [{"ordinal": 0}],
            "scaling_events": [{"at_s": 1.0, "outcome": "applied"}],
            "lifecycle_operations": [{"operation": "start"}],
            "total_ticks": 1,
            "html_report_path": "planner.html",
        },
    }


def test_replay_report_to_dict_preserves_summary_only_shape() -> None:
    report = ReplayReport(
        summary={"completed_requests": 3},
        per_request=None,
        coverage={},
        planner=PlannerReplayDetails(total_ticks=4),
    )

    payload = report.to_dict()

    assert payload["per_request"] is None
    assert payload["planner"]["total_ticks"] == 4
    assert payload["planner"]["ticks"] == []
    assert payload["planner"]["lifecycle_operations"] == []
