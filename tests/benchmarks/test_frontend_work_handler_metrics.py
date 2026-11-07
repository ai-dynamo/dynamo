# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for runtime transport metrics in benchmark reports."""

from pathlib import Path

import pytest

from benchmarks.frontend.scripts.analysis.parsing_util import parse_prometheus_text

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "metric,field",
    [
        (
            "dynamo_work_handler_network_transit_seconds",
            "work_handler_network_transit",
        ),
        (
            "dynamo_work_handler_time_to_first_response_seconds",
            "work_handler_time_to_first_response",
        ),
    ],
)
def test_work_handler_histograms_populate_report(
    tmp_path: Path, metric: str, field: str
):
    path = tmp_path / "metrics.txt"
    path.write_text(
        f'{metric}_bucket{{le="0.1"}} 50\n'
        f'{metric}_bucket{{le="1"}} 100\n'
        f'{metric}_bucket{{le="+Inf"}} 100\n'
        f"{metric}_count 100\n"
    )

    snapshot = parse_prometheus_text(path)

    assert snapshot is not None
    assert getattr(snapshot, field) == pytest.approx(
        {"p50": 0.1, "p95": 0.91, "p99": 0.982}
    )


@pytest.mark.parametrize(
    "sample,expected",
    [
        ("dynamo_request_plane_inflight_requests 7\n", 7),
        ("dynamo_request_plane_inflight_requests 0\n", 0),
        ("# Inflight metric absent\n", 0),
    ],
    ids=["positive", "zero", "missing"],
)
def test_request_plane_inflight_populates_report(
    tmp_path: Path, sample: str, expected: int
):
    path = tmp_path / "metrics.txt"
    path.write_text(sample)

    snapshot = parse_prometheus_text(path)

    assert snapshot is not None
    assert snapshot.request_plane_inflight == expected
