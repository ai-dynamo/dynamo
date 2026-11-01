# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for runtime work-handler metrics in benchmark reports."""

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
