# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.frontend.scripts.analysis.parsing_util import parse_prometheus_text

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "counts, expected",
    [
        ((50, 90, 100), {"p50": 0.1, "p95": 1.0, "p99": 1.0}),
        ((0, 0, 100), {"p50": 1.0, "p95": 1.0, "p99": 1.0}),
        ((20, 100, 100), {"p50": 0.4375, "p95": 0.94375, "p99": 0.98875}),
    ],
    ids=["tail-in-unbounded-bucket", "all-in-unbounded-bucket", "finite-buckets"],
)
@pytest.mark.parametrize("metric", ["stage", "transport"])
def test_histogram_quantiles(tmp_path, counts, expected, metric):
    if metric == "stage":
        name = "dynamo_frontend_stage_duration_seconds"
        labels = 'stage="preprocess",'
        count_labels = '{stage="preprocess"}'
    else:
        name = "dynamo_component_time_to_first_response_seconds"
        labels = ""
        count_labels = "{}"

    lines = [
        f'{name}_bucket{{{labels}le="{bound}"}} {count}'
        for bound, count in zip(["0.1", "1.0", "+Inf"], counts)
    ]
    lines.append(f"{name}_count{count_labels} {counts[-1]}")
    path = tmp_path / "snapshot.txt"
    path.write_text("\n".join(lines) + "\n")

    snapshot = parse_prometheus_text(path)

    if metric == "stage":
        quantiles = snapshot.stage_durations["preprocess"]
    else:
        quantiles = snapshot.work_handler_time_to_first_response
    assert quantiles == pytest.approx(expected)
