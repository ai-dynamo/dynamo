# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from benchmarks.frontend.scripts.analysis.parsing_util import parse_prometheus_text


@pytest.mark.gpu_0
@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize(
    "bucket_labels",
    [
        'stage="preprocess",le="{le}"',
        'le="{le}",stage="preprocess"',
        'worker="0",le="{le}",stage="preprocess"',
        'le="{le}",stage="preprocess",worker="0"',
    ],
)
def test_stage_quantiles_do_not_depend_on_label_order(
    tmp_path: Path, bucket_labels: str
):
    metric = "dynamo_frontend_stage_duration_seconds"
    lines = [
        f"{metric}_bucket{{{bucket_labels.format(le=le)}}} {count}"
        for le, count in [("0.1", 50), ("1", 100), ("+Inf", 100)]
    ]
    lines += [
        f'{metric}_count{{worker="0",stage="preprocess"}} 100',
        f'{metric}_bucket{{stage="route",le="0.1"}} 0',
        f'{metric}_bucket{{stage="route",le="1"}} 10',
        f'{metric}_bucket{{stage="route",le="+Inf"}} 10',
        f'{metric}_count{{stage="route"}} 10',
    ]
    path = tmp_path / "metrics.txt"
    path.write_text("\n".join(lines))

    snapshot = parse_prometheus_text(path)

    assert snapshot is not None
    assert snapshot.stage_durations["preprocess"] == pytest.approx(
        {"p50": 0.1, "p95": 0.91, "p99": 0.982}
    )
    assert snapshot.stage_durations["route"] == pytest.approx(
        {"p50": 0.55, "p95": 0.955, "p99": 0.991}
    )
