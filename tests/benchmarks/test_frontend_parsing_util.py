# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the frontend benchmark Prometheus snapshot parser."""

import math

import pytest

from benchmarks.frontend.scripts.analysis.parsing_util import parse_prometheus_text

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

_METRIC = "dynamo_frontend_stage_duration_seconds"
_BOUNDS = ["0.1", "1.0", "+Inf"]


def _write_snapshot(tmp_path, counts):
    lines = [
        f'{_METRIC}_bucket{{stage="preprocess",le="{le}"}} {count}'
        for le, count in zip(_BOUNDS, counts)
    ]
    lines.append(f'{_METRIC}_count{{stage="preprocess"}} {counts[-1]}')
    path = tmp_path / "prometheus.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize(
    "counts, expected",
    [
        # p50 interpolates inside [0, 0.1]; p95 and p99 fall in the +Inf bucket
        # and must clamp to the highest finite bound (1.0), as Prometheus does.
        ((50, 90, 100), {"p50": 0.1, "p95": 1.0, "p99": 1.0}),
        # Every sample above the finite bounds: all quantiles clamp to 1.0.
        ((0, 0, 100), {"p50": 1.0, "p95": 1.0, "p99": 1.0}),
        # All samples inside finite buckets: linear interpolation is unchanged.
        ((20, 100, 100), {"p50": 0.4375, "p95": 0.94375, "p99": 0.98875}),
    ],
    ids=["tail-in-inf-bucket", "all-in-inf-bucket", "finite-buckets"],
)
def test_stage_quantiles_stay_finite(tmp_path, counts, expected):
    snapshot = parse_prometheus_text(_write_snapshot(tmp_path, counts))

    quantiles = snapshot.stage_durations["preprocess"]
    assert all(math.isfinite(v) for v in quantiles.values()), quantiles
    assert quantiles == pytest.approx(expected)
