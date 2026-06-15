# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``extract_metrics_from_mooncake`` window construction.

The planner's online throughput loop feeds *every* interval (including
zero-traffic ones) to the load predictors, but warmup goes through
``extract_metrics_from_mooncake``. Before the fix it only emitted intervals
that contained requests, so middle gaps were collapsed and warmup diverged
from live behavior. These tests pin the densified behavior: gaps between the
first and last active interval are emitted as empty windows, while leading and
trailing empty intervals are omitted.
"""

from __future__ import annotations

import json

import pytest

from dynamo.planner.offline.trace_data import extract_metrics_from_mooncake

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _write_trace(tmp_path, records):
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(path)


def _rec(timestamp_ms, isl, osl):
    return {"timestamp": timestamp_ms, "input_length": isl, "output_length": osl}


def test_middle_empty_windows_are_preserved(tmp_path):
    # interval=10s; activity in interval 0 (2 reqs) and interval 30 (1 req);
    # intervals 10 and 20 are empty gaps that must be preserved.
    records = [
        _rec(0, 100, 10),
        _rec(5_000, 300, 30),
        _rec(35_000, 500, 50),
    ]
    metrics = extract_metrics_from_mooncake(_write_trace(tmp_path, records), 10)

    assert [m["interval_start"] for m in metrics] == [0, 10, 20, 30]
    assert [m["request_count"] for m in metrics] == [2, 0, 0, 1]
    # active window 0: averages over its two requests
    assert metrics[0]["avg_isl"] == 200.0
    assert metrics[0]["avg_osl"] == 20.0
    # empty middle windows are zero-filled
    for m in (metrics[1], metrics[2]):
        assert m["request_count"] == 0
        assert m["avg_isl"] == 0
        assert m["avg_osl"] == 0
    assert metrics[3]["avg_isl"] == 500.0


def test_leading_and_trailing_empties_are_dropped(tmp_path):
    # First activity at interval 30, last at interval 50, gap at 40.
    records = [
        _rec(35_000, 100, 10),
        _rec(55_000, 200, 20),
    ]
    metrics = extract_metrics_from_mooncake(_write_trace(tmp_path, records), 10)

    # Starts at the first active interval (no leading 0/10/20) and ends at the
    # last active interval; the middle gap (40) is kept.
    assert [m["interval_start"] for m in metrics] == [30, 40, 50]
    assert [m["request_count"] for m in metrics] == [1, 0, 1]


def test_intervals_are_contiguous(tmp_path):
    records = [_rec(0, 10, 1), _rec(125_000, 20, 2)]
    metrics = extract_metrics_from_mooncake(_write_trace(tmp_path, records), 30)
    starts = [m["interval_start"] for m in metrics]
    assert starts == list(range(0, 121, 30))  # 0,30,60,90,120
    assert all(b - a == 30 for a, b in zip(starts, starts[1:]))


def test_empty_trace_returns_empty(tmp_path):
    assert extract_metrics_from_mooncake(_write_trace(tmp_path, []), 10) == []
