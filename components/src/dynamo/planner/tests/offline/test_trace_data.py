# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for planner warmup trace parsing and window construction.

The planner's online throughput loop feeds *every* interval (including
zero-traffic ones) to the load predictors, but warmup goes through
the offline trace helpers. These tests pin the densified behavior: gaps between
the first and last active interval are emitted as empty windows, while leading
and trailing empty intervals are omitted. They also cover automatic Mooncake
and Dynamo request trace v1 detection.
"""

from __future__ import annotations

import gzip
import json

import pytest

from dynamo.planner.offline.trace_data import (
    detect_trace_format,
    extract_metrics_from_mooncake,
    extract_metrics_from_trace,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _write_trace(tmp_path, records, name="trace.jsonl"):
    path = tmp_path / name
    contents = "\n".join(json.dumps(r) for r in records) + "\n"
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as output:
            output.write(contents)
    else:
        path.write_text(contents)
    return str(path)


def _rec(timestamp_ms, isl, osl):
    return {"timestamp": timestamp_ms, "input_length": isl, "output_length": osl}


def _dynamo_rec(timestamp_ms, isl, osl, *, wrapped=False):
    event = {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": timestamp_ms + 1_000,
        "request": {
            "request_id": f"request-{timestamp_ms}",
            "request_received_ms": timestamp_ms,
            "output_tokens": osl,
            "replay": {
                "trace_block_size": 64,
                "input_length": isl,
                "input_sequence_hashes": list(range((isl + 63) // 64)),
            },
        },
    }
    return {"timestamp": timestamp_ms + 1_000, "event": event} if wrapped else event


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


def test_auto_detects_mooncake(tmp_path):
    path = _write_trace(tmp_path, [_rec(0, 100, 10)])

    assert detect_trace_format(path) == "mooncake"
    assert extract_metrics_from_trace(path, 10) == extract_metrics_from_mooncake(
        path, 10
    )


def test_dynamo_v1_matches_lowered_mooncake_windows(tmp_path):
    mooncake_path = _write_trace(
        tmp_path,
        [_rec(0, 100, 10), _rec(5_000, 300, 30), _rec(35_000, 500, 50)],
        "mooncake.jsonl",
    )
    dynamo_path = _write_trace(
        tmp_path,
        [
            _dynamo_rec(100_000, 100, 10),
            _dynamo_rec(105_000, 300, 30, wrapped=True),
            {
                "schema": "dynamo.request.trace.v1",
                "event_type": "tool_end",
                "event_time_unix_ms": 110_000,
            },
            _dynamo_rec(135_000, 500, 50),
        ],
        "dynamo.jsonl.gz",
    )

    assert detect_trace_format(dynamo_path) == "dynamo"
    assert extract_metrics_from_trace(dynamo_path, 10) == extract_metrics_from_trace(
        mooncake_path, 10
    )


def test_dynamo_v1_directory_uses_global_origin_across_shards(tmp_path):
    trace_dir = tmp_path / "trace-shards"
    trace_dir.mkdir()
    _write_trace(
        trace_dir,
        [_dynamo_rec(135_000, 500, 50)],
        "part-2.jsonl.gz",
    )
    _write_trace(
        trace_dir,
        [_dynamo_rec(100_000, 100, 10)],
        "part-1.jsonl",
    )
    metrics = extract_metrics_from_trace(str(trace_dir), 10)

    assert [metric["interval_start"] for metric in metrics] == [0, 10, 20, 30]
    assert [metric["request_count"] for metric in metrics] == [1, 0, 0, 1]


def test_directory_rejects_compressed_and_uncompressed_copies(tmp_path):
    trace_dir = tmp_path / "trace-shards"
    trace_dir.mkdir()
    record = _dynamo_rec(100_000, 100, 10)
    _write_trace(trace_dir, [record], "part-1.jsonl")
    _write_trace(trace_dir, [record], "part-1.jsonl.gz")

    with pytest.raises(ValueError, match="Both compressed and uncompressed"):
        extract_metrics_from_trace(str(trace_dir), 10)


def test_explicit_mooncake_reader_rejects_dynamo(tmp_path):
    path = _write_trace(tmp_path, [_dynamo_rec(100_000, 100, 10)])

    with pytest.raises(ValueError, match="Expected Mooncake warmup trace"):
        extract_metrics_from_mooncake(path, 10)


def test_unknown_trace_format_is_rejected(tmp_path):
    path = _write_trace(tmp_path, [{"timestamp": 0, "prompt_tokens": 100}])

    with pytest.raises(ValueError, match="Unsupported warmup trace format"):
        extract_metrics_from_trace(path, 10)
