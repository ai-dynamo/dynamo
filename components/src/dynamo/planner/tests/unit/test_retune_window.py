# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the retune request-trace window and history readers."""

import gzip
import json
import os

import pytest

from dynamo.mocker.config import normalize_mocker_config
from dynamo.planner.offline.trace_data import extract_traffic_observations_from_trace
from dynamo.planner.retune.window import (
    ROW_COUNT_KEYS,
    ObservationBuckets,
    read_trace_history,
    read_trace_window,
    resolve_trace_files,
    trace_origin_ms,
)
from dynamo.replay import run_trace_replay

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

SCHEMA = "dynamo.request.trace.v1"
_UNSET = object()


def _request_end(
    request_id,
    received_ms=_UNSET,
    *,
    event_ms=None,
    total_ms=None,
    isl=128,
    osl=8,
    block_size=64,
    agent_context=None,
    output_tokens=_UNSET,
):
    request = {"request_id": request_id, "model": "m"}
    if received_ms is not _UNSET and received_ms is not None:
        request["request_received_ms"] = received_ms
    if total_ms is not None:
        request["total_time_ms"] = total_ms
    request["output_tokens"] = osl if output_tokens is _UNSET else output_tokens
    request["replay"] = {
        "trace_block_size": block_size,
        "input_length": isl,
        "input_sequence_hashes": [sum(map(ord, request_id)), 7],
    }
    if event_ms is None:
        base = received_ms if isinstance(received_ms, int) else 0
        event_ms = base + (round(total_ms) if total_ms is not None else 0)
    record = {
        "schema": SCHEMA,
        "event_type": "request_end",
        "event_time_unix_ms": event_ms,
        "request": request,
    }
    if agent_context is not None:
        record["agent_context"] = agent_context
    return record


def _event(event_type, event_ms=1_000, **extra):
    return {
        "schema": SCHEMA,
        "event_type": event_type,
        "event_time_unix_ms": event_ms,
        **extra,
    }


def _envelope(record, timestamp=0):
    return {"timestamp": timestamp, "event": record}


def _write(path, records, *, trailing=""):
    text = "".join(json.dumps(record) + "\n" for record in records) + trailing
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as sink:
            sink.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _window(paths, tmp_path, start=0, end=10_000, as_of=None, **kwargs):
    out = tmp_path / "out" / "window.jsonl"
    result = read_trace_window(
        [str(path) for path in paths],
        start,
        end,
        end if as_of is None else as_of,
        out,
        prune_by_mtime=kwargs.pop("prune_by_mtime", False),
        **kwargs,
    )
    return result, _read(out)


def test_envelope_and_bare_rows_become_sorted_bare_rows(tmp_path):
    trace = _write(
        tmp_path / "trace.jsonl",
        [
            _envelope(_request_end("b", 3_000, total_ms=10.0), timestamp=5),
            _request_end("a", 1_000, total_ms=10.0),
            _envelope(_request_end("c", 2_000, total_ms=10.0)),
        ],
    )

    result, rows = _window([trace], tmp_path)

    assert result.num_requests == 3
    assert result.trace_block_size == 64
    assert set(result.row_counts) == set(ROW_COUNT_KEYS)
    assert [row["request"]["request_id"] for row in rows] == ["a", "c", "b"]
    for row in rows:
        assert set(row) == {"schema", "event_type", "event_time_unix_ms", "request"}
        assert row["request"]["replay"]["input_sequence_hashes"]


def test_missing_received_ms_uses_ais_arrival_rule(tmp_path):
    # Arrival = event - round(total) with Rust rounding (half away from zero).
    trace = _write(
        tmp_path / "trace.jsonl",
        [
            _request_end("late", None, event_ms=5_000, total_ms=2_000.5),
            _request_end("early", None, event_ms=900, total_ms=100.0),
        ],
    )

    result, rows = _window([trace], tmp_path, start=0, end=3_000, as_of=10_000)

    assert result.num_requests == 2
    received = {
        row["request"]["request_id"]: row["request"]["request_received_ms"]
        for row in rows
    }
    assert received == {"early": 800, "late": 2_999}
    # The normalized arrival is written back so trace_data reads the same time.
    assert rows[0]["request"]["request_id"] == "early"


def test_as_of_censors_by_ais_completion_time(tmp_path):
    trace = _write(
        tmp_path / "trace.jsonl",
        [
            _request_end("on-time", 1_000, total_ms=1_000.0),  # completes 2_000
            _request_end("late", 1_500, total_ms=1_000.4),  # completes 2_500
            # No total_time_ms: completion falls back to event_time_unix_ms.
            _request_end("no-total", 1_200, event_ms=9_999),
            _request_end("converted", 1_300, event_ms=1_300),
        ],
    )

    result, rows = _window([trace], tmp_path, start=0, end=2_000, as_of=2_000)

    assert [row["request"]["request_id"] for row in rows] == ["on-time", "converted"]
    assert result.row_counts["censored"] == 2
    assert result.row_counts["censor_unavailable"] == 1


def test_window_bounds_are_half_open(tmp_path):
    trace = _write(
        tmp_path / "trace.jsonl",
        [_request_end(str(at), at) for at in (999, 1_000, 1_999, 2_000)],
    )

    result, rows = _window([trace], tmp_path, start=1_000, end=2_000, as_of=5_000)

    assert [row["request"]["request_id"] for row in rows] == ["1000", "1999"]
    assert result.row_counts["outside_window"] == 2


def test_duplicate_request_ids_keep_the_first_row(tmp_path):
    first = _write(tmp_path / "a.jsonl", [_request_end("dup", 1_000, isl=64)])
    second = _write(
        tmp_path / "b.jsonl",
        [_request_end("dup", 1_500, isl=256), _request_end("other", 1_200)],
    )

    result, rows = _window([first, second], tmp_path)

    assert result.num_requests == 2
    assert result.row_counts["duplicate"] == 1
    dup = next(row for row in rows if row["request"]["request_id"] == "dup")
    assert dup["request"]["replay"]["input_length"] == 64


def test_non_request_rows_are_dropped_and_agent_context_is_stripped(tmp_path):
    trace = _write(
        tmp_path / "trace.jsonl",
        [
            _request_end("a", 1_000, agent_context={"session_id": "s1"}),
            _event("tool_start", agent_context={"session_id": "s1"}),
            _event("tool_end", agent_context={"session_id": "s1"}),
            _event("tool_error"),
            _event("router_selection", selection={"request_id": "a"}),
            _event("request_payload", payload={"request_id": "a"}),
            _envelope(_event("future_event")),
            _request_end("b", 2_000),
        ],
    )

    result, rows = _window([trace], tmp_path)

    assert [row["request"]["request_id"] for row in rows] == ["a", "b"]
    assert all("agent_context" not in row for row in rows)
    counts = result.row_counts
    assert (
        counts["tool"],
        counts["router_selection"],
        counts["request_payload"],
        counts["other_event"],
        counts["agent_context_stripped"],
    ) == (3, 1, 1, 1, 1)


def test_rows_without_output_tokens_are_dropped(tmp_path):
    trace = _write(
        tmp_path / "trace.jsonl",
        [_request_end("a", 1_000, output_tokens=None), _request_end("b", 1_100)],
    )

    result, rows = _window([trace], tmp_path)

    assert [row["request"]["request_id"] for row in rows] == ["b"]
    assert result.row_counts["missing_output_tokens"] == 1


def test_mixed_block_sizes_are_rejected(tmp_path):
    trace = _write(
        tmp_path / "trace.jsonl",
        [_request_end("a", 1_000), _request_end("b", 1_100, block_size=32)],
    )

    with pytest.raises(ValueError, match="trace_block_size"):
        _window([trace], tmp_path)


def test_block_size_outside_the_window_is_not_checked(tmp_path):
    trace = _write(
        tmp_path / "trace.jsonl",
        [_request_end("a", 1_000), _request_end("b", 50_000, block_size=32)],
    )

    result, _ = _window([trace], tmp_path)

    assert result.trace_block_size == 64


def test_truncated_gzip_member_and_partial_line_are_tolerated(tmp_path):
    complete = gzip.compress(
        (json.dumps(_request_end("a", 1_000)) + "\n").encode("utf-8")
    )
    member = gzip.compress(
        (json.dumps(_request_end("b", 1_100)) + "\n").encode("utf-8")
    )
    truncated = member[: len(member) // 2]
    segment = tmp_path / "frontend.000000.jsonl.gz"
    segment.write_bytes(complete + truncated)
    partial = _write(
        tmp_path / "plain.jsonl",
        [_request_end("c", 1_200)],
        trailing='{"schema": "dynamo.request.trac',
    )

    result, rows = _window([segment, partial], tmp_path)

    assert [row["request"]["request_id"] for row in rows] == ["a", "c"]
    assert result.row_counts["truncated_gzip"] == 1
    assert result.row_counts["partial_line"] == 1


@pytest.mark.parametrize("tail", [b"\x1f", b"B"], ids=["magic-prefix", "garbage"])
def test_one_byte_gzip_member_header_is_tolerated_only_as_a_magic_prefix(
    tmp_path, tail
):
    # A live segment read after its writer appended only the first byte of
    # the next gzip member.
    complete = gzip.compress(
        (json.dumps(_request_end("a", 1_000)) + "\n").encode("utf-8")
    )
    segment = tmp_path / "frontend.000000.jsonl.gz"
    segment.write_bytes(complete + tail)

    if tail != b"\x1f":
        with pytest.raises(gzip.BadGzipFile):
            _window([segment], tmp_path)
        return
    result, rows = _window([segment], tmp_path)
    assert [row["request"]["request_id"] for row in rows] == ["a"]
    assert result.row_counts["truncated_gzip"] == 1


def test_corrupt_middle_line_raises(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(_request_end("a", 1_000))
        + "\n{not json}\n"
        + json.dumps(_request_end("b", 1_100))
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        _window([trace], tmp_path)


def test_foreign_schema_raises(tmp_path):
    trace = _write(
        tmp_path / "trace.jsonl",
        [{"timestamp": 0, "input_length": 1, "output_length": 1}],
    )

    with pytest.raises(ValueError, match=SCHEMA):
        _window([trace], tmp_path)


def test_trace_origin_is_the_earliest_ais_arrival_over_all_files(tmp_path):
    first = _write(
        tmp_path / "a.jsonl",
        [_request_end("a", 5_000), _event("tool_end", event_ms=10)],
    )
    second = _write(
        tmp_path / "b.jsonl.gz",
        [
            _request_end("b", None, event_ms=4_500, total_ms=1_000.0),
            _request_end("c", 4_000, total_ms=10.0),
        ],
    )

    counts: dict[str, int] = {}
    assert trace_origin_ms([str(first), str(second)], row_counts=counts) == 3_500.0
    assert counts["tool"] == 1 and counts["agent_context_stripped"] == 0
    assert trace_origin_ms([str(first)]) == 5_000.0
    agentic = _write(
        tmp_path / "agentic.jsonl",
        [_request_end("d", 6_000, agent_context={"session_id": "s1"})],
    )
    trace_origin_ms([str(agentic)], row_counts=counts)
    assert counts["agent_context_stripped"] == 1
    empty = _write(tmp_path / "empty.jsonl", [_event("tool_end")])
    with pytest.raises(ValueError, match="no request_end"):
        trace_origin_ms([str(empty)])


def test_segment_prefix_glob_and_directory_resolution(tmp_path):
    segments = tmp_path / "segments"
    segments.mkdir()
    for index in range(2):
        _write(
            segments / f"frontend.{index:06d}.jsonl.gz",
            [_request_end(f"r{index}", 1_000 + index)],
        )

    by_prefix = resolve_trace_files([str(segments / "frontend")])
    by_suffixed_prefix = resolve_trace_files([str(segments / "frontend.jsonl.gz")])
    by_dir = resolve_trace_files([str(segments), str(segments / "*.jsonl.gz")])

    assert by_prefix == by_suffixed_prefix == by_dir
    assert [path.name for path in by_prefix] == [
        "frontend.000000.jsonl.gz",
        "frontend.000001.jsonl.gz",
    ]
    with pytest.raises(FileNotFoundError):
        resolve_trace_files([str(tmp_path / "missing")])


def test_mtime_pruning_only_when_enabled_with_a_clock_skew_margin(tmp_path):
    start, end = 1_000_000, 2_000_000
    old = _write(tmp_path / "old.jsonl", [_request_end("old", 400_000)])
    skewed = _write(tmp_path / "skewed.jsonl", [_request_end("skewed", start + 4_000)])
    new = _write(tmp_path / "new.jsonl", [_request_end("new", start + 100_000)])
    # mtimes in epoch seconds. old was last written 400 s before the window;
    # skewed's store clock runs behind the frontends', so its mtime is 100 s
    # before the window although it holds an in-window row.
    os.utime(old, (600.0, 600.0))
    os.utime(skewed, (900.0, 900.0))
    os.utime(new, (1_200.0, 1_200.0))
    files = [old, skewed, new]

    unpruned, _ = _window(files, tmp_path, start=start, end=end)
    pruned, rows = _window(files, tmp_path, start=start, end=end, prune_by_mtime=True)

    assert unpruned.num_requests == 2
    assert unpruned.row_counts["pruned_files"] == 0
    assert [row["request"]["request_id"] for row in rows] == ["skewed", "new"]
    assert pruned.row_counts["pruned_files"] == 1


def test_empty_window_writes_an_empty_file(tmp_path):
    trace = _write(tmp_path / "trace.jsonl", [_request_end("a", 50_000)])

    result, rows = _window([trace], tmp_path)

    assert (result.num_requests, result.trace_block_size, rows) == (0, None, [])


def test_history_is_streamed_without_hashes_and_bucketed(tmp_path):
    trace = _write(
        tmp_path / "trace.jsonl",
        [
            _request_end("h0", 1_000, isl=100, osl=10),
            _request_end("h1", 1_500, isl=300, osl=30),
            _request_end("h2", 3_500, isl=200, osl=20),
            _request_end("w0", 4_200, isl=50, osl=5),
        ],
    )
    out = tmp_path / "history.jsonl"
    buckets = ObservationBuckets(end_ms=4_000, interval_ms=1_000, duration_s=1.0)

    history = read_trace_history(
        [str(trace)], 0, 4_000, 4_000, out, prune_by_mtime=False, buckets=buckets
    )

    assert history.num_requests == 3
    assert history.path == str(out)
    rows = _read(out)
    for row in rows:
        assert "input_sequence_hashes" not in row["request"]["replay"]
        assert "trace_block_size" not in row["request"]["replay"]
    # Buckets end at 4_000: [1_000, 2_000), [2_000, 3_000), [3_000, 4_000).
    assert [(o.num_req, o.isl, o.osl) for o in buckets.observations()] == [
        (2.0, 200.0, 20.0),
        (0.0, 0.0, 0.0),
        (1.0, 200.0, 20.0),
    ]
    # The hash-free file is a valid planner warmup trace.
    warmup = extract_traffic_observations_from_trace(str(out), 1)
    assert sum(observation.num_req for observation in warmup) == 3


def test_history_without_output_file_only_buckets(tmp_path):
    trace = _write(tmp_path / "trace.jsonl", [_request_end("h0", 1_000)])
    buckets = ObservationBuckets(end_ms=2_000, interval_ms=1_000, duration_s=1.0)

    history = read_trace_history(
        [str(trace)], 0, 2_000, 2_000, None, prune_by_mtime=False, buckets=buckets
    )

    assert history.path is None
    assert history.num_requests == 1
    assert buckets.bucket_count() == 1


def test_window_file_replays_in_aisimulate(tmp_path):
    trace = _write(
        tmp_path / "trace.jsonl",
        [
            _envelope(_request_end("a", 1_000, total_ms=50.0)),
            _request_end("b", None, event_ms=1_300, total_ms=100.0),
            _request_end("c", 1_100, agent_context={"session_id": "s"}),
            _event("router_selection"),
            _request_end("a", 1_000, total_ms=50.0),
        ],
    )
    result, _ = _window([trace], tmp_path, start=0, end=5_000)

    report = run_trace_replay(
        [result.path],
        trace_format="dynamo",
        extra_engine_args=normalize_mocker_config(
            {"engine": {"block_size": 64, "speedup_ratio": 1000.0}}
        ),
        num_workers=1,
    )

    assert report.summary["completed_requests"] == result.num_requests == 3
