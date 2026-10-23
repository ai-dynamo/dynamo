# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the retune request-trace window and history readers."""

import gzip
import json

import pytest

from dynamo.planner.retune.window import (
    ROW_COUNT_KEYS,
    ObservationBuckets,
    read_trace_history,
    read_trace_window,
    resolve_trace_files,
    trace_origin_ms,
)

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


def _jsonl(*records):
    return "".join(json.dumps(record) + "\n" for record in records).encode("utf-8")


def _write(path, records):
    text = "".join(json.dumps(record) + "\n" for record in records)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as sink:
            sink.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _window(paths, tmp_path, start=0, end=10_000, as_of=None):
    out = tmp_path / "out" / "window.jsonl"
    result = read_trace_window(
        [str(path) for path in paths],
        start,
        end,
        end if as_of is None else as_of,
        out,
    )
    return result, _read(out)


def test_envelope_and_bare_rows_become_sorted_bare_rows(tmp_path):
    first = _write(
        tmp_path / "a.jsonl",
        [
            _envelope(_request_end("b", 3_000, total_ms=10.0), timestamp=5),
            _request_end("a", 1_000, total_ms=10.0, agent_context={"session_id": "s"}),
            _event("tool_start", agent_context={"session_id": "s"}),
            _event("tool_end", agent_context={"session_id": "s"}),
            _event("tool_error"),
            _event("router_selection", selection={"request_id": "a"}),
            _event("request_payload", payload={"request_id": "a"}),
            _envelope(_event("future_event")),
            _envelope(_request_end("c", 2_000, total_ms=10.0, isl=64)),
            _request_end("no-output", 1_100, output_tokens=None),
        ],
    )
    # A request ID repeated in a later file keeps its first row.
    second = _write(tmp_path / "b.jsonl", [_request_end("c", 2_500, isl=256)])

    result, rows = _window([first, second], tmp_path)

    assert result.num_requests == 3
    assert result.trace_block_size == 64
    assert set(result.row_counts) == set(ROW_COUNT_KEYS)
    assert [row["request"]["request_id"] for row in rows] == ["a", "c", "b"]
    for row in rows:
        # No agent_context: AIS replays a row that has one as agentic.
        assert set(row) == {"schema", "event_type", "event_time_unix_ms", "request"}
        assert row["request"]["replay"]["input_sequence_hashes"]
    assert rows[1]["request"]["replay"]["input_length"] == 64
    counts = result.row_counts
    assert (
        counts["agent_context_stripped"],
        counts["missing_output_tokens"],
        counts["duplicate"],
    ) == (1, 1, 1)


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
            # Arrivals are in the half-open window [1_000, 2_000).
            *(
                _request_end(str(at), at, total_ms=0.0)
                for at in (999, 1_000, 1_999, 2_000)
            ),
        ],
    )

    result, rows = _window([trace], tmp_path, start=1_000, end=2_000, as_of=2_000)

    assert [row["request"]["request_id"] for row in rows] == [
        "1000",
        "on-time",
        "converted",
        "1999",
    ]
    assert result.row_counts["censored"] == 2
    assert result.row_counts["censor_unavailable"] == 1


def test_mixed_block_sizes_are_rejected(tmp_path):
    trace = _write(
        tmp_path / "trace.jsonl",
        [_request_end("a", 1_000), _request_end("b", 50_000, block_size=32)],
    )

    # Only the rows inside the window must agree.
    result, _ = _window([trace], tmp_path, end=10_000)
    assert result.trace_block_size == 64
    with pytest.raises(ValueError, match="trace_block_size"):
        _window([trace], tmp_path, end=60_000)


@pytest.mark.parametrize(
    "name, content, error, match",
    [
        (
            "trace.jsonl",
            _jsonl(_request_end("a", 1_000))
            + b"{not json}\n"
            + _jsonl(_request_end("b", 1_100)),
            ValueError,
            "invalid JSON",
        ),
        # A partial last line (a writer cut off mid-row) is not skipped.
        (
            "trace.jsonl",
            _jsonl(_request_end("a", 1_000)) + b'{"schema": "dyn',
            ValueError,
            "invalid JSON",
        ),
        (
            "trace.jsonl",
            _jsonl({"timestamp": 0, "input_length": 1, "output_length": 1}),
            ValueError,
            SCHEMA,
        ),
        # A truncated final gzip member, as the replay's own loader rejects it.
        (
            "trace.jsonl.gz",
            gzip.compress(_jsonl(_request_end("a", 1_000)))[:-8],
            EOFError,
            None,
        ),
    ],
    ids=["middle-line", "partial-final-line", "foreign-schema", "truncated-gzip"],
)
def test_corrupt_input_raises(tmp_path, name, content, error, match):
    trace = tmp_path / name
    trace.write_bytes(content)

    with pytest.raises(error, match=match):
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
    assert counts["agent_context_stripped"] == 0
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


def test_directories_and_globs_resolve_each_file_once(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    for index in range(2):
        _write(
            traces / f"frontend.{index:06d}.jsonl.gz",
            [_request_end(f"r{index}", 1_000 + index)],
        )

    resolved = resolve_trace_files(
        [
            str(traces),
            str(traces / "*.jsonl.gz"),
            str(traces / "frontend.000001.jsonl.gz"),
        ]
    )

    assert [path.name for path in resolved] == [
        "frontend.000000.jsonl.gz",
        "frontend.000001.jsonl.gz",
    ]


def test_history_is_counted_into_every_bucket_set(tmp_path):
    trace = _write(
        tmp_path / "trace.jsonl",
        [
            _request_end("h0", 1_000, isl=100, osl=10),
            _request_end("h1", 2_500, isl=300, osl=30),
            _request_end("h2", 3_500, isl=200, osl=20),
            _request_end("w0", 4_200, isl=50, osl=5),
        ],
    )
    # Bucket sets ending at the window end and at the history end.
    to_window_end = ObservationBuckets(end_ms=5_000, interval_ms=2_000, duration_s=2.0)
    to_history_end = ObservationBuckets(end_ms=4_000, interval_ms=2_000, duration_s=2.0)

    num_requests = read_trace_history(
        [str(trace)], 0, 4_000, 4_000, buckets=(to_window_end, to_history_end)
    )

    assert num_requests == 3
    # [1_000, 3_000) and [3_000, 5_000); w0 is not history.
    assert [(o.num_req, o.isl, o.osl) for o in to_window_end.observations()] == [
        (2.0, 200.0, 20.0),
        (1.0, 200.0, 20.0),
    ]
    # [0, 2_000) and [2_000, 4_000).
    assert [(o.num_req, o.isl, o.osl) for o in to_history_end.observations()] == [
        (1.0, 100.0, 10.0),
        (2.0, 250.0, 25.0),
    ]
    assert read_trace_history([str(trace)], 0, 4_000, 4_000) == 3
