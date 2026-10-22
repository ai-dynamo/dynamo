# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-trace window reader shared by the replay and live retune hosts.

Both hosts read ``dynamo.request.trace.v1`` files (frontend segments live, the
replay's own trace in simulation) and hand AISimulate a clean, bare-row file:

* only ``request_end`` rows; tool, ``router_selection`` and ``request_payload``
  rows are dropped and ``agent_context`` is stripped (open-loop replay);
* request times follow the AISimulate rule: arrival is
  ``request_received_ms``, else ``event_time_unix_ms - round(total_time_ms)``;
  completion is ``request_received_ms + round(total_time_ms)`` when both are
  present, else ``event_time_unix_ms``;
* a row is selected when its arrival is in ``[start_ms, end_ms)`` and its
  completion is at or before ``as_of_ms`` (censoring);
* repeated ``request_id`` values keep the first row; one ``trace_block_size``;
* a truncated final gzip member or a partial final line is tolerated and
  counted; corruption anywhere else raises.

The window file keeps ``input_sequence_hashes`` (inner replays need them for
KV reuse) and is sorted in memory. The history reader streams and writes
hash-free rows: the normalized ``request_received_ms``, ``input_length`` and
``output_tokens``, which is all predictor warmup needs.
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, NamedTuple

from dynamo.planner.core.types import TrafficObservation
from dynamo.planner.offline.trace_data import (
    _reject_duplicate_copies,
    _resolve_trace_paths,
)

TRACE_SCHEMA = "dynamo.request.trace.v1"

# Every row_counts key; values count rows unless noted.
ROW_COUNT_KEYS: tuple[str, ...] = (
    "tool",  # tool_start / tool_end / tool_error rows dropped
    "router_selection",  # router_selection rows dropped
    "request_payload",  # request_payload rows dropped
    "other_event",  # rows with any other event type dropped
    "missing_output_tokens",  # request_end rows without output_tokens dropped
    "outside_window",  # arrival outside [start_ms, end_ms)
    "censored",  # completion after as_of_ms
    "duplicate",  # repeated request_id among selected rows
    "agent_context_stripped",  # kept rows whose agent_context was removed
    "censor_unavailable",  # kept rows without total_time_ms: as_of cannot censor
    "partial_line",  # unparsable final line without a newline, tolerated
    "truncated_gzip",  # files whose final gzip member is truncated (files)
    "pruned_files",  # files skipped by modification time (files)
)

_TOOL_EVENTS = frozenset({"tool_start", "tool_end", "tool_error"})
_DROPPED_EVENTS = {
    "router_selection": "router_selection",
    "request_payload": "request_payload",
}
# ``gzip`` reads a member's 2-byte magic and raises this for a 1-byte tail: a
# writer that has appended only the first byte of the next member.
_PARTIAL_GZIP_MAGIC = "Not a gzipped file (%r)" % b"\x1f"
# Live segment mtimes are stamped by the trace store's clock, request times by
# the frontends'; prune only files older than the window start by this margin.
_MTIME_PRUNE_MARGIN_MS = 300_000.0


class WindowFile(NamedTuple):
    """Result of ``read_trace_window``.

    ``trace_block_size`` is the single embedded block size of the selected
    rows, or ``None`` when no row was selected.
    """

    path: str
    num_requests: int
    trace_block_size: int | None
    row_counts: dict[str, int]


class HistoryFile(NamedTuple):
    """Result of ``read_trace_history``; ``path`` is ``None`` when not written."""

    path: str | None
    num_requests: int
    row_counts: dict[str, int]


@dataclass(frozen=True)
class _RequestRow:
    request_id: str
    arrival_ms: int
    completion_ms: int
    event_time_ms: int
    total_time_ms: float | None
    input_length: int
    output_tokens: int
    trace_block_size: int
    input_sequence_hashes: list[Any] | None
    model: str | None
    had_agent_context: bool


class ObservationBuckets:
    """Per-interval traffic aggregates aligned to end at ``end_ms``.

    Bucket ``j`` covers trace milliseconds ``[end_ms - (j+1)*interval_ms,
    end_ms - j*interval_ms)``. ``observations()`` returns every bucket from the
    earliest non-empty one through the bucket that ends at ``end_ms``, in time
    order; buckets without requests are zero observations, matching
    ``offline.trace_data``. ``duration_s`` is the planner-clock length of one
    bucket (``interval_ms / (1000 * arrival_speedup_ratio)``).
    """

    def __init__(self, end_ms: float, interval_ms: float, duration_s: float) -> None:
        if not (math.isfinite(interval_ms) and interval_ms > 0):
            raise ValueError("interval_ms must be positive and finite")
        if not math.isfinite(end_ms):
            raise ValueError("end_ms must be finite")
        self.end_ms = float(end_ms)
        self.interval_ms = float(interval_ms)
        self.duration_s = float(duration_s)
        self._buckets: dict[int, list[int]] = {}

    def add(self, arrival_ms: float, input_length: int, output_tokens: int) -> None:
        """Count one request; arrivals at or after ``end_ms`` are ignored."""

        if arrival_ms >= self.end_ms:
            return
        # arrival in [end - (j+1)*I, end - j*I)  <=>  j = ceil((end - arrival) / I) - 1
        index = math.ceil((self.end_ms - arrival_ms) / self.interval_ms) - 1
        aggregate = self._buckets.setdefault(index, [0, 0, 0])
        aggregate[0] += 1
        aggregate[1] += input_length
        aggregate[2] += output_tokens

    def bucket_count(self) -> int:
        """Number of observations ``observations()`` returns."""

        return max(self._buckets) + 1 if self._buckets else 0

    def observations(self) -> list[TrafficObservation]:
        result: list[TrafficObservation] = []
        for index in range(self.bucket_count() - 1, -1, -1):
            count, total_isl, total_osl = self._buckets.get(index, (0, 0, 0))
            result.append(
                TrafficObservation(
                    duration_s=self.duration_s,
                    num_req=float(count),
                    isl=total_isl / count if count else 0.0,
                    osl=total_osl / count if count else 0.0,
                )
            )
        return result


def resolve_trace_files(paths: Sequence[str | os.PathLike[str]]) -> list[Path]:
    """Expand trace files, directories, globs and frontend segment prefixes.

    A path that is neither an existing file, a directory nor a glob is read
    as a frontend segment prefix: ``<prefix>.NNNNNN.jsonl.gz`` with any
    ``.jsonl``/``.jsonl.gz`` suffix of the prefix removed. A file listed more
    than once is read once.
    """

    resolved: list[Path] = []
    seen: set[Path] = set()
    for entry in paths:
        text = os.fspath(entry)
        path = Path(text)
        if path.exists() or glob.has_magic(text):
            matches = _resolve_trace_paths(text)
        else:
            prefix = text
            for suffix in (".jsonl.gz", ".jsonl"):
                if prefix.endswith(suffix):
                    prefix = prefix[: -len(suffix)]
                    break
            matches = _reject_duplicate_copies(
                [
                    Path(match)
                    for match in glob.glob(f"{glob.escape(prefix)}.[0-9]*.jsonl.gz")
                ]
            )
            if not matches:
                raise FileNotFoundError(f"no trace file or segment matched {text!r}")
        for match in matches:
            if match not in seen:
                seen.add(match)
                resolved.append(match)
    return resolved


def trace_origin_ms(
    paths: Sequence[str | os.PathLike[str]],
    *,
    row_counts: dict[str, int] | None = None,
) -> float:
    """Earliest AISimulate arrival over the ``request_end`` rows of ``paths``.

    This is trace time at planner-clock zero for a replay of ``paths``:
    AISimulate rebases a Dynamo trace to its first arrival. ``row_counts``,
    when given, receives the read's row counts with every parsed request
    counted as kept, so ``agent_context_stripped`` counts the requests that
    carry ``agent_context`` (an agentic Dynamo trace).
    """

    counts = _new_counts()
    origin: int | None = None
    for path in resolve_trace_files(paths):
        for record, location in _iter_records(path, counts):
            event = _request_end_event(record, location, counts)
            if event is None:
                continue
            row = _parse_request_end(event, location, need_hashes=False)
            if row is None:
                counts["missing_output_tokens"] += 1
                continue
            _count_kept(row, counts)
            if origin is None or row.arrival_ms < origin:
                origin = row.arrival_ms
    if row_counts is not None:
        row_counts.update(counts)
    if origin is None:
        raise ValueError(f"trace has no request_end rows: {list(map(str, paths))}")
    return float(origin)


def read_trace_window(
    paths: Sequence[str | os.PathLike[str]],
    start_ms: float,
    end_ms: float,
    as_of_ms: float,
    out_path: str | os.PathLike[str],
    *,
    prune_by_mtime: bool,
    buckets: ObservationBuckets | None = None,
) -> WindowFile:
    """Write the requests that arrived in ``[start_ms, end_ms)`` to ``out_path``.

    Rows whose completion is after ``as_of_ms`` are censored. The output is a
    bare-row ``dynamo.request.trace.v1`` file sorted by (arrival, completion,
    request_id) with ``request_received_ms`` set to the normalized arrival, so
    AISimulate replays it as is. ``prune_by_mtime`` skips files last modified
    more than five minutes (a clock-skew margin) before ``start_ms``; only live
    frontend segments, written in completion order, may use it. ``buckets``,
    when given, receives every kept row.
    """

    counts = _new_counts()
    selected: dict[str, _RequestRow] = {}
    for row in _iter_selected_rows(
        paths, start_ms, end_ms, as_of_ms, counts, prune_by_mtime, need_hashes=True
    ):
        if row.request_id in selected:
            counts["duplicate"] += 1
            continue
        selected[row.request_id] = row

    rows = sorted(
        selected.values(),
        key=lambda row: (row.arrival_ms, row.completion_ms, row.request_id),
    )
    block_sizes = {row.trace_block_size for row in rows}
    if len(block_sizes) > 1:
        raise ValueError(
            "trace window mixes trace_block_size values "
            f"{sorted(block_sizes)}; replay needs one block size"
        )
    with _open_output(out_path) as sink:
        for row in rows:
            _count_kept(row, counts)
            if buckets is not None:
                buckets.add(row.arrival_ms, row.input_length, row.output_tokens)
            sink.write(_dumps(_window_record(row)))
            sink.write("\n")
    return WindowFile(
        path=os.fspath(out_path),
        num_requests=len(rows),
        trace_block_size=next(iter(block_sizes)) if block_sizes else None,
        row_counts=counts,
    )


def read_trace_history(
    paths: Sequence[str | os.PathLike[str]],
    start_ms: float,
    end_ms: float,
    as_of_ms: float,
    out_path: str | os.PathLike[str] | None,
    *,
    prune_by_mtime: bool,
    buckets: ObservationBuckets | None = None,
) -> HistoryFile:
    """Stream the requests that arrived in ``[start_ms, end_ms)``.

    Selection and request times are those of ``read_trace_window``. Rows are
    never held in memory: each is written to ``out_path`` (when set) as a
    hash-free ``request_end`` row, in input order, and added to ``buckets``
    (when set). Request IDs are not de-duplicated, because that would need
    the whole history's IDs in memory; the window file is de-duplicated.
    """

    counts = _new_counts()
    num_requests = 0
    with _open_output(out_path) if out_path is not None else nullcontext() as sink:
        for row in _iter_selected_rows(
            paths,
            start_ms,
            end_ms,
            as_of_ms,
            counts,
            prune_by_mtime,
            need_hashes=False,
        ):
            num_requests += 1
            _count_kept(row, counts)
            if buckets is not None:
                buckets.add(row.arrival_ms, row.input_length, row.output_tokens)
            if sink is not None:
                sink.write(_dumps(_history_record(row)))
                sink.write("\n")
    return HistoryFile(
        path=None if out_path is None else os.fspath(out_path),
        num_requests=num_requests,
        row_counts=counts,
    )


def _new_counts() -> dict[str, int]:
    return dict.fromkeys(ROW_COUNT_KEYS, 0)


def _count_kept(row: _RequestRow, counts: dict[str, int]) -> None:
    if row.had_agent_context:
        counts["agent_context_stripped"] += 1
    if row.total_time_ms is None:
        counts["censor_unavailable"] += 1


def _open_output(path: str | os.PathLike[str]) -> IO[str]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w", encoding="utf-8")


def _dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, separators=(",", ":"))


def _iter_selected_rows(
    paths: Sequence[str | os.PathLike[str]],
    start_ms: float,
    end_ms: float,
    as_of_ms: float,
    counts: dict[str, int],
    prune_by_mtime: bool,
    *,
    need_hashes: bool,
) -> Iterator[_RequestRow]:
    for path in resolve_trace_files(paths):
        # A file's mtime is at or after every row written to it, and rows
        # complete after they arrive: an older file has no arrival >= start.
        # The margin absorbs skew between the store's and the frontends' clocks.
        if (
            prune_by_mtime
            and path.stat().st_mtime * 1000.0 < start_ms - _MTIME_PRUNE_MARGIN_MS
        ):
            counts["pruned_files"] += 1
            continue
        for record, location in _iter_records(path, counts):
            event = _request_end_event(record, location, counts)
            if event is None:
                continue
            row = _parse_request_end(event, location, need_hashes=need_hashes)
            if row is None:
                counts["missing_output_tokens"] += 1
                continue
            if not start_ms <= row.arrival_ms < end_ms:
                counts["outside_window"] += 1
                continue
            if row.completion_ms > as_of_ms:
                counts["censored"] += 1
                continue
            yield row


def _iter_records(
    path: Path, counts: dict[str, int]
) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield JSON objects, tolerating only a truncated tail.

    A truncated final gzip member ends the file early (``EOFError``, or a
    1-byte member header); an unparsable final line without a newline is a
    write in progress. Both are counted. A malformed line anywhere else, or a
    corrupt gzip stream, raises.
    """

    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as source:
        line_number = 0
        try:
            for raw in source:
                line_number += 1
                if not raw.strip():
                    continue
                location = f"{path}:{line_number}"
                try:
                    record = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    if not raw.endswith(b"\n"):
                        counts["partial_line"] += 1
                        continue
                    raise ValueError(f"invalid JSON at {location}: {error}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"trace row at {location} must be a JSON object")
                yield record, location
        except EOFError:
            counts["truncated_gzip"] += 1
        except gzip.BadGzipFile as error:
            # Only a strict prefix of the next member's magic is a write in
            # progress; any other bad header is corruption.
            if str(error) != _PARTIAL_GZIP_MAGIC:
                raise
            counts["truncated_gzip"] += 1


def _request_end_event(
    record: dict[str, Any], location: str, counts: dict[str, int]
) -> dict[str, Any] | None:
    """Return the ``request_end`` event of a bare or enveloped row, or drop it."""

    event = record.get("event", record)
    if not isinstance(event, dict):
        raise ValueError(f"trace row at {location} has a non-object event")
    if event.get("schema") != TRACE_SCHEMA:
        raise ValueError(
            f"expected {TRACE_SCHEMA} at {location}, got {event.get('schema')!r}"
        )
    event_type = event.get("event_type")
    if event_type == "request_end":
        return event
    if event_type in _TOOL_EVENTS:
        counts["tool"] += 1
    else:
        counts[_DROPPED_EVENTS.get(str(event_type), "other_event")] += 1
    return None


def _parse_request_end(
    event: dict[str, Any], location: str, *, need_hashes: bool
) -> _RequestRow | None:
    """Parse a ``request_end`` event; ``None`` when it has no output_tokens."""

    request = event.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"request_end is missing request metrics at {location}")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError(f"request_id must be a non-empty string at {location}")
    replay = request.get("replay")
    if not isinstance(replay, dict):
        raise ValueError(f"request_end is missing replay metrics at {location}")

    event_time = _uint(event.get("event_time_unix_ms"), "event_time_unix_ms", location)
    received = request.get("request_received_ms")
    if received is not None:
        received = _uint(received, "request_received_ms", location)
    total = request.get("total_time_ms")
    rounded_total: int | None = None
    if total is not None:
        if isinstance(total, bool) or not isinstance(total, (int, float)):
            raise ValueError(f"total_time_ms must be numeric at {location}")
        if not math.isfinite(total) or total < 0:
            raise ValueError(f"total_time_ms must be finite and >= 0 at {location}")
        # Rust f64::round: half away from zero.
        rounded_total = math.floor(float(total) + 0.5)
    if received is not None and rounded_total is not None:
        completion = received + rounded_total
    else:
        completion = event_time
    arrival = (
        received if received is not None else max(event_time - (rounded_total or 0), 0)
    )

    output_tokens = request.get("output_tokens")
    if output_tokens is None:
        return None
    output_tokens = _uint(output_tokens, "output_tokens", location)
    input_length = _uint(replay.get("input_length"), "replay.input_length", location)
    block_size = 0
    hashes = None
    if need_hashes:
        block_size = _uint(
            replay.get("trace_block_size"), "replay.trace_block_size", location
        )
        if block_size == 0:
            raise ValueError(f"replay.trace_block_size must be positive at {location}")
        hashes = replay.get("input_sequence_hashes")
        if not isinstance(hashes, list):
            raise ValueError(
                f"replay.input_sequence_hashes must be a list at {location}"
            )
    model = request.get("model")
    return _RequestRow(
        request_id=request_id,
        arrival_ms=arrival,
        completion_ms=completion,
        event_time_ms=event_time,
        total_time_ms=None if total is None else float(total),
        input_length=input_length,
        output_tokens=output_tokens,
        trace_block_size=block_size,
        input_sequence_hashes=hashes,
        model=model if isinstance(model, str) else None,
        had_agent_context=event.get("agent_context") is not None,
    )


def _uint(value: Any, field: str, location: str) -> int:
    """A non-negative integer; integral floats (converted traces) are accepted."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a non-negative integer at {location}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field} must be a non-negative integer at {location}")
        value = int(value)
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer at {location}")
    return value


def _window_record(row: _RequestRow) -> dict[str, Any]:
    request: dict[str, Any] = {"request_id": row.request_id}
    if row.model is not None:
        request["model"] = row.model
    request["output_tokens"] = row.output_tokens
    request["request_received_ms"] = row.arrival_ms
    if row.total_time_ms is not None:
        request["total_time_ms"] = row.total_time_ms
    request["replay"] = {
        "trace_block_size": row.trace_block_size,
        "input_length": row.input_length,
        "input_sequence_hashes": row.input_sequence_hashes,
    }
    return {
        "schema": TRACE_SCHEMA,
        "event_type": "request_end",
        "event_time_unix_ms": row.event_time_ms,
        "request": request,
    }


def _history_record(row: _RequestRow) -> dict[str, Any]:
    return {
        "schema": TRACE_SCHEMA,
        "event_type": "request_end",
        "event_time_unix_ms": row.event_time_ms,
        "request": {
            "request_id": row.request_id,
            "output_tokens": row.output_tokens,
            "request_received_ms": row.arrival_ms,
            "replay": {"input_length": row.input_length},
        },
    }
