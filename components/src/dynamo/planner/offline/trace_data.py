# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

TraceFormat = Literal["mooncake", "dynamo"]
TraceRow = Tuple[float, int, int]


def _reject_duplicate_copies(paths: Sequence[Path]) -> List[Path]:
    selected: Dict[str, Path] = {}
    for path in paths:
        key = str(path)[:-3] if path.name.endswith(".jsonl.gz") else str(path)
        current = selected.get(key)
        if current is not None and current != path:
            raise ValueError(
                f"Both compressed and uncompressed warmup trace copies matched: "
                f"{current} and {path}. Select one representation with a glob."
            )
        selected[key] = path
    return sorted(selected.values())


def _resolve_trace_paths(dataset: str) -> List[Path]:
    path = Path(dataset)
    if path.is_dir():
        paths = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and (
                candidate.name.endswith(".jsonl")
                or candidate.name.endswith(".jsonl.gz")
            )
        )
    elif glob.has_magic(dataset):
        paths = sorted(Path(match) for match in glob.glob(dataset, recursive=True))
        paths = [candidate for candidate in paths if candidate.is_file()]
    else:
        paths = [path]

    paths = _reject_duplicate_copies(paths)
    if not paths:
        raise ValueError(f"No warmup trace files matched {dataset!r}")
    return paths


def _open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _load_json_line(path: Path, line_number: int, line: str) -> Dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"Trace row at {path}:{line_number} must be a JSON object")
    return record


def _detect_record_format(record: Dict[str, Any]) -> Optional[TraceFormat]:
    event = record.get("event", record)
    if isinstance(event, dict) and event.get("schema") == "dynamo.request.trace.v1":
        return "dynamo"
    if all(field in record for field in ("timestamp", "input_length", "output_length")):
        return "mooncake"
    return None


def detect_trace_format(dataset: str) -> Optional[TraceFormat]:
    """Detect a Mooncake or Dynamo request-trace-v1 warmup dataset.

    ``dataset`` may be a file, a glob, or a directory containing JSONL/JSONL.GZ
    shards. Empty datasets return ``None``.
    """
    for path in _resolve_trace_paths(dataset):
        with _open_trace(path) as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = _load_json_line(path, line_number, line)
                trace_format = _detect_record_format(record)
                if trace_format is None:
                    raise ValueError(
                        f"Unsupported warmup trace format at {path}:{line_number}"
                    )
                return trace_format
    return None


def _number(record: Dict[str, Any], field: str, location: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric at {location}")
    return float(value)


def _length(record: Dict[str, Any], field: str, location: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer at {location}")
    return value


def _parse_mooncake_row(record: Dict[str, Any], location: str) -> TraceRow:
    return (
        _number(record, "timestamp", location),
        _length(record, "input_length", location),
        _length(record, "output_length", location),
    )


def _parse_dynamo_row(record: Dict[str, Any], location: str) -> Optional[TraceRow]:
    event = record.get("event", record)
    if not isinstance(event, dict) or event.get("schema") != "dynamo.request.trace.v1":
        raise ValueError(f"Expected dynamo.request.trace.v1 at {location}")
    if event.get("event_type") != "request_end":
        return None

    request = event.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"request_end is missing request payload at {location}")
    replay = request.get("replay")
    if not isinstance(replay, dict):
        raise ValueError(f"request payload is missing replay metrics at {location}")
    return (
        _number(request, "request_received_ms", location),
        _length(replay, "input_length", location),
        _length(request, "output_tokens", location),
    )


def _iter_trace_rows(
    paths: Sequence[Path], trace_format: TraceFormat
) -> Iterator[TraceRow]:
    for path in paths:
        with _open_trace(path) as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                location = f"{path}:{line_number}"
                record = _load_json_line(path, line_number, line)
                actual_format = _detect_record_format(record)
                if actual_format != trace_format:
                    raise ValueError(
                        f"Mixed or unsupported warmup trace format at {location}: "
                        f"expected {trace_format}, found {actual_format or 'unknown'}"
                    )
                row = (
                    _parse_mooncake_row(record, location)
                    if trace_format == "mooncake"
                    else _parse_dynamo_row(record, location)
                )
                if row is not None:
                    yield row


def _extract_metrics(
    paths: Sequence[Path],
    trace_format: TraceFormat,
    throughput_adjustment_interval_seconds: int,
) -> List[Dict[str, Any]]:
    if throughput_adjustment_interval_seconds <= 0:
        raise ValueError("throughput_adjustment_interval_seconds must be positive")

    # Mooncake timestamps are already relative to the trace start. Dynamo v1
    # carries absolute request timestamps, so find the global origin first.
    # This extra streaming pass keeps direct-v1 warmup equivalent to lowering
    # the same requests to Mooncake without retaining a large trace in memory.
    origin_ms = 0.0
    if trace_format == "dynamo":
        origin_ms = min(
            (
                timestamp_ms
                for timestamp_ms, _, _ in _iter_trace_rows(paths, trace_format)
            ),
            default=0.0,
        )

    interval_ms = throughput_adjustment_interval_seconds * 1000
    interval_groups: Dict[int, List[int]] = defaultdict(lambda: [0, 0, 0])
    for timestamp_ms, input_length, output_length in _iter_trace_rows(
        paths, trace_format
    ):
        interval_index = int((timestamp_ms - origin_ms) // interval_ms)
        interval_start = interval_index * throughput_adjustment_interval_seconds
        aggregate = interval_groups[interval_start]
        aggregate[0] += 1
        aggregate[1] += input_length
        aggregate[2] += output_length

    metrics: List[Dict[str, Any]] = []
    if not interval_groups:
        return metrics

    sorted_starts = sorted(interval_groups)
    for interval_start in range(
        sorted_starts[0],
        sorted_starts[-1] + 1,
        throughput_adjustment_interval_seconds,
    ):
        request_count, total_isl, total_osl = interval_groups.get(
            interval_start, (0, 0, 0)
        )
        metrics.append(
            {
                "interval_start": interval_start,
                "request_count": request_count,
                "avg_isl": total_isl / request_count if request_count else 0,
                "avg_osl": total_osl / request_count if request_count else 0,
            }
        )

    return metrics


def extract_metrics_from_trace(
    dataset: str, throughput_adjustment_interval_seconds: int
) -> List[Dict[str, Any]]:
    """Extract warmup metrics from auto-detected Mooncake or Dynamo v1 traces.

    Dynamo v1 inputs may be plain or gzip-compressed and may be supplied as one
    file, a glob, or a directory of shards. Non-request events are ignored.
    """
    paths = _resolve_trace_paths(dataset)
    trace_format = detect_trace_format(dataset)
    if trace_format is None:
        return []
    return _extract_metrics(paths, trace_format, throughput_adjustment_interval_seconds)


def extract_metrics_from_mooncake(
    dataset: str, throughput_adjustment_interval_seconds: int
) -> List[Dict[str, Any]]:
    """Extract warmup metrics from a Mooncake-style JSONL trace.

    Kept for compatibility with callers that explicitly require Mooncake. New
    planner warmup code should call :func:`extract_metrics_from_trace`.
    """
    paths = _resolve_trace_paths(dataset)
    trace_format = detect_trace_format(dataset)
    if trace_format is None:
        return []
    if trace_format != "mooncake":
        raise ValueError(f"Expected Mooncake warmup trace, detected {trace_format}")
    return _extract_metrics(paths, trace_format, throughput_adjustment_interval_seconds)
