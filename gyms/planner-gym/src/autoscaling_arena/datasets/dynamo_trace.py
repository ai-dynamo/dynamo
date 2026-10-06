# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, dependency-free helpers for ``dynamo.request.trace.v1`` datasets."""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

TRACE_SCHEMA = "dynamo.request.trace.v1"


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    received_ms: int
    end_ms: int
    input_length: int
    output_length: int
    trace_block_size: int
    input_sequence_hashes: tuple[int, ...]
    session_id: str | None = None
    trajectory_id: str | None = None
    session_type_id: str | None = None


@dataclass(frozen=True)
class TraceEvent:
    event_type: str
    timestamp_ms: int
    request: RequestMetrics | None

    @property
    def is_request(self) -> bool:
        return self.request is not None


def open_trace(path: Path) -> BinaryIO:
    """Open an uncompressed or gzip-compressed JSONL shard as bytes."""
    return gzip.open(path, "rb") if path.name.endswith(".gz") else path.open("rb")


def _loads(line: bytes, location: str) -> dict:
    try:
        value = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid JSON at {location}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"trace row is not an object at {location}")
    return value


def parse_trace_event(line: bytes, location: str) -> TraceEvent | None:
    """Parse one trace row.

    Verification probes have no replay timestamp and are ignored. Request rows
    receive full replay-field validation; other v1 events are retained by the
    splitter using ``event_time_unix_ms``.
    """
    record = _loads(line, location)
    if set(record) == {"verification"}:
        return None

    event = record.get("event", record)
    if not isinstance(event, dict) or event.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"unsupported trace schema at {location}")
    event_type = event.get("event_type")
    if not isinstance(event_type, str) or not event_type:
        raise ValueError(f"event_type is missing at {location}")

    if event_type != "request_end":
        timestamp_ms = event.get("event_time_unix_ms")
        if not isinstance(timestamp_ms, int):
            raise ValueError(f"event_time_unix_ms is missing at {location}")
        return TraceEvent(event_type, timestamp_ms, None)

    request = event.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"request_end is missing request payload at {location}")
    replay = request.get("replay")
    if not isinstance(replay, dict):
        raise ValueError(f"request payload is missing replay metrics at {location}")

    request_id = request.get("request_id")
    received_ms = request.get("request_received_ms")
    output_length = request.get("output_tokens")
    input_length = replay.get("input_length")
    block_size = replay.get("trace_block_size")
    hashes = replay.get("input_sequence_hashes")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(f"request_id is missing at {location}")
    if not isinstance(received_ms, int):
        raise ValueError(f"request_received_ms is missing at {location}")
    if not isinstance(output_length, int) or output_length < 0:
        raise ValueError(f"invalid output_tokens at {location}")
    if not isinstance(input_length, int) or input_length < 0:
        raise ValueError(f"invalid replay input_length at {location}")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"invalid replay trace_block_size at {location}")
    if not isinstance(hashes, list) or not all(
        isinstance(value, int) for value in hashes
    ):
        raise ValueError(f"input_sequence_hashes is missing or invalid at {location}")
    expected_hashes = math.ceil(input_length / block_size)
    if len(hashes) != expected_hashes:
        raise ValueError(
            f"input_length {input_length} with block size {block_size} requires "
            f"{expected_hashes} hashes, got {len(hashes)} at {location}"
        )

    total_time_ms = request.get("total_time_ms")
    if isinstance(total_time_ms, (int, float)):
        duration_ms = math.floor(max(0.0, float(total_time_ms)) + 0.5)
    else:
        event_time_ms = event.get("event_time_unix_ms")
        if not isinstance(event_time_ms, int):
            raise ValueError(f"event_time_unix_ms is missing at {location}")
        duration_ms = max(0, event_time_ms - received_ms)

    agent_context = event.get("agent_context")
    if agent_context is not None and not isinstance(agent_context, dict):
        raise ValueError(f"agent_context is invalid at {location}")

    def optional_context_id(field: str) -> str | None:
        if agent_context is None or field not in agent_context:
            return None
        value = agent_context[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"agent_context.{field} is invalid at {location}")
        return value

    metrics = RequestMetrics(
        request_id=request_id,
        received_ms=received_ms,
        end_ms=received_ms + duration_ms,
        input_length=input_length,
        output_length=output_length,
        trace_block_size=block_size,
        input_sequence_hashes=tuple(hashes),
        session_id=optional_context_id("session_id"),
        trajectory_id=optional_context_id("trajectory_id"),
        session_type_id=optional_context_id("session_type_id"),
    )
    return TraceEvent(event_type, received_ms, metrics)


def discover_trace_shards(
    root: Path, *, prefer_uncompressed: bool = False
) -> list[Path]:
    """Discover each logical JSONL shard exactly once.

    A dataset may contain both ``trace-X.jsonl`` and ``trace-X.jsonl.gz``.
    Analysis can prefer the uncompressed copy; replay consumers can prefer gzip.
    """
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)

    by_logical_path: dict[str, dict[str, Path]] = {}
    for path in root.rglob("*.jsonl"):
        by_logical_path.setdefault(str(path), {})["plain"] = path
    for path in root.rglob("*.jsonl.gz"):
        logical = str(path)[:-3]
        by_logical_path.setdefault(logical, {})["gzip"] = path

    selected = []
    for variants in by_logical_path.values():
        if prefer_uncompressed:
            selected.append(variants.get("plain") or variants["gzip"])
        else:
            selected.append(variants.get("gzip") or variants["plain"])
    return sorted(selected)


def iter_trace_lines(path: Path) -> Iterator[tuple[int, bytes]]:
    with open_trace(path) as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                yield line_number, line


def compressed_source(path: Path) -> Path:
    """Return the gzip artifact corresponding to a scanned shard."""
    if path.name.endswith(".jsonl.gz"):
        return path
    candidate = path.with_name(f"{path.name}.gz")
    if not candidate.is_file():
        raise FileNotFoundError(f"missing compressed source shard {candidate}")
    return candidate
