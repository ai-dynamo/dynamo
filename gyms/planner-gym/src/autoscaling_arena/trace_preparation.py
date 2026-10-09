# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible request-count scaling without compressing the arrival timeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import closing
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from autoscaling_arena.workloads.generator import validate_mooncake_trace


@dataclass(frozen=True)
class TracePreparationConfig:
    """Select original-time [start_ms, end_ms), then scale request count.

    The first copy retains original hashes and timestamps. Additional copies
    use independent deterministic hash namespaces. Fractional scaling samples
    individual requests, not sessions. Optional phase jitter is constant per
    copy (or per session within a copy), except for tapering near window edges.
    Tapering preserves bounds and within-session ordering; interior gaps are
    unchanged. Missing session identifiers use a separate phase per source row.
    Hash namespaces belong to one prepared output. To share cache identity
    across windows, prepare the combined interval before slicing it.
    """

    start_ms: float | None = None
    end_ms: float | None = None
    rebase: bool = False
    request_scale: float = 1.0
    seed: int = 0
    jitter_ms: float = 0.0
    session_id_field: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("start_ms", self.start_ms),
            ("end_ms", self.end_ms),
            ("request_scale", self.request_scale),
            ("jitter_ms", self.jitter_ms),
        ):
            if value is None and name in {"start_ms", "end_ms"}:
                continue
            try:
                valid_number = (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value >= 0
                )
            except OverflowError:
                valid_number = False
            if not valid_number:
                raise ValueError(
                    f"trace preparation {name} must be finite and nonnegative"
                )
        if self.request_scale == 0:
            raise ValueError("trace preparation request_scale must be positive")
        if self.end_ms is not None and self.end_ms <= (self.start_ms or 0):
            raise ValueError("trace preparation end_ms must exceed start_ms")
        if not isinstance(self.rebase, bool):
            raise ValueError("trace preparation rebase must be a boolean")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("trace preparation seed must be a nonnegative integer")
        if self.session_id_field is not None and (
            not isinstance(self.session_id_field, str) or not self.session_id_field
        ):
            raise ValueError(
                "trace preparation session_id_field must be a nonempty string"
            )
        if self.session_id_field in {
            "timestamp",
            "input_length",
            "output_length",
            "hash_ids",
        }:
            raise ValueError("session_id_field cannot name a replay metric")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TracePreparationConfig:
        """Reject unknown options instead of silently changing trace semantics."""
        if not isinstance(value, Mapping):
            raise ValueError("trace preparation must be a mapping")
        if not all(isinstance(key, str) for key in value):
            raise ValueError("trace preparation option names must be strings")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown trace preparation options: {sorted(unknown)}")
        return cls(**value)

    @property
    def is_identity(self) -> bool:
        return (
            self.start_ms is None
            and self.end_ms is None
            and not self.rebase
            and self.request_scale == 1
            and self.jitter_ms == 0
        )


@dataclass(frozen=True)
class TracePreparationResult:
    path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(*parts: Any) -> bytes:
    return hashlib.sha256(_canonical(parts)).digest()


def _fraction(*parts: Any) -> float:
    # Use the 53 bits exactly representable in a float, always strictly below 1.
    return (int.from_bytes(_digest(*parts)[:8], "big") >> 11) / (1 << 53)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("rb") as handle:
        for ordinal, line in enumerate(handle):
            if line.strip():
                yield ordinal, json.loads(line)


def _shifted_timestamp(
    timestamp: float, *, phase: float, start: float, end: float, jitter: float
) -> float:
    if not jitter:
        return timestamp
    taper = min(1.0, (timestamp - start) / jitter, (end - timestamp) / jitter)
    return timestamp + phase * max(0.0, taper)


def _scaled_rows(
    source: Path,
    config: TracePreparationConfig,
    *,
    start: float,
    end: float,
    connection: sqlite3.Connection,
) -> Iterator[tuple[float, int, int, bytes]]:
    next_hash = 0

    @lru_cache(maxsize=8192)
    def mapped_hash(copy_index: int, original: int) -> int:
        nonlocal next_hash
        original_bytes = original.to_bytes(8, "big")
        existing = connection.execute(
            "SELECT mapped FROM namespaces WHERE copy = ? AND original = ?",
            (copy_index, original_bytes),
        ).fetchone()
        if existing is not None:
            return int.from_bytes(existing[0], "big")
        while next_hash < 1 << 64:
            candidate = next_hash.to_bytes(8, "big")
            next_hash += 1
            inserted = connection.execute(
                "INSERT OR IGNORE INTO reserved_hashes VALUES (?)", (candidate,)
            ).rowcount
            if inserted:
                connection.execute(
                    "INSERT INTO namespaces VALUES (?, ?, ?)",
                    (copy_index, original_bytes, candidate),
                )
                return int.from_bytes(candidate, "big")
        raise ValueError("trace preparation exhausted the u64 hash namespace")

    @lru_cache(maxsize=8192)
    def phase_shift(copy_index: int, session_key: tuple | None) -> float:
        if config.jitter_ms == 0:
            return 0.0
        return config.jitter_ms * (
            2 * _fraction("phase", config.seed, copy_index, session_key) - 1
        )

    full_copies = math.floor(config.request_scale)
    fraction = config.request_scale - full_copies
    for ordinal, record in _rows(source):
        timestamp = record["timestamp"]
        if timestamp < start or (config.end_ms is not None and timestamp >= end):
            continue
        copies = full_copies + (
            fraction > 0 and _fraction("thinning", config.seed, ordinal) < fraction
        )
        session = None
        if config.session_id_field is not None:
            session = record.get(config.session_id_field)
            if session is not None and (
                isinstance(session, bool) or not isinstance(session, (str, int))
            ):
                raise ValueError("session identifier must be a string or integer")
        phase_key = ("row", ordinal) if session is None else ("session", session)
        for copy_index in range(copies):
            transformed = dict(record)
            adjusted_timestamp = timestamp
            if copy_index:
                transformed["hash_ids"] = [
                    mapped_hash(copy_index, item) for item in record["hash_ids"]
                ]
                for field in {"request_id", config.session_id_field} - {None}:
                    if field in record:
                        transformed[field] = _digest(
                            "identifier", config.seed, copy_index, record[field]
                        ).hex()
                phase = phase_shift(
                    copy_index, phase_key if config.session_id_field else None
                )
                adjusted_timestamp = _shifted_timestamp(
                    timestamp,
                    phase=phase,
                    start=start,
                    end=end,
                    jitter=config.jitter_ms,
                )
            if config.rebase:
                adjusted_timestamp -= start
            transformed["timestamp"] = adjusted_timestamp
            yield (
                adjusted_timestamp,
                ordinal,
                copy_index,
                _canonical(transformed) + b"\n",
            )


def prepare_trace(
    source: Path,
    output: Path,
    *,
    config: TracePreparationConfig,
    block_size: int,
) -> TracePreparationResult:
    """Write a sorted replay JSONL and a path-redacted provenance sidecar.

    Input must already be sorted. The source is never modified and existing
    output files are never overwritten. Identity preparation copies bytes
    exactly. Active transformations use a disk-backed sort with an 8 MiB SQLite
    page cache; memory does not grow with total request count or scale. Budget
    temporary disk space for the scaled trace, sort and final output.

    The declared measurement window is retained in the manifest even when
    thinning removes its first or last arrival. An omitted end is inferred
    from the last source arrival, not a claim about trailing idle time.
    """
    if not isinstance(config, TracePreparationConfig):
        raise ValueError("config must be a TracePreparationConfig")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("trace block size must be a positive integer")
    manifest_path = output.with_name(output.name + ".manifest.json")
    if source.resolve() in {output.resolve(), manifest_path.resolve()}:
        raise ValueError("trace preparation cannot overwrite its source")
    for path in (output, manifest_path):
        if os.path.lexists(path):
            raise FileExistsError(path)
    source_sha = _sha256(source)
    source_count = validate_mooncake_trace(
        source, block_size=block_size, presorted=True
    )
    first_ms = last_ms = None
    selected_count = 0
    for _, record in _rows(source):
        timestamp = record["timestamp"]
        if first_ms is None:
            first_ms = timestamp
        last_ms = timestamp
        if (config.start_ms is None or timestamp >= config.start_ms) and (
            config.end_ms is None or timestamp < config.end_ms
        ):
            selected_count += 1
    if not selected_count or first_ms is None or last_ms is None:
        raise ValueError("trace preparation window contains no requests")
    start = config.start_ms if config.start_ms is not None else first_ms
    end = config.end_ms if config.end_ms is not None else last_ms
    if end < start:
        raise ValueError("trace preparation window ends before it starts")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".trace-preparation-", dir=output.parent
    ) as temp:
        staging = Path(temp)
        staged_output = staging / "trace.jsonl"
        output_first_ms = output_last_ms = None
        if config.is_identity:
            shutil.copyfile(source, staged_output)
            output_count = source_count
            output_first_ms, output_last_ms = first_ms, last_ms
        else:
            output_count = 0
            with closing(sqlite3.connect(staging / "sort.sqlite3")) as connection:
                connection.execute("PRAGMA cache_size = -8192")
                connection.execute("PRAGMA temp_store = FILE")
                connection.execute(
                    "CREATE TABLE rows (timestamp REAL, ordinal INTEGER, copy INTEGER, body BLOB)"
                )
                connection.execute(
                    "CREATE TABLE reserved_hashes (value BLOB PRIMARY KEY) WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE namespaces (copy INTEGER, original BLOB, mapped BLOB, "
                    "PRIMARY KEY (copy, original)) WITHOUT ROWID"
                )
                for _, record in _rows(source):
                    for value in record["hash_ids"]:
                        if not 0 <= value < 1 << 64:
                            raise ValueError(
                                "trace hashes must fit in an unsigned 64-bit integer"
                            )
                    connection.executemany(
                        "INSERT OR IGNORE INTO reserved_hashes VALUES (?)",
                        ((value.to_bytes(8, "big"),) for value in record["hash_ids"]),
                    )
                connection.executemany(
                    "INSERT INTO rows VALUES (?, ?, ?, ?)",
                    _scaled_rows(
                        source, config, start=start, end=end, connection=connection
                    ),
                )
                connection.commit()
                with staged_output.open("wb") as handle:
                    for timestamp, body in connection.execute(
                        "SELECT timestamp, body FROM rows ORDER BY timestamp, ordinal, copy"
                    ):
                        if output_first_ms is None:
                            output_first_ms = timestamp
                        output_last_ms = timestamp
                        output_count += 1
                        handle.write(body)
            if not output_count:
                raise ValueError("trace preparation thinning selected no requests")
        if _sha256(source) != source_sha:
            raise ValueError("source changed during trace preparation")
        output_sha = _sha256(staged_output)
        if config.is_identity and output_sha != source_sha:
            raise ValueError("source changed during trace preparation")
        preparation = asdict(config)
        manifest = {
            "schema": "autoscaling_arena.trace_preparation.v1",
            "config": preparation,
            "config_sha256": hashlib.sha256(_canonical(preparation)).hexdigest(),
            "block_size": block_size,
            "hash_namespace_scope": "prepared_output",
            "source": {
                "sha256": source_sha,
                "request_count": source_count,
                "first_timestamp_ms": first_ms,
                "last_timestamp_ms": last_ms,
            },
            "window": {
                "source_start_ms": start,
                "source_end_ms": end,
                "end_exclusive": config.end_ms is not None,
                "duration_ms": end - start,
                "output_start_ms": 0 if config.rebase else start,
                "output_end_ms": end - start if config.rebase else end,
                "selected_source_requests": selected_count,
            },
            "output": {
                "sha256": output_sha,
                "request_count": output_count,
                "first_timestamp_ms": output_first_ms,
                "last_timestamp_ms": output_last_ms,
            },
        }
        staged_manifest = staging / "manifest.json"
        staged_manifest.write_bytes(_canonical(manifest) + b"\n")
        # Hard-link publication is atomic and refuses to replace an existing
        # destination, including one created after our initial existence check.
        os.link(staged_output, output)
        try:
            os.link(staged_manifest, manifest_path)
        except OSError:
            output.unlink()
            raise
    return TracePreparationResult(output, manifest_path, manifest)
