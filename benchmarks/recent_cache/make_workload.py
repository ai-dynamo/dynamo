# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeat a timestamped Mooncake window with shared or private prompt copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def field(row: dict[str, Any], canonical: str, alias: str) -> str:
    keys = [key for key in (canonical, alias) if key in row]
    if len(keys) != 1:
        raise ValueError(f"each row must contain exactly one of {canonical}, {alias}")
    return keys[0]


def read_window(source: Path, count: int, block_size: int) -> tuple:
    rows = []
    source_digest = hashlib.sha256()
    window_digest = hashlib.sha256()
    hash_mapping: dict[int, int] = {}
    prefix_nodes: dict[tuple[int, int], int] = {}
    session_ids: set[str] = set()
    partial_inputs = 0
    with source.open("rb") as handle:
        for line in handle:
            source_digest.update(line)
            if not line.strip() or len(rows) >= count:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("source rows must be JSON objects")
            if any(
                key in row
                for key in (
                    "schema",
                    "dependencies",
                    "agent_context",
                    "wait_for",
                    "tool_wait_ms",
                    "delay",
                    "delay_ms",
                )
            ):
                raise ValueError("agentic/dependency traces are not supported")
            time_key = field(row, "timestamp", "created_time")
            timestamp = number(row[time_key], time_key)
            input_key = field(row, "input_length", "input_tokens")
            output_key = field(row, "output_length", "output_tokens")
            input_length = integer(row[input_key], input_key, 1)
            integer(row[output_key], output_key)
            hashes = row.get("hash_ids")
            if not isinstance(hashes, list):
                raise ValueError("each row requires a hash_ids array")
            if len(hashes) < (input_length + block_size - 1) // block_size:
                raise ValueError(
                    "hash_ids must cover the entire input at trace block size"
                )
            for source_hash in hashes:
                integer(source_hash, "hash_ids entry")
                if source_hash > 2**64 - 1:
                    raise ValueError("hash_ids entries must fit u64")
                hash_mapping.setdefault(source_hash, len(hash_mapping))
            parent = 0
            for source_hash in hashes[: input_length // block_size]:
                key = (parent, source_hash)
                parent = prefix_nodes.setdefault(key, len(prefix_nodes) + 1)
            partial_inputs += int(input_length % block_size != 0)
            for key in ("request_id", "session_id"):
                if row.get(key) is not None and not isinstance(row[key], str):
                    raise ValueError(f"{key} must be a string or null")
            session_id = row.get("session_id")
            if session_id is not None:
                if session_id in session_ids:
                    raise ValueError("multi-turn session traces are not supported")
                session_ids.add(session_id)
            rows.append((row, time_key, timestamp))
            window_digest.update(line)
    if len(rows) != count:
        raise ValueError(f"requested {count} source rows, found {len(rows)}")
    stats = {
        "sha256": source_digest.hexdigest(),
        "selected_rows_sha256": window_digest.hexdigest(),
        "unique_hash_ids": len(hash_mapping),
        "unique_chained_complete_prefix_blocks": len(prefix_nodes),
        "requests_with_partial_final_block": partial_inputs,
    }
    return rows, hash_mapping, stats


def generate(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("first", "copies", "cycles", "trace_block_size", "engine_block_size"):
        integer(vars(args)[name], name, 1)
    period_ms = number(
        number(args.period_seconds, "period_seconds") * 1000, "period_ms"
    )
    copy_offset_ms = number(args.copy_offset_ms, "copy_offset_ms")
    if period_ms == 0:
        raise ValueError("period_seconds must be positive")
    source = args.source.resolve()
    output = args.output_dir.resolve()
    repository = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repository):
        raise ValueError("output-dir must be outside the repository")
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    rows, hash_mapping, stats = read_window(source, args.first, args.trace_block_size)
    start = min(timestamp for _, _, timestamp in rows)
    span = max(timestamp for _, _, timestamp in rows) - start
    number(
        (args.cycles - 1) * period_ms + span + (args.copies - 1) * copy_offset_ms,
        "last arrival",
    )
    if period_ms < span + (args.copies - 1) * copy_offset_ms:
        raise ValueError("period is shorter than the source window plus copy offsets")
    if args.sharing == "private" and len(hash_mapping) * args.copies > 2**32:
        raise ValueError("private hash namespace exceeds u32 replay identities")

    # Sort the overlaid window once; periods never overlap by construction.
    schedule = sorted(
        (timestamp - start + copy * copy_offset_ms, index, copy)
        for index, (_, _, timestamp) in enumerate(rows)
        for copy in range(args.copies)
    )
    output.mkdir(parents=True, exist_ok=False)
    trace_path = output / "trace.jsonl"
    generated_digest = hashlib.sha256()
    with trace_path.open("xb") as handle:
        for cycle in range(args.cycles):
            for timestamp, index, copy in schedule:
                original, time_key, _ = rows[index]
                row = original.copy()
                row[time_key] = cycle * period_ms + timestamp
                for key in ("request_id", "session_id"):
                    if row.get(key) is not None:
                        row[key] = f"cycle{cycle}:copy{copy}:{row[key]}"
                if row.get("request_id") is not None:
                    row["request_id"] = f"row{index}:{row['request_id']}"
                if args.sharing == "private":
                    row["hash_ids"] = [
                        copy * len(hash_mapping) + hash_mapping[source_hash]
                        for source_hash in original["hash_ids"]
                    ]
                encoded = (
                    json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                ).encode()
                handle.write(encoded)
                generated_digest.update(encoded)

    unique_blocks = stats["unique_chained_complete_prefix_blocks"]
    if args.sharing == "private":
        unique_blocks *= args.copies
    manifest = {
        "source": {"path": str(source), "url": args.source_url, **stats},
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in ("source", "source_url", "output_dir")
        },
        "generated": {
            "trace": str(trace_path),
            "sha256": generated_digest.hexdigest(),
            "bytes": trace_path.stat().st_size,
            "requests": args.first * args.copies * args.cycles,
            "source_window_span_ms": span,
            "last_arrival_ms": (args.cycles - 1) * period_ms + schedule[-1][0],
            "unique_chained_complete_prefix_blocks": unique_blocks,
            "approximate_complete_prefix_engine_blocks": unique_blocks
            * args.trace_block_size
            / args.engine_block_size,
        },
        "semantics": {
            "sharing": (
                "shared preserves source hash_ids; private consistently namespaces "
                "each copy; both reuse identities across cycles"
            ),
            "identifiers": (
                "request/session IDs, when present, are distinct per copy and cycle; "
                "requests also include the source-row index"
            ),
            "timing": (
                "source arrivals normalized to zero; original within-window gaps "
                "retained; copies overlaid with offsets; cycles separated by the "
                "configured period"
            ),
            "capacity_estimate": (
                "prefix count is exact for complete chained source blocks, not "
                "distinct raw hash IDs; engine-block conversion is approximate and "
                "excludes partial source blocks, generated output, live requests, "
                "and actual residency/replication"
            ),
        },
    }
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-url", help="optional public provenance URL")
    parser.add_argument("--first", type=int, default=32)
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--period-seconds", type=float, default=60)
    parser.add_argument("--copy-offset-ms", type=float, default=10)
    parser.add_argument("--sharing", choices=("shared", "private"), default="shared")
    parser.add_argument("--trace-block-size", type=int, default=512)
    parser.add_argument(
        "--engine-block-size", type=int, default=64, help="capacity estimate only"
    )
    args = parser.parse_args()
    try:
        manifest = generate(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(manifest["generated"], indent=2))


if __name__ == "__main__":
    main()
