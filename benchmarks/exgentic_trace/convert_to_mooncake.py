# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO


@dataclass
class ConversionStats:
    sessions: int = 0
    requests: int = 0
    failed_spans: int = 0
    non_llm_spans: int = 0
    zero_token_spans: int = 0
    overlapping_spans: int = 0
    prefix_resets: int = 0


def _timestamp_ms(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp() * 1000.0


def _parquet_files(inputs: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.parquet")))
        elif path.suffix == ".parquet" and path.is_file():
            files.append(path)
        else:
            raise ValueError(f"expected a Parquet file or directory, got {path}")
    if not files:
        raise ValueError("no Parquet files found")
    return files


def iter_parquet_rows(inputs: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required; run with `uv run --with pyarrow python ...`"
        ) from error

    for path in _parquet_files(inputs):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=64, columns=["session_id", "spans"]
        ):
            yield from batch.to_pylist()


def convert_rows(
    rows: Iterable[dict[str, Any]], output: TextIO, block_size: int
) -> ConversionStats:
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    stats = ConversionStats()
    seen_sessions: set[str] = set()
    next_hash_id = 1

    for row_index, row in enumerate(rows, 1):
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"row {row_index} has no session_id")
        if session_id in seen_sessions:
            raise ValueError(f"duplicate session_id {session_id!r}")
        seen_sessions.add(session_id)

        spans = []
        for span in row.get("spans") or []:
            if span.get("type") != "llm_call":
                stats.non_llm_spans += 1
                continue
            if (span.get("status") or {}).get("code") == 2:
                stats.failed_spans += 1
                continue

            attributes = span.get("attributes") or {}
            input_tokens = attributes.get("gen_ai.usage.input_tokens")
            output_tokens = attributes.get("gen_ai.usage.output_tokens")
            if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
                raise ValueError(
                    f"session {session_id!r} has an LLM span without integer token counts"
                )
            if input_tokens <= 0 or output_tokens <= 0:
                stats.zero_token_spans += 1
                continue

            start_ms = _timestamp_ms(span["start_time"])
            end_ms = _timestamp_ms(span["end_time"])
            if end_ms < start_ms:
                raise ValueError(
                    f"session {session_id!r} has a span ending before it starts"
                )
            spans.append((start_ms, end_ms, input_tokens, output_tokens))

        spans.sort()
        if not spans:
            continue

        stats.sessions += 1
        hash_ids: list[int] = []
        previous_input_tokens = 0
        previous_end_ms: float | None = None

        for start_ms, end_ms, input_tokens, output_tokens in spans:
            if input_tokens < previous_input_tokens:
                hash_ids = []
                stats.prefix_resets += 1
            while len(hash_ids) * block_size < input_tokens:
                hash_ids.append(next_hash_id)
                next_hash_id += 1

            mooncake_row: dict[str, Any] = {
                "session_id": session_id,
                "input_length": input_tokens,
                "output_length": output_tokens,
                "hash_ids": hash_ids.copy(),
            }
            if previous_end_ms is None:
                mooncake_row["timestamp"] = start_ms
            else:
                delay_ms = start_ms - previous_end_ms
                if delay_ms < 0:
                    stats.overlapping_spans += 1
                    delay_ms = 0.0
                mooncake_row["delay"] = delay_ms

            output.write(json.dumps(mooncake_row, separators=(",", ":")) + "\n")
            stats.requests += 1
            previous_input_tokens = input_tokens
            previous_end_ms = end_ms

    if stats.requests == 0:
        raise ValueError("input did not contain any replayable LLM spans")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Exgentic agent-llm-traces Parquet shards to Mooncake JSONL"
    )
    parser.add_argument("input", nargs="+", help="Parquet file or directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args(argv)

    try:
        with args.output.open("w", encoding="utf-8") as output:
            stats = convert_rows(iter_parquet_rows(args.input), output, args.block_size)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(
        f"Wrote {stats.requests} requests across {stats.sessions} sessions to {args.output}",
        file=sys.stderr,
    )
    print(
        f"Skipped: {stats.failed_spans} failed, {stats.non_llm_spans} non-LLM, "
        f"{stats.zero_token_spans} zero-token; clamped {stats.overlapping_spans} "
        f"overlaps; reset {stats.prefix_resets} prefixes",
        file=sys.stderr,
    )
    print(f"Trace block size: {args.block_size}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
