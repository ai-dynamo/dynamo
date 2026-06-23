# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Iterator, TextIO


@dataclass
class ConversionStats:
    sessions: int = 0
    requests: int = 0
    failed_spans: int = 0
    non_llm_spans: int = 0
    zero_token_spans: int = 0
    overlapping_spans: int = 0
    prefix_resets: int = 0


def canonical_model(value: str) -> str:
    lowered = value.casefold()
    for prefix in ("openai/azure/", "azure/", "aws/", "gcp/"):
        if lowered.startswith(prefix):
            return value[len(prefix) :]
    return value


def _timestamp_ms(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp() * 1000.0


def _parquet_files(trace_file: str | Path) -> list[Path]:
    path = Path(trace_file)
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
    elif path.suffix == ".parquet" and path.is_file():
        files = [path]
    else:
        raise ValueError(f"expected a Parquet file or directory, got {path}")
    if not files:
        raise ValueError(f"no Parquet files found under {path}")
    return files


def iter_parquet_rows(trace_file: str | Path) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "trace_format='exgentic' requires pyarrow; install ai-dynamo[replay] "
            "or run `uv pip install pyarrow`"
        ) from error

    columns = [
        "harness",
        "models",
        "session_id",
        "spans.list.element.start_time",
        "spans.list.element.end_time",
        "spans.list.element.attributes.gen_ai.usage.input_tokens",
        "spans.list.element.attributes.gen_ai.usage.output_tokens",
        "spans.list.element.status.code",
        "spans.list.element.type",
    ]
    for path in _parquet_files(trace_file):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pylist()


def convert_rows(
    rows: Iterable[dict[str, Any]],
    output: TextIO,
    block_size: int,
    harness: str | None = None,
    model: str | None = None,
) -> ConversionStats:
    if block_size <= 0:
        raise ValueError("trace_block_size must be positive")

    stats = ConversionStats()
    seen_sessions: set[str] = set()
    combinations: set[tuple[str, str]] = set()
    harness_filter = harness.casefold() if harness else None
    model_filter = canonical_model(model).casefold() if model else None
    next_hash_id = 1

    for row_index, row in enumerate(rows, 1):
        row_harness = row.get("harness")
        session_id = row.get("session_id")
        if not isinstance(row_harness, str) or not row_harness:
            raise ValueError(f"row {row_index} has no harness")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"row {row_index} has no session_id")

        row_models = {
            canonical_model(value)
            for value in row.get("models") or []
            if isinstance(value, str) and value
        }
        combinations.update((row_harness, value) for value in row_models)
        if harness_filter and row_harness.casefold() != harness_filter:
            continue
        if model_filter and not any(
            value.casefold() == model_filter for value in row_models
        ):
            continue
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
        available = ", ".join(
            f"{available_harness}/{available_model}"
            for available_harness, available_model in sorted(combinations)
        )
        raise ValueError(
            f"no replayable Exgentic spans matched harness={harness!r}, model={model!r}; "
            f"available combinations: {available}"
        )
    return stats


@contextmanager
def prepare_trace(
    trace_file: str | Path,
    block_size: int,
    harness: str | None = None,
    model: str | None = None,
) -> Iterator[Path]:
    with TemporaryDirectory(prefix="dynamo-exgentic-") as directory:
        output_path = Path(directory) / "trace.mooncake.jsonl"
        with output_path.open("w", encoding="utf-8") as output:
            convert_rows(
                iter_parquet_rows(trace_file),
                output,
                block_size,
                harness=harness,
                model=model,
            )
        yield output_path
