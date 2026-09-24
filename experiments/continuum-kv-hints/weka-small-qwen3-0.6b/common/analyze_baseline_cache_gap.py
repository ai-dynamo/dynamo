# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Localize the gap between WEKA theoretical and observed prefix reuse."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SourceKey = tuple[str, str, int, int | None]


@dataclass
class SourceRequest:
    hashes: list[int]
    stream: str
    block_size: int
    global_hits: int = 0
    stream_hits: int = 0


def prefix_hits(hashes: list[int], seen: set[int]) -> int:
    for index, block_hash in enumerate(hashes):
        if block_hash not in seen:
            return index
    return len(hashes)


def build_source_index(
    dataset: Path,
) -> tuple[dict[SourceKey, SourceRequest], dict[str, dict[str, Any]]]:
    traces = {
        trace["id"]: trace
        for trace in map(json.loads, dataset.read_text().splitlines())
    }
    source: dict[SourceKey, SourceRequest] = {}
    for trace_id, trace in traces.items():
        events: list[tuple[float, int, int, int, SourceKey]] = []
        block_size = int(trace["block_size"])
        for outer_index, item in enumerate(trace["requests"]):
            if item.get("type") == "subagent":
                stream = f"subagent:{item.get('agent_id', outer_index)}"
                for inner_index, request in enumerate(item["requests"]):
                    key = (
                        trace_id,
                        "weka_subagent",
                        outer_index,
                        inner_index,
                    )
                    source[key] = SourceRequest(request["hash_ids"], stream, block_size)
                    events.append((request["t"], outer_index, 1, inner_index, key))
                continue

            key = (trace_id, "weka_main", outer_index, None)
            source[key] = SourceRequest(item["hash_ids"], "main", block_size)
            source[(trace_id, "weka_flat", outer_index, None)] = SourceRequest(
                item["hash_ids"], f"flat:{outer_index}", block_size
            )
            events.append((item["t"], outer_index, 0, 0, key))

        global_seen: set[int] = set()
        stream_seen: dict[str, set[int]] = defaultdict(set)
        for *_, key in sorted(events):
            request = source[key]
            request.global_hits = prefix_hits(request.hashes, global_seen)
            request.stream_hits = prefix_hits(
                request.hashes, stream_seen[request.stream]
            )
            global_seen.update(request.hashes)
            stream_seen[request.stream].update(request.hashes)
            if key[1] == "weka_main":
                flat = source[(key[0], "weka_flat", key[2], None)]
                flat.global_hits = request.global_hits
    return source, traces


def source_key(record: dict[str, Any]) -> SourceKey:
    metadata = record["metadata"]
    return (
        metadata["source_trace_id"],
        metadata["source_kind"],
        int(metadata["source_outer_idx"]),
        metadata.get("source_inner_idx"),
    )


def idle_bucket(seconds: float | None) -> str:
    if seconds is None:
        return "first-turn"
    if seconds < 1:
        return "<1s"
    if seconds < 10:
        return "1-10s"
    if seconds < 60:
        return "10-60s"
    return ">=60s"


def summarize(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    values: Counter[str] = Counter()
    for row in rows:
        values.update(
            requests=1,
            total_blocks=row["total_blocks"],
            theoretical_blocks=row["theoretical_blocks"],
            same_stream_theoretical_blocks=row["same_stream_theoretical_blocks"],
            observed_blocks=row["observed_blocks"],
            positive_gap_blocks=row["positive_gap_blocks"],
            same_stream_positive_gap_blocks=row["same_stream_positive_gap_blocks"],
        )
    total = values["total_blocks"]
    values["theoretical_hit_pct"] = (
        100 * values["theoretical_blocks"] / total if total else 0
    )
    values["observed_hit_pct"] = 100 * values["observed_blocks"] / total if total else 0
    return dict(values)


def summarize_loss_shape(rows: list[dict[str, Any]]) -> dict[str, int]:
    eligible = [row for row in rows if row["theoretical_blocks"] > 0]
    full = [row for row in eligible if row["observed_blocks"] == 0]
    partial = [
        row
        for row in eligible
        if 0 < row["observed_blocks"] < row["theoretical_blocks"]
    ]
    return {
        "eligible_requests": len(eligible),
        "full_prefix_loss_requests": len(full),
        "full_prefix_loss_blocks": sum(row["positive_gap_blocks"] for row in full),
        "partial_prefix_loss_requests": len(partial),
        "partial_prefix_loss_blocks": sum(
            row["positive_gap_blocks"] for row in partial
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source, traces = build_source_index(args.dataset)
    records = [
        json.loads(line) for line in args.profile.read_text().splitlines() if line
    ]
    records_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_session[record["metadata"]["x_correlation_id"]].append(record)

    idle_by_request: dict[str, float | None] = {}
    for session_records in records_by_session.values():
        session_records.sort(key=lambda record: record["metadata"]["turn_index"])
        previous = None
        for record in session_records:
            metadata = record["metadata"]
            idle_by_request[metadata["x_request_id"]] = (
                None
                if previous is None
                else max(
                    0,
                    (
                        metadata["request_start_ns"]
                        - previous["metadata"]["request_end_ns"]
                    )
                    / 1e9,
                )
            )
            previous = record

    rows: list[dict[str, Any]] = []
    for record in records:
        metadata = record["metadata"]
        request = source[source_key(record)]
        observed_tokens = int(
            record["metrics"]["usage_prompt_cache_read_tokens"]["value"]
        )
        observed_blocks = observed_tokens // request.block_size
        trace = traces[metadata["source_trace_id"]]
        outer_index = int(metadata["source_outer_idx"])
        follows_subagent = (
            metadata["source_kind"] == "weka_main"
            and outer_index > 0
            and trace["requests"][outer_index - 1].get("type") == "subagent"
        )
        idle_seconds = idle_by_request[metadata["x_request_id"]]
        rows.append(
            {
                "request_id": metadata["x_request_id"],
                "runtime_session_id": metadata["x_correlation_id"],
                "source_trace_id": metadata["source_trace_id"],
                "source_kind": metadata["source_kind"],
                "source_outer_idx": outer_index,
                "source_inner_idx": metadata.get("source_inner_idx"),
                "turn_index": metadata["turn_index"],
                "idle_seconds": idle_seconds,
                "idle_bucket": idle_bucket(idle_seconds),
                "follows_subagent": follows_subagent,
                "total_blocks": len(request.hashes),
                "theoretical_blocks": request.global_hits,
                "same_stream_theoretical_blocks": request.stream_hits,
                "observed_blocks": observed_blocks,
                "positive_gap_blocks": max(0, request.global_hits - observed_blocks),
                "same_stream_positive_gap_blocks": max(
                    0, request.stream_hits - observed_blocks
                ),
            }
        )

    by_source_kind = {
        key: summarize([row for row in rows if row["source_kind"] == key])
        for key in sorted({row["source_kind"] for row in rows})
    }
    by_idle_bucket = {
        key: summarize([row for row in rows if row["idle_bucket"] == key])
        for key in ("first-turn", "<1s", "1-10s", "10-60s", ">=60s")
    }
    root_after_subagent = [row for row in rows if row["follows_subagent"]]
    summary = {
        "all_requests": summarize(rows),
        "loss_shape": summarize_loss_shape(rows),
        "by_source_kind": by_source_kind,
        "by_idle_bucket": by_idle_bucket,
        "root_requests_following_subagent": summarize(root_after_subagent),
        "root_after_subagent_loss_shape": summarize_loss_shape(root_after_subagent),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline-cache-gap-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (args.output_dir / "baseline-cache-gap-requests.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
