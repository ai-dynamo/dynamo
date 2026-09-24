#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

SourceKey = tuple[str, str, int, int | None]


@dataclass(frozen=True)
class Candidate:
    source_trace_id: str
    source_kind: str
    source_outer_idx: int
    source_inner_idx: int | None
    request_end_seconds: float
    future_ttl_seconds: float
    prefix_blocks: int
    cached_blocks: int
    missing_blocks: int
    removal_pressure_5s: float

    @property
    def source_key(self) -> SourceKey:
        return (
            self.source_trace_id,
            self.source_kind,
            self.source_outer_idx,
            self.source_inner_idx,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--server-metrics", type=Path, required=True)
    parser.add_argument("--min-missing-blocks", type=int, default=24)
    parser.add_argument("--min-removal-pressure-5s", type=float, default=4.0)
    parser.add_argument("--fixed-ttl-seconds", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def source_key(record: dict) -> SourceKey:
    metadata = record["metadata"]
    return (
        metadata["source_trace_id"],
        metadata["source_kind"],
        int(metadata["source_outer_idx"]),
        metadata.get("source_inner_idx"),
    )


def load_source_paths(dataset: Path) -> dict[SourceKey, list[int]]:
    paths: dict[SourceKey, list[int]] = {}
    for trace in map(json.loads, dataset.read_text().splitlines()):
        trace_id = trace["id"]
        for outer_index, item in enumerate(trace["requests"]):
            if item.get("type") == "subagent":
                for inner_index, request in enumerate(item["requests"]):
                    paths[(trace_id, "weka_subagent", outer_index, inner_index)] = list(
                        request["hash_ids"]
                    )
            else:
                path = list(item["hash_ids"])
                paths[(trace_id, "weka_main", outer_index, None)] = path
                paths[(trace_id, "weka_flat", outer_index, None)] = path
    return paths


def load_removal_pressure(server_metrics: Path) -> list[tuple[float, float]]:
    payload = json.loads(server_metrics.read_text())
    series = payload["metrics"]["dynamo_component_kv_cache_events_applied"]["series"]
    by_end: defaultdict[float, float] = defaultdict(float)
    for item in series:
        labels = item["labels"]
        if labels.get("event_type") != "removed" or labels.get("status") != "ok":
            continue
        for timeslice in item["timeslices"]:
            by_end[int(timeslice["end_ns"]) / 1e9] += float(timeslice["rate"])
    return sorted(by_end.items())


def trailing_mean(
    samples: list[tuple[float, float]], now: float, window_seconds: float
) -> float:
    values = [rate for end, rate in samples if now - window_seconds < end <= now]
    return sum(values) / len(values) if values else 0.0


def load_candidates(
    dataset: Path, profile: Path, server_metrics: Path
) -> list[Candidate]:
    paths = load_source_paths(dataset)
    pressure = load_removal_pressure(server_metrics)
    records = [json.loads(line) for line in profile.read_text().splitlines() if line]
    by_session: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_session[record["metadata"]["x_correlation_id"]].append(record)

    candidates: list[Candidate] = []
    for session_records in by_session.values():
        session_records.sort(key=lambda item: int(item["metadata"]["turn_index"]))
        for current, following in zip(session_records, session_records[1:]):
            key = source_key(current)
            end = int(current["metadata"]["request_end_ns"]) / 1e9
            next_start = int(following["metadata"]["request_start_ns"]) / 1e9
            prefix_blocks = len(paths[key])
            cached_tokens = int(
                current["metrics"]["usage_prompt_cache_read_tokens"]["value"]
            )
            cached_blocks = min(prefix_blocks, cached_tokens // 64)
            candidates.append(
                Candidate(
                    source_trace_id=key[0],
                    source_kind=key[1],
                    source_outer_idx=key[2],
                    source_inner_idx=key[3],
                    request_end_seconds=end,
                    future_ttl_seconds=max(0.001, next_start - end),
                    prefix_blocks=prefix_blocks,
                    cached_blocks=cached_blocks,
                    missing_blocks=max(0, prefix_blocks - cached_blocks),
                    removal_pressure_5s=trailing_mean(pressure, end, 5.0),
                )
            )
    return sorted(candidates, key=lambda candidate: candidate.request_end_seconds)


def schedule_entry(candidate: Candidate, ttl_seconds: float) -> dict:
    return {
        "source_trace_id": candidate.source_trace_id,
        "source_kind": candidate.source_kind,
        "source_outer_idx": candidate.source_outer_idx,
        "source_inner_idx": candidate.source_inner_idx,
        "ttl_ms": math.ceil(ttl_seconds * 1000),
        "block_start": 0,
        "block_count": candidate.prefix_blocks,
    }


def write_schedule(path: Path, policy: str, entries: list[dict]) -> None:
    path.write_text(json.dumps({"policy": policy, "entries": entries}, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args.dataset, args.profile, args.server_metrics)
    selected = [
        candidate
        for candidate in candidates
        if candidate.missing_blocks >= args.min_missing_blocks
        and candidate.removal_pressure_5s >= args.min_removal_pressure_5s
    ]

    fixed_entries = [
        schedule_entry(candidate, args.fixed_ttl_seconds) for candidate in selected
    ]
    future_entries = [
        schedule_entry(candidate, candidate.future_ttl_seconds)
        for candidate in selected
    ]
    write_schedule(
        args.output_dir / "fixed-10s.json",
        "heuristic_whole_prefix_fixed_ttl",
        fixed_entries,
    )
    write_schedule(
        args.output_dir / "future-derived-ttl.json",
        "heuristic_whole_prefix_future_derived_ttl",
        future_entries,
    )

    rows = [
        {
            **asdict(candidate),
            "selected": candidate in selected,
        }
        for candidate in candidates
    ]
    with (args.output_dir / "candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "selector": {
            "min_missing_blocks": args.min_missing_blocks,
            "min_removal_pressure_5s": args.min_removal_pressure_5s,
            "target": "whole_prefix",
        },
        "fixed_ttl_seconds": args.fixed_ttl_seconds,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_source_keys": [list(candidate.source_key) for candidate in selected],
        "limitations": [
            "Miss depth and removal pressure are replayed from the paired no-hint control rather than queried from a live policy-altered router.",
            "The future-derived TTL uses the exact next same-session request time.",
            "The fixed and future-derived schedules contain the same requests and whole-prefix block ranges; only TTL differs.",
            "The vLLM 25% retention cap remains authoritative at runtime.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
