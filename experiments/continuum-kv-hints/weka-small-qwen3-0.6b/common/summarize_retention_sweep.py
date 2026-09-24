# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare one no-hint baseline with retained-G1 occupancy candidates."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUEST_ID = re.compile(r'action_type="kv\.retain".*x_request_id="([^"]+)"')


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


SourceKey = tuple[str, str, int, int | None]


def source_key(record: dict[str, Any]) -> SourceKey:
    metadata = record["metadata"]
    return (
        metadata["source_trace_id"],
        metadata["source_kind"],
        int(metadata["source_outer_idx"]),
        metadata.get("source_inner_idx"),
    )


def record_metric(record: dict[str, Any], name: str) -> float:
    return float(record.get("metrics", {}).get(name, {}).get("value", 0))


def profile_metric(
    profile: dict[str, Any], name: str, statistic: str = "avg"
) -> float | None:
    value = profile.get(name)
    return value.get(statistic) if isinstance(value, dict) else None


def summarize(
    label: str,
    case_dir: Path,
    baseline_records: dict[SourceKey, list[dict[str, Any]]],
    baseline_gap_tokens: dict[SourceKey, float],
) -> dict[str, Any]:
    profile = json.loads((case_dir / "aiperf/profile.json").read_text())
    occupancy = json.loads((case_dir / "retention-occupancy-summary.json").read_text())
    worker_log = (case_dir / "worker.log").read_text(encoding="utf-8", errors="replace")
    row: dict[str, Any] = {
        "case": label,
        "request_count": profile_metric(profile, "request_count"),
        "request_error_rate_pct": profile_metric(profile, "request_error_rate"),
        "request_throughput_rps": profile_metric(profile, "request_throughput"),
        "observed_prompt_cache_read_pct": profile_metric(
            profile, "overall_usage_prompt_cache_read_pct"
        ),
        "prompt_cache_read_tokens": profile_metric(
            profile, "usage_prompt_cache_read_tokens", "sum"
        ),
        "ttft_p50_ms": profile_metric(profile, "time_to_first_token", "p50"),
        "ttft_p99_ms": profile_metric(profile, "time_to_first_token", "p99"),
        "latency_p50_ms": profile_metric(profile, "request_latency", "p50"),
        "latency_p99_ms": profile_metric(profile, "request_latency", "p99"),
        "mean_retained_pct": 100
        * float(occupancy["time_weighted_mean_retained_fraction"]),
        "p95_retained_pct": 100
        * float(occupancy["time_weighted_p95_retained_fraction"]),
        "peak_retained_pct": 100 * float(occupancy["peak_retained_fraction"]),
        "retained_change_count": occupancy["change_reasons"].get("retained", 0),
        "skipped_retention_action_count": worker_log.count(
            "Skipping KV retention lease"
        ),
        "forced_reuse_change_count": occupancy["change_reasons"].get("forced_reuse", 0),
        "artifact_dir": str(case_dir),
    }

    action_log = case_dir / "policy-actions.log"
    if not action_log.exists() or not baseline_records:
        return row

    candidate_records = load_records(case_dir / "aiperf/profile.jsonl")
    records_by_id = {
        record["metadata"]["x_request_id"]: record for record in candidate_records
    }
    records_by_session_turn = {
        (
            record["metadata"]["x_correlation_id"],
            int(record["metadata"]["turn_index"]),
        ): record
        for record in candidate_records
    }
    protected_request_ids = REQUEST_ID.findall(action_log.read_text())
    comparisons: list[tuple[float, float, float, float, float]] = []
    actions_without_completed_resume = 0
    for request_id in protected_request_ids:
        protected = records_by_id[request_id]
        metadata = protected["metadata"]
        resume_key = (
            metadata["x_correlation_id"],
            int(metadata["turn_index"]) + 1,
        )
        candidate = records_by_session_turn.get(resume_key)
        if candidate is None:
            actions_without_completed_resume += 1
            continue
        source = source_key(candidate)
        baseline_candidates = baseline_records.get(source, [])
        if not baseline_candidates:
            continue
        baseline_cache = statistics.fmean(
            record_metric(record, "usage_prompt_cache_read_tokens")
            for record in baseline_candidates
        )
        baseline_ttft = statistics.fmean(
            record_metric(record, "time_to_first_token")
            for record in baseline_candidates
        )
        comparisons.append(
            (
                baseline_cache,
                record_metric(candidate, "usage_prompt_cache_read_tokens"),
                baseline_ttft,
                record_metric(candidate, "time_to_first_token"),
                baseline_gap_tokens.get(source, 0),
            )
        )

    row["targeted_resume_count"] = len(comparisons)
    row[
        "targeted_action_without_completed_resume_count"
    ] = actions_without_completed_resume
    row["targeted_cache_read_token_delta"] = sum(
        candidate - baseline for baseline, candidate, *_ in comparisons
    )
    recoverable_tokens = sum(comparison[4] for comparison in comparisons)
    row["targeted_recoverable_cache_read_tokens"] = recoverable_tokens
    row["targeted_recovery_pct"] = (
        100 * row["targeted_cache_read_token_delta"] / recoverable_tokens
        if recoverable_tokens
        else None
    )
    row["targeted_ttft_avg_delta_ms"] = (
        statistics.fmean(
            candidate - baseline for _, _, baseline, candidate, _ in comparisons
        )
        if comparisons
        else None
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-gap-requests", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="LABEL=CASE_DIR",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_records: dict[SourceKey, list[dict[str, Any]]] = defaultdict(list)
    for record in load_records(args.baseline / "aiperf/profile.jsonl"):
        baseline_records[source_key(record)].append(record)

    baseline_gap_tokens: dict[SourceKey, float] = {}
    if args.baseline_gap_requests:
        gap_rows: dict[SourceKey, list[float]] = defaultdict(list)
        with args.baseline_gap_requests.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (
                    row["source_trace_id"],
                    row["source_kind"],
                    int(row["source_outer_idx"]),
                    int(row["source_inner_idx"]) if row["source_inner_idx"] else None,
                )
                gap_rows[key].append(float(row["positive_gap_blocks"]) * 64)
        baseline_gap_tokens = {
            key: statistics.fmean(values) for key, values in gap_rows.items()
        }

    rows = [summarize("no-hints", args.baseline, {}, {})]
    for value in args.candidate:
        label, separator, path = value.partition("=")
        if not separator:
            parser.error(f"candidate must be LABEL=CASE_DIR: {value}")
        rows.append(
            summarize(
                label,
                Path(path),
                baseline_records,
                baseline_gap_tokens,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retention-cap-summary.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    with (args.output_dir / "retention-cap-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    columns = (
        "case",
        "request_count",
        "request_throughput_rps",
        "observed_prompt_cache_read_pct",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "mean_retained_pct",
        "p95_retained_pct",
        "peak_retained_pct",
        "targeted_cache_read_token_delta",
        "targeted_ttft_avg_delta_ms",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = " | ".join(str(row.get(column, "")) for column in columns)
        lines.append(f"| {values} |")
    (args.output_dir / "retention-cap-summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
