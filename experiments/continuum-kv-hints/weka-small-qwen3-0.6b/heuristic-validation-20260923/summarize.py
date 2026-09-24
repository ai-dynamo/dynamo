#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

CONDITIONS = {
    "baseline": "No hints",
    "heuristic-fixed-10s": "Heuristic, fixed 10s TTL",
    "heuristic-future-ttl": "Heuristic, future-derived TTL",
}

METRICS = (
    "cache_hit_pct",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "itl_p50_ms",
    "itl_p90_ms",
    "request_latency_p50_ms",
    "request_latency_p90_ms",
    "total_token_throughput",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def avg(profile: dict, key: str) -> float:
    return float(profile[key]["avg"])


def percentile(profile: dict, key: str, percentile_name: str) -> float:
    return float(profile[key][percentile_name])


def source_key(record: dict) -> tuple[str, str, int, int | None]:
    metadata = record["metadata"]
    return (
        metadata["source_trace_id"],
        metadata["source_kind"],
        int(metadata["source_outer_idx"]),
        metadata.get("source_inner_idx"),
    )


def arrival_order(profile_jsonl: Path) -> list[tuple[str, str, int, int | None]]:
    records = [
        json.loads(line) for line in profile_jsonl.read_text().splitlines() if line
    ]
    records.sort(key=lambda record: int(record["metadata"]["request_start_ns"]))
    return [source_key(record) for record in records]


def cached_tokens_by_source(profile_jsonl: Path) -> dict[tuple, int]:
    records = [
        json.loads(line) for line in profile_jsonl.read_text().splitlines() if line
    ]
    return {
        source_key(record): int(
            record["metrics"]["usage_prompt_cache_read_tokens"]["value"]
        )
        for record in records
    }


def same_position_pct(reference: list[tuple], candidate: list[tuple]) -> float:
    if not reference or len(reference) != len(candidate):
        return 0.0
    equal = sum(left == right for left, right in zip(reference, candidate))
    return 100.0 * equal / len(reference)


def count_matching(path: Path, needle: str) -> int:
    if not path.exists():
        return 0
    return sum(needle in line for line in path.read_text(errors="replace").splitlines())


def read_run(
    run_dir: Path,
    round_number: int,
    condition: str,
    baseline_order: list[tuple],
    baseline_cache: dict[tuple, int],
    selected_keys: set[tuple],
) -> dict:
    profile = json.loads((run_dir / "aiperf/profile.json").read_text())
    occupancy_path = run_dir / "retention-occupancy-summary.json"
    occupancy = (
        json.loads(occupancy_path.read_text()) if occupancy_path.exists() else {}
    )
    order = arrival_order(run_dir / "aiperf/profile.jsonl")
    run_cache = cached_tokens_by_source(run_dir / "aiperf/profile.jsonl")
    if set(run_cache) != set(baseline_cache):
        raise ValueError(
            f"Request identity mismatch between {run_dir} and its baseline"
        )
    aggregate_cache_delta = sum(run_cache.values()) - sum(baseline_cache.values())
    selected_cache_delta = sum(
        run_cache[key] - baseline_cache[key] for key in selected_keys
    )
    action_count = count_matching(
        run_dir / "policy-actions.log", 'action_type="kv.retain"'
    )
    skipped_count = count_matching(
        run_dir / "worker.log", "Skipping KV retention lease"
    )
    summary_path = (
        run_dir.parent / "schedules" / f"round-{round_number}" / "summary.json"
    )
    selector_summary = (
        json.loads(summary_path.read_text()) if summary_path.exists() else {}
    )
    return {
        "round": round_number,
        "condition": condition,
        "condition_label": CONDITIONS[condition],
        "completed_requests": int(avg(profile, "completed_request_count")),
        "cache_hit_pct": avg(profile, "overall_usage_prompt_cache_read_pct"),
        "ttft_p50_ms": percentile(profile, "time_to_first_token", "p50"),
        "ttft_p90_ms": percentile(profile, "time_to_first_token", "p90"),
        "itl_p50_ms": percentile(profile, "inter_token_latency", "p50"),
        "itl_p90_ms": percentile(profile, "inter_token_latency", "p90"),
        "request_latency_p50_ms": percentile(profile, "request_latency", "p50"),
        "request_latency_p90_ms": percentile(profile, "request_latency", "p90"),
        "total_token_throughput": avg(profile, "total_token_throughput"),
        "retain_actions": action_count,
        "retain_actions_admitted": action_count - skipped_count,
        "retain_actions_skipped_by_cap": skipped_count,
        "selected_requests": int(selector_summary.get("selected_count", 0))
        if condition != "baseline"
        else 0,
        "peak_retained_blocks": int(occupancy.get("peak_retained_blocks", 0)),
        "peak_retained_fraction": float(occupancy.get("peak_retained_fraction", 0.0)),
        "mean_retained_fraction": float(
            occupancy.get("time_weighted_mean_retained_fraction", 0.0)
        ),
        "same_arrival_position_as_paired_baseline_pct": same_position_pct(
            baseline_order, order
        ),
        "aggregate_cached_token_delta": aggregate_cache_delta,
        "selected_cached_token_delta": selected_cache_delta,
        "nonselected_cached_token_delta": aggregate_cache_delta - selected_cache_delta,
        "selected_requests_improved": sum(
            run_cache[key] > baseline_cache[key] for key in selected_keys
        ),
        "selected_requests_regressed": sum(
            run_cache[key] < baseline_cache[key] for key in selected_keys
        ),
        "prometheus_reset_warnings": count_matching(
            run_dir / "aiperf.log", "counter reset"
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    aggregates = []
    for condition, label in CONDITIONS.items():
        condition_rows = [row for row in rows if row["condition"] == condition]
        aggregate = {
            "condition": condition,
            "condition_label": label,
            "rounds": len(condition_rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in condition_rows]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_stddev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
            aggregate[f"{metric}_min"] = min(values)
            aggregate[f"{metric}_max"] = max(values)
        for metric in (
            "retain_actions",
            "retain_actions_admitted",
            "retain_actions_skipped_by_cap",
            "selected_requests",
            "peak_retained_blocks",
            "peak_retained_fraction",
            "mean_retained_fraction",
            "same_arrival_position_as_paired_baseline_pct",
            "aggregate_cached_token_delta",
            "selected_cached_token_delta",
            "nonselected_cached_token_delta",
            "selected_requests_improved",
            "selected_requests_regressed",
        ):
            aggregate[f"{metric}_mean"] = statistics.fmean(
                float(row[metric]) for row in condition_rows
            )
        aggregates.append(aggregate)
    return aggregates


def write_markdown(path: Path, aggregates: list[dict]) -> None:
    lines = [
        "| Condition | Cache hit % | TTFT P50 / P90 (ms) | ITL P50 / P90 (ms) | Request latency P50 / P90 (ms) | Total tok/s | Peak retained blocks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            "| {label} | {cache:.2f} | {ttft50:.1f} / {ttft90:.1f} | {itl50:.2f} / {itl90:.2f} | {lat50:.1f} / {lat90:.1f} | {throughput:,.2f} | {retained:.1f} |".format(
                label=row["condition_label"],
                cache=row["cache_hit_pct_mean"],
                ttft50=row["ttft_p50_ms_mean"],
                ttft90=row["ttft_p90_ms_mean"],
                itl50=row["itl_p50_ms_mean"],
                itl90=row["itl_p90_ms_mean"],
                lat50=row["request_latency_p50_ms_mean"],
                lat90=row["request_latency_p90_ms_mean"],
                throughput=row["total_token_throughput_mean"],
                retained=row["peak_retained_blocks_mean"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    round_numbers = sorted(
        int(path.name.removeprefix("round-").removesuffix("-baseline"))
        for path in args.run_root.glob("round-*-baseline")
    )
    if not round_numbers:
        raise FileNotFoundError(
            f"No round-*-baseline directories under {args.run_root}"
        )
    for round_number in round_numbers:
        baseline_dir = args.run_root / f"round-{round_number}-baseline"
        baseline_order = arrival_order(baseline_dir / "aiperf/profile.jsonl")
        baseline_cache = cached_tokens_by_source(baseline_dir / "aiperf/profile.jsonl")
        summary_path = (
            args.run_root / "schedules" / f"round-{round_number}" / "summary.json"
        )
        selector_summary = json.loads(summary_path.read_text())
        selected_keys = {tuple(key) for key in selector_summary["selected_source_keys"]}
        for condition in CONDITIONS:
            run_dir = args.run_root / f"round-{round_number}-{condition}"
            if not run_dir.exists():
                raise FileNotFoundError(run_dir)
            rows.append(
                read_run(
                    run_dir,
                    round_number,
                    condition,
                    baseline_order,
                    baseline_cache,
                    selected_keys,
                )
            )

    aggregates = aggregate_rows(rows)
    write_csv(args.output_dir / "per-run.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregates)
    (args.output_dir / "per-run.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregates, indent=2) + "\n"
    )
    write_markdown(args.output_dir / "aggregate-table.md", aggregates)


if __name__ == "__main__":
    main()
