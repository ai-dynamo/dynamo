# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize one WEKA/Qwen KV-hint ablation matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CASES = (
    "a-no-hints",
    "b-parent-retain",
    "c-root-final-evict",
    "d-combined",
)


def metric(profile: dict[str, Any], name: str, statistic: str = "avg") -> Any:
    value = profile.get(name)
    if not isinstance(value, dict):
        return None
    return value.get(statistic)


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def summarize_case(case_dir: Path) -> dict[str, Any]:
    profile_path = case_dir / "aiperf" / "profile.json"
    profile = (
        json.loads(profile_path.read_text(encoding="utf-8"))
        if profile_path.exists()
        else {}
    )
    action_log = case_dir / "policy-actions.log"
    actions = action_log.read_text(encoding="utf-8") if action_log.exists() else ""
    env = read_env(case_dir / "run.env")
    exit_code_path = case_dir / "aiperf-exit-code.txt"
    records_path = case_dir / "aiperf" / "profile.jsonl"
    worker_log_path = case_dir / "worker.log"
    worker_log = (
        worker_log_path.read_text(encoding="utf-8", errors="replace")
        if worker_log_path.exists()
        else ""
    )
    requests_with_cache_hits = 0
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            cache_read = (
                record.get("metrics", {})
                .get("usage_prompt_cache_read_tokens", {})
                .get("value", 0)
            )
            requests_with_cache_hits += cache_read > 0

    return {
        "case": case_dir.name,
        "policy_mode": env.get("policy_mode"),
        "aiperf_exit_code": (
            int(exit_code_path.read_text(encoding="utf-8").strip())
            if exit_code_path.exists()
            else None
        ),
        "request_count": metric(profile, "request_count"),
        "completed_request_count": metric(profile, "completed_request_count"),
        "request_error_rate_pct": metric(profile, "request_error_rate"),
        "request_throughput_rps": metric(profile, "request_throughput"),
        "input_token_throughput_tps": metric(profile, "input_token_throughput"),
        "output_token_throughput_tps": metric(profile, "output_token_throughput"),
        "ttft_avg_ms": metric(profile, "time_to_first_token"),
        "ttft_p50_ms": metric(profile, "time_to_first_token", "p50"),
        "ttft_p90_ms": metric(profile, "time_to_first_token", "p90"),
        "ttft_p99_ms": metric(profile, "time_to_first_token", "p99"),
        "latency_avg_ms": metric(profile, "request_latency"),
        "latency_p50_ms": metric(profile, "request_latency", "p50"),
        "latency_p90_ms": metric(profile, "request_latency", "p90"),
        "latency_p99_ms": metric(profile, "request_latency", "p99"),
        "theoretical_prefix_cache_hit_pct": metric(
            profile, "theoretical_prefix_cache_hit"
        ),
        "observed_prompt_cache_read_pct": metric(
            profile, "overall_usage_prompt_cache_read_pct"
        ),
        "prompt_cache_read_tokens_avg": metric(
            profile, "usage_prompt_cache_read_tokens"
        ),
        "prompt_cache_read_tokens_sum": metric(
            profile, "usage_prompt_cache_read_tokens", "sum"
        ),
        "requests_with_cache_hits": requests_with_cache_hits,
        "retain_action_count": actions.count("kv.retain"),
        "evict_action_count": actions.count("kv.evict"),
        "missing_remove_warning_count": worker_log.count(
            "Failed to find block to remove"
        ),
        "profile_json": str(profile_path),
        "records_jsonl": str(records_path),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    columns = (
        "case",
        "request_count",
        "request_error_rate_pct",
        "request_throughput_rps",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "latency_p50_ms",
        "latency_p99_ms",
        "observed_prompt_cache_read_pct",
        "prompt_cache_read_tokens_sum",
        "retain_action_count",
        "evict_action_count",
        "missing_remove_warning_count",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    rows = [summarize_case(args.run_dir / case) for case in CASES]
    json_path = args.run_dir / "matrix-summary.json"
    csv_path = args.run_dir / "matrix-summary.csv"
    markdown_path = args.run_dir / "matrix-summary.md"

    json_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(markdown(rows), encoding="utf-8")

    print(markdown_path)


if __name__ == "__main__":
    main()
