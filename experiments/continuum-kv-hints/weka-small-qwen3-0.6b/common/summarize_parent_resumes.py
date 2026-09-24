# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the parent turns immediately following oracle retain actions."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

REQUEST_ID = re.compile(r'action_type="kv\.retain".*x_request_id="([^"]+)"')


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def metric(record: dict[str, Any], name: str) -> float:
    return float(record.get("metrics", {}).get(name, {}).get("value", 0))


def record_key(record: dict[str, Any]) -> tuple[str, int]:
    metadata = record["metadata"]
    return metadata["conversation_id"], int(metadata["turn_index"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    baseline_records = load_records(args.run_dir / "a-no-hints/aiperf/profile.jsonl")
    retain_records = load_records(args.run_dir / "b-parent-retain/aiperf/profile.jsonl")
    baseline_by_key = {record_key(record): record for record in baseline_records}
    retain_by_key = {record_key(record): record for record in retain_records}
    retain_by_request = {
        record["metadata"]["x_request_id"]: record for record in retain_records
    }
    action_log = (args.run_dir / "b-parent-retain/policy-actions.log").read_text()
    protected_request_ids = REQUEST_ID.findall(action_log)

    rows: list[dict[str, Any]] = []
    for request_id in protected_request_ids:
        protected = retain_by_request[request_id]
        conversation_id, turn_index = record_key(protected)
        resume_key = conversation_id, turn_index + 1
        baseline = baseline_by_key[resume_key]
        retained = retain_by_key[resume_key]
        input_tokens = metric(retained, "usage_prompt_tokens")
        baseline_read = metric(baseline, "usage_prompt_cache_read_tokens")
        retained_read = metric(retained, "usage_prompt_cache_read_tokens")
        rows.append(
            {
                "conversation_id": conversation_id,
                "protected_turn_index": turn_index,
                "resume_turn_index": turn_index + 1,
                "resume_source_outer_idx": retained["metadata"].get("source_outer_idx"),
                "input_tokens": input_tokens,
                "baseline_cache_read_tokens": baseline_read,
                "retained_cache_read_tokens": retained_read,
                "cache_read_token_delta": retained_read - baseline_read,
                "baseline_cache_read_pct": 100 * baseline_read / input_tokens,
                "retained_cache_read_pct": 100 * retained_read / input_tokens,
                "baseline_ttft_ms": metric(baseline, "time_to_first_token"),
                "retained_ttft_ms": metric(retained, "time_to_first_token"),
            }
        )

    summary = {
        "resume_request_count": len(rows),
        "baseline_cache_read_tokens": sum(
            row["baseline_cache_read_tokens"] for row in rows
        ),
        "retained_cache_read_tokens": sum(
            row["retained_cache_read_tokens"] for row in rows
        ),
        "cache_read_token_delta": sum(row["cache_read_token_delta"] for row in rows),
        "baseline_ttft_avg_ms": statistics.fmean(
            row["baseline_ttft_ms"] for row in rows
        ),
        "retained_ttft_avg_ms": statistics.fmean(
            row["retained_ttft_ms"] for row in rows
        ),
    }
    summary["ttft_avg_delta_ms"] = (
        summary["retained_ttft_avg_ms"] - summary["baseline_ttft_avg_ms"]
    )

    (args.run_dir / "parent-resume-comparison.json").write_text(
        json.dumps({"summary": summary, "requests": rows}, indent=2) + "\n"
    )
    with (args.run_dir / "parent-resume-comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
