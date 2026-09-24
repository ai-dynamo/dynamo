# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attribute a retention candidate's cache effects to targeted and other requests."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACTION = re.compile(
    r"session_id=(?P<session>\S+) worker_id=(?P<worker>\d+) "
    r"lineage_block_count=(?P<blocks>\d+) "
    r"lineage_count=(?P<lineages>\d+) .*"
    r"ttl_seconds=(?P<ttl>[\d.]+) .*?"
    r'(?:retention_reason="(?P<reason>[^"]+)" )?method=.*'
    r'x_request_id="(?P<request_id>[^"]+)"'
)
SKIPPED_ACTION = re.compile(r"Skipping KV retention lease \('(?P<request_key>[^']+)'")
SourceKey = tuple[str, str, int, int | None]


@dataclass(frozen=True)
class Action:
    sequence: int
    request_id: str
    session_id: str
    worker_id: str
    reason: str
    blocks: int
    lineages: int
    ttl_seconds: float


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def source_key(record: dict[str, Any]) -> SourceKey:
    metadata = record["metadata"]
    return (
        metadata["source_trace_id"],
        metadata["source_kind"],
        int(metadata["source_outer_idx"]),
        metadata.get("source_inner_idx"),
    )


def metric(record: dict[str, Any], name: str) -> float:
    return float(record.get("metrics", {}).get(name, {}).get("value", 0))


def parse_actions(path: Path) -> list[Action]:
    actions = []
    for line in path.read_text().splitlines():
        match = ACTION.search(line)
        if match:
            actions.append(
                Action(
                    sequence=len(actions),
                    request_id=match["request_id"],
                    session_id=match["session"],
                    worker_id=match["worker"],
                    reason=match["reason"] or "legacy_parent_spawn",
                    blocks=int(match["blocks"]),
                    lineages=int(match["lineages"]),
                    ttl_seconds=float(match["ttl"]),
                )
            )
    return actions


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def summarize_actions(actions: list[Action], cap_blocks: int) -> dict[str, Any]:
    blocks = [float(action.blocks) for action in actions]
    lineages = [float(action.lineages) for action in actions]
    ttls = [action.ttl_seconds for action in actions]
    return {
        "count": len(actions),
        "zero_block_count": sum(action.blocks == 0 for action in actions),
        "multi_lineage_count": sum(action.lineages > 1 for action in actions),
        "over_cap_count": sum(action.blocks > cap_blocks for action in actions),
        "ttl_below_2s_count": sum(action.ttl_seconds < 2 for action in actions),
        "blocks_mean": statistics.fmean(blocks) if blocks else None,
        "blocks_p50": percentile(blocks, 0.5),
        "blocks_p95": percentile(blocks, 0.95),
        "blocks_max": max(blocks, default=None),
        "lineages_mean": statistics.fmean(lineages) if lineages else None,
        "lineages_p50": percentile(lineages, 0.5),
        "lineages_p95": percentile(lineages, 0.95),
        "lineages_max": max(lineages, default=None),
        "ttl_mean_seconds": statistics.fmean(ttls) if ttls else None,
        "ttl_p50_seconds": percentile(ttls, 0.5),
        "ttl_p95_seconds": percentile(ttls, 0.95),
        "ttl_max_seconds": max(ttls, default=None),
    }


def summarize_comparisons(rows: list[dict[str, float]]) -> dict[str, Any]:
    cache_delta = sum(row["cache_delta"] for row in rows)
    recoverable = sum(row["recoverable_tokens"] for row in rows)
    return {
        "request_count": len(rows),
        "cache_read_token_delta": cache_delta,
        "baseline_cache_read_tokens": sum(row["baseline_cache"] for row in rows),
        "candidate_cache_read_tokens": sum(row["candidate_cache"] for row in rows),
        "recoverable_cache_read_tokens": recoverable,
        "recovery_pct": 100 * cache_delta / recoverable if recoverable else None,
        "ttft_avg_delta_ms": statistics.fmean(row["ttft_delta"] for row in rows)
        if rows
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-gap-requests", type=Path, required=True)
    parser.add_argument("--retention-cap-blocks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_by_source: dict[SourceKey, list[dict[str, Any]]] = defaultdict(list)
    for record in load_records(args.baseline / "aiperf/profile.jsonl"):
        baseline_by_source[source_key(record)].append(record)

    gap_by_source: dict[SourceKey, list[float]] = defaultdict(list)
    with args.baseline_gap_requests.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["source_trace_id"],
                row["source_kind"],
                int(row["source_outer_idx"]),
                int(row["source_inner_idx"]) if row["source_inner_idx"] else None,
            )
            gap_by_source[key].append(float(row["positive_gap_blocks"]) * 64)
    mean_gap_by_source = {
        key: statistics.fmean(values) for key, values in gap_by_source.items()
    }

    candidate_records = load_records(args.candidate / "aiperf/profile.jsonl")
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
    actions = parse_actions(args.candidate / "policy-actions.log")
    skipped_request_keys = {
        match["request_key"]
        for line in (args.candidate / "worker.log")
        .read_text(encoding="utf-8", errors="replace")
        .splitlines()
        if (match := SKIPPED_ACTION.search(line))
    }

    beneficiary_reason_by_id: dict[str, str] = {}
    beneficiary_action_by_id: dict[str, Action] = {}
    actions_without_resume = 0
    for action in actions:
        source = records_by_id.get(action.request_id)
        if source is None:
            continue
        metadata = source["metadata"]
        beneficiary = records_by_session_turn.get(
            (metadata["x_correlation_id"], int(metadata["turn_index"]) + 1)
        )
        if beneficiary is None:
            actions_without_resume += 1
            continue
        request_key = f"{action.session_id}-{action.worker_id}-{action.sequence}"
        status = "rejected" if request_key in skipped_request_keys else "admitted"
        beneficiary_id = beneficiary["metadata"]["x_request_id"]
        beneficiary_reason_by_id[beneficiary_id] = f"{action.reason}:{status}"
        beneficiary_action_by_id[beneficiary_id] = action

    comparisons: list[dict[str, Any]] = []
    beneficiary_details: list[dict[str, Any]] = []
    for record in candidate_records:
        baseline_records = baseline_by_source.get(source_key(record), [])
        if not baseline_records:
            continue
        baseline_cache = statistics.fmean(
            metric(item, "usage_prompt_cache_read_tokens") for item in baseline_records
        )
        baseline_ttft = statistics.fmean(
            metric(item, "time_to_first_token") for item in baseline_records
        )
        request_id = record["metadata"]["x_request_id"]
        comparisons.append(
            {
                "reason": beneficiary_reason_by_id.get(request_id, "non_target"),
                "baseline_cache": baseline_cache,
                "candidate_cache": metric(record, "usage_prompt_cache_read_tokens"),
                "cache_delta": metric(record, "usage_prompt_cache_read_tokens")
                - baseline_cache,
                "ttft_delta": metric(record, "time_to_first_token") - baseline_ttft,
                "recoverable_tokens": mean_gap_by_source.get(source_key(record), 0),
            }
        )
        action = beneficiary_action_by_id.get(request_id)
        if action is not None:
            beneficiary_details.append(
                {
                    "sequence": action.sequence,
                    "reason": action.reason,
                    "status": beneficiary_reason_by_id[request_id].rsplit(":", 1)[1],
                    "lineage_blocks": action.blocks,
                    "lineages": action.lineages,
                    "ttl_seconds": action.ttl_seconds,
                    "baseline_gap_tokens": mean_gap_by_source.get(
                        source_key(record), 0
                    ),
                    "cache_read_token_delta": metric(
                        record, "usage_prompt_cache_read_tokens"
                    )
                    - baseline_cache,
                    "ttft_delta_ms": metric(record, "time_to_first_token")
                    - baseline_ttft,
                }
            )

    actions_by_reason: dict[str, list[Action]] = defaultdict(list)
    actions_by_reason_and_status: dict[str, list[Action]] = defaultdict(list)
    for action in actions:
        actions_by_reason[action.reason].append(action)
        request_key = f"{action.session_id}-{action.worker_id}-{action.sequence}"
        status = "rejected" if request_key in skipped_request_keys else "admitted"
        actions_by_reason_and_status[f"{action.reason}:{status}"].append(action)
    comparisons_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        comparisons_by_reason[row["reason"]].append(row)

    output = {
        "actions": {
            "all": summarize_actions(actions, args.retention_cap_blocks),
            "by_reason": {
                reason: summarize_actions(rows, args.retention_cap_blocks)
                for reason, rows in sorted(actions_by_reason.items())
            },
            "by_reason_and_status": {
                reason: summarize_actions(rows, args.retention_cap_blocks)
                for reason, rows in sorted(actions_by_reason_and_status.items())
            },
            "without_completed_resume_count": actions_without_resume,
        },
        "matched_request_effects": {
            "all": summarize_comparisons(comparisons),
            "by_reason": {
                reason: summarize_comparisons(rows)
                for reason, rows in sorted(comparisons_by_reason.items())
            },
            "beneficiary_details": beneficiary_details,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
