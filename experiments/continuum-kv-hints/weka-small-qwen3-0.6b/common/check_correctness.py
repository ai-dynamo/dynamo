# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check one exact single-trace combined-policy replay."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def average(profile: dict[str, Any], name: str) -> float:
    return float(profile[name]["avg"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--expected-requests", type=int, default=35)
    parser.add_argument("--expected-retains", type=int, default=1)
    parser.add_argument("--expected-evictions", type=int, default=1)
    args = parser.parse_args()

    profile = json.loads(
        (args.case_dir / "aiperf" / "profile.json").read_text(encoding="utf-8")
    )
    records = [
        json.loads(line)
        for line in (args.case_dir / "aiperf" / "profile.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    actions = (args.case_dir / "policy-actions.log").read_text(encoding="utf-8")
    kv_events = (args.case_dir / "kv-events.log").read_text(encoding="utf-8")
    worker_log = (args.case_dir / "worker.log").read_text(encoding="utf-8")

    request_count = int(average(profile, "request_count"))
    completed_count = int(average(profile, "completed_request_count"))
    error_rate = average(profile, "request_error_rate")
    retain_count = actions.count("kv.retain")
    eviction_count = actions.count("kv.evict")
    cached_reads = [
        int(record["metrics"]["usage_prompt_cache_read_tokens"]["value"])
        for record in records
        if "usage_prompt_cache_read_tokens" in record["metrics"]
    ]

    assert request_count == args.expected_requests, request_count
    assert completed_count == args.expected_requests, completed_count
    assert error_rate == 0, error_rate
    assert retain_count == args.expected_retains, retain_count
    assert eviction_count == args.expected_evictions, eviction_count
    assert len(cached_reads) == args.expected_requests, len(cached_reads)
    assert any(value > 0 for value in cached_reads), "no observed cache reuse"

    root_sessions = {
        record["metadata"]["root_correlation_id"]
        for record in records
        if record["metadata"].get("agent_depth") == 0
    }
    action_sessions = set(re.findall(r"session_id=([^ ]+)", actions))
    assert action_sessions <= root_sessions, (action_sessions, root_sessions)
    assert kv_events, "no removal-event evidence was captured"
    assert "GPU KV cache usage: 0.0%" in worker_log

    summary = {
        "request_count": request_count,
        "request_error_rate_pct": error_rate,
        "retain_action_count": retain_count,
        "evict_action_count": eviction_count,
        "cache_read_tokens_sum": sum(cached_reads),
        "requests_with_cache_hits": sum(value > 0 for value in cached_reads),
        "root_session_count": len(root_sessions),
        "action_sessions_are_roots": True,
        "removal_event_evidence": True,
        "gpu_cache_reached_zero_pct": True,
    }
    output = args.case_dir / "correctness-summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
