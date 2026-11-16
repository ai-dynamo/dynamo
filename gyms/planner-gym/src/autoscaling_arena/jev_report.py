# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller overhead kept separate from simulated serving metrics."""

import json
import math
from pathlib import Path


def summarize_decisions(path: Path) -> dict:
    calls = errors = gated = input_tokens = missing_usage = 0
    models = set()
    latencies = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            calls += 1
            errors += row["status"] == "error"
            gated += bool(row["gated_pools"])
            latencies.append(row["wall_latency_s"])
            if row.get("model"):
                models.add(row["model"])
            tokens = (row.get("usage") or {}).get("input_tokens")
            if type(tokens) is int and tokens >= 0:
                input_tokens += tokens
            else:
                missing_usage += 1
    latencies.sort()
    return {
        "calls": calls,
        "errors": errors,
        "confidence_gated_ticks": gated,
        "models": sorted(models),
        "reported_input_tokens": input_tokens,
        "calls_without_token_usage": missing_usage,
        "total_wall_latency_s": sum(latencies),
        "p50_wall_latency_s": latencies[math.ceil(calls * 0.5) - 1] if calls else None,
        "p95_wall_latency_s": latencies[math.ceil(calls * 0.95) - 1] if calls else None,
        "max_wall_latency_s": max(latencies, default=None),
        "latency_accounting": "measured_wall_time_excluded_from_simulated_time",
    }
