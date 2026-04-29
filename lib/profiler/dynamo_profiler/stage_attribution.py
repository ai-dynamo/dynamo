# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage attribution analyzer.

Consumes merge_result.json from the Rust merger and produces per-stage
latency distributions and p99 critical path attribution.

Can also consume a trace_index.json with richer per-request stage data
when available from the NVTX instrumentation pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

log = logging.getLogger("stage_attribution")

DEFAULT_STAGES = [
    {"name": "tokenize",            "kind": "serial",   "ttft_phase": True},
    {"name": "route",               "kind": "serial",   "ttft_phase": True},
    {"name": "dispatch_lag",        "kind": "queue",    "ttft_phase": True},
    {"name": "prefill_queue_wait",  "kind": "queue",    "ttft_phase": True},
    {"name": "prefill_compute",     "kind": "serial",   "ttft_phase": True},
    {"name": "kv_transfer",         "kind": "serial",   "ttft_phase": True},
    {"name": "decode_queue_wait",   "kind": "queue",    "ttft_phase": True},
    {"name": "decode_first_token",  "kind": "serial",   "ttft_phase": True},
    {"name": "decode_stream",       "kind": "decode",   "ttft_phase": False},
]


@dataclass
class StagePercentiles:
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    p99_9_ms: float = 0.0
    mean_ms: float = 0.0
    stddev_ms: float = 0.0
    count: int = 0
    coverage: float = 0.0


@dataclass
class CriticalPathAttribution:
    percentile: float
    target_ttft_ms: float
    sample_count: int
    per_stage_pct: dict[str, float] = field(default_factory=dict)
    per_stage_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class AttributionReport:
    total_requests: int
    requests_with_complete_stages: int
    ttft_percentiles: StagePercentiles = field(default_factory=StagePercentiles)
    per_stage_percentiles: dict[str, StagePercentiles] = field(default_factory=dict)
    p50_critical_path: Optional[CriticalPathAttribution] = None
    p95_critical_path: Optional[CriticalPathAttribution] = None
    p99_critical_path: Optional[CriticalPathAttribution] = None
    p99_9_critical_path: Optional[CriticalPathAttribution] = None
    unaccounted_ms_p99: float = 0.0
    stage_miss_counts: dict[str, int] = field(default_factory=dict)


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _stats(values_ms: list[float], total_request_count: int) -> StagePercentiles:
    if not values_ms:
        return StagePercentiles(coverage=0.0)
    sv = sorted(values_ms)
    return StagePercentiles(
        p50_ms=_percentile(sv, 50),
        p90_ms=_percentile(sv, 90),
        p95_ms=_percentile(sv, 95),
        p99_ms=_percentile(sv, 99),
        p99_9_ms=_percentile(sv, 99.9),
        mean_ms=statistics.mean(values_ms),
        stddev_ms=statistics.stdev(values_ms) if len(values_ms) > 1 else 0.0,
        count=len(values_ms),
        coverage=len(values_ms) / total_request_count if total_request_count else 0,
    )


def extract_stage_durations(trace_index: dict, stage_defs: list[dict]) -> list[dict]:
    stage_names = [s["name"] for s in stage_defs]
    by_request: list[dict] = []

    for trace_id, info in trace_index.items():
        stages_in_request = info.get("stages", {})
        request_durations = {"trace_id": trace_id}

        observed = []
        for name in stage_names:
            if name in stages_in_request:
                s = stages_in_request[name]
                observed.append((
                    name,
                    s["first_ts_ns"],
                    s.get("end_ts_ns"),
                    s.get("duration_ns"),
                ))
        observed.sort(key=lambda x: x[1])

        for i, (name, start, end, dur) in enumerate(observed):
            if dur is not None:
                duration_ns = dur
            elif end is not None:
                duration_ns = end - start
            else:
                if i + 1 < len(observed):
                    duration_ns = observed[i + 1][1] - start
                else:
                    duration_ns = info.get("last_ts_ns", start) - start
            if duration_ns < 0:
                continue
            request_durations[name] = duration_ns / 1e6

        by_request.append(request_durations)

    return by_request


def compute_attribution(
    trace_index: dict, stage_defs: list[dict]
) -> AttributionReport:
    durations_per_request = extract_stage_durations(trace_index, stage_defs)
    total = len(durations_per_request)
    report = AttributionReport(total_requests=total, requests_with_complete_stages=0)

    stage_names = [s["name"] for s in stage_defs]
    ttft_phase_names = [s["name"] for s in stage_defs if s.get("ttft_phase")]
    per_stage_values: dict[str, list[float]] = defaultdict(list)
    miss_counts = defaultdict(int)

    for req in durations_per_request:
        for name in stage_names:
            if name in req:
                per_stage_values[name].append(req[name])
            else:
                miss_counts[name] += 1

    for name, values in per_stage_values.items():
        report.per_stage_percentiles[name] = _stats(values, total)
    report.stage_miss_counts = dict(miss_counts)

    core_required = {"tokenize", "prefill_compute", "decode_first_token"}
    core_required_present = core_required & {s["name"] for s in stage_defs}

    ttft_values = []
    complete_requests = []
    for req in durations_per_request:
        if not core_required_present.issubset(set(req.keys()) - {"trace_id"}):
            continue
        ttft = sum(req[name] for name in ttft_phase_names if name in req)
        ttft_values.append(ttft)
        complete_requests.append(req)
    report.requests_with_complete_stages = len(complete_requests)
    report.ttft_percentiles = _stats(ttft_values, total)

    if ttft_values:
        sorted_ttft = sorted(ttft_values)
        for p in [50, 95, 99, 99.9]:
            target = _percentile(sorted_ttft, p)
            margin = max(target * 0.05, 1.0)
            min_samples = max(3, len(complete_requests) // 100)
            near_p = [
                req for req in complete_requests
                if abs(sum(req[n] for n in ttft_phase_names if n in req) - target) <= margin
            ]
            if len(near_p) < min_samples:
                with_dist = [
                    (abs(sum(req[n] for n in ttft_phase_names if n in req) - target), req)
                    for req in complete_requests
                ]
                with_dist.sort(key=lambda x: x[0])
                near_p = [r for _, r in with_dist[:min_samples]]
            if not near_p:
                continue
            avg_per_stage = {}
            for n in ttft_phase_names:
                vals = [req[n] for req in near_p if n in req]
                if vals:
                    avg_per_stage[n] = statistics.mean(vals)
            avg_total = sum(avg_per_stage.values())
            cp = CriticalPathAttribution(
                percentile=p,
                target_ttft_ms=target,
                sample_count=len(near_p),
                per_stage_ms=avg_per_stage,
                per_stage_pct={
                    n: (v / avg_total * 100.0) if avg_total else 0.0
                    for n, v in avg_per_stage.items()
                },
            )
            if p == 50:
                report.p50_critical_path = cp
            elif p == 95:
                report.p95_critical_path = cp
            elif p == 99:
                report.p99_critical_path = cp
            elif p == 99.9:
                report.p99_9_critical_path = cp

    return report


def compare_with_baseline(
    current: AttributionReport, baseline: AttributionReport
) -> dict:
    deltas = {
        "ttft": {
            "p50_delta_ms": current.ttft_percentiles.p50_ms - baseline.ttft_percentiles.p50_ms,
            "p95_delta_ms": current.ttft_percentiles.p95_ms - baseline.ttft_percentiles.p95_ms,
            "p99_delta_ms": current.ttft_percentiles.p99_ms - baseline.ttft_percentiles.p99_ms,
            "p99_pct_change": (
                (current.ttft_percentiles.p99_ms - baseline.ttft_percentiles.p99_ms)
                / baseline.ttft_percentiles.p99_ms * 100
                if baseline.ttft_percentiles.p99_ms else 0
            ),
        },
        "per_stage": {},
    }
    all_stages = set(current.per_stage_percentiles) | set(baseline.per_stage_percentiles)
    for name in all_stages:
        cur = current.per_stage_percentiles.get(name, StagePercentiles())
        base = baseline.per_stage_percentiles.get(name, StagePercentiles())
        deltas["per_stage"][name] = {
            "p50_delta_ms": cur.p50_ms - base.p50_ms,
            "p99_delta_ms": cur.p99_ms - base.p99_ms,
            "p99_pct_change": (
                (cur.p99_ms - base.p99_ms) / base.p99_ms * 100 if base.p99_ms else 0
            ),
        }
    stage_deltas = [(name, d["p99_delta_ms"]) for name, d in deltas["per_stage"].items()]
    deltas["biggest_regressions"] = sorted(
        [(n, d) for n, d in stage_deltas if d > 0], key=lambda x: -x[1]
    )[:5]
    deltas["biggest_improvements"] = sorted(
        [(n, d) for n, d in stage_deltas if d < 0], key=lambda x: x[1]
    )[:5]
    return deltas


def to_serializable(report: AttributionReport) -> dict:
    d = {
        "total_requests": report.total_requests,
        "complete_requests": report.requests_with_complete_stages,
        "ttft_percentiles": {
            "p50": report.ttft_percentiles.p50_ms,
            "p95": report.ttft_percentiles.p95_ms,
            "p99": report.ttft_percentiles.p99_ms,
            "p99.9": report.ttft_percentiles.p99_9_ms,
        },
        "per_stage_percentiles": {
            k: {"p50": v.p50_ms, "p95": v.p95_ms, "p99": v.p99_ms}
            for k, v in report.per_stage_percentiles.items()
        },
        "stage_order": list(report.per_stage_percentiles.keys()),
    }
    if report.p99_critical_path:
        cp = report.p99_critical_path
        d["p99_critical_path"] = {
            "ttft_ms": cp.target_ttft_ms,
            "sample_count": cp.sample_count,
            "per_stage_pct": cp.per_stage_pct,
            "per_stage_ms": cp.per_stage_ms,
        }
    return d


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--stages-config")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    with open(args.trace_index) as f:
        trace_index = json.load(f)

    stage_defs = DEFAULT_STAGES
    if args.stages_config:
        import yaml
        with open(args.stages_config) as f:
            stage_defs = yaml.safe_load(f)["stages"]

    report = compute_attribution(trace_index, stage_defs)
    output = {"report": to_serializable(report)}

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    sys.exit(main() or 0)
