#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a set of live endpoints with the Arena bench suite (online backend).

Takes an endpoints config (name / url / model / description), replays the Arena
workloads against each endpoint via the AIPerf CLI, scores against the SLO
profiles, and exports a comparison leaderboard. Needs the ``aiperf`` CLI on PATH
(or pass --aiperf-bin); does NOT need the Dynamo build.

    python scripts/run_endpoint_bench.py \\
        --endpoints configs/endpoints.example.yaml \\
        --workloads flat staircase --profiles interactive relaxed

Note: the online backend runs in REAL time (AIPerf replays each trace at its
timestamps), so prefer short workloads / a few endpoints for quick comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    from autoscaling_arena.scorecard import DEFAULT_PROFILES

    p = argparse.ArgumentParser(
        description="Benchmark live endpoints with the Arena suite."
    )
    p.add_argument("--endpoints", required=True, help="endpoints config JSON")
    p.add_argument(
        "--workloads",
        nargs="+",
        default=["flat", "staircase"],
        help="registry workload names, or 'all'",
    )
    p.add_argument(
        "--profiles",
        nargs="+",
        choices=[profile.name for profile in DEFAULT_PROFILES],
        default=[profile.name for profile in DEFAULT_PROFILES],
    )
    p.add_argument("--artifact-root", default=str(_REPO / "runs" / "online"))
    p.add_argument("--aiperf-bin", default="aiperf")
    p.add_argument("--tokenizer", default=None, help="tokenizer for AIPerf (e.g. gpt2)")
    p.add_argument(
        "--timeout-s", type=float, default=None, help="timeout per AIPerf run"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="cap each workload to its earliest N requests (shorter real-time runs)",
    )
    p.add_argument(
        "--arrival-speedup",
        type=float,
        default=1.0,
        help="compress trace timestamps by this ratio so AIPerf replays faster than real time (>1 also raises load intensity)",
    )
    p.add_argument("--out", default=None, help="path to save results JSON")
    args = p.parse_args()
    if args.timeout_s is not None and (
        not math.isfinite(args.timeout_s) or args.timeout_s <= 0
    ):
        p.error("--timeout-s must be positive and finite")

    from autoscaling_arena.runners.real import (
        format_endpoint_leaderboard,
        load_endpoints,
        run_endpoint_leaderboard,
    )
    from autoscaling_arena.workloads import get_workload, list_workloads

    endpoints = load_endpoints(args.endpoints)
    profiles = tuple(p for p in DEFAULT_PROFILES if p.name in args.profiles)
    wl_names = list_workloads() if args.workloads == ["all"] else args.workloads

    # Materialize workloads to AIPerf-replayable Mooncake traces.
    trace_dir = _REPO / "runs" / "traces"
    workloads = {name: get_workload(name) for name in wl_names}
    workload_traces = {
        n: str(
            workload.materialize(
                trace_dir,
                seed=args.seed,
                max_requests=args.max_requests,
                arrival_speedup=args.arrival_speedup,
            )
        )
        for n, workload in workloads.items()
    }

    print(
        f"Benchmarking {len(endpoints)} endpoints × {len(wl_names)} workloads × {len(profiles)} profiles "
        f"({len(endpoints) * len(wl_names) * len(profiles)} AIPerf runs)..."
    )

    def progress(ep, wl, prof):
        print(f"  aiperf: {ep:<16} {wl:<14} [{prof}] ...", flush=True)

    results = run_endpoint_leaderboard(
        endpoints=endpoints,
        workload_traces=workload_traces,
        workload_block_sizes={
            name: workload.block_size for name, workload in workloads.items()
        },
        profiles=profiles,
        artifact_root=args.artifact_root,
        aiperf_bin=args.aiperf_bin,
        tokenizer=args.tokenizer,
        timeout_s=args.timeout_s,
        on_progress=progress,
    )

    for prof in profiles:
        print(format_endpoint_leaderboard(results, profile=prof.name))

    out = args.out or str(_REPO / "runs" / "online_leaderboard.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps([r.__dict__ for r in results], indent=2, default=str)
    )
    print(f"\nFull results JSON: {out}")
    return 1 if any(result.returncode != 0 for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
