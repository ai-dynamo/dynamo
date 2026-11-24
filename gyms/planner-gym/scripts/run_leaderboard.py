#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an autoscaler comparison and print a leaderboard.

The Arena's headline use case. Sweeps the default roster (Planner / KEDA /
reactive / static) across the chosen workloads on a pinned gpt-oss-120b config,
scores each against the SLO profiles, and prints a ranking by Efficiency
(goodput/average GPU). Self-bootstrapping; needs the Dynamo venv + HYBRID for
gpt-oss.

    export DYNAMO_DIR=/absolute/path/to/dynamo
    source .venv/bin/activate
    python scripts/run_leaderboard.py \\
        --workloads staircase flash_crowd mooncake --profile interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


# Default load-only SLA Planner config. It is model-agnostic; rival adapters
# ignore the planner-specific fields.
# GPU counts/budget are filled from the chosen substrate preset at runtime.
def _substrate_config(sub) -> str:
    return json.dumps(
        {
            "mode": "disagg",
            "optimization_target": "sla",
            "ttft_ms": 2000,
            "itl_ms": 50,
            "enable_load_scaling": True,
            "enable_throughput_scaling": False,
            "pre_deployment_sweeping_mode": "none",
            "load_adjustment_interval_seconds": 5,
            "load_min_observations": 5,
            "load_scaling_down_sensitivity": 80,
            "min_endpoint": 1,
            "max_gpu_budget": sub.default_gpu_budget,
            "prefill_engine_num_gpu": sub.gpus_per_worker,
            "decode_engine_num_gpu": sub.gpus_per_worker,
            "report_filename": "_leaderboard_run.html",
        }
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run an Autoscaling Arena leaderboard sweep."
    )
    p.add_argument(
        "--workloads",
        nargs="+",
        default=["mooncake", "staircase", "flash_crowd"],
        help="workload names, or 'all'",
    )
    p.add_argument(
        "--profile",
        default="interactive",
        help="SLO profile to rank by (interactive / agentic / relaxed)",
    )
    from autoscaling_arena.substrates import SUBSTRATES, get_substrate

    p.add_argument(
        "--substrate",
        choices=sorted(SUBSTRATES),
        default="gpt_oss",
        help="model/hardware preset (gpt_oss=h200/vllm)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="cap each workload to its earliest N requests (faster sweeps)",
    )
    p.add_argument(
        "--arrival-speedup",
        type=float,
        default=1.0,
        help="compress arrival times by this ratio (replay runtime); >1 = faster + higher load",
    )
    p.add_argument("--out", default=None, help="path to save the full results JSON")
    args = p.parse_args()
    sub = get_substrate(args.substrate)
    engine_args = sub.engine_args()

    from autoscaling_arena.leaderboard import (
        default_autoscalers,
        format_leaderboard,
        run_leaderboard,
    )
    from autoscaling_arena.scorecard import DEFAULT_PROFILES
    from autoscaling_arena.workloads import list_workloads

    by_name = {p.name: p for p in DEFAULT_PROFILES}
    if args.profile not in by_name:
        p.error(f"unknown profile '{args.profile}'; known: {sorted(by_name)}")
    score_profile = by_name[args.profile]

    workloads = list_workloads() if args.workloads == ["all"] else args.workloads
    trace_dir = _REPO / "runs" / "traces"

    def progress(autoscaler: str, workload: str) -> None:
        print(f"  running {autoscaler:<13} on {workload} ...", flush=True)

    print(
        f"Sweeping {len(default_autoscalers())} autoscalers × {len(workloads)} workloads "
        f"({len(default_autoscalers()) * len(workloads)} matches) — "
        f"{sub.model} / {sub.system} / {sub.backend} {sub.backend_version}..."
    )
    rows = run_leaderboard(
        autoscalers=default_autoscalers(),
        workload_names=workloads,
        prefill_engine_args=engine_args,
        decode_engine_args=engine_args,
        substrate_config=_substrate_config(sub),
        trace_dir=trace_dir,
        profiles=(score_profile,),
        score_profile=score_profile,
        seed=args.seed,
        max_requests=args.max_requests,
        arrival_speedup_ratio=args.arrival_speedup,
        on_progress=progress,
    )

    print(format_leaderboard(rows, profile=args.profile))

    out = args.out or str(_REPO / "runs" / "leaderboard.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=2))
    print(f"\nFull results JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
