#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a single Arena match: one autoscaler × one config × one workload.

Self-bootstrapping (adds ``src/`` to ``sys.path``), so it runs without installing
the package. Requires a built Dynamo simulation environment for
gpt-oss. Example:

    export DYNAMO_DIR=/absolute/path/to/dynamo
    source .venv/bin/activate
    python scripts/run_match.py \\
        --trace /path/to/trace.jsonl \\
        --autoscaler static --num-prefill 4 --num-decode 1

This is the single-match primitive used by the leaderboard sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def main() -> int:
    from autoscaling_arena.substrates import SUBSTRATES, get_substrate
    from autoscaling_arena.workloads import validate_mooncake_trace

    p = argparse.ArgumentParser(description="Run one Autoscaling Arena match.")
    p.add_argument("--autoscaler", choices=("static", "reactive"), default="static")
    p.add_argument(
        "--trace",
        required=True,
        help="Mooncake-format JSONL trace supplied by the user",
    )
    p.add_argument(
        "--substrate",
        choices=sorted(SUBSTRATES),
        default="gpt_oss",
        help="model/hardware preset (gpt_oss=h200/vllm)",
    )
    p.add_argument("--num-prefill", type=int, default=4)
    p.add_argument("--num-decode", type=int, default=1)
    p.add_argument(
        "--trace-block-size",
        type=int,
        default=512,
        help="Mooncake block size; MUST match the trace (synthetic/anchor default: 512)",
    )
    p.add_argument(
        "--presorted",
        action="store_true",
        help="declare that the input trace is already timestamp-sorted",
    )
    p.add_argument("--ttft-ms", type=float, default=2000.0)
    p.add_argument("--itl-ms", type=float, default=50.0)
    p.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="cap the trace to its earliest N requests (faster run)",
    )
    p.add_argument(
        "--arrival-speedup",
        type=float,
        default=1.0,
        help="compress arrival times by this ratio (replay runtime); >1 = faster + higher load",
    )
    p.add_argument("--report-json", default=None)
    args = p.parse_args()

    if args.max_requests is not None and args.max_requests <= 0:
        p.error("--max-requests must be positive")
    if not math.isfinite(args.arrival_speedup) or args.arrival_speedup <= 0:
        p.error("--arrival-speedup must be a finite positive number")

    trace_path = Path(args.trace).expanduser().resolve()
    try:
        validate_mooncake_trace(
            trace_path,
            block_size=args.trace_block_size,
            presorted=args.presorted,
        )
    except ValueError as exc:
        p.error(f"invalid --trace: {exc}")
    from autoscaling_arena.adapters import ReactiveAutoscaler, StaticAutoscaler
    from autoscaling_arena.runners import run_arena_replay
    from autoscaling_arena.workloads import Workload

    sub = get_substrate(args.substrate)
    ea = sub.engine_args()
    substrate = json.dumps(
        {
            "mode": "disagg",
            "optimization_target": "sla",
            "ttft_ms": args.ttft_ms,
            "itl_ms": args.itl_ms,
            "prefill_engine_num_gpu": sub.gpus_per_worker,
            "decode_engine_num_gpu": sub.gpus_per_worker,
            "max_gpu_budget": sub.default_gpu_budget,
            "report_filename": f"arena_{args.autoscaler}.html",
        }
    )

    if args.autoscaler == "static":
        factory = lambda c, k: StaticAutoscaler(  # noqa: E731
            num_prefill=args.num_prefill, num_decode=args.num_decode, mode="disagg"
        )
    else:
        factory = lambda c, k: ReactiveAutoscaler(  # noqa: E731
            mode="disagg", capabilities=k
        )

    trace_workspace = tempfile.TemporaryDirectory(
        prefix="autoscaling-arena-single-trace-"
    )
    try:
        trace = Workload(
            name="recorded-trace",
            description="User-supplied recorded trace",
            block_size=args.trace_block_size,
            static_trace=trace_path,
            presorted=args.presorted,
        ).materialize(
            Path(trace_workspace.name),
            max_requests=args.max_requests,
            force_copy_static=True,
        )
        report = run_arena_replay(
            trace_file=str(trace),
            autoscaler=factory,
            substrate_config=substrate,
            prefill_engine_args=ea,
            decode_engine_args=ea,
            num_prefill_workers=args.num_prefill,
            num_decode_workers=args.num_decode,
            arrival_speedup_ratio=args.arrival_speedup,
            trace_block_size=args.trace_block_size,
            report_json=args.report_json,
        )
    finally:
        trace_workspace.cleanup()

    r = report.trace_report
    print(
        f"\n=== {args.autoscaler} {args.num_prefill}P{args.num_decode}D "
        f"| {sub.model} / {sub.system} / {sub.backend} {sub.backend_version} "
        f"({sub.gpus_per_worker} GPU/worker) ==="
    )
    print(
        f"  mean_ttft={r['mean_ttft_ms']:.1f}ms  p99_ttft={r['p99_ttft_ms']:.1f}ms  "
        f"mean_itl={r['mean_itl_ms']:.1f}ms  rps={r['request_throughput_rps']:.2f}"
    )
    print(f"  ticks={report.total_ticks}  scaling_events={len(report.scaling_events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
