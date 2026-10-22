# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Manual static-vs-retuned replay comparison for planner retune.

This script lives outside the automated test tree (pytest does not collect
it). It replays one Dynamo request trace twice per outer run: once with the
static configuration and once with ``planner.retune`` set, and reports:

1. the goodput-per-GPU delta (retuned minus static), mean and 95% CI over the
   outer runs (replay routing is random, so use at least 3 runs);
2. decision agreement across runs (the same ``chosen`` point per window);
3. with ``--decision-latency`` listing several values, the delta per value.

Scenario 1 (fixed replicas, router-only retune) uses no static planner:

    python components/src/dynamo/planner/tests/manual/retune/compare.py \\
        --trace day5.jsonl --mode disagg \\
        --prefill-engine-args prefill.json --decode-engine-args decode.json \\
        --router-config router.json --num-prefill-workers 2 --num-decode-workers 4 \\
        --sla-ttft-ms 500 --sla-itl-ms 30 \\
        --retune '{"interval_seconds": 600, "router": "default", "max_parallel": 8}'

Scenario 2 passes the autoscaling planner as ``--planner`` (JSON or a path);
the static run uses it without ``retune``. Both scenarios score retune
candidates with ``--sla-ttft-ms``/``--sla-itl-ms`` unless ``--retune`` sets
``sla_ttft_ms``/``sla_itl_ms``, so retune optimizes what the report measures.

Retune sweeps start worker processes with ``spawn``, so this module must only
run under ``if __name__ == "__main__"``.
"""

import argparse
import json
import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from aisimulate.sweeper import OptimizationTarget, objective_value

from dynamo.llm import KvRouterConfig
from dynamo.replay import run_trace_replay
from dynamo.replay.config import load_engine_args

logger = logging.getLogger(__name__)

# Two-sided 95% Student-t quantiles by degrees of freedom; normal beyond 30.
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _json_arg(value: str | None) -> Any:
    """Inline JSON, or the path of a JSON file."""

    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _mean_ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, math.nan
    quantile = _T95.get(len(values) - 1, 1.96)
    return mean, quantile * statistics.stdev(values) / math.sqrt(len(values))


def _goodput_per_gpu(report: Any) -> float:
    return objective_value(report.summary, OptimizationTarget.GOODPUT_PER_GPU)


def _common_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trace_format": "dynamo",
        "router_mode": args.router_mode,
        "router_config": (
            None
            if args.router_config is None
            else KvRouterConfig.from_json(
                Path(args.router_config).read_text(encoding="utf-8")
            )
        ),
        "arrival_speedup_ratio": args.arrival_speedup_ratio,
        "sla_ttft_ms": args.sla_ttft_ms,
        "sla_itl_ms": args.sla_itl_ms,
    }
    if args.mode == "disagg":
        kwargs.update(
            prefill_engine_args=load_engine_args(_json_arg(args.prefill_engine_args)),
            decode_engine_args=load_engine_args(_json_arg(args.decode_engine_args)),
            num_prefill_workers=args.num_prefill_workers,
            num_decode_workers=args.num_decode_workers,
        )
    else:
        kwargs.update(
            extra_engine_args=load_engine_args(_json_arg(args.extra_engine_args)),
            num_workers=args.num_workers,
        )
    return kwargs


def _retune_planner(
    args: argparse.Namespace, base: dict[str, Any] | None, latency: float
) -> dict[str, Any]:
    retune = {**_json_arg(args.retune), "decision_latency_seconds": latency}
    # Score candidates with the SLA this driver reports goodput against.
    retune.setdefault("sla_ttft_ms", args.sla_ttft_ms)
    retune.setdefault("sla_itl_ms", args.sla_itl_ms)
    if base is None:
        # Scenario 1: retune-only planner with fixed replicas.
        base = {
            "mode": args.mode,
            "enable_load_scaling": False,
            "enable_throughput_scaling": False,
            "ttft_ms": args.sla_ttft_ms,
            "itl_ms": args.sla_itl_ms,
            "report_interval_hours": None,
        }
    return {**base, "retune": retune}


def _compare(args: argparse.Namespace) -> dict[str, Any]:
    traces = [str(path) for path in args.trace]
    common = _common_kwargs(args)
    base_planner = _json_arg(args.planner)
    # The static run never retunes, even when --planner carries a retune
    # section (one config file may drive both runs).
    static_planner = (
        None
        if base_planner is None
        else {key: value for key, value in base_planner.items() if key != "retune"}
    )
    latencies = args.decision_latency or [0.0]
    deltas: dict[float, list[float]] = defaultdict(list)
    chosen: dict[float, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    runs: list[dict[str, Any]] = []
    for run in range(args.runs):
        static = run_trace_replay(traces, planner_config=static_planner, **common)
        static_score = _goodput_per_gpu(static)
        for latency in latencies:
            retuned = run_trace_replay(
                traces,
                planner_config=_retune_planner(args, static_planner, latency),
                **common,
            )
            retuned_score = _goodput_per_gpu(retuned)
            events = retuned.planner.retune_events if retuned.planner else []
            deltas[latency].append(retuned_score - static_score)
            for event in events:
                if event.get("outcome") in ("applied", "kept", "dry_run"):
                    chosen[latency][event["window_index"]].append(
                        json.dumps(event["chosen"], sort_keys=True)
                    )
            runs.append(
                {
                    "run": run,
                    "decision_latency_seconds": latency,
                    "static_goodput_per_gpu": static_score,
                    "retuned_goodput_per_gpu": retuned_score,
                    "retune_events": events,
                }
            )
            logger.info(
                "run %d latency %ss: static %.4f retuned %.4f (%d decisions)",
                run,
                latency,
                static_score,
                retuned_score,
                len(events),
            )
    summary = {}
    for latency in latencies:
        mean, half_width = _mean_ci(deltas[latency])
        windows = chosen[latency]
        agreeing = sum(len(set(points)) == 1 for points in windows.values())
        summary[str(latency)] = {
            "delta_goodput_per_gpu_mean": mean,
            "delta_goodput_per_gpu_ci95": half_width,
            "windows": len(windows),
            "windows_with_agreeing_decisions": agreeing,
        }
    return {"summary": summary, "runs": runs}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--trace", nargs="+", type=Path, required=True)
    parser.add_argument("--mode", choices=("agg", "disagg"), required=True)
    parser.add_argument("--extra-engine-args", help="agg engine args (JSON or path)")
    parser.add_argument("--prefill-engine-args", help="JSON or path")
    parser.add_argument("--decode-engine-args", help="JSON or path")
    parser.add_argument("--router-config", help="KvRouterConfig JSON file")
    parser.add_argument(
        "--router-mode", choices=("kv_router", "round_robin"), default="kv_router"
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-prefill-workers", type=int, default=1)
    parser.add_argument("--num-decode-workers", type=int, default=1)
    parser.add_argument("--arrival-speedup-ratio", type=float, default=1.0)
    parser.add_argument("--sla-ttft-ms", type=float, required=True)
    parser.add_argument("--sla-itl-ms", type=float, required=True)
    parser.add_argument(
        "--planner",
        help="Scenario 2 planner config (JSON or path); omit for Scenario 1",
    )
    parser.add_argument(
        "--retune", required=True, help="planner.retune section (JSON or path)"
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--decision-latency",
        type=float,
        nargs="*",
        help="decision_latency_seconds values to compare (default: 0)",
    )
    parser.add_argument("--output", type=Path, help="write the full results here")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = _parser().parse_args()
    results = _compare(args)
    print(json.dumps(results["summary"], indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
