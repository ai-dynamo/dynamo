#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process regression guardrail for the unified-backend Rust↔Python bridge.

Drives ``sample_engine.py`` through the real ``PyLLMEngine`` bridge +
``EngineAdapter`` and a GIL-free Rust floor through the *same* adapter path,
with no NATS / etcd / frontend. The throughput / latency delta is the bridge +
GIL cost; sweeping concurrency surfaces the GIL-contention curve. Run it before
and after touching the bridge to catch regressions.

Build the wheel with the harness entries first:

    cd lib/bindings/python && maturin develop --features bench-harness

Then run:

    python -m benchmarks.unified_backend.bench_bridge --quick
    python -m benchmarks.unified_backend.bench_bridge --json results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, replace

from .workload import DEFAULT_CONCURRENCY, Workload, default_sweep

# Keys returned by the Rust bench entries (see lib/bindings/python/rust/bench.rs).
BENCH_FUNCS = ("bench_unified_python_engine", "bench_unified_rust_floor")


def _require_bench_api():
    try:
        import dynamo._core as core
    except ImportError as e:  # pragma: no cover - import guard
        sys.exit(f"could not import dynamo._core: {e}")
    missing = [n for n in BENCH_FUNCS if not hasattr(core, n)]
    if missing:
        sys.exit(
            f"dynamo._core is missing the bench entries {missing}.\n"
            "Rebuild the wheel with the harness feature:\n"
            "    cd lib/bindings/python && maturin develop --features bench-harness"
        )
    return core


def _make_sample_engine(model: str, w: Workload):
    from dynamo.common.backend.sample_engine import SampleLLMEngine

    # delay is seconds on the Python side; the Rust floor takes ms.
    return SampleLLMEngine(
        model_name=model,
        max_tokens=w.max_tokens,
        delay=w.per_token_delay_ms / 1000.0,
    )


async def _run_point(core, model: str, w: Workload, concurrency: int):
    loop = asyncio.get_running_loop()
    floor = await core.bench_unified_rust_floor(
        model,
        w.prompt_len,
        w.max_tokens,
        w.logprobs_k,
        w.per_token_delay_ms,
        concurrency,
        w.total_requests,
    )
    unified = await core.bench_unified_python_engine(
        _make_sample_engine(model, w),
        loop,
        model,
        w.prompt_len,
        w.max_tokens,
        w.logprobs_k,
        concurrency,
        w.total_requests,
    )
    return {"floor": floor, "unified": unified}


def _overhead_pct(unified: dict, floor: dict) -> float:
    """How much throughput the bridge gives up vs the GIL-free floor."""
    f = floor["tokens_per_sec"]
    if f <= 0:
        return 0.0
    return (f - unified["tokens_per_sec"]) / f * 100.0


_HEADER = (
    f"{'workload':<14}{'conc':>5}{'floor t/s':>14}{'unified t/s':>14}"
    f"{'overhead%':>11}{'u.ttft p50':>12}{'u.itl p50':>11}{'u.itl p99':>11}"
)


def _print_row(w: Workload, concurrency: int, r: dict) -> None:
    floor, unified = r["floor"], r["unified"]
    print(
        f"{w.name:<14}{concurrency:>5}"
        f"{floor['tokens_per_sec']:>14,.0f}{unified['tokens_per_sec']:>14,.0f}"
        f"{_overhead_pct(unified, floor):>10.1f}%"
        f"{unified['ttft_p50_ms']:>12.2f}{unified['itl_p50_ms']:>11.3f}"
        f"{unified['itl_p99_ms']:>11.3f}"
    )


async def main_async(args) -> list[dict]:
    core = _require_bench_api()
    sweep = default_sweep()
    if args.quick:
        sweep = [replace(sweep[0], total_requests=args.total_requests or 64)]
    elif args.total_requests:
        sweep = [replace(w, total_requests=args.total_requests) for w in sweep]

    print(_HEADER)
    print("-" * len(_HEADER))

    results: list[dict] = []
    for w in sweep:
        for concurrency in args.concurrency:
            r = await _run_point(core, args.model, w, concurrency)
            _print_row(w, concurrency, r)
            results.append(
                {
                    "workload": asdict(w),
                    "concurrency": concurrency,
                    "overhead_pct": _overhead_pct(r["unified"], r["floor"]),
                    **r,
                }
            )
        print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {len(results)} rows to {args.json}")
    return results


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="bench-model")
    p.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=DEFAULT_CONCURRENCY,
        help="concurrency levels to sweep",
    )
    p.add_argument(
        "--total-requests",
        type=int,
        default=None,
        help="override total requests per point (default: per-workload value)",
    )
    p.add_argument("--json", default=None, help="write full results to this JSON file")
    p.add_argument(
        "--quick",
        action="store_true",
        help="one workload, 64 requests/point — a fast smoke run",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    asyncio.run(main_async(_parse_args(argv)))


if __name__ == "__main__":
    main()
