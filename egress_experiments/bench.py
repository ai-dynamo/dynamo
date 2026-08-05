#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one number: **items/second through the asyncio loop thread**.

    python3 -m egress_experiments.bench                       # every architecture
    python3 -m egress_experiments.bench --architecture mine   # just yours
    python3 -m egress_experiments.bench --json

Method
------
Deliberate saturation. The engine is driven far faster than the loop can
possibly drain, so the loop is never idle and the rate at which responses come
out the far end *is* its throughput. Nothing is computed from ``Costs``: the
number is measured from delivery timestamps, so an architecture cannot win by
editing a constant.

Saturation is confirmed, not assumed. If the backlog -- responses handed to the
loop but not yet delivered -- is not growing, the loop was keeping up and the
measurement is a lower bound; the benchmark escalates the offered load and
retries. If it never saturates it says so instead of reporting a number.

Geometry
--------
Closed loop at ``concurrency == batch`` so exactly ``batch`` requests are live,
with ``max_tokens`` large enough that none of them finishes during the run.
``stream_interval=1``, so the engine offers ``batch / iteration_s`` responses
per second. That isolates the loop: no arrivals, no completions, no ramp beyond
the first iteration -- just a firehose of responses and one thread draining it.

``--max-backlog`` bounds memory. Because the offered rate is held at roughly
twice whatever the loop can do, the backlog grows at about the loop's own rate,
so the run self-terminates after a fixed number of *items* rather than a fixed
number of seconds. Fast architectures therefore cost no more memory than slow
ones -- they just finish sooner.

Reading the output
------------------
``items/s`` is the score. ``work µs/item on the loop`` is the same measurement
from the other side -- total modelled work charged to the loop thread, divided
by items delivered -- and the two should be consistent: an architecture that
halves the per-item loop work should roughly double items/s. If they disagree,
the architecture is spending loop time on something other than modelled work
(scheduling overhead, wakeups, GC) and that is worth knowing.

``work µs by thread`` is the conservation check. Total modelled work per item
should be roughly INVARIANT across architectures: the point is to move it off
the loop, not to delete it. A large drop in the total is either a genuine
amortisation (batching) that should be explained, or a bug.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from egress_experiments import architectures
from egress_experiments.costs import Costs
from egress_experiments.fake_trtllm.engine import BatchConfig, ConstantIteration
from egress_experiments.harness import SimConfig, SimResult, run_simulation

#: Batches to escalate through until the loop is demonstrably saturated. Each
#: step roughly doubles the offered rate; baseline saturates at the first.
BATCH_LADDER = (240, 600, 1500, 4000, 10_000)
#: Engine iteration. Small so the offered rate is high without a huge batch.
ITERATION_MS = 10.0
#: Bounds memory: the run ends when the loop is this far behind.
MAX_BACKLOG = 40_000
#: Long enough that no request finishes mid-run at any plausible speed.
MAX_TOKENS = 1_000_000
#: Ignore the first second: the batch is still filling and the ladder step
#: before this one may still be draining.
WARMUP_S = 1.0
#: Length of the scored slice. Every architecture is measured over the same
#: duration at the same point in the run, so a design that merely drains its
#: deque more slowly cannot score differently for that reason alone.
MEASURE_S = 6.0
#: Hard bound on every run. Saturating runs end sooner, on MAX_BACKLOG; this is
#: what stops a NON-saturating one from running forever, since closed-loop
#: requests never finish and a loop that is keeping up never builds a backlog.
DURATION_S = 12.0


@dataclass
class BenchResult:
    architecture: str
    description: str
    #: THE metric.
    items_per_s: float
    #: Offered/absorbed, for context.
    offered_per_s: float
    batch: int
    #: Measurement window.
    window_s: float
    items_in_window: int
    #: Saturation evidence.
    saturated: bool
    backlog_growth_per_s: float
    backlog_max: int
    #: Work accounting.
    loop_thread: str
    work_us_per_item_on_loop: float
    work_us_per_item_total: float
    work_us_by_thread: Dict[str, float] = field(default_factory=dict)
    #: Structural counters, for cross-checking against the diagram's claims.
    deque_entries_per_item: float = 0.0
    blocking_gil_acq_per_item: float = 0.0
    arch_report: Dict[str, Any] = field(default_factory=dict)
    #: Where the meter was ticked from. Anything but the loop is a bug in the
    #: architecture, not a result.
    meter_threads: Dict[str, int] = field(default_factory=dict)
    note: str = ""


def _config(architecture: str, batch: int, costs: Costs) -> SimConfig:
    return SimConfig(
        architecture=architecture,
        arrival="closed",
        concurrency=batch,
        requests=batch,
        max_tokens=MAX_TOKENS,
        isl=8,
        batch=BatchConfig(total=batch),
        iteration=ConstantIteration(ITERATION_MS),
        stream_interval=1,
        max_backlog=MAX_BACKLOG,
        duration_s=DURATION_S,
        costs=costs,
        lag_ms=5.0,
    )


def _measure(result: SimResult, warmup_s: float) -> tuple[float, float, int]:
    """items/s over a FIXED post-warmup slice, from LOOP-EXIT timestamps.

    Fixed rather than "everything after the warmup", because the run's total
    length is architecture-dependent and that made the comparison unfair.
    ``max_backlog`` cancels ingress, not the loop: the meter keeps ticking
    while the ready deque drains, and a design whose deque holds whole-batch
    callbacks drains far more slowly. Measured windows of 13.0 s and 18.3 s
    for two architectures on the same config -- different lengths covering
    different phases of the run, with different contenders active.

    A fixed slice gives every architecture the same duration at the same point
    in the run. Found by the batched-loop experiment.
    """
    times = result.loop_item_times
    if len(times) < 100:
        return 0.0, 0.0, len(times)
    start = times[0] + int(warmup_s * 1e9)
    end = start + int(MEASURE_S * 1e9)
    windowed = [t for t in times if start <= t <= end]
    if len(windowed) < 100:
        # Run was shorter than warmup+MEASURE_S; fall back to everything after
        # the warmup rather than scoring off a handful of samples.
        windowed = [t for t in times if t >= start] or times
    span_s = (windowed[-1] - windowed[0]) / 1e9
    if span_s <= 0:
        return 0.0, 0.0, len(windowed)
    return len(windowed) / span_s, span_s, len(windowed)


def run_bench(
    architecture: str,
    costs: Optional[Costs] = None,
    ladder: tuple = BATCH_LADDER,
    warmup_s: float = WARMUP_S,
) -> BenchResult:
    costs = costs or Costs()
    arch = architectures.get(architecture)
    last: Optional[SimResult] = None
    note = ""

    for batch in ladder:
        result = run_simulation(_config(architecture, batch, costs))
        last = result
        if result.errors:
            note = f"errors: {result.errors[:2]}"
            break
        if result.backlog_growing:
            break
        note = (
            f"not saturated at batch {batch} "
            f"({result.config.offered_response_rate:,.0f}/s offered); escalating"
        )
    else:
        note = (
            f"NEVER SATURATED up to batch {ladder[-1]} "
            f"({last.config.offered_response_rate:,.0f}/s offered) -- the number "
            "below is a lower bound, not this architecture's ceiling"
        )

    assert last is not None
    items_per_s, window_s, items = _measure(last, warmup_s)

    by_thread = dict(last.spin_us_by_thread)
    loop_thread = "MainThread"
    loop_us = by_thread.get(loop_thread, 0.0)
    total_us = sum(by_thread.values())
    # Per ITEM THROUGH THE LOOP, not per item out the far end: under saturation
    # the tokio-side consumer lags and the two differ by a lot.
    delivered = max(1, len(last.loop_item_times))

    return BenchResult(
        architecture=architecture,
        description=arch.description,
        items_per_s=items_per_s,
        offered_per_s=last.config.offered_response_rate,
        batch=last.config.batch.total_batch,
        window_s=window_s,
        items_in_window=items,
        saturated=last.backlog_growing,
        backlog_growth_per_s=last.backlog_growth_per_s,
        backlog_max=last.backlog_max,
        loop_thread=loop_thread,
        work_us_per_item_on_loop=loop_us / delivered,
        work_us_per_item_total=total_us / delivered,
        work_us_by_thread={k: round(v / delivered, 2) for k, v in by_thread.items()},
        deque_entries_per_item=last.probe["enqueues"] / delivered,
        blocking_gil_acq_per_item=last.blocking_gil_acquisitions / delivered,
        arch_report=last.arch_report,
        meter_threads=last.loop_meter_threads,
        note=note,
    )


def _print(results: List[BenchResult], baseline: Optional[BenchResult]) -> None:
    width = 78
    print()
    print("=" * width)
    print("  LOOP THROUGHPUT -- items/second through the asyncio loop thread")
    print("=" * width)
    print(
        f"  saturation bench · closed loop · si=1 · iteration {ITERATION_MS:g} ms"
        f" · max-backlog {MAX_BACKLOG:,}"
    )
    print("-" * width)
    print(
        f"{'architecture':<26}{'items/s':>12}{'vs base':>10}"
        f"{'loop µs/item':>14}{'all µs/item':>14}"
    )
    print("-" * width)
    for r in results:
        rel = (
            f"{r.items_per_s / baseline.items_per_s:.2f}x"
            if baseline and baseline.items_per_s
            else "-"
        )
        flag = "" if r.saturated else "  (unsat)"
        print(
            f"{r.architecture:<26}{r.items_per_s:>12,.0f}{rel:>10}"
            f"{r.work_us_per_item_on_loop:>14.2f}{r.work_us_per_item_total:>14.2f}{flag}"
        )
    print("-" * width)
    print()
    for r in results:
        print(f"{r.architecture}  --  {r.description}")
        print(
            f"  batch {r.batch:,} · offered {r.offered_per_s:,.0f}/s"
            f" · window {r.window_s:.2f} s · {r.items_in_window:,} items"
        )
        print(
            f"  saturated {r.saturated} · backlog max {r.backlog_max:,}"
            f" growth {r.backlog_growth_per_s:+,.0f}/s"
        )
        print(
            f"  deque entries/item {r.deque_entries_per_item:.3f}"
            f" · spawn_blocking GIL/item {r.blocking_gil_acq_per_item:.3f}"
        )
        print(f"  work µs/item by thread: {r.work_us_by_thread}")
        off_loop = {k: v for k, v in r.meter_threads.items() if k != r.loop_thread}
        if off_loop:
            print(f"  !! meter ticked OFF the loop: {off_loop}")
        if r.arch_report:
            print(f"  arch: {r.arch_report}")
        if r.note:
            print(f"  NOTE: {r.note}")
        print()
    print("Conservation check: 'all µs/item' should stay roughly constant across")
    print("architectures. Moving work off the loop lowers 'loop µs/item' without")
    print("lowering the total; a large drop in the total is either an explained")
    print("amortisation or a bug.")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench", description="Loop throughput: items/s through the asyncio thread"
    )
    parser.add_argument(
        "--architecture",
        action="append",
        default=None,
        help="repeatable; default is every registered architecture",
    )
    parser.add_argument(
        "--baseline",
        default="baseline-push",
        help="architecture the ratios are taken against",
    )
    parser.add_argument("--warmup-s", type=float, default=WARMUP_S)
    parser.add_argument("--cost-scale", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list", action="store_true", help="list architectures")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.list:
        for name in architectures.names():
            print(f"{name:<26}{architectures.get(name).description}")
        return 0

    wanted = args.architecture or architectures.names()
    costs = Costs().with_scale(args.cost_scale)
    results = [run_bench(name, costs, warmup_s=args.warmup_s) for name in wanted]
    baseline = next((r for r in results if r.architecture == args.baseline), None)

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2, default=str))
    else:
        _print(results, baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
