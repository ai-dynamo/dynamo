#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the simulation and print the diagram's aggregates.

    python3 -m egress_experiments.run_experiment                 # pull vs push
    python3 -m egress_experiments.run_experiment --gil-noise 42  # 45-thread regime
    python3 -m egress_experiments.run_experiment --egress push --json

Numbers are reported next to the si=40 capture they model. Where the
simulation cannot reproduce a figure it says so rather than printing a
lookalike -- see the CAVEATS block it prints at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from egress_experiments.costs import SERVE_LOOP_US_PER_RESPONSE, Costs
from egress_experiments.dynamo_sim.gil_noise import GilNoiseConfig
from egress_experiments.dynamo_sim.worker import USING_REAL_PUSH_EGRESS
from egress_experiments.fake_trtllm import ipc
from egress_experiments.fake_trtllm.engine import (
    BatchConfig,
    BatchDependentIteration,
    ConstantIteration,
)
from egress_experiments.harness import SimConfig, SimResult, run_simulation


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_experiment",
        description="Simulate the dynamo column of ASYNCIO_GIL_PATH.md",
    )
    p.add_argument(
        "--egress",
        choices=("pull", "push", "both"),
        default="both",
        help="pull = default Rust bridge; push = DYN_TRTLLM_PUSH_EGRESS=1",
    )
    p.add_argument("--requests", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--isl", type=int, default=1024)

    ingress = p.add_argument_group("ingress (requests off the wire)")
    ingress.add_argument(
        "--qps",
        type=float,
        default=None,
        help="offered load; default is the steady-state qps that fills the batch",
    )
    ingress.add_argument(
        "--arrival",
        choices=("constant", "poisson", "closed"),
        default="constant",
        help="arrival process; 'closed' holds --concurrency in flight instead",
    )
    ingress.add_argument("--arrival-seed", type=int, default=1234)
    ingress.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="closed-loop only; default is the engine's total batch",
    )

    engine = p.add_argument_group("engine")
    engine.add_argument(
        "--batch-total",
        type=int,
        default=None,
        help="responses per iteration (default 132, the measured batch)",
    )
    engine.add_argument(
        "--batch-per-rank",
        type=int,
        default=None,
        help="per-rank batch; use with --ranks instead of --batch-total",
    )
    engine.add_argument("--ranks", type=int, default=1)
    engine.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        metavar="S",
        help="TRT-LLM stream_interval: one response per S iterations carrying "
        "S tokens. Divides the response rate without touching the token rate. "
        "The 355778 capture ran 40; si=1 offers 40x the responses.",
    )
    engine.add_argument(
        "--max-backlog",
        type=int,
        default=None,
        metavar="N",
        help="abort once the loop is N responses behind the engine; without it "
        "an overloaded run queues until the process dies",
    )
    engine.add_argument(
        "--engine",
        choices=("constant", "batch-dependent"),
        default="constant",
        help="iteration-time model",
    )
    engine.add_argument(
        "--iteration-ms",
        type=float,
        default=34.0,
        help="constant model only",
    )
    engine.add_argument(
        "--engine-base-ms",
        type=float,
        default=22.296,
        help="batch-dependent model: fixed per-iteration cost",
    )
    engine.add_argument(
        "--engine-per-request-us",
        type=float,
        default=29.811,
        help="batch-dependent model: marginal cost per sequence per rank",
    )

    p.add_argument(
        "--gil-noise",
        type=int,
        default=0,
        metavar="N",
        help="extra GIL-capable threads; the decode worker had 50 total",
    )
    p.add_argument(
        "--gil-noise-hold-us",
        type=float,
        default=20.0,
        help="GIL held per wake-up by each noise thread",
    )
    p.add_argument(
        "--gil-noise-period-us",
        type=float,
        default=200.0,
        help="sleep between wake-ups; N x hold/period is the GIL demand added",
    )
    p.add_argument(
        "--blocking-threads",
        type=int,
        default=8,
        help="threads in the spawn_blocking pool (engine.rs:85)",
    )
    p.add_argument(
        "--cost-scale",
        type=float,
        default=1.0,
        help="multiply every measured stage cost (sanity check on structure)",
    )
    p.add_argument("--json", action="store_true", help="emit raw JSON instead")
    return p.parse_args(argv)


def _config(args: argparse.Namespace, egress: str) -> SimConfig:
    if args.batch_per_rank is not None:
        batch = BatchConfig(per_rank=args.batch_per_rank, ranks=args.ranks)
    else:
        batch = BatchConfig(total=args.batch_total or 132, ranks=args.ranks)

    if args.engine == "batch-dependent":
        iteration: object = BatchDependentIteration(
            base_ms=args.engine_base_ms,
            per_request_us=args.engine_per_request_us,
        )
    else:
        iteration = ConstantIteration(iteration_ms=args.iteration_ms)

    return SimConfig(
        egress=egress,
        requests=args.requests,
        max_tokens=args.max_tokens,
        isl=args.isl,
        arrival=args.arrival,
        qps=args.qps,
        arrival_seed=args.arrival_seed,
        concurrency=args.concurrency,
        batch=batch,
        iteration=iteration,  # type: ignore[arg-type]
        stream_interval=args.stream_interval,
        max_backlog=args.max_backlog,
        blocking_threads=args.blocking_threads,
        costs=Costs().with_scale(args.cost_scale),
        gil_noise=GilNoiseConfig(
            threads=args.gil_noise,
            hold_us=args.gil_noise_hold_us,
            period_us=args.gil_noise_period_us,
        ),
    )


def _row(label: str, values: List[str], width: int = 22) -> str:
    return f"{label:<34}" + "".join(f"{v:>{width}}" for v in values)


def _print_report(results: List[SimResult]) -> None:
    cfg = results[0].config
    modes = [r.config.egress for r in results]

    print()
    print("=" * (34 + 22 * len(results)))
    print("  dynamo column of ASYNCIO_GIL_PATH.md -- simulated")
    print("=" * (34 + 22 * len(results)))
    print(f"  INGRESS  {cfg.describe_ingress()} · {cfg.requests} requests")
    print(
        f"  ENGINE   {cfg.batch.describe()} · {cfg.iteration.describe()}"
        f" · {cfg.max_tokens} tok/req"
    )
    print(
        f"           full-batch iteration {cfg.full_batch_iteration_ms:.2f} ms"
        f" · residency {cfg.residency_s:.1f} s"
        f" · steady-state {cfg.steady_state_qps:.0f} qps"
    )
    print(
        f"           responses/iteration = batch/si ="
        f" {cfg.batch.total_batch}/{cfg.stream_interval} ="
        f" {cfg.responses_per_iteration:.1f}"
        f"  ->  {cfg.offered_response_rate:,.0f} responses/s offered"
    )
    print(
        f"  IPC backend {'zmq' if ipc.USING_ZMQ else 'socketpair (PAIR)'}"
        f" · real push_egress.py {'YES' if USING_REAL_PUSH_EGRESS else 'NO'}"
    )
    print(
        f"  extra GIL-capable threads: {results[0].gil_noise_threads}"
        f"{'  <-- contention regime OFF' if not results[0].gil_noise_threads else ''}"
    )
    print("-" * (34 + 22 * len(results)))
    print(_row("", [m for m in modes]))
    print("-" * (34 + 22 * len(results)))

    print(_row("responses delivered", [f"{r.responses}" for r in results]))
    print(_row("wall clock (s)", [f"{r.wall_s:.2f}" for r in results]))
    print(
        _row(
            "steady-state window (s)",
            [
                f"{r.window_s:.2f}" + ("" if r.window_valid else " (whole run)")
                for r in results
            ],
        )
    )
    print(
        _row(
            "response demand (/s)",
            [f"{r.response_demand_per_s:,.0f}" for r in results],
        )
    )
    print()
    print(
        _row(
            "per-response cost on the loop",
            [f"{r.loop_us_per_response:.2f} us" for r in results],
        )
    )
    print(_row("  vs serve's 1.94 us", [f"{r.serve_ratio:.1f}x" for r in results]))
    print(
        _row("loop capacity (/s)", [f"{r.loop_capacity_per_s:,.0f}" for r in results])
    )
    print(_row("loop load", [f"{100 * r.loop_load:.1f} %" for r in results]))
    print()
    print(
        _row(
            "OFFERED load (offer/capacity)",
            [
                f"{100 * r.offered_load:.1f} %"
                + (" OVER" if r.offered_load > 1 else "")
                for r in results
            ],
        )
    )
    print()
    print(_row("backlog: max", [f"{r.backlog_max:,}" for r in results]))
    print(_row("backlog: final", [f"{r.backlog_final:,}" for r in results]))
    print(
        _row(
            "backlog: growth (/s)",
            [
                f"{r.backlog_growth_per_s:+,.0f}"
                + (" GROWING" if r.backlog_growing else "")
                for r in results
            ],
        )
    )
    print()
    print(_row("achieved qps", [f"{r.achieved_qps:,.0f}" for r in results]))
    print(_row("IPC messages (engine iters)", [f"{r.ipc_messages}" for r in results]))
    # One quantity, two names: the engine's batch IS the number of responses
    # sharing a single ready-deque entry, because the whole iteration crosses
    # as one IPC message and costs one notify_many.
    print(
        _row(
            "emitted/iter = deque entry",
            [f"{r.mean_engine_batch:.1f} ({100 * r.batch_fill:.0f}%)" for r in results],
        )
    )
    print(
        _row(
            "  of which delivered",
            [f"{r.mean_delivered_per_message:.1f}" for r in results],
        )
    )
    print(
        _row("observed iteration (ms)", [f"{r.mean_iteration_ms:.2f}" for r in results])
    )
    print(
        _row(
            "loop hand-offs (rust->loop)",
            [f"{r.loop_handoffs}" for r in results],
        )
    )
    print(
        _row(
            "  per response",
            [f"{r.loop_handoffs / max(1, r.responses):.3f}" for r in results],
        )
    )
    print(
        _row("total call_soon_threadsafe", [f"{r.probe['enqueues']}" for r in results])
    )
    print(
        _row(
            "spawn_blocking GIL acq",
            [f"{r.blocking_gil_acquisitions:,}" for r in results],
        )
    )
    print(
        _row(
            "  per response",
            [
                f"{r.blocking_gil_acquisitions / max(1, r.responses):.3f}"
                for r in results
            ],
        )
    )
    print()
    print(
        _row("admission wait p50 (ms)", [f"{r.queue_wait['p50']:.2f}" for r in results])
    )
    print(
        _row("admission wait p90 (ms)", [f"{r.queue_wait['p90']:.2f}" for r in results])
    )
    print(
        _row("admission wait p99 (ms)", [f"{r.queue_wait['p99']:.2f}" for r in results])
    )
    print(
        _row("admission wait max (ms)", [f"{r.queue_wait['max']:.2f}" for r in results])
    )
    print()
    print(
        _row("loop LAG p50 (ms)", [f"{r.probe['lag']['p50_ms']:.2f}" for r in results])
    )
    print(
        _row("loop LAG p90 (ms)", [f"{r.probe['lag']['p90_ms']:.2f}" for r in results])
    )
    print()
    print(_row("TTFT p50 (ms)", [f"{r.ttft['p50']:.1f}" for r in results]))
    print(_row("TPOT p50 (ms)", [f"{r.tpot['p50']:.2f}" for r in results]))
    print("-" * (34 + 22 * len(results)))

    print()
    print("Reference -- job 355778, decode rank 0, npw=0, push egress.")
    print("Measured directly off nsys_355778_disagg_gen-rank0.sqlite; re-derive with")
    print("  python3 -m egress_experiments.capture_params <that .sqlite>")
    print(
        f"  per-response cost on the loop  85.35 us   (serve {SERVE_LOOP_US_PER_RESPONSE} us, 44.0x)"
    )
    print("    handle_response 23.97 + build_response 50.65 + push_send 10.72 (p50)")
    print("  loop capacity                  11,717/s")
    print("  response demand                 3,841/s")
    print("  loop load                          32.8 %")
    print("  batch = responses/deque entry      200.1")
    print("  engine iteration                   52.10 ms")
    print("  ingress                            42.4 qps, arrival CV 1.81")
    print("  tokens per request                  88.9")
    print("  GIL-capable threads                   50")
    print()
    print("From the document's prose (different windows/jobs, kept for contrast):")
    print("  admission wait   p50 0.86 · p90 13.66 · p99 21.78 · max 26.60 ms")
    print("    (queue_probe on job 355971, not this capture)")
    print("  TTFT p50 214 ms · TPOT p50 46.4 ms   (client-side, whole run)")

    print()
    print("CAVEATS")
    for r in results:
        if r.backlog_aborted:
            print(f"  · {r.config.egress}: ABORTED at --max-backlog. The loop could")
            print(
                f"    not keep up: {r.config.offered_response_rate:,.0f} responses/s"
                f" offered against {r.loop_capacity_per_s:,.0f}/s of capacity"
            )
            print(
                f"    ({100 * r.offered_load:.0f} %). Raise --stream-interval or"
                " shrink the batch."
            )
        elif r.overloaded:
            print(
                f"  · {r.config.egress}: offered load {100 * r.offered_load:.0f} %"
                f" and the backlog grew {r.backlog_growth_per_s:+,.0f}/s."
            )
            print("    Latency figures above are a lower bound -- the run never")
            print("    reached steady state.")
    if not results[0].window_valid:
        print("  · No steady-state window: the run was too short for the batch")
        print("    to fill before arrivals stopped, so every figure below is a")
        print("    whole-run average over ramp-up and drain. Raise --requests.")
    print("  · The engine is opaque: a child process that sleeps for an iteration.")
    print("    GPU time, batching policy and KV behaviour are NOT modelled.")
    print("  · The 'tokio' side is Python and holds the GIL where real tokio does")
    print("    not, so it charges the push path for work that is free on the real")
    print("    worker. Any push win reported here is a lower bound.")
    if not results[0].gil_noise_threads:
        print("  · --gil-noise 0: only 3 GIL-capable threads, versus the decode")
        print("    worker's ~45. Cross-thread GIL acquisition -- the mechanism the")
        print("    pull path pays for per response -- is therefore nearly free here.")
        print("    Structural results hold; the pull/push latency gap needs")
        print("    --gil-noise 42 to appear.")
    for r in results:
        if r.batch_fill < 0.8:
            print(
                f"  · {r.config.egress}: the engine batch only reached "
                f"{r.mean_engine_batch:.0f} of {r.config.batch.total_batch} "
                f"({100 * r.batch_fill:.0f}%). Ramp-up and drain dominate a"
            )
            print("    short run; raise --requests or --qps to reach steady")
            print("    state. Loop load is understated until it does.")
        if r.errors:
            print(f"  · {r.config.egress}: {len(r.errors)} error(s): {r.errors[:3]}")
    print()


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    modes = ["pull", "push"] if args.egress == "both" else [args.egress]
    results = [run_simulation(_config(args, mode)) for mode in modes]

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "egress": r.config.egress,
                        "ingress": r.config.describe_ingress(),
                        "offered_qps": r.config.offered_qps,
                        "achieved_qps": r.achieved_qps,
                        "engine": r.config.iteration.describe(),
                        "stream_interval": r.config.stream_interval,
                        "responses_per_iteration": r.config.responses_per_iteration,
                        "offered_response_rate": r.config.offered_response_rate,
                        "offered_load": r.offered_load,
                        "backlog_max": r.backlog_max,
                        "backlog_final": r.backlog_final,
                        "backlog_growth_per_s": r.backlog_growth_per_s,
                        "backlog_aborted": r.backlog_aborted,
                        "batch_total": r.config.batch.total_batch,
                        "batch_per_rank": r.config.batch.per_rank_batch,
                        "ranks": r.config.batch.ranks,
                        "observed_engine_batch": r.mean_engine_batch,
                        "observed_iteration_ms": r.mean_iteration_ms,
                        "wall_s": r.wall_s,
                        "responses": r.responses,
                        "response_demand_per_s": r.response_demand_per_s,
                        "loop_us_per_response": r.loop_us_per_response,
                        "loop_capacity_per_s": r.loop_capacity_per_s,
                        "loop_load": r.loop_load,
                        "ipc_messages": r.ipc_messages,
                        "responses_per_deque_entry": r.responses_per_deque_entry,
                        "loop_handoffs": r.loop_handoffs,
                        "enqueues": r.probe["enqueues"],
                        "queue_wait_ms": r.queue_wait,
                        "ttft_ms": r.ttft,
                        "tpot_ms": r.tpot,
                        "lag_ms": r.probe["lag"],
                        "gil_noise_threads": r.gil_noise_threads,
                        "errors": r.errors,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
    else:
        _print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
