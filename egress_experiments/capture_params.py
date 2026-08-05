#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive simulation parameters from an nsys capture, and emit the command.

    python3 -m egress_experiments.capture_params \\
        /tmp/p355778/355778-*/decode_worker_0/nsys_355778_disagg_gen-rank0.sqlite

Everything the simulation needs is measurable in the ``.sqlite`` nsys export,
so none of it has to be taken on trust from the prose:

===========================  ============================================
parameter                    measured from
===========================  ============================================
ingress qps                  ``trtllm:generate_locally`` start timestamps
arrival burstiness (CV)      their inter-arrival distribution
tokens per request           responses / requests
engine iteration time        ``_handle_responses`` inter-arrival
responses per iteration      ``handle_response`` bucketed by iteration
loop stage costs             p50 of the three loop ranges
GIL-capable threads          distinct tids that ever held ``Holding GIL``
===========================  ============================================

The diagram's stage figures are **p50**, not mean. On this capture the two
differ by 32 % (85.35 vs 112.64 us for the three-stage total), so the statistic
matters more than the extra digit suggests.

Requires only stdlib ``sqlite3``. Point it at any rank's export; the diagram
used decode rank 0.
"""

from __future__ import annotations

import argparse
import bisect
import re
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

#: The three loop stages the diagram totals to 85.34 us.
LOOP_STAGES = ("handle_response", "trtllm:build_response", "trtllm:push_send")
#: Stages before ``llm.generate_async`` returns, all on the loop.
INGRESS_STAGES = (
    "trtllm:normalize_request",
    "trtllm:setup_disagg_params",
    "trtllm:prepare_input",
    "trtllm:sampling_params",
)
#: The executor's per-iteration range.
ITERATION_RANGE = "_handle_responses"
#: The handler's per-request range. Its END timestamps are unusable (push-mode
#: ranges span awaits and nsys leaves some unclosed), but starts are fine.
REQUEST_RANGE = "trtllm:generate_locally"


def _p50(values: List[int]) -> float:
    return statistics.median(values) / 1e3 if values else 0.0


@dataclass
class CaptureParams:
    sqlite_path: str
    requests: int
    request_span_s: float
    qps: float
    arrival_cv: float
    iterations: int
    iteration_ms: float
    responses: int
    responses_per_iteration: float
    tokens_per_request: float
    gil_threads: int
    stage_p50_us: Dict[str, float]
    stage_mean_us: Dict[str, float]
    ingress_p50_us: float
    engine_submit_p50_us: float
    engine_gen_reqs: Optional[int]

    @property
    def loop_p50_total_us(self) -> float:
        return sum(self.stage_p50_us.get(s, 0.0) for s in LOOP_STAGES)

    @property
    def loop_mean_total_us(self) -> float:
        return sum(self.stage_mean_us.get(s, 0.0) for s in LOOP_STAGES)

    @property
    def response_demand_per_s(self) -> float:
        return self.responses_per_iteration / (self.iteration_ms / 1000.0)

    @property
    def loop_load(self) -> float:
        return self.response_demand_per_s * self.loop_p50_total_us / 1e6


#: Any of these stages taking more than a second is not a measurement, it is a
#: range nsys never saw closed -- either the profiler stopped mid-range
#: (``--duration``) or the range spans awaits and the process exited inside it.
#: The real 355778 capture has this on ``trtllm:generate_locally`` and
#: ``trtllm:push_egress``, whose means come out as ~97,000 seconds. p50 shrugs
#: it off; the mean does not, so drop them and say how many.
_UNCLOSED_NS = 1_000_000_000

_dropped: Dict[str, int] = {}


def _durations(conn: sqlite3.Connection, name: str) -> List[int]:
    raw = [
        row[0]
        for row in conn.execute(
            "select e.end - e.start from NVTX_EVENTS e "
            "join StringIds s on s.id = e.textId "
            "where s.value = ? and e.end is not null",
            (name,),
        )
    ]
    kept = [d for d in raw if 0 <= d < _UNCLOSED_NS]
    if len(kept) != len(raw):
        _dropped[name] = len(raw) - len(kept)
    return kept


def _starts(conn: sqlite3.Connection, name: str) -> List[int]:
    return [
        row[0]
        for row in conn.execute(
            "select e.start from NVTX_EVENTS e join StringIds s on s.id = e.textId "
            "where s.value = ? order by e.start",
            (name,),
        )
    ]


def extract(sqlite_path: str) -> CaptureParams:
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)

    iterations = _starts(conn, ITERATION_RANGE)
    if len(iterations) < 2:
        raise SystemExit(
            f"{sqlite_path}: only {len(iterations)} '{ITERATION_RANGE}' ranges; "
            "is this a decode worker capture with the NVTX patch armed?"
        )
    gaps = [(b - a) / 1e6 for a, b in zip(iterations, iterations[1:])]
    iteration_ms = statistics.median(gaps)

    responses = _starts(conn, "handle_response")
    per_iteration: List[int] = [0] * len(iterations)
    for timestamp in responses:
        index = bisect.bisect_right(iterations, timestamp) - 1
        if 0 <= index < len(iterations):
            per_iteration[index] += 1
    populated = [n for n in per_iteration if n]

    requests = _starts(conn, REQUEST_RANGE)
    request_span_s = (requests[-1] - requests[0]) / 1e9 if len(requests) > 1 else 0.0
    inter_arrival = [(b - a) / 1e6 for a, b in zip(requests, requests[1:])]
    mean_ia = statistics.mean(inter_arrival) if inter_arrival else 0.0
    cv = (statistics.pstdev(inter_arrival) / mean_ia) if mean_ia else 0.0

    stage_p50 = {}
    stage_mean = {}
    for stage in LOOP_STAGES:
        durations = _durations(conn, stage)
        if durations:
            stage_p50[stage] = _p50(durations)
            stage_mean[stage] = statistics.mean(durations) / 1e3

    ingress_p50 = sum(_p50(_durations(conn, s)) for s in INGRESS_STAGES)
    submit_p50 = _p50(_durations(conn, "trtllm:engine_submit"))

    gil_threads = conn.execute(
        "select count(distinct e.globalTid) from NVTX_EVENTS e "
        "join StringIds s on s.id = e.textId where s.value = 'Holding GIL'"
    ).fetchone()[0]

    # The executor names its batch in the range text, one per forward step:
    #   "[Executor] _forward_step 35057: 0 ctx reqs, 976 gen reqs"
    # Joined to NVTX_EVENTS and bounded by the iteration window: StringIds
    # also holds text from ramp-up, whose smaller batches would drag the
    # median well below the steady-state value.
    gen_reqs = None
    counts = [
        int(match.group(1))
        for (text,) in conn.execute(
            "select s.value from NVTX_EVENTS e join StringIds s on s.id = e.textId "
            "where s.value like '%_forward_step%gen reqs%' "
            "and e.start between ? and ?",
            (iterations[0], iterations[-1]),
        )
        if (match := re.search(r"(\d+)\s+gen reqs", text))
    ]
    if counts:
        gen_reqs = int(statistics.median(counts))

    conn.close()
    return CaptureParams(
        sqlite_path=sqlite_path,
        requests=len(requests),
        request_span_s=request_span_s,
        qps=(len(requests) / request_span_s) if request_span_s else 0.0,
        arrival_cv=cv,
        iterations=len(iterations),
        iteration_ms=iteration_ms,
        responses=len(responses),
        responses_per_iteration=(statistics.mean(populated) if populated else 0.0),
        tokens_per_request=(len(responses) / len(requests)) if requests else 0.0,
        gil_threads=gil_threads,
        stage_p50_us=stage_p50,
        stage_mean_us=stage_mean,
        ingress_p50_us=ingress_p50,
        engine_submit_p50_us=submit_p50,
        engine_gen_reqs=gen_reqs,
    )


def command(
    params: CaptureParams,
    requests: int,
    gil_noise: bool,
    ranks: int = 8,
    stream_interval: int = 40,
    max_tokens: int = 400,
) -> str:
    """Emit the run command in the capture's real geometry.

    ``max_tokens`` is the one figure deliberately NOT taken from the capture.
    Steady state there implies ~3,600 tokens per request, i.e. a 189 s
    residency -- the batch would take over three minutes to fill. Shrinking it
    keeps ``batch = qps x residency`` intact at a runnable scale; the qps is
    rescaled to match, and every quantity that sets loop load (batch,
    stream_interval, iteration time) stays exact.
    """
    per_rank = params.engine_gen_reqs or 986
    # qps that holds the same batch with the shortened residency.
    qps = (per_rank * ranks) / (max_tokens * params.iteration_ms / 1000.0)
    parts = [
        "python3 -m egress_experiments.run_experiment",
        "  --egress push",
        f"  --qps {qps:.0f}",
        "  --arrival poisson",
        f"  --batch-per-rank {per_rank} --ranks {ranks}",
        f"  --stream-interval {stream_interval}",
        "  --engine constant",
        f"  --iteration-ms {params.iteration_ms:.1f}",
        f"  --max-tokens {max_tokens}",
        f"  --requests {requests}",
    ]
    if gil_noise:
        # Aim the extra threads at the capture's GIL occupancy rather than at
        # its thread count: N threads at hold/period duty each add N*duty of
        # GIL demand, and 47 threads at the default 10 % duty would saturate
        # it several times over.
        extra = max(0, params.gil_threads - 3)  # loop + dispatch + tokio
        parts += [
            f"  --gil-noise {extra}",
            "  --gil-noise-hold-us 20",
            "  --gil-noise-period-us 1450",
        ]
    return " \\\n".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="capture_params",
        description="Derive simulation parameters from an nsys sqlite export",
    )
    parser.add_argument("sqlite", help="nsys .sqlite export of a decode worker rank")
    parser.add_argument(
        "--ranks",
        type=int,
        default=8,
        help="attention-DP ranks (server-gen-si40.yaml: tensor_parallel_size 8, "
        "enable_attention_dp true)",
    )
    parser.add_argument(
        "--stream-interval",
        type=int,
        default=40,
        help="TRT-LLM stream_interval (server-gen-si40.yaml:62)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="requests for the emitted command (default: enough for steady state)",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    p = extract(args.sqlite)

    print()
    print("=" * 78)
    print(f"  measured from {p.sqlite_path}")
    print("=" * 78)
    print()
    print("INGRESS")
    print(f"  requests                    {p.requests}")
    print(f"  span                        {p.request_span_s:.3f} s")
    print(f"  qps                         {p.qps:.1f}")
    print(
        f"  inter-arrival CV            {p.arrival_cv:.2f}"
        "   (0 = constant, 1 = poisson)"
    )
    print(f"  tokens per request          {p.tokens_per_request:.1f}")
    print()
    print("ENGINE")
    print(f"  iterations                  {p.iterations}")
    print(f"  iteration time (p50)        {p.iteration_ms:.2f} ms")
    print(f"  responses per iteration     {p.responses_per_iteration:.1f}")
    print(f"  responses total             {p.responses}")
    if p.engine_gen_reqs:
        # The executor's per-forward-step count is the PER-RANK batch. Multiply
        # by the ADP ranks and divide by stream_interval to get what actually
        # reaches Python -- which is the number that sets loop load.
        total = p.engine_gen_reqs * args.ranks
        predicted = total / args.stream_interval
        err = 100 * (p.responses_per_iteration - predicted) / predicted
        print(f"  executor 'gen reqs'/rank    {p.engine_gen_reqs}")
        print(
            f"  x {args.ranks} ADP ranks / si {args.stream_interval}"
            f"        = {predicted:.1f} responses/iteration"
            f"   vs {p.responses_per_iteration:.1f} measured  ({err:+.1f}%)"
        )
        print(f"  -> decode batch in flight   {total:,}")
    print()
    print("LOOP STAGES                      p50          mean")
    for stage in LOOP_STAGES:
        if stage in p.stage_p50_us:
            print(
                f"  {stage:<28} {p.stage_p50_us[stage]:7.2f} us  "
                f"{p.stage_mean_us[stage]:7.2f} us"
            )
    print(
        f"  {'TOTAL':<28} {p.loop_p50_total_us:7.2f} us  "
        f"{p.loop_mean_total_us:7.2f} us"
    )
    print("  ^ the diagram quotes the p50 column")
    if _dropped:
        for name, count in sorted(_dropped.items()):
            print(f"  dropped {count} unclosed '{name}' range(s) from the mean")
    print()
    print("INGRESS STAGES (on the loop)")
    print(f"  pre-submit stages (p50 sum)  {p.ingress_p50_us:7.2f} us")
    print(f"  trtllm:engine_submit (p50)   {p.engine_submit_p50_us:7.2f} us")
    print()
    print("DERIVED")
    print(f"  response demand             {p.response_demand_per_s:,.0f} /s")
    print(f"  loop capacity               {1e6 / p.loop_p50_total_us:,.0f} /s")
    print(f"  loop load                   {100 * p.loop_load:.1f} %")
    print(f"  GIL-capable threads         {p.gil_threads}")
    print()

    # Steady state needs one full residency to fill the batch, then a window
    # at least as long as the capture's.
    per_rank = p.engine_gen_reqs or 986
    scaled_tokens = 400
    scaled_qps = (per_rank * args.ranks) / (scaled_tokens * p.iteration_ms / 1000.0)
    residency_s = scaled_tokens * p.iteration_ms / 1000.0
    default_requests = int(scaled_qps * (residency_s + 2 * p.request_span_s))
    requests = args.requests or max(50, default_requests)

    print("-" * 78)
    print("  COMMAND -- structural + loop-cost measurements")
    print("-" * 78)
    print(command(p, requests, False, args.ranks, args.stream_interval, scaled_tokens))
    print()
    print("-" * 78)
    print("  COMMAND -- same, with the contention regime approximated")
    print("-" * 78)
    print(command(p, requests, True, args.ranks, args.stream_interval, scaled_tokens))
    print()
    print("-" * 78)
    print("  COMMAND -- same batch, stream_interval=1: the loop cannot keep up")
    print("-" * 78)
    print(
        command(p, requests, False, args.ranks, 1, scaled_tokens)
        + " \\\n  --max-backlog 200000"
    )
    print()
    print("NOTES")
    print("  · 'gen reqs' is the executor's PER-RANK decode batch. What reaches")
    print("    the worker's Python is gen_reqs x ranks / stream_interval, and")
    print("    that closes the geometry to ~1.5%. stream_interval is not in the")
    print("    trace -- it is read from server-gen-si40.yaml:62 and passed in")
    print("    via --stream-interval, so the check above is only as good as it.")
    print("  · max_tokens in the emitted commands is scaled down. Steady state")
    print("    here implies ~3,600 tokens/request (a 189 s residency); the qps")
    print("    is rescaled to hold the same batch at a runnable scale.")
    print(f"  · measured arrival CV is {p.arrival_cv:.2f}; --arrival poisson is CV 1,")
    print("    the closest of the available processes. Real traffic was burstier.")
    print("  · the GIL totals (acquisitions, hold, wait) are NOT reproducible by")
    print("    the simulation. They are a property of a real CPython process with")
    print(f"    {p.gil_threads} GIL-capable threads under nsys; --gil-noise only")
    print("    approximates the regime.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
