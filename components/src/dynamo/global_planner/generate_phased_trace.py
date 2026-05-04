# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a synthetic mooncake-format trace that alternates between
prefill-heavy and decode-heavy phases.

This is intended for offline-replay testing of disagg planner behaviour
under a tight ``max_gpu_budget`` — workloads that swing the load between
the prefill and decode pools so you can observe the local planner
rebalancing replicas across phases.

Phase shape (per request):
  - Prefill-heavy: large ISL, small OSL → most compute is in prefill.
  - Decode-heavy: small ISL, large OSL → most compute is in decode.

The output JSONL matches ``traces/mooncake_trace.jsonl`` schema:

    {"timestamp": <ms>, "input_length": N, "output_length": M,
     "hash_ids": [...]}

``hash_ids`` are monotonically increasing per request (no KV cache reuse
modelled). Each id represents ``--block-size`` tokens.

Example:

    python -m dynamo.global_planner.generate_phased_trace \\
        --output /tmp/phased_trace.jsonl \\
        --phases 4 --phase-seconds 90 --rps 8 \\
        --block-size 512 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass


@dataclass
class PhaseSpec:
    name: str
    isl_mean: int
    isl_jitter: int  # ±jitter
    osl_mean: int
    osl_jitter: int


# Two opposing shapes; we cycle them per phase.
PHASE_PREFILL_HEAVY = PhaseSpec(
    name="prefill_heavy",
    isl_mean=12000,
    isl_jitter=2000,
    osl_mean=200,
    osl_jitter=80,
)
PHASE_DECODE_HEAVY = PhaseSpec(
    name="decode_heavy",
    isl_mean=500,
    isl_jitter=200,
    osl_mean=8000,
    osl_jitter=2000,
)


def _phase_for(idx: int) -> PhaseSpec:
    return PHASE_PREFILL_HEAVY if idx % 2 == 0 else PHASE_DECODE_HEAVY


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a phased prefill/decode trace (mooncake JSONL)."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the JSONL trace.",
    )
    parser.add_argument(
        "--phases",
        type=int,
        default=4,
        help="Number of phases. Phase N is prefill-heavy if even, decode-heavy if odd.",
    )
    parser.add_argument(
        "--phase-seconds",
        type=int,
        default=90,
        help="Duration of each phase in seconds (default: 90).",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=8.0,
        help="Average requests per second (uniform within each phase).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Tokens per hash_id (default: 512).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility.",
    )
    return parser


def _generate(args: argparse.Namespace) -> tuple[int, list[dict]]:
    rng = random.Random(args.seed)
    interval_ms = 1000.0 / args.rps
    next_hash_id = 0
    records: list[dict] = []

    for phase_idx in range(args.phases):
        spec = _phase_for(phase_idx)
        phase_start_ms = phase_idx * args.phase_seconds * 1000
        n_requests = int(args.rps * args.phase_seconds)
        for k in range(n_requests):
            ts_ms = int(phase_start_ms + k * interval_ms + rng.uniform(0, interval_ms))
            isl = max(
                1,
                int(rng.gauss(spec.isl_mean, max(1, spec.isl_jitter / 2))),
            )
            isl = max(spec.isl_mean - spec.isl_jitter, isl)
            isl = min(spec.isl_mean + spec.isl_jitter, isl)
            osl = max(
                1,
                int(rng.gauss(spec.osl_mean, max(1, spec.osl_jitter / 2))),
            )
            osl = max(spec.osl_mean - spec.osl_jitter, osl)
            osl = min(spec.osl_mean + spec.osl_jitter, osl)
            n_blocks = max(1, (isl + args.block_size - 1) // args.block_size)
            hash_ids = list(range(next_hash_id, next_hash_id + n_blocks))
            next_hash_id += n_blocks
            records.append(
                {
                    "timestamp": ts_ms,
                    "input_length": isl,
                    "output_length": osl,
                    "hash_ids": hash_ids,
                }
            )
    records.sort(key=lambda r: r["timestamp"])
    return next_hash_id, records


def _print_summary(args: argparse.Namespace, records: list[dict]) -> None:
    print(f"Wrote {len(records)} records to {args.output}")
    print(f"Total span: {records[-1]['timestamp'] / 1000:.1f}s")
    for phase_idx in range(args.phases):
        spec = _phase_for(phase_idx)
        lo = phase_idx * args.phase_seconds * 1000
        hi = (phase_idx + 1) * args.phase_seconds * 1000
        in_phase = [r for r in records if lo <= r["timestamp"] < hi]
        if not in_phase:
            continue
        avg_isl = sum(r["input_length"] for r in in_phase) / len(in_phase)
        avg_osl = sum(r["output_length"] for r in in_phase) / len(in_phase)
        print(
            f"  phase {phase_idx} ({spec.name}, "
            f"{lo/1000:.0f}–{hi/1000:.0f}s): "
            f"{len(in_phase)} reqs, avg ISL={avg_isl:.0f}, avg OSL={avg_osl:.0f}"
        )


def main() -> int:
    args = _build_parser().parse_args()
    _, records = _generate(args)
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r))
            f.write("\n")
    _print_summary(args, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
