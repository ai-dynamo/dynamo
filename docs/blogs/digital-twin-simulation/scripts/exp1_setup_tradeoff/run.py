# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exp 1: Setup-tradeoff sweep — static deployments vs planner-driven runs.

Two sweeps are aggregated into a single `results.json`:

A) **Static-deployment baseline** (no `--planner-config`). Worker counts are
   fixed, so each row gives one (GPU-hours, TTFT, ITL) point on the static
   Pareto curve. We sweep agg `--num-workers` ∈ {2..8} and disagg balanced
   ratios (P, D) ∈ {(1,1), (1,2), (2,2), (2,3), (3,3), (3,4)}.

B) **Planner runs** with `optimization_target` ∈ {throughput, latency, sla}
   for both agg and disagg. The `sla` variant is swept across TTFT targets
   {500, 750, 1000, 1500, 2500, 5000} ms while ITL stays at 50 ms.

All runs use Qwen/Qwen3-32B BF16, TP=2, vLLM 0.12.0 on H200-SXM, with
`startup_time=60` s for fair comparison (no effect on static runs since
worker count never changes).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script: `python exp1_setup_tradeoff/run.py`.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common import (  # noqa: E402
    DEFAULT_GPU_PER_REPLICA,
    ReplayInvocation,
    base_engine_args,
    base_planner_config,
    run_sweep,
)

EXP_NAME = "exp1_setup_tradeoff"
STARTUP_TIME_S = 60

AGG_STATIC_WORKERS = [2, 3, 4, 5, 6, 7, 8]
DISAGG_STATIC_PD = [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)]

SLA_TTFT_VALUES_MS = [500, 750, 1000, 1500, 2500, 5000]
SLA_ITL_MS = 50

PLANNER_TARGETS = ["throughput", "latency"]  # SLA is handled separately


def build_invocations() -> list[ReplayInvocation]:
    invs: list[ReplayInvocation] = []

    # ---- Sweep A: static deployments, agg ----
    for n in AGG_STATIC_WORKERS:
        tag = f"static_agg_w{n:02d}"
        invs.append(
            ReplayInvocation(
                tag=tag,
                params={"sweep": "static", "mode": "agg", "num_workers": n},
                extra_engine_args=base_engine_args(startup_time=STARTUP_TIME_S),
                planner_config=None,
                cli_extra=["--num-workers", str(n)],
            )
        )

    # ---- Sweep A: static deployments, disagg ----
    for p, d in DISAGG_STATIC_PD:
        tag = f"static_disagg_p{p}d{d}"
        prefill_engine = {
            **base_engine_args(startup_time=STARTUP_TIME_S),
            "worker_type": "prefill",
        }
        decode_engine = {
            **base_engine_args(startup_time=STARTUP_TIME_S),
            "worker_type": "decode",
        }
        invs.append(
            ReplayInvocation(
                tag=tag,
                params={
                    "sweep": "static",
                    "mode": "disagg",
                    "num_prefill": p,
                    "num_decode": d,
                },
                prefill_engine_args=prefill_engine,
                decode_engine_args=decode_engine,
                planner_config=None,
                cli_extra=[
                    "--num-prefill-workers",
                    str(p),
                    "--num-decode-workers",
                    str(d),
                ],
            )
        )

    # ---- Sweep B: planner runs, agg ----
    for target in PLANNER_TARGETS:
        tag = f"planner_agg_{target}"
        cfg = base_planner_config(
            f"{tag}.html",
            mode="agg",
            optimization_target=target,
        )
        invs.append(
            ReplayInvocation(
                tag=tag,
                params={
                    "sweep": "planner",
                    "mode": "agg",
                    "optimization_target": target,
                },
                extra_engine_args=base_engine_args(startup_time=STARTUP_TIME_S),
                planner_config=cfg,
                cli_extra=["--num-workers", "2"],
            )
        )
    for ttft in SLA_TTFT_VALUES_MS:
        tag = f"planner_agg_sla_ttft{ttft:04d}"
        cfg = base_planner_config(
            f"{tag}.html",
            mode="agg",
            optimization_target="sla",
            ttft=ttft,
            itl=SLA_ITL_MS,
        )
        invs.append(
            ReplayInvocation(
                tag=tag,
                params={
                    "sweep": "planner",
                    "mode": "agg",
                    "optimization_target": "sla",
                    "ttft_ms": ttft,
                    "itl_ms": SLA_ITL_MS,
                },
                extra_engine_args=base_engine_args(startup_time=STARTUP_TIME_S),
                planner_config=cfg,
                cli_extra=["--num-workers", "2"],
            )
        )

    # ---- Sweep B: planner runs, disagg ----
    for target in PLANNER_TARGETS:
        tag = f"planner_disagg_{target}"
        cfg = base_planner_config(
            f"{tag}.html",
            mode="disagg",
            optimization_target=target,
        )
        common_engine = base_engine_args(startup_time=STARTUP_TIME_S)
        invs.append(
            ReplayInvocation(
                tag=tag,
                params={
                    "sweep": "planner",
                    "mode": "disagg",
                    "optimization_target": target,
                },
                prefill_engine_args=common_engine,
                decode_engine_args=common_engine,
                planner_config=cfg,
                cli_extra=[
                    "--num-prefill-workers",
                    "1",
                    "--num-decode-workers",
                    "1",
                ],
            )
        )
    for ttft in SLA_TTFT_VALUES_MS:
        tag = f"planner_disagg_sla_ttft{ttft:04d}"
        cfg = base_planner_config(
            f"{tag}.html",
            mode="disagg",
            optimization_target="sla",
            ttft=ttft,
            itl=SLA_ITL_MS,
        )
        common_engine = base_engine_args(startup_time=STARTUP_TIME_S)
        invs.append(
            ReplayInvocation(
                tag=tag,
                params={
                    "sweep": "planner",
                    "mode": "disagg",
                    "optimization_target": "sla",
                    "ttft_ms": ttft,
                    "itl_ms": SLA_ITL_MS,
                },
                prefill_engine_args=common_engine,
                decode_engine_args=common_engine,
                planner_config=cfg,
                cli_extra=[
                    "--num-prefill-workers",
                    "1",
                    "--num-decode-workers",
                    "1",
                ],
            )
        )

    return invs


def _on_done(row: dict) -> None:
    ttft = row["ttft_ms"] or {}
    itl = row["itl_ms"] or {}
    print(
        f"[done {row['tag']:36s}] wall={row['wall_time_s']:5.1f}s "
        f"ttft_avg={ttft.get('avg')}  ttft_p90={ttft.get('p90')}  "
        f"itl_avg={itl.get('avg')}  itl_p90={itl.get('p90')}  "
        f"gpu_h={row['gpu_hours']}  rc={row['returncode']}",
        flush=True,
    )


if __name__ == "__main__":
    invs = build_invocations()
    print(f"[{EXP_NAME}] {len(invs)} runs total", flush=True)
    print(
        f"  GPU per replica = {DEFAULT_GPU_PER_REPLICA}, startup_time = {STARTUP_TIME_S}s",
        flush=True,
    )
    run_sweep(invs, EXP_NAME, max_workers=12, on_done=_on_done)
