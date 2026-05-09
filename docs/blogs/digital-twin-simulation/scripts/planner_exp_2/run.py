# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exp 2: Load-based scaling interval sweep — responsiveness vs oscillation.

Disable throughput-based scaling and keep only load-based scaling enabled.
Sweep `load_adjustment_interval` and observe how TTFT/ITL track latency vs
how often the planner toggles workers (the oscillation proxy). Engine
startup time is **0** here so this experiment isolates the interval knob;
its interaction with cold-start is the subject of Exp 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common import (  # noqa: E402
    ReplayInvocation,
    base_engine_args,
    base_planner_config,
    run_sweep,
)

EXP_NAME = "planner_exp_2"
LOAD_INTERVAL_VALUES_S = [1, 2, 5, 10, 20, 30, 60, 120, 300]


def build_invocations() -> list[ReplayInvocation]:
    invs: list[ReplayInvocation] = []
    for s in LOAD_INTERVAL_VALUES_S:
        tag = f"load_interval_{s:03d}s"
        cfg = base_planner_config(
            f"{tag}.html",
            mode="agg",
            optimization_target="sla",
            ttft=1500,
            itl=50,
            enable_throughput_scaling=False,
            enable_load_scaling=True,
            load_adjustment_interval=s,
        )
        invs.append(
            ReplayInvocation(
                tag=tag,
                params={
                    "load_adjustment_interval_s": s,
                    "mode": "agg",
                    "optimization_target": "sla",
                    "startup_time_s": 0,
                },
                extra_engine_args=base_engine_args(startup_time=None),
                planner_config=cfg,
                cli_extra=["--num-workers", "2"],
            )
        )
    return invs


def _on_done(row: dict) -> None:
    ttft = row["ttft_ms"] or {}
    itl = row["itl_ms"] or {}
    print(
        f"[done {row['tag']:24s}] wall={row['wall_time_s']:5.1f}s "
        f"ttft_avg={ttft.get('avg')}  ttft_p90={ttft.get('p90')}  "
        f"itl_avg={itl.get('avg')}  itl_p90={itl.get('p90')}  "
        f"gpu_h={row['gpu_hours']}  events_up/down={row['scale_up_events']}/{row['scale_down_events']}",
        flush=True,
    )


if __name__ == "__main__":
    invs = build_invocations()
    print(f"[{EXP_NAME}] {len(invs)} runs total", flush=True)
    run_sweep(invs, EXP_NAME, max_workers=12, on_done=_on_done)
