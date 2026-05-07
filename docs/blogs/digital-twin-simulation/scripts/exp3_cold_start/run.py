# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exp 3: Engine cold-start time sweep — when does the planner stop keeping up?

Sweep `startup_time` from 0 to 300 s in 30-second steps with the SLA-mode
planner (`ttft=1500`, `itl=50`, both throughput- and load-based scaling
enabled). Mirrors the earlier Llama-3.1-8B sweep, but on Qwen3-32B so it
lines up with the rest of the digital twin blog.
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

EXP_NAME = "exp3_cold_start"
STARTUP_VALUES_S = list(range(0, 301, 30))


def build_invocations() -> list[ReplayInvocation]:
    invs: list[ReplayInvocation] = []
    for s in STARTUP_VALUES_S:
        tag = f"startup_{s:03d}s"
        cfg = base_planner_config(
            f"{tag}.html",
            mode="agg",
            optimization_target="sla",
            ttft=1500,
            itl=50,
            enable_throughput_scaling=True,
            enable_load_scaling=True,
            load_adjustment_interval=10,
            throughput_adjustment_interval=300,
        )
        invs.append(
            ReplayInvocation(
                tag=tag,
                params={
                    "startup_time_s": s,
                    "mode": "agg",
                    "optimization_target": "sla",
                },
                extra_engine_args=base_engine_args(startup_time=s if s > 0 else None),
                planner_config=cfg,
                cli_extra=["--num-workers", "2"],
            )
        )
    return invs


def _on_done(row: dict) -> None:
    ttft = row["ttft_ms"] or {}
    itl = row["itl_ms"] or {}
    print(
        f"[done {row['tag']:18s}] wall={row['wall_time_s']:5.1f}s "
        f"ttft_avg={ttft.get('avg')}  ttft_p90={ttft.get('p90')}  "
        f"itl_avg={itl.get('avg')}  itl_p90={itl.get('p90')}  "
        f"gpu_h={row['gpu_hours']}  events_up/down={row['scale_up_events']}/{row['scale_down_events']}",
        flush=True,
    )


if __name__ == "__main__":
    invs = build_invocations()
    print(f"[{EXP_NAME}] {len(invs)} runs total", flush=True)
    run_sweep(invs, EXP_NAME, max_workers=12, on_done=_on_done)
