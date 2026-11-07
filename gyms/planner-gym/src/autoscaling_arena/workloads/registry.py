# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The v1 workload registry — 8 workloads spanning all three axes.

All synthetic workloads target ~1000 requests over ~180 s (matching the Mooncake
anchor's scale) so leaderboard rows are comparable. Five vary the **arrival**
axis at a fixed prefill-heavy shape; ``decode_heavy`` exercises the **shape**
axis (the P-vs-D scaling decision); ``shared_prefix`` exercises the **prefix**
axis (cache reuse). Full agentic multi-turn is deferred to v2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from autoscaling_arena.workloads import axes
from autoscaling_arena.workloads.generator import Workload

_DURATION = 180.0


# The optional Mooncake anchor ships with Dynamo. Prefer an explicit location,
# then look for a sibling checkout above this source tree.
def _dynamo_dir() -> Path:
    configured = os.environ.get("DYNAMO_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent, parent / "dynamo"):
            if (candidate / "lib" / "bench" / "testdata").is_dir():
                return candidate
    return (Path.cwd() / "dynamo").resolve()


_DYNAMO_DIR = _dynamo_dir()
_MOONCAKE_TRACE = (
    _DYNAMO_DIR / "lib" / "bench" / "testdata" / "mooncake_trace_1000.jsonl"
)


def _synthetic(name, description, arrival, shape, prefix_factory) -> Workload:
    return Workload(
        name=name,
        description=description,
        duration_s=_DURATION,
        arrival=arrival,
        shape=shape,
        prefix_factory=prefix_factory,
    )


WORKLOADS: Dict[str, Workload] = {
    # --- anchor (real trace) ---
    "mooncake": Workload(
        name="mooncake",
        description="Real Mooncake trace (anchor): prefill-heavy, long-context, real prefix reuse.",
        static_trace=_MOONCAKE_TRACE,
    ),
    # --- arrival axis (prefill-heavy shape, no sharing) ---
    "flat": _synthetic(
        "flat",
        "Constant load — steady-state baseline (oscillation should be ~0).",
        axes.flat(5.6),
        axes.prefill_heavy,
        axes.NoSharing,
    ),
    "staircase": _synthetic(
        "staircase",
        "Load steps up +1 req/s every 20 s — provisioning lag / under-provisioning.",
        axes.staircase(base=1.0, step=1.0, step_interval_s=20.0),
        axes.prefill_heavy,
        axes.NoSharing,
    ),
    "square_wave": _synthetic(
        "square_wave",
        "On/off bursts (1<->10 req/s, 40 s period) — stability / thrash.",
        axes.square_wave(low=1.0, high=10.0, period_s=40.0),
        axes.prefill_heavy,
        axes.NoSharing,
    ),
    "flash_crowd": _synthetic(
        "flash_crowd",
        "Flat with a single sharp spike (3->30 req/s for 15 s) — responsiveness / recovery.",
        axes.flash_crowd(base=3.0, spike=30.0, spike_at_s=80.0, width_s=15.0),
        axes.prefill_heavy,
        axes.NoSharing,
    ),
    "diurnal": _synthetic(
        "diurnal",
        "Sinusoidal load (mean 5.6, amp 4, one cycle) — scale-down discipline.",
        axes.diurnal(mean=5.6, amplitude=4.0, period_s=_DURATION),
        axes.prefill_heavy,
        axes.NoSharing,
    ),
    # --- shape axis ---
    "decode_heavy": _synthetic(
        "decode_heavy",
        "Short input, long output on a staircase — the P-vs-D scaling decision.",
        axes.staircase(base=1.0, step=1.0, step_interval_s=20.0),
        axes.decode_heavy,
        axes.NoSharing,
    ),
    # --- prefix axis ---
    "shared_prefix": _synthetic(
        "shared_prefix",
        "RAG-style: flat load, balanced shape, 8-block shared prefix — cache-reuse axis.",
        axes.flat(5.6),
        axes.balanced,
        lambda: axes.SharedPrefix(shared_blocks=8),
    ),
}


def get_workload(name: str) -> Workload:
    if name not in WORKLOADS:
        raise KeyError(f"unknown workload '{name}'; known: {sorted(WORKLOADS)}")
    return WORKLOADS[name]


def list_workloads() -> List[str]:
    return list(WORKLOADS)
