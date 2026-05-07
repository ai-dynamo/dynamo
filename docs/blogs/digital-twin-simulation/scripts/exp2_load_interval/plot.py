# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot Exp 2 results: latency / oscillation / cost vs load_adjustment_interval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common import RESULTS_ROOT  # noqa: E402

EXP_NAME = "exp2_load_interval"
GRAY = "#8C8C8C"


def load() -> list[dict]:
    p = RESULTS_ROOT / EXP_NAME / "results.json"
    rows = json.loads(p.read_text())
    rows.sort(key=lambda r: r["params"]["load_adjustment_interval_s"])
    return rows


def main() -> None:
    rows = load()
    xs = [r["params"]["load_adjustment_interval_s"] for r in rows]
    ttft_avg = [(r["ttft_ms"] or {}).get("avg") for r in rows]
    ttft_p90 = [(r["ttft_ms"] or {}).get("p90") for r in rows]
    itl_avg = [(r["itl_ms"] or {}).get("avg") for r in rows]
    itl_p90 = [(r["itl_ms"] or {}).get("p90") for r in rows]
    events = [r["scale_up_events"] + r["scale_down_events"] for r in rows]
    gpu_h = [r["gpu_hours"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(xs, ttft_avg, "o-", label="avg")
    ax.plot(xs, ttft_p90, "s-", label="p90")
    ax.axhline(1500, ls="--", color=GRAY, alpha=0.6, label="SLA 1500ms")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("load_adjustment_interval (s, log)")
    ax.set_ylabel("TTFT (ms, log)")
    ax.set_title("TTFT vs interval")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(xs, itl_avg, "o-", label="avg")
    ax.plot(xs, itl_p90, "s-", label="p90")
    ax.axhline(50, ls="--", color=GRAY, alpha=0.6, label="SLA 50ms")
    ax.set_xscale("log")
    ax.set_xlabel("load_adjustment_interval (s, log)")
    ax.set_ylabel("ITL (ms)")
    ax.set_title("ITL vs interval")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(xs, events, "o-", color="C3")
    ax.set_xscale("log")
    ax.set_xlabel("load_adjustment_interval (s, log)")
    ax.set_ylabel("scale_up + scale_down events")
    ax.set_title("Oscillation vs interval")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 1]
    ax.plot(xs, gpu_h, "o-", color="C2")
    ax.set_xscale("log")
    ax.set_xlabel("load_adjustment_interval (s, log)")
    ax.set_ylabel("Cumulative GPU-hours")
    ax.set_title("GPU-hours vs interval")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Exp 2 — Qwen3-32B / TP=2 / H200 / vLLM 0.12.0 — toolagent_trace, "
        "load-only planner, startup_time=0",
        y=1.0,
    )
    plt.tight_layout()
    out = RESULTS_ROOT / EXP_NAME / "plot.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
