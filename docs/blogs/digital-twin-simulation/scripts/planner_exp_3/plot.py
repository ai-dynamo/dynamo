# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot Exp 3 results: TTFT/ITL/GPU-hours/scaling-events vs engine startup time."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common import image_path, results_json_path  # noqa: E402

EXP_NAME = "planner_exp_3"
GRAY = "#8C8C8C"


def load() -> list[dict]:
    rows = json.loads(results_json_path(EXP_NAME).read_text())
    rows.sort(key=lambda r: r["params"]["startup_time_s"])
    return rows


def main() -> None:
    rows = load()
    xs = [r["params"]["startup_time_s"] for r in rows]
    ttft_avg = [(r["ttft_ms"] or {}).get("avg") for r in rows]
    ttft_p90 = [(r["ttft_ms"] or {}).get("p90") for r in rows]
    itl_avg = [(r["itl_ms"] or {}).get("avg") for r in rows]
    itl_p90 = [(r["itl_ms"] or {}).get("p90") for r in rows]
    gpu_h = [r["gpu_hours"] for r in rows]
    n_up = [r["scale_up_events"] for r in rows]
    n_down = [r["scale_down_events"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(xs, ttft_avg, "o-", label="avg")
    ax.plot(xs, ttft_p90, "s-", label="p90")
    ax.axhline(1500, ls="--", color=GRAY, alpha=0.6, label="SLA 1500ms")
    ax.set_yscale("log")
    ax.set_xlabel("Engine startup_time (s)")
    ax.set_ylabel("TTFT (ms, log)")
    ax.set_title("TTFT vs engine startup time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(xs, itl_avg, "o-", label="avg")
    ax.plot(xs, itl_p90, "s-", label="p90")
    ax.axhline(50, ls="--", color=GRAY, alpha=0.6, label="SLA 50ms")
    ax.set_xlabel("Engine startup_time (s)")
    ax.set_ylabel("ITL (ms)")
    ax.set_title("ITL vs engine startup time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(xs, gpu_h, "o-", color="C2")
    ax.set_xlabel("Engine startup_time (s)")
    ax.set_ylabel("Cumulative GPU-hours")
    ax.set_title("GPU-hours vs engine startup time")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(xs, n_up, "o-", color="C3", label="scale_up")
    ax.plot(xs, n_down, "s-", color="C4", label="scale_down")
    ax.set_xlabel("Engine startup_time (s)")
    ax.set_ylabel("Event count")
    ax.set_title("Planner scaling events vs engine startup time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(
        "Exp 3 — Qwen3-32B / TP=2 / H200 / vLLM — toolagent_trace, "
        "agg planner SLA(ttft=1500, itl=50)",
        y=1.0,
    )
    plt.tight_layout()
    out = image_path(EXP_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
