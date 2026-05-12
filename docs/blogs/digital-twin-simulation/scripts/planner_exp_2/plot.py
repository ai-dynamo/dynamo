# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot Exp 2 results: p90 TTFT vs load_adjustment_interval.

Single-panel focus on the headline metric (p90 TTFT). Secondary metrics
(ITL, scaling-event count, GPU-hours) are reported in the blog text
rather than the figure to keep the visual lean.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common import image_path, results_json_path  # noqa: E402

EXP_NAME = "planner_exp_2"
NV_GREEN = "#76B900"
SLA_GRAY = "#8C8C8C"
SWEET_LO_S = 5
SWEET_HI_S = 10


def load() -> list[dict]:
    rows = json.loads(results_json_path(EXP_NAME).read_text())
    rows.sort(key=lambda r: r["params"]["load_adjustment_interval_s"])
    return rows


def main() -> None:
    rows = load()
    xs = [r["params"]["load_adjustment_interval_s"] for r in rows]
    ttft_p90 = [(r["ttft_ms"] or {}).get("p90") for r in rows]
    events = [r["scale_up_events"] + r["scale_down_events"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Highlight the 5–10 s sweet spot as a soft green band behind the curve.
    ax.axvspan(SWEET_LO_S, SWEET_HI_S, alpha=0.15, color=NV_GREEN, zorder=0)
    ax.text(
        (SWEET_LO_S * SWEET_HI_S) ** 0.5,  # geometric midpoint on log axis
        0.97,
        "5–10 s sweet spot",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=NV_GREEN,
    )

    # SLA reference line.
    ax.axhline(1500, ls="--", color=SLA_GRAY, alpha=0.8, label="TTFT SLA 1500 ms")

    # Main curve.
    ax.plot(
        xs,
        ttft_p90,
        "o-",
        color="#0072B2",
        lw=2,
        markersize=7,
        label="p90 TTFT",
        zorder=3,
    )

    # Annotate each point with its scaling-event count so the cost
    # of oscillation stays visible in the figure as well as the text.
    for x, y, n in zip(xs, ttft_p90, events):
        ax.annotate(
            f"{n} events",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            fontsize=7,
            ha="center",
            color="#444444",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("load_adjustment_interval (s, log)")
    ax.set_ylabel("p90 TTFT (ms, log)")
    ax.set_title(
        "Exp 2 — p90 TTFT vs load_adjustment_interval\n"
        "Qwen3-32B / TP=2 / H200 / vLLM, load-only planner, startup_time=0",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    out = image_path(EXP_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
