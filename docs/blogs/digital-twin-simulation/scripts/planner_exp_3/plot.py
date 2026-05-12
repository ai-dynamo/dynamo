# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot Exp 3 results: p90 TTFT vs engine startup_time.

Single-panel focus on the headline metric (p90 TTFT) with the SLA line
and the startup-delay cliff annotated. GPU-hours and scaling-event
counts are reported in the blog text rather than the figure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common import image_path, results_json_path  # noqa: E402

EXP_NAME = "planner_exp_3"
SLA_GRAY = "#8C8C8C"
CLIFF_RED = "#C8102E"
CLIFF_S = 200  # approximate startup_time at which p90 TTFT diverges


def load() -> list[dict]:
    rows = json.loads(results_json_path(EXP_NAME).read_text())
    rows.sort(key=lambda r: r["params"]["startup_time_s"])
    return rows


def main() -> None:
    rows = load()
    xs = [r["params"]["startup_time_s"] for r in rows]
    ttft_p90 = [(r["ttft_ms"] or {}).get("p90") for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # SLA reference line.
    ax.axhline(1500, ls="--", color=SLA_GRAY, alpha=0.8, label="TTFT SLA 1500 ms")

    # Cliff marker — a thin vertical line + label at the empirical breaking point.
    ax.axvline(CLIFF_S, ls=":", color=CLIFF_RED, alpha=0.8, lw=1.5)
    ax.text(
        CLIFF_S - 5,
        0.97,
        f"SLA cliff ≈ {CLIFF_S} s",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=CLIFF_RED,
    )

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

    ax.set_yscale("log")
    ax.set_xlabel("Engine startup_time (s)")
    ax.set_ylabel("p90 TTFT (ms, log)")
    ax.set_title(
        "Exp 3 — p90 TTFT vs engine startup_time\n"
        "Qwen3-32B / TP=2 / H200 / vLLM, agg planner SLA(ttft=1500, itl=50)",
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
