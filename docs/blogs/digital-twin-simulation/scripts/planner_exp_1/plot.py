# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot Exp 1 results: planner runs against a static-deployment Pareto curve."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common import image_path, results_json_path  # noqa: E402

EXP_NAME = "planner_exp_1"
NV_GREEN = "#76B900"
INK = "#1A1A1A"
GRAY = "#8C8C8C"

TARGET_MARKERS = {"throughput": "s", "latency": "^", "sla": "D"}
MODE_COLORS = {"agg": "#0072B2", "disagg": "#D55E00"}


def load() -> list[dict]:
    return json.loads(results_json_path(EXP_NAME).read_text())


def main() -> None:
    rows = load()
    by_kind: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        params = r["params"]
        if params["sweep"] == "static":
            key = (params["mode"], "static")
        else:
            key = (params["mode"], params["optimization_target"])
        by_kind.setdefault(key, []).append(r)

    fig, (ax_ttft, ax_itl) = plt.subplots(1, 2, figsize=(14, 6))

    def gpu_count(row):
        p = row["params"]
        if p["mode"] == "agg":
            return p["num_workers"] * 2
        return (p["num_prefill"] + p["num_decode"]) * 2

    def draw_pareto(ax, metric: str, *, log_y: bool):
        """Plot {metric}_p90 vs cumulative GPU-hours, static curves + planner scatter."""
        for mode in ["agg", "disagg"]:
            static_rows = sorted(
                by_kind.get((mode, "static"), []),
                key=lambda r: r["gpu_hours"] or 0.0,
            )
            if static_rows:
                xs = [r["gpu_hours"] for r in static_rows]
                ys = [(r[metric] or {}).get("p90") for r in static_rows]
                ax.plot(
                    xs,
                    ys,
                    "-o",
                    color=MODE_COLORS[mode],
                    lw=2,
                    markersize=6,
                    alpha=0.6,
                    label=f"{mode} static",
                )
                for r, x, y in zip(static_rows, xs, ys):
                    if y is None:
                        continue
                    ax.annotate(
                        f"{gpu_count(r)} GPU",
                        (x, y),
                        textcoords="offset points",
                        xytext=(6, 4),
                        fontsize=7,
                        color=MODE_COLORS[mode],
                    )

            for target in ["throughput", "latency", "sla"]:
                pts = by_kind.get((mode, target), [])
                if not pts:
                    continue
                label = f"{mode} planner-{target}"
                marker = TARGET_MARKERS[target]
                xs = [r["gpu_hours"] for r in pts]
                ys = [(r[metric] or {}).get("p90") for r in pts]
                ax.scatter(
                    xs,
                    ys,
                    marker=marker,
                    s=55,
                    edgecolors=MODE_COLORS[mode],
                    facecolors="none",
                    linewidths=1.6,
                    label=label,
                    zorder=5,
                )

        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Cumulative GPU-hours")
        ax.grid(True, which="both", alpha=0.3)

        # The "lower-left = better" indicator (right-isosceles triangle flush
        # to the corner) is drawn separately after tight_layout() in
        # add_better_indicator(), once the axes box is finalized.

    draw_pareto(ax_ttft, "ttft_ms", log_y=True)
    ax_ttft.set_ylabel("p90 TTFT (ms, log)")
    ax_ttft.set_title("p90 TTFT")
    # Single shared legend on the first panel only; two columns to keep it compact.
    ax_ttft.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    draw_pareto(ax_itl, "itl_ms", log_y=False)
    ax_itl.set_ylabel("p90 ITL (ms)")
    ax_itl.set_title("p90 ITL")

    # Pull xmin / ymin lower so the lower-left corner has visual headroom
    # for the "better" indicator without overlapping data and annotations.
    for ax in (ax_ttft, ax_itl):
        ax.set_xlim(left=2.5)
    ax_ttft.set_ylim(bottom=500)  # log scale; planner-SLA point sits at ~2200
    ax_itl.set_ylim(bottom=-60)  # linear scale; smallest data point ~15ms

    fig.suptitle(
        "Exp 1 — GPU-hours vs latency: planner vs static, agg vs disagg\n"
        "Qwen3-32B / TP=2 / H200 / vLLM, planner-SLA target TTFT=1500 ms / ITL=50 ms, startup=60 s",
        fontsize=11,
        y=1.0,
    )
    plt.tight_layout()

    # Add the right-isosceles "better" indicator AFTER tight_layout so we can
    # compute the axes' display aspect ratio and pick leg lengths in
    # axes-fraction that map to equal-length legs in display pixels.
    leg_inches = 0.34  # physical leg length on the page (75% of original 0.45")
    fig_w_in, fig_h_in = fig.get_size_inches()
    for ax in (ax_ttft, ax_itl):
        pos = ax.get_position()
        ax_w_in = pos.width * fig_w_in
        ax_h_in = pos.height * fig_h_in
        x_leg = leg_inches / ax_w_in  # axes-fraction along bottom axis
        y_leg = leg_inches / ax_h_in  # axes-fraction along left axis
        triangle = Polygon(
            [(0.0, 0.0), (x_leg, 0.0), (0.0, y_leg)],
            transform=ax.transAxes,
            facecolor=NV_GREEN,
            edgecolor=NV_GREEN,
            zorder=10,
            clip_on=False,
        )
        ax.add_patch(triangle)
        # "better" text in NV_GREEN, parallel to the hypotenuse (rotation
        # -45° matches the hypotenuse's upper-left → lower-right direction)
        # and offset just past the hypotenuse so it doesn't overlap the
        # triangle. Anchor is along the line from the corner through the
        # hypotenuse midpoint, pushed out to ~0.8× of the leg length.
        ax.text(
            x_leg * 0.8,
            y_leg * 0.8,
            "better",
            transform=ax.transAxes,
            ha="center",
            va="center",
            rotation=-45,
            fontsize=11,
            fontweight="bold",
            color=NV_GREEN,
        )
    out = image_path(EXP_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
