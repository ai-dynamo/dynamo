# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Plot the KVBM G2 host-memory cache experiment.

Reads data.csv (long format: one row per (source, concurrency) with sources
`baseline` and `g2`) and emits kvbm_g2_exp.png to ../images/. The figure has
two panels:
  - Mean TTFT vs concurrency (baseline vs G2, log y)
  - Throughput / GPU vs Interactivity (TPS / User) Pareto, baseline vs G2

Usage:
    python plot.py
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

# --- NVIDIA brand palette ---
NV_GREEN = "#76B900"
INK = "#1A1A1A"
GRAY = "#8C8C8C"

HERE = os.path.dirname(os.path.abspath(__file__))


def load_csv(path):
    """Return {source: {conc: {metric: value}}}."""
    out = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            c = int(row["concurrency"])
            out[row["source"]][c] = {
                "tps_gpu": float(row["tps_gpu"]),
                "tps_user": float(row["tps_user"]),
                "tpot_ms": float(row["tpot_ms"]),
                "ttft_ms": float(row["ttft_ms"]),
            }
    return out


def series(data, source, metric, concs):
    return [data[source][c][metric] for c in concs]


# --- Style helpers (kept local to match fidelity_aic_vs_mocker_vs_hw) ---
def apply_nv_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.4)
    ax.tick_params(colors=INK, length=4, width=1.0)
    ax.set_facecolor("white")


def add_title(fig, title, subtitle=None, sub_subtitle=None):
    fig.text(0.5, 0.95, title, fontsize=17, fontweight="bold", color=INK, ha="center")
    if subtitle:
        fig.text(
            0.5,
            0.91,
            subtitle,
            fontsize=12,
            color=NV_GREEN,
            fontweight="600",
            ha="center",
        )
    if sub_subtitle:
        # Sit just below subtitle if present, otherwise take its spot under the title.
        y = 0.88 if subtitle else 0.91
        fig.text(0.5, y, sub_subtitle, fontsize=10, color=GRAY, ha="center")


def plot_lines(ax, concs, baseline, g2, label_baseline, label_g2):
    ax.plot(
        concs,
        baseline,
        "-o",
        color=GRAY,
        linewidth=2.2,
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=1.4,
        label=label_baseline,
        zorder=3,
    )
    ax.plot(
        concs,
        g2,
        "-D",
        color=NV_GREEN,
        linewidth=2.6,
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=1.4,
        label=label_g2,
        zorder=4,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=os.path.join(HERE, "data.csv"))
    ap.add_argument(
        "--out", default=os.path.join(HERE, "..", "..", "images", "kvbm_g2_exp.png")
    )
    args = ap.parse_args()

    data = load_csv(args.csv)
    concs = sorted(set(data["baseline"]) & set(data["g2"]))

    base_ttft = series(data, "baseline", "ttft_ms", concs)
    g2_ttft = series(data, "g2", "ttft_ms", concs)
    ttft_red = [(b - g) / b * 100 for b, g in zip(base_ttft, g2_ttft)]

    base_tps_gpu = series(data, "baseline", "tps_gpu", concs)
    g2_tps_gpu = series(data, "g2", "tps_gpu", concs)
    base_tps_user = series(data, "baseline", "tps_user", concs)
    g2_tps_user = series(data, "g2", "tps_user", concs)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )

    fig, (ax_ttft, ax_pareto) = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.81, bottom=0.13, wspace=0.22)

    # --- Mean TTFT panel (log y) ---
    plot_lines(
        ax_ttft, concs, base_ttft, g2_ttft, "Baseline (G1 only)", "With G2 (32K blocks)"
    )
    for c, y, r in zip(concs, g2_ttft, ttft_red):
        ax_ttft.annotate(
            f"-{r:.1f}%",
            (c, y),
            textcoords="offset points",
            xytext=(10, -4),
            fontsize=10,
            color=NV_GREEN,
            fontweight="bold",
        )
    ax_ttft.set_xlabel("Concurrency", fontsize=11, fontweight="bold", labelpad=10)
    ax_ttft.set_ylabel(
        "Mean TTFT (ms, log)", fontsize=11, fontweight="bold", labelpad=10
    )
    ax_ttft.set_xticks(concs)
    ax_ttft.set_xticklabels([f"c={c}" for c in concs])
    ax_ttft.set_yscale("log")
    apply_nv_style(ax_ttft)
    ax_ttft.legend(fontsize=10, loc="upper left", frameon=False)

    # --- Pareto panel: TPS/GPU vs TPS/User ---
    ax_pareto.plot(
        base_tps_user,
        base_tps_gpu,
        "-o",
        color=GRAY,
        linewidth=2.2,
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=1.4,
        label="Baseline (G1 only)",
        zorder=3,
    )
    ax_pareto.plot(
        g2_tps_user,
        g2_tps_gpu,
        "-D",
        color=NV_GREEN,
        linewidth=2.6,
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=1.4,
        label="With G2 (32K blocks)",
        zorder=4,
    )
    for c, x, y in zip(concs, base_tps_user, base_tps_gpu):
        ax_pareto.annotate(
            f"c={c}",
            (x, y),
            textcoords="offset points",
            xytext=(-6, -16),
            fontsize=8,
            color=GRAY,
            fontweight="bold",
        )
    for c, x, y in zip(concs, g2_tps_user, g2_tps_gpu):
        ax_pareto.annotate(
            f"c={c}",
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color=NV_GREEN,
            fontweight="bold",
        )
    ax_pareto.set_xlabel(
        "Interactivity (TPS per User)", fontsize=11, fontweight="bold", labelpad=10
    )
    ax_pareto.set_ylabel(
        "Throughput (TPS per GPU)", fontsize=11, fontweight="bold", labelpad=10
    )
    all_x = base_tps_user + g2_tps_user
    all_y = base_tps_gpu + g2_tps_gpu
    ax_pareto.set_xlim(min(all_x) - 15, max(all_x) + 25)
    ax_pareto.set_ylim(min(all_y) - 10, max(all_y) + 15)
    apply_nv_style(ax_pareto)
    ax_pareto.legend(fontsize=10, loc="upper right", frameon=False)

    add_title(
        fig,
        "Modeling KVBM's G2 Host-Memory Cache Tier",
        sub_subtitle="B200 MiniMax-M2.5   TP=4   1 worker   Mooncake trace   offline replay",
    )

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
