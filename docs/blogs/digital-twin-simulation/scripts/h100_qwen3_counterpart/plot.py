# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Plot H100 Qwen3-32B silicon vs mocker replay vs AIC results.

Reads data.csv (long format: one row per (source, concurrency)) and emits two
figures:
  - h100_hw_mocker_aic_pareto.png
  - h100_hw_mocker_aic_4panel.png

Also prints a MAPE table comparing mocker replay and AIC against silicon.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

NV_GREEN = "#76B900"
INK = "#1A1A1A"
GRAY = "#8C8C8C"

HERE = os.path.dirname(os.path.abspath(__file__))

PARETO_NAME = "h100_hw_mocker_aic_pareto.png"
PANEL_NAME = "h100_hw_mocker_aic_4panel.png"


def load_csv(path):
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


def apply_nv_style(ax, *, axis_arrows=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
        ax.spines[spine].set_linewidth(1.4)
    ax.tick_params(colors=INK, length=4, width=1.0)
    ax.set_facecolor("white")
    if axis_arrows:
        ax.plot(
            1,
            0,
            ">",
            transform=ax.get_yaxis_transform(),
            color=INK,
            clip_on=False,
            markersize=8,
        )
        ax.plot(
            0,
            1,
            "^",
            transform=ax.get_xaxis_transform(),
            color=INK,
            clip_on=False,
            markersize=8,
        )


def add_title(fig, title, subtitle, sub_subtitle):
    fig.text(0.5, 0.965, title, fontsize=18, fontweight="bold", color=INK, ha="center")
    fig.text(
        0.5, 0.928, subtitle, fontsize=12, color=NV_GREEN, fontweight="600", ha="center"
    )
    fig.text(0.5, 0.902, sub_subtitle, fontsize=10, color=GRAY, ha="center")


STYLES = {
    "silicon": ("Silicon", INK, "o", "-"),
    "mocker": ("Mocker Replay", NV_GREEN, "D", "-"),
    "aic": ("AIC", GRAY, "s", "--"),
}
SOURCE_ORDER = ["silicon", "mocker", "aic"]


def plot_pareto(data, concs, out_path):
    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.10)

    all_x = []
    for src in SOURCE_ORDER:
        label, color, marker, ls = STYLES[src]
        xs = series(data, src, "tps_user", concs)
        ys = series(data, src, "tps_gpu", concs)
        all_x.extend(xs)
        ax.plot(xs, ys, ls, color=color, linewidth=2.4, zorder=3)
        ax.scatter(
            xs,
            ys,
            color=color,
            s=90,
            marker=marker,
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
            label=label,
        )
        for c, x, y in zip(concs, xs, ys):
            ax.annotate(
                f"c={c}",
                (x, y),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    ax.set_xlabel(
        "Interactivity (TPS per User)", fontsize=12, fontweight="bold", labelpad=10
    )
    ax.set_ylabel(
        "Throughput (TPS per GPU)", fontsize=12, fontweight="bold", labelpad=10
    )
    apply_nv_style(ax, axis_arrows=False)
    ax.set_xlim(max(0, min(all_x) - 5), max(all_x) + 8)
    ax.set_ylim(bottom=0)
    ax.legend(
        fontsize=10,
        loc="upper right",
        frameon=False,
        handletextpad=0.6,
        labelspacing=0.7,
    )

    add_title(
        fig,
        "H100 Qwen3-32B Performance",
        "Mocker Replay vs AIC vs Silicon",
        "TP=2   ISL/OSL 1K/1K",
    )
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_4panel(data, concs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.subplots_adjust(
        left=0.08, right=0.97, top=0.88, bottom=0.08, hspace=0.28, wspace=0.20
    )

    panels = [
        (axes[0, 0], "tps_gpu", "Output TPS / GPU"),
        (axes[0, 1], "tps_user", "Output TPS / User"),
        (axes[1, 0], "tpot_ms", "Mean TPOT (ms)"),
        (axes[1, 1], "ttft_ms", "Mean TTFT (ms)"),
    ]
    for ax, metric, ylabel in panels:
        for src in SOURCE_ORDER:
            label, color, marker, ls = STYLES[src]
            ys = series(data, src, metric, concs)
            ax.plot(concs, ys, ls, color=color, linewidth=2.2, zorder=3)
            ax.scatter(
                concs,
                ys,
                color=color,
                s=70,
                marker=marker,
                edgecolors="white",
                linewidths=1.2,
                zorder=5,
                label=label,
            )
        ax.set_xlabel("Concurrency", fontsize=11, fontweight="bold", labelpad=14)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", labelpad=14)
        apply_nv_style(ax)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc="best", frameon=False)

    add_title(
        fig,
        "H100 Qwen3-32B Performance",
        "Mocker Replay vs AIC vs Silicon",
        "TP=2   ISL/OSL 1K/1K",
    )
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_mape(data, concs):
    print(f"\n=== MAPE vs silicon (concurrencies: {concs}) ===")
    print(f"{'Metric':<10} {'Mocker':>10} {'AIC':>10}")
    for metric in ("tps_gpu", "tps_user", "tpot_ms", "ttft_ms"):
        actual = series(data, "silicon", metric, concs)
        mocker = series(data, "mocker", metric, concs)
        aic = series(data, "aic", metric, concs)
        mocker_mape = sum(abs(p - a) / a for p, a in zip(mocker, actual)) / len(actual)
        aic_mape = sum(abs(p - a) / a for p, a in zip(aic, actual)) / len(actual)
        print(f"{metric:<10} {mocker_mape * 100:>9.1f}% {aic_mape * 100:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=os.path.join(HERE, "data.csv"))
    parser.add_argument("--out-dir", default=os.path.join(HERE, "..", "..", "images"))
    args = parser.parse_args()

    data = load_csv(args.csv)
    concs = sorted(set(data["silicon"]) & set(data["mocker"]) & set(data["aic"]))

    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    plot_pareto(data, concs, os.path.join(args.out_dir, PARETO_NAME))
    plot_4panel(data, concs, os.path.join(args.out_dir, PANEL_NAME))
    print_mape(data, concs)


if __name__ == "__main__":
    main()
