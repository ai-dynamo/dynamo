"""
Plot the KV Router experiment.

Reads data.csv (long format: one row per (router_mode, concurrency)) and emits
kv_router_exp.png to ../../images/. The figure has two panels:
  - Mean TTFT vs concurrency (round-robin vs KV Router, log y)
  - Throughput / GPU vs interactivity (TPS / user) Pareto

Usage:
    python plot.py
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

NV_GREEN = "#76B900"
INK = "#1A1A1A"
GRAY = "#8C8C8C"
BLUE = "#0071C5"

HERE = os.path.dirname(os.path.abspath(__file__))

STYLES = {
    "round_robin": ("Round robin", GRAY, "o", "-"),
    "kv_router": ("KV Router", NV_GREEN, "D", "-"),
}
ROUTER_ORDER = ["round_robin", "kv_router"]


def load_csv(path):
    out = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            concurrency = int(row["concurrency"])
            out[row["router_mode"]][concurrency] = {
                "tps_gpu": float(row["tps_gpu"]),
                "tps_cluster": float(row["tps_cluster"]),
                "tps_user": float(row["tps_user"]),
                "tpot_ms": float(row["tpot_ms"]),
                "ttft_ms": float(row["ttft_ms"]),
                "prefix_cache_reused_ratio": float(row["prefix_cache_reused_ratio"]),
            }
    return out


def series(data, router_mode, metric, concs):
    return [data[router_mode][c][metric] for c in concs]


def apply_nv_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
        ax.spines[spine].set_linewidth(1.4)
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
        y = 0.88 if subtitle else 0.91
        fig.text(0.5, y, sub_subtitle, fontsize=10, color=GRAY, ha="center")


def plot_metric_lines(ax, concs, data, metric):
    for router_mode in ROUTER_ORDER:
        label, color, marker, linestyle = STYLES[router_mode]
        ax.plot(
            concs,
            series(data, router_mode, metric, concs),
            linestyle,
            color=color,
            linewidth=2.5,
            marker=marker,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.4,
            label=label,
            zorder=3,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=os.path.join(HERE, "data.csv"))
    parser.add_argument(
        "--out", default=os.path.join(HERE, "..", "..", "images", "kv_router_exp.png")
    )
    args = parser.parse_args()

    data = load_csv(args.csv)
    concs = sorted(set(data["round_robin"]) & set(data["kv_router"]))

    rr_ttft = series(data, "round_robin", "ttft_ms", concs)
    kv_ttft = series(data, "kv_router", "ttft_ms", concs)
    ttft_reduction = [(rr - kv) / rr * 100 for rr, kv in zip(rr_ttft, kv_ttft)]

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

    plot_metric_lines(ax_ttft, concs, data, "ttft_ms")
    for c, y, reduction in zip(concs, kv_ttft, ttft_reduction):
        ax_ttft.annotate(
            f"-{reduction:.0f}%",
            (c, y),
            textcoords="offset points",
            xytext=(10, -4),
            fontsize=10,
            color=NV_GREEN,
            fontweight="bold",
        )
    ax_ttft.set_xlabel("Concurrency", fontsize=11, fontweight="bold", labelpad=10)
    ax_ttft.set_ylabel("Mean TTFT (ms)", fontsize=11, fontweight="bold", labelpad=10)
    ax_ttft.set_xticks(concs)
    ax_ttft.set_xticklabels([f"c={c}" for c in concs])
    ax_ttft.set_yscale("log")
    ax_ttft.set_yticks([200, 500, 1000, 2000, 5000])
    ax_ttft.set_yticklabels(["200", "500", "1k", "2k", "5k"])
    apply_nv_style(ax_ttft)
    ax_ttft.legend(fontsize=10, loc="upper left", frameon=False)

    all_x = []
    all_y = []
    for router_mode in ROUTER_ORDER:
        label, color, marker, linestyle = STYLES[router_mode]
        xs = series(data, router_mode, "tps_user", concs)
        ys = series(data, router_mode, "tps_gpu", concs)
        all_x.extend(xs)
        all_y.extend(ys)
        ax_pareto.plot(
            xs,
            ys,
            linestyle,
            color=color,
            linewidth=2.5,
            marker=marker,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.4,
            label=label,
            zorder=3,
        )
        for c, x, y in zip(concs, xs, ys):
            offset = (8, 8) if router_mode == "kv_router" else (-10, -16)
            ax_pareto.annotate(
                f"c={c}",
                (x, y),
                textcoords="offset points",
                xytext=offset,
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    ax_pareto.set_xlabel(
        "Interactivity (TPS per User)", fontsize=11, fontweight="bold", labelpad=10
    )
    ax_pareto.set_ylabel(
        "Throughput (TPS per GPU)", fontsize=11, fontweight="bold", labelpad=10
    )
    ax_pareto.set_xlim(max(0, min(all_x) - 8), max(all_x) + 12)
    ax_pareto.set_ylim(max(0, min(all_y) - 12), max(all_y) + 18)
    apply_nv_style(ax_pareto)
    ax_pareto.legend(fontsize=10, loc="upper right", frameon=False)

    rr_prefix = series(data, "round_robin", "prefix_cache_reused_ratio", concs)
    kv_prefix = series(data, "kv_router", "prefix_cache_reused_ratio", concs)
    prefix_note = (
        f"Prefix reuse: round robin {sum(rr_prefix) / len(rr_prefix):.2f}, "
        f"KV Router {sum(kv_prefix) / len(kv_prefix):.2f}"
    )
    ax_pareto.text(
        0.02,
        0.04,
        prefix_note,
        transform=ax_pareto.transAxes,
        fontsize=9,
        color=BLUE,
        fontweight="bold",
    )

    add_title(
        fig,
        "Modeling KV-Aware Routing",
        sub_subtitle="B200 MiniMax-M2.5   TP=4   8 workers   Mooncake trace, offline replay",
    )

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
