"""
Plot the H200 Kimi router counterpart data for the digital-twin blog draft.

Reads router_data.csv and emits h200_kv_router_exp.png.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

NV_GREEN = "#76B900"
INK = "#1A1A1A"
GRAY = "#8C8C8C"
BLUE = "#0071C5"

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parents[1] / "images"

STYLES = {
    "round_robin": ("Round robin", GRAY, "o", "-"),
    "kv_router": ("KV Router", NV_GREEN, "D", "-"),
}
ROUTER_ORDER = ["round_robin", "kv_router"]


def load_router_data(path):
    out = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            mode = row["router_mode"]
            if mode not in STYLES:
                continue
            concurrency = int(row["concurrency"])
            out[mode][concurrency] = {
                "tps_gpu": float(row["tps_gpu"]),
                "tps_user": float(row["tps_user"]),
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


def add_title(fig, title, sub_subtitle=None):
    fig.text(0.5, 0.95, title, fontsize=17, fontweight="bold", color=INK, ha="center")
    if sub_subtitle:
        fig.text(0.5, 0.91, sub_subtitle, fontsize=10, color=GRAY, ha="center")


def plot_router(data_path, out_path):
    data = load_router_data(data_path)
    concs = sorted(set(data["round_robin"]) & set(data["kv_router"]))
    x_positions = list(range(len(concs)))

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

    for mode in ROUTER_ORDER:
        label, color, marker, linestyle = STYLES[mode]
        ax_ttft.plot(
            x_positions,
            series(data, mode, "ttft_ms", concs),
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

    rr_ttft = series(data, "round_robin", "ttft_ms", concs)
    kv_ttft = series(data, "kv_router", "ttft_ms", concs)
    for x, y, rr in zip(x_positions, kv_ttft, rr_ttft):
        reduction = (rr - y) / rr * 100
        ax_ttft.annotate(
            f"-{reduction:.0f}%",
            (x, y),
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
    ax_ttft.set_xticks(x_positions)
    ax_ttft.set_xticklabels([f"c={c}" for c in concs])
    ax_ttft.set_yscale("log")
    ax_ttft.set_yticks([1000, 2000, 5000, 10000])
    ax_ttft.set_yticklabels(["1k", "2k", "5k", "10k"])
    apply_nv_style(ax_ttft)
    ax_ttft.legend(fontsize=10, loc="upper left", frameon=False)

    all_x = []
    all_y = []
    for mode in ROUTER_ORDER:
        label, color, marker, linestyle = STYLES[mode]
        xs = series(data, mode, "tps_user", concs)
        ys = series(data, mode, "tps_gpu", concs)
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
            offset = (8, 8) if mode == "kv_router" else (-10, -16)
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
    ax_pareto.set_xlim(max(0, min(all_x) - 2), max(all_x) + 4)
    ax_pareto.set_ylim(max(0, min(all_y) - 3), max(all_y) + 5)
    apply_nv_style(ax_pareto)
    ax_pareto.legend(fontsize=10, loc="upper right", frameon=False)

    rr_prefix = series(data, "round_robin", "prefix_cache_reused_ratio", concs)
    kv_prefix = series(data, "kv_router", "prefix_cache_reused_ratio", concs)
    prefix_note = (
        f"Prefix reuse avg: round robin {sum(rr_prefix) / len(rr_prefix):.2f}, "
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
        "H200 Kimi: Modeling KV-Aware Routing",
        sub_subtitle="H200-SXM   moonshotai/Kimi-K2.5   vLLM 0.19.0   TP=4   8 workers   toolagent trace",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=HERE / "router_data.csv")
    parser.add_argument("--out", default=IMAGES / "h200_kv_router_exp.png")
    args = parser.parse_args()

    plot_router(Path(args.csv), Path(args.out))


if __name__ == "__main__":
    main()
