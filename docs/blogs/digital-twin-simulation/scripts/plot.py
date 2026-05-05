"""
Plot B200 MiniMax-M2.5 hardware vs mocker vs AIC results.

Reads data.csv (long format: one row per (source, concurrency)) and emits two
figures:
  - hw_mocker_aic_pareto.png   (Throughput / GPU vs. Throughput / User)
  - hw_mocker_aic_4panel.png   (TPS/GPU, TPS/User, TPOT, TTFT vs. concurrency)

Also prints a MAPE table comparing mocker and AIC against hardware.

Usage:
    python plot.py [--keep-c4]

By default c=4 is excluded because the hardware TTFT measurement at c=4 is a
known outlier (cold-start / first-batch effect). Pass --keep-c4 to include it.
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


# --- Style helpers ---
def apply_nv_style(ax, *, axis_arrows=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.4)
    ax.tick_params(colors=INK, length=4, width=1.0)
    ax.set_facecolor("white")
    if axis_arrows:
        ax.plot(1, 0, ">", transform=ax.get_yaxis_transform(),
                color=INK, clip_on=False, markersize=8)
        ax.plot(0, 1, "^", transform=ax.get_xaxis_transform(),
                color=INK, clip_on=False, markersize=8)


def add_title(fig, title, subtitle, sub_subtitle):
    fig.text(0.5, 0.965, title, fontsize=18, fontweight="bold", color=INK, ha="center")
    fig.text(0.5, 0.928, subtitle, fontsize=12, color=NV_GREEN, fontweight="600", ha="center")
    fig.text(0.5, 0.902, sub_subtitle, fontsize=10, color=GRAY, ha="center")


# Source styling: (label, color, marker, linestyle)
STYLES = {
    "hardware": ("Hardware", INK, "o", "-"),
    "mocker": ("Mocker", NV_GREEN, "D", "-"),
    "aic": ("AIC", GRAY, "s", "--"),
}
SOURCE_ORDER = ["hardware", "mocker", "aic"]


# --- Plots ---
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
        ax.scatter(xs, ys, color=color, s=90, marker=marker,
                   edgecolors="white", linewidths=1.5, zorder=5, label=label)
        for c, x, y in zip(concs, xs, ys):
            ax.annotate(f"c={c}", (x, y), textcoords="offset points",
                        xytext=(8, 8), fontsize=8, color=color, fontweight="bold")

    ax.set_xlabel("Interactivity (TPS per User)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Throughput (TPS per GPU)", fontsize=12, fontweight="bold", labelpad=10)
    apply_nv_style(ax, axis_arrows=False)
    ax.set_xlim(max(0, min(all_x) - 5), max(all_x) + 8)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, loc="upper right", frameon=False,
              handletextpad=0.6, labelspacing=0.7)

    add_title(fig, "B200 MiniMax-M2.5 Performance",
              "Mocker vs AIC vs Hardware", "TP=4   ISL/OSL 1K/1K")
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_4panel(data, concs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.08,
                        hspace=0.28, wspace=0.20)

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
            ax.scatter(concs, ys, color=color, s=70, marker=marker,
                       edgecolors="white", linewidths=1.2, zorder=5, label=label)
        ax.set_xlabel("Concurrency", fontsize=11, fontweight="bold", labelpad=14)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", labelpad=14)
        apply_nv_style(ax)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc="best", frameon=False)

    add_title(fig, "B200 MiniMax-M2.5 Performance",
              "Mocker vs AIC vs Hardware", "TP=4   ISL/OSL 1K/1K")
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_mape(data, concs):
    print(f"\n=== MAPE vs hardware (concurrencies: {concs}) ===")
    print(f"{'Metric':<10} {'Mocker':>10} {'AIC':>10}")
    for metric in ("tps_gpu", "tps_user", "tpot_ms", "ttft_ms"):
        hw = series(data, "hardware", metric, concs)
        mk = series(data, "mocker", metric, concs)
        ac = series(data, "aic", metric, concs)
        mk_mape = sum(abs(p - a) / a for p, a in zip(mk, hw)) / len(hw) * 100
        ac_mape = sum(abs(p - a) / a for p, a in zip(ac, hw)) / len(hw) * 100
        print(f"{metric:<10} {mk_mape:>9.1f}% {ac_mape:>9.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-c4", action="store_true",
                    help="Include c=4 (default: excluded as TTFT outlier).")
    ap.add_argument("--csv", default=os.path.join(HERE, "data.csv"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "images"))
    args = ap.parse_args()

    data = load_csv(args.csv)
    all_concs = sorted(set(data["hardware"]) & set(data["mocker"]) & set(data["aic"]))
    concs = all_concs if args.keep_c4 else [c for c in all_concs if c != 4]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    })
    plot_pareto(data, concs, os.path.join(args.out_dir, "hw_mocker_aic_pareto.png"))
    plot_4panel(data, concs, os.path.join(args.out_dir, "hw_mocker_aic_4panel.png"))
    print_mape(data, concs)


if __name__ == "__main__":
    main()
