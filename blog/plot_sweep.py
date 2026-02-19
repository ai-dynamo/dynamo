#!/usr/bin/env python3
"""Plot sweep results from the JSON file produced by mooncake_bench --sweep."""

import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", help="Path to sweep_plot.json")
    parser.add_argument(
        "-o", "--output", default="sweep_plot.png", help="Output image path"
    )
    parser.add_argument(
        "--title",
        default="Achieved vs Payload Throughput",
        help="Plot title",
    )
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 7))

    global_min = float("inf")
    global_max = float("-inf")

    for name, steps in data.items():
        for s in steps:
            offered = s["offered_block_throughput"]
            achieved = s["block_throughput"]
            global_min = min(global_min, offered, achieved)
            global_max = max(global_max, offered, achieved)

    axis_min = global_min * 0.9
    axis_max = global_max * 1.1

    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        "--",
        color="gray",
        alpha=0.5,
        linewidth=1,
        label="_nolegend_",
    )

    for i, (name, steps) in enumerate(data.items()):
        color = COLORS[i % len(COLORS)]
        offered = [s["offered_block_throughput"] for s in steps]
        achieved = [s["block_throughput"] for s in steps]

        ax.plot(offered, achieved, "-o", color=color, label=name, markersize=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}"))

    ax.set_xlabel("Payload Throughput", fontsize=16)
    ax.set_ylabel("Achieved Throughput", fontsize=16)
    ax.set_title(f"{args.title} (million ops/s)", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="black", fontsize=13)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
