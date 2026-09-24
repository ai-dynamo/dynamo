#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#666666", "#76B900", "#0077C8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def grouped_percentiles(
    ax,
    rows: list[dict],
    prefix: str,
    title: str,
    ylabel: str,
    show_errors: bool,
) -> None:
    x = np.arange(2)
    width = 0.24
    for index, row in enumerate(rows):
        values = [
            float(row[f"{prefix}_p50_ms_mean"]),
            float(row[f"{prefix}_p90_ms_mean"]),
        ]
        errors = [
            float(row[f"{prefix}_p50_ms_stddev"]),
            float(row[f"{prefix}_p90_ms_stddev"]),
        ]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            yerr=errors if show_errors else None,
            capsize=3,
            color=COLORS[index],
            label=row["condition_label"],
        )
    ax.set_xticks(x, ["P50", "P90"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    args = parse_args()
    with args.aggregate_csv.open() as stream:
        rows = list(csv.DictReader(stream))

    labels = [row["condition_label"] for row in rows]
    x = np.arange(len(rows))
    rounds = int(rows[0]["rounds"])
    show_errors = rounds > 1
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    axes[0, 0].bar(
        x,
        [float(row["cache_hit_pct_mean"]) for row in rows],
        yerr=[float(row["cache_hit_pct_stddev"]) for row in rows]
        if show_errors
        else None,
        capsize=4,
        color=COLORS,
    )
    axes[0, 0].set_xticks(x, labels, rotation=12, ha="right")
    axes[0, 0].set_title("Cache Hit")
    axes[0, 0].set_ylabel("Percent")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(
        x,
        [float(row["total_token_throughput_mean"]) for row in rows],
        yerr=[float(row["total_token_throughput_stddev"]) for row in rows]
        if show_errors
        else None,
        capsize=4,
        color=COLORS,
    )
    axes[0, 1].set_xticks(x, labels, rotation=12, ha="right")
    axes[0, 1].set_title("Total Throughput")
    axes[0, 1].set_ylabel("Tokens/s")
    axes[0, 1].grid(axis="y", alpha=0.25)

    grouped_percentiles(
        axes[1, 0],
        rows,
        "ttft",
        "Time to First Token",
        "Milliseconds",
        show_errors,
    )
    grouped_percentiles(
        axes[1, 1],
        rows,
        "itl",
        "Inter-Token Latency",
        "Milliseconds",
        show_errors,
    )
    handles, legend_labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=False,
    )
    suffix = (
        f"mean +/- sample standard deviation, n={rounds}"
        if rounds > 1
        else "single paired run"
    )
    fig.suptitle(f"Experiment 7: Heuristic Whole-Prefix Retention ({suffix})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
