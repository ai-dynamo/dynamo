# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Generate a waterfall timeline plot from streaming timing data.

Usage:
    python3 plot_stream_timeline.py --input raw/timing-no-dispatch.jsonl --output plots/timeline.png
"""

import argparse
import json
import sys

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "matplotlib not installed. Install with: uv pip install matplotlib",
        file=sys.stderr,
    )
    sys.exit(1)


def load_runs(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_waterfall(
    runs: list, output: str, title: str = "Streaming Tool Call Timeline"
):
    """Plot a horizontal waterfall for each run showing phase durations."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(runs) * 0.6)))

    colors = {
        "TTFT": "#4CAF50",
        "Reasoning": "#2196F3",
        "Tool call generation": "#FF9800",
        "Post-tool to finish": "#F44336",
    }

    for i, run in enumerate(runs):
        y = len(runs) - i - 1
        ttft = run["first_token_ms"]
        reasoning_end = run.get("reasoning_end_ms", ttft)
        tool_complete = run.get("tool_call_complete_ms", reasoning_end)
        done = run["done_ms"]

        # Phases
        phases = [
            ("TTFT", 0, ttft),
            ("Reasoning", ttft, reasoning_end),
            ("Tool call generation", reasoning_end, tool_complete),
            ("Post-tool to finish", tool_complete, done),
        ]

        for label, start, end in phases:
            width = end - start
            if width > 0:
                ax.barh(
                    y,
                    width,
                    left=start,
                    height=0.6,
                    color=colors[label],
                    edgecolor="white",
                    linewidth=0.5,
                )

        # Annotate the dispatch opportunity
        gap = done - tool_complete
        if gap > 5:
            ax.annotate(
                f"{gap:.0f}ms\nsaved",
                xy=(tool_complete + gap / 2, y),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([f"Run {r['run']}" for r in reversed(runs)])
    ax.set_xlabel("Time (ms)")
    ax.set_title(title)

    # Legend
    patches = [mpatches.Patch(color=c, label=lbl) for lbl, c in colors.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output}", file=sys.stderr)


def plot_summary_bars(runs: list, output: str):
    """Plot mean timing with the dispatch gap highlighted."""
    import statistics

    ttfts = [r["first_token_ms"] for r in runs]
    tool_completes = [r.get("tool_call_complete_ms", r["done_ms"]) for r in runs]
    dones = [r["done_ms"] for r in runs]
    gaps = [d - t for d, t in zip(dones, tool_completes)]

    fig, ax = plt.subplots(figsize=(8, 4))

    labels = ["TTFT", "Tool Call\nComplete", "Stream\nDone"]
    means = [
        statistics.mean(ttfts),
        statistics.mean(tool_completes),
        statistics.mean(dones),
    ]
    stds = [
        statistics.stdev(ttfts),
        statistics.stdev(tool_completes),
        statistics.stdev(dones),
    ]

    ax.bar(
        labels,
        means,
        yerr=stds,
        capsize=5,
        color=["#4CAF50", "#FF9800", "#F44336"],
        alpha=0.85,
    )

    # Annotate gap
    mean_gap = statistics.mean(gaps)
    ax.annotate(
        f"Dispatch saves\n~{mean_gap:.0f}ms avg",
        xy=(2, means[2]),
        xytext=(2.4, means[2] - 20),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
        ha="center",
    )

    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Streaming Timing Summary (n={len(runs)})")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Summary plot saved to {output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL timing file")
    parser.add_argument("--output", required=True, help="Output PNG for waterfall")
    parser.add_argument("--summary", default=None, help="Output PNG for summary bars")
    args = parser.parse_args()

    runs = load_runs(args.input)
    plot_waterfall(runs, args.output)
    if args.summary:
        plot_summary_bars(runs, args.summary)


if __name__ == "__main__":
    main()
