#!/usr/bin/env python3
"""Compare dispatch-off vs dispatch-on timing data."""
import json
import statistics
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed", file=sys.stderr)
    sys.exit(1)


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    off = load("streaming-actionable-state/raw/timing-no-dispatch.jsonl")
    on = load("streaming-actionable-state/raw/timing-with-dispatch.jsonl")

    def gaps(runs):
        return [r["done_ms"] - r.get("tool_call_complete_ms", r["done_ms"]) for r in runs]

    def ttfts(runs):
        return [r["first_token_ms"] for r in runs]

    def tool_times(runs):
        return [r.get("tool_call_complete_ms", r["done_ms"]) for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 1. TTFT comparison
    ax = axes[0]
    data = [ttfts(off), ttfts(on)]
    bp = ax.boxplot(data, labels=["Dispatch OFF", "Dispatch ON"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#FF9800")
    bp["boxes"][1].set_facecolor("#4CAF50")
    ax.set_ylabel("ms")
    ax.set_title("TTFT (no difference expected)")

    # 2. Tool call complete time
    ax = axes[1]
    data = [tool_times(off), tool_times(on)]
    bp = ax.boxplot(data, labels=["Dispatch OFF", "Dispatch ON"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#FF9800")
    bp["boxes"][1].set_facecolor("#4CAF50")
    ax.set_ylabel("ms")
    ax.set_title("Time to Tool Call Complete")

    # 3. Gap: tool complete → stream done
    ax = axes[2]
    data = [gaps(off), gaps(on)]
    bp = ax.boxplot(data, labels=["Dispatch OFF", "Dispatch ON"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#F44336")
    bp["boxes"][1].set_facecolor("#4CAF50")
    ax.set_ylabel("ms")
    ax.set_title("Wasted Gap: Tool Complete → Done")

    mean_off = statistics.mean(gaps(off))
    mean_on = statistics.mean(gaps(on))
    ax.annotate(f"OFF: {mean_off:.0f}ms avg\nON: {mean_on:.0f}ms avg",
                xy=(1.5, max(max(gaps(off)), max(gaps(on))) * 0.85),
                ha="center", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.suptitle("Streaming Tool Dispatch: OFF vs ON (n=10 each)", fontsize=13)
    plt.tight_layout()
    plt.savefig("streaming-actionable-state/plots/dispatch-comparison.png", dpi=150, bbox_inches="tight")
    print("Saved dispatch-comparison.png", file=sys.stderr)


if __name__ == "__main__":
    main()
