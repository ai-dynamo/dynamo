"""
Plot the H200 Kimi counterpart data for the digital-twin blog draft.

Reads the normalized CSV/JSON files in this directory and emits three figures:
  - h200_kv_router_exp.png
  - h200_kvbm_g2_diagnostic.png
  - h200_optimizer_candidates.png
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

NV_GREEN = "#76B900"
INK = "#1A1A1A"
GRAY = "#8C8C8C"
BLUE = "#0071C5"
ORANGE = "#E87722"
RED = "#D62728"

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parents[1] / "images"

ROUTER_STYLES = {
    "round_robin": ("Round robin", GRAY, "o", "-"),
    "kv_router": ("KV Router", NV_GREEN, "D", "-"),
    "kv_router_aic": ("KV Router + AIC load", BLUE, "s", "-"),
}
ROUTER_ORDER = ["round_robin", "kv_router", "kv_router_aic"]


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


def load_router_data(path):
    out = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            concurrency = int(row["concurrency"])
            mode = row["router_mode"]
            out[mode][concurrency] = {
                "tps_gpu": float(row["tps_gpu"]),
                "tps_cluster": float(row["tps_cluster"]),
                "tps_user": float(row["tps_user"]),
                "tpot_ms": float(row["tpot_ms"]),
                "ttft_ms": float(row["ttft_ms"]),
                "e2e_ms": float(row["e2e_ms"]),
                "prefix_cache_reused_ratio": float(row["prefix_cache_reused_ratio"]),
            }
    return out


def load_source_data(path):
    out = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            concurrency = int(row["concurrency"])
            source = row["source"]
            out[source][concurrency] = {
                "tps_gpu": float(row["tps_gpu"]),
                "tps_user": float(row["tps_user"]),
                "tpot_ms": float(row["tpot_ms"]),
                "ttft_ms": float(row["ttft_ms"]),
                "e2e_ms": float(row["e2e_ms"]),
            }
    return out


def series(data, key, metric, concs):
    return [data[key][c][metric] for c in concs]


def plot_router(data_path, out_path):
    data = load_router_data(data_path)
    concs = sorted(set.intersection(*(set(data[mode]) for mode in ROUTER_ORDER)))
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
        label, color, marker, linestyle = ROUTER_STYLES[mode]
        ax_ttft.plot(
            x_positions,
            series(data, mode, "ttft_ms", concs),
            linestyle,
            color=color,
            linewidth=2.5,
            marker=marker,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.3,
            label=label,
        )

    rr_ttft = series(data, "round_robin", "ttft_ms", concs)
    kv_ttft = series(data, "kv_router", "ttft_ms", concs)
    for x, y, rr in zip(x_positions, kv_ttft, rr_ttft):
        reduction = (rr - y) / rr * 100
        ax_ttft.annotate(
            f"-{reduction:.0f}%",
            (x, y),
            textcoords="offset points",
            xytext=(8, -5),
            fontsize=9,
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
    ax_ttft.legend(fontsize=9, loc="upper left", frameon=False)

    all_x = []
    all_y = []
    for mode in ROUTER_ORDER:
        label, color, marker, linestyle = ROUTER_STYLES[mode]
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
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.3,
            label=label,
        )
        if mode != "kv_router":
            for c, x, y in zip(concs, xs, ys):
                offset = (8, 7) if mode != "round_robin" else (-16, -15)
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
    ax_pareto.legend(fontsize=9, loc="upper right", frameon=False)

    rr_prefix = series(data, "round_robin", "prefix_cache_reused_ratio", concs)
    kv_prefix = series(data, "kv_router", "prefix_cache_reused_ratio", concs)
    aic_prefix = series(data, "kv_router_aic", "prefix_cache_reused_ratio", concs)
    prefix_note = (
        f"Prefix reuse avg: RR {sum(rr_prefix) / len(rr_prefix):.2f}, "
        f"KV {sum(kv_prefix) / len(kv_prefix):.2f}, "
        f"KV+AIC {sum(aic_prefix) / len(aic_prefix):.2f}"
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


def plot_kvbm(data_path, out_path):
    data = load_source_data(data_path)
    concs = sorted(set(data["baseline"]) & set(data["g2"]))
    x_positions = list(range(len(concs)))

    fig, (ax_ttft, ax_pareto) = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.81, bottom=0.13, wspace=0.22)

    for source, label, color, marker in (
        ("baseline", "Baseline (G1 only)", GRAY, "o"),
        ("g2", "With G2 (32K blocks)", NV_GREEN, "D"),
    ):
        ax_ttft.plot(
            x_positions,
            series(data, source, "ttft_ms", concs),
            "-",
            color=color,
            linewidth=2.4,
            marker=marker,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.3,
            label=label,
        )
    ax_ttft.set_xlabel("Concurrency", fontsize=11, fontweight="bold", labelpad=10)
    ax_ttft.set_ylabel(
        "Mean TTFT (ms, log)", fontsize=11, fontweight="bold", labelpad=10
    )
    ax_ttft.set_xticks(x_positions)
    ax_ttft.set_xticklabels([f"c={c}" for c in concs])
    ax_ttft.set_yscale("log")
    apply_nv_style(ax_ttft)
    ax_ttft.legend(fontsize=9, loc="upper left", frameon=False)
    ax_ttft.text(
        0.05,
        0.09,
        "Diagnostic only: G2 and baseline overlap because\nthis setup does not create G1 pressure.",
        transform=ax_ttft.transAxes,
        fontsize=9,
        color=RED,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
    )

    all_x = []
    all_y = []
    for source, label, color, marker in (
        ("baseline", "Baseline (G1 only)", GRAY, "o"),
        ("g2", "With G2 (32K blocks)", NV_GREEN, "D"),
    ):
        xs = series(data, source, "tps_user", concs)
        ys = series(data, source, "tps_gpu", concs)
        all_x.extend(xs)
        all_y.extend(ys)
        ax_pareto.plot(
            xs,
            ys,
            "-",
            color=color,
            linewidth=2.4,
            marker=marker,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.3,
            label=label,
        )
        for c, x, y in zip(concs, xs, ys):
            ax_pareto.annotate(
                f"c={c}",
                (x, y),
                textcoords="offset points",
                xytext=(8, 7),
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
    ax_pareto.legend(fontsize=9, loc="upper right", frameon=False)

    add_title(
        fig,
        "H200 Kimi KVBM Diagnostic",
        sub_subtitle="H200-SXM   moonshotai/Kimi-K2.5   TP=4   1 worker   G1=16K blocks   toolagent trace",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_optimizer(evaluated_path, summary_path, out_path):
    rows = []
    with evaluated_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    **row,
                    "output_throughput_tok_s": float(row["output_throughput_tok_s"]),
                    "mean_ttft_ms": float(row["mean_ttft_ms"]),
                    "mean_tpot_ms": float(row["mean_tpot_ms"]),
                    "mean_e2e_latency_ms": float(row["mean_e2e_latency_ms"]),
                    "violation_penalty": float(row["violation_penalty"]),
                    "prefill_tp": int(float(row["prefill_tp"])),
                    "decode_tp": int(float(row["decode_tp"])),
                    "prefill_workers": int(float(row["prefill_workers"])),
                    "decode_workers": int(float(row["decode_workers"])),
                }
            )
    summary = json.loads(summary_path.read_text())
    best = summary["report"]["best_infeasible"]
    sla = summary["report"]["setup"]["sla"]
    top = sorted(rows, key=lambda r: r["violation_penalty"])[:8]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.82, bottom=0.13)

    ax.scatter(
        [r["mean_ttft_ms"] for r in rows],
        [r["output_throughput_tok_s"] for r in rows],
        s=34,
        color=GRAY,
        alpha=0.35,
        label="Evaluated candidates",
    )
    ax.scatter(
        [r["mean_ttft_ms"] for r in top],
        [r["output_throughput_tok_s"] for r in top],
        s=70,
        color=BLUE,
        edgecolor="white",
        linewidth=1.0,
        label="Lowest SLA-violation candidates",
        zorder=4,
    )
    ax.scatter(
        [best["mean_ttft_ms"]],
        [best["output_throughput_tok_s"]],
        s=140,
        color=NV_GREEN,
        edgecolor="white",
        linewidth=1.3,
        marker="D",
        label="Best near-miss",
        zorder=5,
    )
    ax.axvline(sla["mean_ttft_ms"], color=RED, linewidth=2, linestyle="--")
    ax.text(
        sla["mean_ttft_ms"],
        0.14,
        " TTFT SLA",
        transform=ax.get_xaxis_transform(),
        color=RED,
        fontsize=9,
        fontweight="bold",
        rotation=90,
        va="bottom",
        ha="right",
    )
    ax.annotate(
        "2/1 TP, 5P/6D\nprefill_load_scale=0.5\nTTFT +58 ms over SLA",
        (best["mean_ttft_ms"], best["output_throughput_tok_s"]),
        textcoords="offset points",
        xytext=(38, -36),
        arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 1.2},
        fontsize=9,
        color=INK,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
    )
    ax.set_xscale("log")
    ax.set_xlabel("Mean TTFT (ms, log)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel(
        "Output throughput (tok/s)", fontsize=11, fontweight="bold", labelpad=10
    )
    ax.set_xlim(3000, max(r["mean_ttft_ms"] for r in rows) * 1.4)
    apply_nv_style(ax)
    ax.legend(fontsize=9, loc="lower left", frameon=False)
    add_title(
        fig,
        "H200 Kimi Replay Optimization Candidates",
        sub_subtitle="16-GPU budget   objective=throughput   TTFT<=4000 ms, TPOT<=75 ms, E2E<=20000 ms",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=HERE)
    parser.add_argument("--out-dir", default=IMAGES)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    plot_router(data_dir / "router_data.csv", out_dir / "h200_kv_router_exp.png")
    plot_kvbm(
        data_dir / "kvbm_diagnostic_data.csv", out_dir / "h200_kvbm_g2_diagnostic.png"
    )
    plot_optimizer(
        data_dir / "optimize_evaluated.csv",
        data_dir / "optimize_summary.json",
        out_dir / "h200_optimizer_candidates.png",
    )


if __name__ == "__main__":
    main()
