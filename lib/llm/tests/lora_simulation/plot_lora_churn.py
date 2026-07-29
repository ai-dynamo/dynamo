#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualize LoRA routing-target churn: HRW vs Random vs MCF.

Reads CSV files exported by the Rust simulation test and generates
matplotlib charts showing per-tick route-target churn, cumulative churn, and load patterns for all
three algorithms. Replica sets describe controller routing intent, not observed backend adapter
residency or physical cache load/unload operations.

Usage:
    # 1. Export CSVs from Rust tests:
    cargo test --test lora_simulation -- test_export_csv --ignored --nocapture

    # 2. Generate plots:
    python lib/llm/tests/lora_simulation/plot_lora_churn.py

    # Or save to PNG instead of showing interactively:
    python lib/llm/tests/lora_simulation/plot_lora_churn.py --save

    # Plot a single scenario:
    python lib/llm/tests/lora_simulation/plot_lora_churn.py --scenario hot_lora_poisson --save
"""

import argparse
import csv
import sys
from pathlib import Path

# Plotting dependencies (matplotlib, numpy) are optional: reading CSVs works without them and
# only the plotting entrypoints need them. Per .ai/python-guidelines.md these imports live at
# module scope; the guard turns a missing install into a clear message from main() instead of an
# import-time crash.
try:
    import matplotlib

    if "--save" in sys.argv:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    from matplotlib.patches import Patch

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    matplotlib = None
    plt = None
    ticker = None
    np = None
    Patch = None
    MATPLOTLIB_AVAILABLE = False

# ── Locate CSV directory ────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR / ".." / ".." / ".." / ".."
CSV_DIR = REPO_ROOT / "target" / "lora_sim_csv"

SCENARIOS = [
    "hot_lora_poisson",
    "daily",
    "spike",
    "mmpp",
]


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def read_meta(path: Path) -> dict:
    meta = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta[row["key"]] = row["value"]
    return meta


def build_title(name: str, meta: dict) -> str:
    """Build a descriptive title for a scenario."""
    n = meta.get("num_backends", "?")
    k = meta.get("slots_per_backend", "?")
    total_slots = meta.get("total_slots", "?")
    total_loras = meta.get("total_loras", "?")
    loras_used = meta.get("loras_used", total_loras)
    concurrent = meta.get("concurrent_loras", "?")
    lt_mean = meta.get("lifetime_mean", "0")
    lt_stddev = meta.get("lifetime_stddev", "0.0")

    # Lifetime info
    if lt_mean != "0" and lt_mean != "?":
        lifetime_str = f"lifetime={lt_mean}t (σ={lt_stddev})"
    else:
        lifetime_str = ""

    load_model = meta.get("load_model", "")
    base_line2 = (
        f"N={n}×K={k}={total_slots} slots, L={loras_used} LoRAs (pool={total_loras})"
    )

    if load_model == "diurnal":
        zipf_s = meta.get("zipf_s", "?")
        peak = meta.get("peak_total_load", "?")
        trough = meta.get("trough_total_load", "?")
        tpd = meta.get("ticks_per_day", "?")
        line1 = (
            f"Scenario: {name}  |  Daily: Zipf(s={zipf_s}), "
            f"peak={peak}, trough={trough}, T={tpd}t/day"
        )
        line2 = base_line2
    elif load_model == "zipf_poisson":
        zipf_s = meta.get("zipf_s", "?")
        avg_load = meta.get("avg_total_load", "?")
        line1 = f"Scenario: {name}  |  Hot-LoRA Poisson: Zipf(s={zipf_s}), λ_total={avg_load}"
        line2 = base_line2
    elif load_model == "flash_crowd":
        base_load = meta.get("base_total_load", "?")
        spike_mult = meta.get("spike_multiplier", "?")
        half_life = meta.get("decay_half_life", "?")
        flashes = meta.get("flash_ticks", "?")
        line1 = (
            f"Scenario: {name}  |  Spike: base={base_load}, "
            f"{spike_mult}× spike, t½={half_life}, events@{flashes}"
        )
        line2 = base_line2
    elif load_model == "mmpp":
        rates = meta.get("state_rates", "?")
        states = meta.get("state_names", "?")
        line1 = f"Scenario: {name}  |  MMPP: states={states}, rates={rates}"
        line2 = base_line2
    else:
        c_pct = meta.get("c_pct", "?")
        line1 = f"Scenario: {name}  |  C={c_pct}% slot usage"
        line2 = f"N={n}×K={k}={total_slots} slots, L={loras_used} LoRAs, C={concurrent}"

    if lifetime_str:
        line2 += f", {lifetime_str}"

    return f"{line1}\n{line2}"


def plot_scenario(name: str, csv_dir: Path, save: bool, out_dir: Path):
    """Generate a multi-panel figure for a single scenario."""

    churn_file = csv_dir / f"{name}_churn.csv"
    load_file = csv_dir / f"{name}_load.csv"
    summary_file = csv_dir / f"{name}_summary.csv"
    meta_file = csv_dir / f"{name}_meta.csv"

    if not churn_file.exists():
        print(f"  ⚠ Skipping '{name}': {churn_file} not found")
        return

    churn_data = read_csv(churn_file)
    load_data = read_csv(load_file)
    meta = read_meta(meta_file) if meta_file.exists() else {}
    summary = {}
    if summary_file.exists():
        for row in read_csv(summary_file):
            summary[row["metric"]] = {
                "hrw": row.get("hrw", "0"),
                "random": row.get("random", "0"),
                "mcf": row.get("mcf", "0"),
            }

    ticks = [int(r["tick"]) for r in churn_data]
    hrw_churn = [int(r["hrw_churn"]) for r in churn_data]
    random_churn = [int(r["random_churn"]) for r in churn_data]
    mcf_churn = [int(r.get("mcf_churn", 0)) for r in churn_data]
    hrw_cum = [int(r["hrw_cumulative"]) for r in churn_data]
    random_cum = [int(r["random_cumulative"]) for r in churn_data]
    mcf_cum = [int(r.get("mcf_cumulative", 0)) for r in churn_data]

    # LoRA adds/removes per tick
    hrw_adds = [int(r.get("hrw_lora_adds", 0)) for r in churn_data]
    _random_adds = [int(r.get("random_lora_adds", 0)) for r in churn_data]
    mcf_adds = [int(r.get("mcf_lora_adds", 0)) for r in churn_data]
    _hrw_removes = [int(r.get("hrw_lora_removes", 0)) for r in churn_data]
    _random_removes = [int(r.get("random_lora_removes", 0)) for r in churn_data]
    _mcf_removes = [int(r.get("mcf_lora_removes", 0)) for r in churn_data]

    load_ticks = [int(r["tick"]) for r in load_data]
    total_load = [int(r["total_load"]) for r in load_data]
    active_loras = [int(r["active_loras"]) for r in load_data]

    title = build_title(name, meta)

    # Summary annotation
    summary_parts = []
    if summary:
        vals = summary.get("total_churn", {})
        try:
            h, r, m = int(vals["hrw"]), int(vals["random"]), int(vals["mcf"])
            r_pct = f" ({(1 - h/r)*100:.0f}%↓)" if r > 0 else ""
            m_pct = f" ({(1 - m/r)*100:.0f}%↓)" if r > 0 else ""
            summary_parts.append(
                f"Churn — HRW: {h}{r_pct}  MCF: {m}{m_pct}  Random: {r}"
            )
        except (ValueError, KeyError):
            pass
        adds = summary.get("lora_additions", {})
        rems = summary.get("lora_removals", {})
        summary_parts.append(
            f"LoRA adds: HRW={adds.get('hrw','?')} / MCF={adds.get('mcf','?')} / Random={adds.get('random','?')}"
        )
        summary_parts.append(
            f"LoRA removes: HRW={rems.get('hrw','?')} / MCF={rems.get('mcf','?')} / Random={rems.get('random','?')}"
        )
    summary_text = "\n".join(summary_parts)

    # ── Colors ───────────────────────────────────────────────────────────
    colors = {
        "hrw": "#2196F3",
        "random": "#F44336",
        "mcf": "#4CAF50",
        "load": "#78909C",
        "active": "#FF9800",
    }

    num_panels = 3
    fig, axes = plt.subplots(num_panels, 1, figsize=(16, 4 * num_panels), sharex=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # ── Panel 1: Per-tick churn (bar chart with 3 algorithms) ────────────
    ax1 = axes[0]
    bar_width = 0.25
    x_hrw = [t - bar_width for t in ticks]
    x_rand = [t for t in ticks]
    x_mcf = [t + bar_width for t in ticks]
    ax1.bar(x_hrw, hrw_churn, bar_width, label="HRW", color=colors["hrw"], alpha=0.8)
    ax1.bar(
        x_rand,
        random_churn,
        bar_width,
        label="Random",
        color=colors["random"],
        alpha=0.8,
    )
    ax1.bar(x_mcf, mcf_churn, bar_width, label="MCF", color=colors["mcf"], alpha=0.8)
    ax1.set_ylabel("Routing-target churn (adds + removes)")
    ax1.set_title("Per-Tick Churn")
    ax1.legend(loc="upper right")
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Panel 2: Cumulative churn + LoRA adds/removes ────────────────────
    ax2 = axes[1]
    ax2.plot(ticks, hrw_cum, label="HRW cumulative", color=colors["hrw"], linewidth=2)
    ax2.plot(
        ticks,
        random_cum,
        label="Random cumulative",
        color=colors["random"],
        linewidth=2,
    )
    ax2.plot(ticks, mcf_cum, label="MCF cumulative", color=colors["mcf"], linewidth=2)
    ax2.fill_between(ticks, mcf_cum, random_cum, alpha=0.10, color=colors["random"])
    ax2.fill_between(ticks, hrw_cum, mcf_cum, alpha=0.08, color=colors["mcf"])
    ax2.set_ylabel("Cumulative Churn")
    ax2.set_title("Cumulative Churn + LoRA Adds/Removes Over Time")
    ax2.legend(loc="upper left")

    # Overlay LoRA adds/removes as step markers on a secondary y-axis
    ax2_twin = ax2.twinx()
    hrw_cum_adds = list(np.cumsum(hrw_adds))
    mcf_cum_adds = list(np.cumsum(mcf_adds))
    ax2_twin.step(
        ticks,
        hrw_cum_adds,
        where="post",
        color=colors["hrw"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label="HRW adds (cum)",
    )
    ax2_twin.step(
        ticks,
        mcf_cum_adds,
        where="post",
        color=colors["mcf"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label="MCF adds (cum)",
    )
    ax2_twin.set_ylabel("LoRA Adds (cumulative)", fontsize=9)
    ax2_twin.tick_params(axis="y", labelsize=8)
    ax2_twin.legend(loc="center right", fontsize=8)

    if summary_text:
        ax2.annotate(
            summary_text,
            xy=(0.98, 0.05),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9),
        )

    # ── Panel 3: Load pattern (area + line) ──────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(
        load_ticks, total_load, alpha=0.3, color=colors["load"], label="Total Load"
    )
    ax3.plot(load_ticks, total_load, color=colors["load"], linewidth=1.5)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(
        load_ticks,
        active_loras,
        color=colors["active"],
        linewidth=2,
        linestyle="--",
        label="Active LoRAs",
    )
    ax3_twin.set_ylabel("Active LoRAs", color=colors["active"])
    ax3_twin.tick_params(axis="y", labelcolor=colors["active"])
    ax3.set_ylabel("Total Load (requests)")
    ax3.set_title("Load Pattern")

    # Reference lines for target concurrency
    _total_slots_val = int(meta.get("total_slots", 0))
    concurrent = int(meta.get("concurrent_loras", 0))

    load_model = meta.get("load_model", "")
    if load_model == "diurnal":
        # Draw the diurnal load envelope on the primary (load) axis

        tpd = int(meta.get("ticks_per_day", 100))
        peak_l = float(meta.get("peak_total_load", 50))
        trough_l = float(meta.get("trough_total_load", 10))
        t_arr = np.arange(int(meta.get("total_ticks", 200)))
        amp = (peak_l - trough_l) / 2.0
        base = (peak_l + trough_l) / 2.0
        envelope = base - amp * np.cos(2 * np.pi * (t_arr % tpd) / tpd)
        ax3.plot(
            t_arr,
            envelope,
            color="#E91E63",
            linewidth=2,
            linestyle="-.",
            alpha=0.8,
            label="Diurnal envelope",
        )
        # Day/night shading
        for day_start in range(0, int(meta.get("total_ticks", 200)), tpd):
            ax3.axvspan(day_start, day_start + tpd // 4, alpha=0.05, color="navy")
            ax3.axvspan(
                day_start + 3 * tpd // 4, day_start + tpd, alpha=0.05, color="navy"
            )
    elif load_model == "flash_crowd":
        # Mark flash event times with vertical lines
        flash_str = meta.get("flash_ticks", "")
        if flash_str:
            for ft in flash_str.split(";"):
                ft_val = int(ft.strip())
                ax3.axvline(
                    x=ft_val, color="#E91E63", linewidth=2, linestyle="--", alpha=0.7
                )
                ax3.annotate(
                    f"FLASH @{ft_val}",
                    xy=(ft_val, 0.95),
                    xycoords=("data", "axes fraction"),
                    fontsize=8,
                    color="#E91E63",
                    fontweight="bold",
                    ha="center",
                    rotation=90,
                )
    elif load_model == "mmpp":
        # Shade background by MMPP state using the states CSV

        states_file = csv_dir / "mmpp_states.csv"
        if states_file.exists():
            states_data = read_csv(states_file)
            state_colors = {"calm": "#E3F2FD", "busy": "#FFF9C4", "surge": "#FFCDD2"}
            prev_state = None
            block_start = 0
            for row in states_data:
                t = int(row["tick"])
                sname = row["state_name"]
                if sname != prev_state:
                    if prev_state is not None:
                        ax3.axvspan(
                            block_start,
                            t,
                            alpha=0.3,
                            color=state_colors.get(prev_state, "#EEEEEE"),
                            label=prev_state if t < 5 or block_start == 0 else None,
                        )
                    block_start = t
                    prev_state = sname
            # Final block
            if prev_state is not None:
                total_t = int(meta.get("total_ticks", 200))
                ax3.axvspan(
                    block_start,
                    total_t,
                    alpha=0.3,
                    color=state_colors.get(prev_state, "#EEEEEE"),
                )
            # Add state legend entries

            state_patches = [
                Patch(facecolor=c, alpha=0.3, label=s) for s, c in state_colors.items()
            ]
            ax3.legend(
                handles=state_patches, loc="upper left", fontsize=8, title="MMPP State"
            )
    elif concurrent > 0:
        ax3_twin.axhline(
            y=concurrent, color=colors["active"], linestyle=":", alpha=0.5, linewidth=1
        )
        ax3_twin.annotate(
            f"C={concurrent}",
            xy=(0.01, concurrent),
            xycoords=("axes fraction", "data"),
            fontsize=8,
            color=colors["active"],
            alpha=0.7,
        )

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax_bottom = axes[-1]
    ax_bottom.set_xlabel("Tick")

    plt.tight_layout()

    if save:
        out_path = out_dir / f"lora_churn_{name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_comparison_summary(csv_dir: Path, save: bool, out_dir: Path):
    """Generate a summary bar chart of total churn (log scale) across all scenarios."""

    scenarios_found = []
    hrw_totals = []
    random_totals = []
    mcf_totals = []
    labels = []

    for name in SCENARIOS:
        summary_file = csv_dir / f"{name}_summary.csv"
        meta_file = csv_dir / f"{name}_meta.csv"
        if not summary_file.exists():
            continue

        summary = {}
        for row in read_csv(summary_file):
            summary[row["metric"]] = {
                "hrw": row.get("hrw", "0"),
                "random": row.get("random", "0"),
                "mcf": row.get("mcf", "0"),
            }

        meta = read_meta(meta_file) if meta_file.exists() else {}

        vals = summary.get("total_churn", {})
        hrw_totals.append(int(vals.get("hrw", 0)))
        random_totals.append(int(vals.get("random", 0)))
        mcf_totals.append(int(vals.get("mcf", 0)))

        labels.append(_short_label(name, meta))
        scenarios_found.append(name)

    if not scenarios_found:
        print("  ⚠ No scenario data found for summary chart")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(
        "LoRA Allocation: Total Churn — HRW vs Random vs MCF\n"
        "Fixed cluster: N=8 × K=4 = 32 resident slots  |  Log scale",
        fontsize=14,
        fontweight="bold",
    )

    x = np.arange(len(scenarios_found))
    bar_width = 0.25

    bars_hrw = ax.bar(
        x - bar_width,
        hrw_totals,
        bar_width,
        label="HRW",
        color="#2196F3",
        edgecolor="white",
    )
    bars_random = ax.bar(
        x,
        random_totals,
        bar_width,
        label="Random",
        color="#F44336",
        edgecolor="white",
    )
    bars_mcf = ax.bar(
        x + bar_width,
        mcf_totals,
        bar_width,
        label="MCF",
        color="#4CAF50",
        edgecolor="white",
    )

    ax.set_yscale("log")
    ax.set_ylim(bottom=30)

    for i, (h, r, m) in enumerate(zip(hrw_totals, random_totals, mcf_totals)):
        # Annotate MCF vs HRW and MCF vs Random
        if r > 0 and h > 0:
            mcf_vs_hrw = (1 - m / h) * 100
            mcf_vs_rand = (1 - m / r) * 100
            ax.annotate(
                f"MCF: −{mcf_vs_hrw:.2f}% vs HRW\n−{mcf_vs_rand:.2f}% vs Rand",
                xy=(i + bar_width, m),
                xytext=(12, 8),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="#2E7D32",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="white", ec="#4CAF50", alpha=0.8
                ),
            )
        # Value labels on each bar
        for bar_set, val in [(bars_hrw, h), (bars_random, r), (bars_mcf, m)]:
            ax.text(
                bar_set[i].get_x() + bar_set[i].get_width() / 2,
                val * 1.08,
                f"{val:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Total routing-target churn (adds + removes) — log scale")
    ax.set_title("Total Churn by Load Pattern")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3, which="both")

    plt.tight_layout()

    if save:
        out_path = out_dir / "lora_churn_summary.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


# ============================================================================
# New MCF-focused visualizations
# ============================================================================


def plot_route_pressure(csv_dir: Path, save: bool, out_dir: Path):
    """Scatter route-target pressure vs churn, one dot per (scenario, algorithm).

    Route targets are controller intent, not observed resident adapters. Values above 100% mean
    the routing table names more targets than the resident cache can hold simultaneously.
    """

    colors = {"HRW": "#2196F3", "Random": "#F44336", "MCF": "#4CAF50"}
    markers = {"HRW": "o", "Random": "s", "MCF": "D"}

    # Collect data points: (util%, total_churn, algo, scenario_label)
    points: dict[str, list] = {"HRW": [], "Random": [], "MCF": []}

    for name in SCENARIOS:
        summary_file = csv_dir / f"{name}_summary.csv"
        replicas_file = csv_dir / f"{name}_replicas.csv"
        meta_file = csv_dir / f"{name}_meta.csv"
        if not summary_file.exists() or not replicas_file.exists():
            continue

        meta = read_meta(meta_file) if meta_file.exists() else {}
        total_slots = int(meta.get("total_slots", 0))
        if total_slots <= 0:
            continue

        replica_data = read_csv(replicas_file)

        def average_route_pressure(algo_prefix: str) -> float:
            route_targets = []
            for row in replica_data:
                route_targets.append(
                    sum(
                        int(count) * int(column.rsplit("_r", 1)[1])
                        for column, count in row.items()
                        if column.startswith(f"{algo_prefix}_r")
                    )
                )
            return 100.0 * np.mean(route_targets) / total_slots

        summary = {}
        for row in read_csv(summary_file):
            summary[row["metric"]] = row
        total_churn = summary.get("total_churn", {})
        label = _short_label(name, meta)

        for algo in ["HRW", "Random", "MCF"]:
            churn = int(total_churn.get(algo.lower(), 0))
            points[algo].append((average_route_pressure(algo.lower()), churn, label))

    if not any(points.values()):
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for algo in ["Random", "HRW", "MCF"]:
        pts = points[algo]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(
            xs,
            ys,
            label=algo,
            color=colors[algo],
            marker=markers[algo],
            s=100,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )
        # Annotate each point with scenario label
        for x, y, lbl in pts:
            ax.annotate(
                lbl.replace("\n", " "),
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.7,
            )

    ax.set_xlabel("Average route targets / resident slot capacity (%)", fontsize=12)
    ax.set_ylabel("Total routing-target churn (adds + removes)", fontsize=12)
    ax.set_title(
        "Routing-Target Pressure vs Churn\n"
        "Values above 100% indicate soft over-subscription and potential cache swaps",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Log scale on y to handle Random's massive churn
    ax.set_yscale("log")
    ax.set_ylim(bottom=10)

    plt.tight_layout()

    if save:
        p = out_dir / "routing_pressure_vs_churn.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {p}")
        plt.close(fig)
    else:
        plt.show()


def _short_label(name: str, meta: dict) -> str:
    """Build a short label for a scenario (for bar chart x-axis)."""
    load_model = meta.get("load_model", "")
    if load_model == "diurnal":
        return "Daily"
    elif load_model == "zipf_poisson":
        return "Hot-LoRA\nPoisson"
    elif load_model == "flash_crowd":
        spike = meta.get("spike_multiplier", "?")
        return f"Spike\n{spike}×"
    elif load_model == "mmpp":
        return "MMPP\n3-state"
    else:
        c_pct = meta.get("c_pct", "?")
        return f"C={c_pct}%"


def main():
    parser = argparse.ArgumentParser(description="Visualize LoRA allocation churn")
    parser.add_argument(
        "--save", action="store_true", help="Save PNGs instead of showing interactively"
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=str(CSV_DIR),
        help=f"Directory containing CSV files (default: {CSV_DIR})",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Plot a single scenario (hot_lora_poisson, daily, spike, mmpp)",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = csv_dir / "plots"

    if not csv_dir.exists():
        print(f"ERROR: CSV directory not found: {csv_dir}")
        print()
        print("Run the CSV export first:")
        print(
            "  cargo test --test lora_simulation -- test_export_csv --ignored --nocapture"
        )
        sys.exit(1)

    # Plotting requires matplotlib (imported at module scope, guarded by MATPLOTLIB_AVAILABLE).
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib is required. Install with:")
        print("  pip install matplotlib")
        sys.exit(1)
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSVs from: {csv_dir}")
    print()

    scenarios = [args.scenario] if args.scenario else SCENARIOS

    # Per-scenario plots
    for name in scenarios:
        print(f"Plotting scenario: {name}")
        plot_scenario(name, csv_dir, args.save, out_dir)

    # Cross-scenario comparison charts
    if not args.scenario:
        print("Plotting summary comparison...")
        plot_comparison_summary(csv_dir, args.save, out_dir)

        print("Plotting routing-target pressure...")
        plot_route_pressure(csv_dir, args.save, out_dir)

    if args.save:
        print(f"\nAll plots saved to: {out_dir}")
    else:
        print("\nClose plot windows to continue.")


if __name__ == "__main__":
    main()
