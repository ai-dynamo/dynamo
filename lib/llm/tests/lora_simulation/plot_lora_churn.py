#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualize LoRA allocation churn: HRW vs Random vs MCF.

Reads CSV files exported by the Rust simulation test and generates
matplotlib charts showing per-tick churn, cumulative churn, load patterns,
and LoRA lifecycle timelines for all three algorithms.

Usage:
    # 1. Export CSVs from Rust tests:
    cargo test --test lora_simulation -- test_export_csv --ignored --nocapture

    # 2. Generate plots:
    python lib/llm/tests/lora_simulation/plot_lora_churn.py

    # Or save to PNG instead of showing interactively:
    python lib/llm/tests/lora_simulation/plot_lora_churn.py --save

    # Plot a single scenario:
    python lib/llm/tests/lora_simulation/plot_lora_churn.py --scenario c20_low --save
"""

import argparse
import csv
import sys
from pathlib import Path

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
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    churn_file = csv_dir / f"{name}_churn.csv"
    load_file = csv_dir / f"{name}_load.csv"
    summary_file = csv_dir / f"{name}_summary.csv"
    meta_file = csv_dir / f"{name}_meta.csv"
    lifecycle_file = csv_dir / f"{name}_lifecycle.csv"
    replicas_file = csv_dir / f"{name}_replicas.csv"

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

    lifecycle = read_csv(lifecycle_file) if lifecycle_file.exists() else []
    _replicas_data = read_csv(replicas_file) if replicas_file.exists() else []

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
        "lifecycle": "#7E57C2",
    }

    has_lifecycle = len(lifecycle) > 0
    # 1 metrics panel + 3 base panels + lifecycle
    num_panels = 4 + (1 if has_lifecycle else 0)
    fig, axes = plt.subplots(num_panels, 1, figsize=(16, 4 * num_panels), sharex=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # ── Panel 1 (top): Metrics — total churn + churn-free ratio ────────
    ax_metrics = axes[0]
    _plot_scenario_metrics(
        ax_metrics,
        ticks,
        hrw_churn,
        random_churn,
        mcf_churn,
        summary,
        colors,
    )

    # ── Panel 2: Per-tick churn (bar chart with 3 algorithms) ────────────
    ax1 = axes[1]
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
    ax1.set_ylabel("Churn (loads + unloads)")
    ax1.set_title("Per-Tick Churn")
    ax1.legend(loc="upper right")
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Panel 3: Cumulative churn + LoRA adds/removes ────────────────────
    ax2 = axes[2]
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

    # ── Panel 4: Load pattern (area + line) ──────────────────────────────
    ax3 = axes[3]
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
        import numpy as np

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
        import numpy as np

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
            from matplotlib.patches import Patch

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

    # ── Panel 5: LoRA Lifecycle Timeline (Gantt chart) ───────────────────
    panel_idx = 4
    if has_lifecycle:
        ax4 = axes[panel_idx]
        _plot_lifecycle_gantt(ax4, lifecycle, load_data, meta, colors)
        panel_idx += 1

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


def _plot_scenario_metrics(
    ax,
    ticks,
    hrw_churn,
    random_churn,
    mcf_churn,
    summary,
    colors,
):
    """Draw a two-part metrics panel: total churn (left, log) + churn-free ratio (right)."""
    import numpy as np

    n_ticks = len(ticks)
    hrw_zero = sum(1 for c in hrw_churn if c == 0)
    rand_zero = sum(1 for c in random_churn if c == 0)
    mcf_zero = sum(1 for c in mcf_churn if c == 0)

    hrw_total = sum(hrw_churn)
    rand_total = sum(random_churn)
    mcf_total = sum(mcf_churn)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x = np.arange(3)
    bar_colors = [colors["mcf"], colors["hrw"], colors["random"]]

    # ── Left half: Total churn (log scale) ──────────────────────────────
    left = ax.inset_axes([0.02, 0.1, 0.45, 0.85])
    totals = [mcf_total, hrw_total, rand_total]
    bars_l = left.barh(x, totals, color=bar_colors, edgecolor="white", height=0.6)
    left.set_yticks(x)
    left.set_yticklabels(["MCF", "HRW", "Random"], fontsize=10, fontweight="bold")
    left.set_xscale("log")
    left.set_xlabel("Total Churn (log scale)")
    left.set_title("Total Churn (loads + unloads)", fontsize=10, fontweight="bold")
    left.grid(axis="x", alpha=0.3, which="both")
    for bar, val in zip(bars_l, totals):
        left.text(
            bar.get_width() * 1.15,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    # MCF reduction annotation
    if hrw_total > 0:
        pct = (1 - mcf_total / hrw_total) * 100
        left.annotate(
            f"−{pct:.0f}% vs HRW",
            xy=(mcf_total, 0),
            xytext=(mcf_total * 3, -0.3),
            fontsize=8,
            color="#2E7D32",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1.2),
        )

    # ── Right half: Churn-free tick ratio ───────────────────────────────
    right = ax.inset_axes([0.55, 0.1, 0.43, 0.85])
    ratios = [
        100 * mcf_zero / max(n_ticks, 1),
        100 * hrw_zero / max(n_ticks, 1),
        100 * rand_zero / max(n_ticks, 1),
    ]
    bars_r = right.barh(x, ratios, color=bar_colors, edgecolor="white", height=0.6)
    right.set_yticks(x)
    right.set_yticklabels(["MCF", "HRW", "Random"], fontsize=10, fontweight="bold")
    right.set_xlim(0, 110)
    right.set_xlabel("Churn-Free Ticks (%)")
    right.set_title(
        "Stability: % of ticks with zero churn", fontsize=10, fontweight="bold"
    )
    right.axvline(x=100, color="gray", linestyle=":", alpha=0.3)
    right.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars_r, ratios):
        right.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )


def _plot_lifecycle_gantt(
    ax, lifecycle: list[dict], load_data: list[dict], meta: dict, colors: dict
):
    """Draw a Gantt-style timeline of each LoRA's active period with load heatmap."""
    import matplotlib.pyplot as plt

    # Parse lifecycle data
    loras = []
    for row in lifecycle:
        loras.append(
            {
                "name": row["lora_name"],
                "start": int(row["start_tick"]),
                "end": int(row["end_tick"]),
                "peak_load": int(row["peak_load"]),
                "index": int(row["lora_index"]),
            }
        )

    # Sort by start tick, then by name
    loras.sort(key=lambda x: (x["start"], x["name"]))

    # Build per-LoRA per-tick load matrix from load_data
    total_ticks = int(meta.get("total_ticks", len(load_data)))

    # Get LoRA column names from load CSV (columns after tick, total_load, active_loras)
    if load_data:
        lora_cols = [
            k
            for k in load_data[0].keys()
            if k not in ("tick", "total_load", "active_loras")
        ]
    else:
        lora_cols = []

    # Map lora_name → per-tick load array
    lora_load_map = {}
    for col in lora_cols:
        loads = [int(row.get(col, 0)) for row in load_data]
        lora_load_map[col] = loads

    # Color map: load intensity
    cmap = plt.cm.YlOrRd
    max_peak = max((lora["peak_load"] for lora in loras), default=1)

    # Limit display to first 30 LoRAs for readability
    display_loras = loras[:30]
    n = len(display_loras)

    y_labels = []
    for i, lora in enumerate(display_loras):
        y = n - 1 - i  # top-to-bottom
        start = lora["start"]
        end = lora["end"]
        name = lora["name"]
        y_labels.append(name)

        # Draw per-tick colored segments
        loads = lora_load_map.get(name, [])
        for t in range(start, min(end, total_ticks)):
            load_val = loads[t] if t < len(loads) else 0
            intensity = load_val / max_peak if max_peak > 0 else 0
            color = cmap(intensity * 0.85)  # cap at 85% to avoid pure red
            ax.barh(y, 1, left=t, height=0.7, color=color, edgecolor="none")

        # Draw border
        ax.barh(
            y,
            end - start,
            left=start,
            height=0.7,
            color="none",
            edgecolor="#666",
            linewidth=0.5,
        )

        # Label peak load
        mid = (start + end) / 2
        ax.text(
            mid,
            y,
            f"pk={lora['peak_load']}",
            ha="center",
            va="center",
            fontsize=6,
            color="black",
            alpha=0.8,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(reversed(y_labels), fontsize=7)
    ax.set_title("LoRA Lifecycle Timeline (color = load intensity)")
    ax.set_xlim(0, total_ticks)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_peak))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, aspect=30)
    cbar.set_label("Load (requests)", fontsize=8)


def _plot_replica_distribution(
    ax_hrw, ax_random, ax_mcf, replicas_data: list[dict], ticks: list[int], colors: dict
):
    """Draw stacked bar charts showing replica count distribution over time.

    X-axis = tick, Y-axis = number of LoRAs with that replica count.
    One panel per algorithm: HRW, Random, MCF.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Discover all replica-count columns
    sample = replicas_data[0] if replicas_data else {}
    hrw_cols = sorted(
        [k for k in sample if k.startswith("hrw_r")],
        key=lambda c: int(c.split("_r")[1]),
    )
    random_cols = sorted(
        [k for k in sample if k.startswith("random_r")],
        key=lambda c: int(c.split("_r")[1]),
    )
    mcf_cols = sorted(
        [k for k in sample if k.startswith("mcf_r")],
        key=lambda c: int(c.split("_r")[1]),
    )

    # Extract replica counts
    replica_ticks = [int(r["tick"]) for r in replicas_data]

    # Color palette for replica counts (diverging from light to dark)
    max_cols = max(len(hrw_cols), len(random_cols), len(mcf_cols), 1)
    palette = plt.cm.viridis(np.linspace(0.15, 0.95, max_cols))

    def _stacked_bar(ax, cols, title):
        bottom = np.zeros(len(replica_ticks))
        for i, col in enumerate(cols):
            vals = np.array([int(r.get(col, 0)) for r in replicas_data], dtype=float)
            replica_num = int(col.split("_r")[1])
            ax.bar(
                replica_ticks,
                vals,
                bottom=bottom,
                width=0.9,
                color=palette[i % len(palette)],
                alpha=0.85,
                label=f"r={replica_num}",
            )
            bottom += vals
        ax.set_ylabel("Number of LoRAs")
        ax.set_title(title)
        ax.legend(
            loc="upper right", fontsize=8, ncol=min(len(cols), 6), title="Replicas"
        )
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    _stacked_bar(ax_hrw, hrw_cols, "HRW: LoRA Replica Count Distribution")
    _stacked_bar(ax_random, random_cols, "Random: LoRA Replica Count Distribution")
    _stacked_bar(ax_mcf, mcf_cols, "MCF: LoRA Replica Count Distribution")


def plot_comparison_summary(csv_dir: Path, save: bool, out_dir: Path):
    """Generate a summary bar chart of total churn (log scale) across all scenarios."""
    import matplotlib.pyplot as plt
    import numpy as np

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
        "Fixed cluster: N=8 × K=4 = 32 slots  |  Log scale",
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

    ax.set_ylabel("Total Churn (loads + unloads)  — log scale")
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


def plot_churn_free_ratio(csv_dir: Path, save: bool, out_dir: Path):
    """Bar chart: % of ticks with zero churn per algorithm, across all scenarios."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels = []
    hrw_ratios, random_ratios, mcf_ratios = [], [], []
    scenarios_found = []

    for name in SCENARIOS:
        churn_file = csv_dir / f"{name}_churn.csv"
        meta_file = csv_dir / f"{name}_meta.csv"
        if not churn_file.exists():
            continue
        data = read_csv(churn_file)
        meta = read_meta(meta_file) if meta_file.exists() else {}
        n_ticks = len(data)
        if n_ticks == 0:
            continue

        hrw_zero = sum(1 for r in data if int(r.get("hrw_churn", 0)) == 0)
        rand_zero = sum(1 for r in data if int(r.get("random_churn", 0)) == 0)
        mcf_zero = sum(1 for r in data if int(r.get("mcf_churn", 0)) == 0)

        hrw_ratios.append(100.0 * hrw_zero / n_ticks)
        random_ratios.append(100.0 * rand_zero / n_ticks)
        mcf_ratios.append(100.0 * mcf_zero / n_ticks)
        labels.append(_short_label(name, meta))
        scenarios_found.append(name)

    if not scenarios_found:
        return

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - w, hrw_ratios, w, label="HRW", color="#2196F3", alpha=0.85)
    b2 = ax.bar(x, random_ratios, w, label="Random", color="#F44336", alpha=0.85)
    b3 = ax.bar(x + w, mcf_ratios, w, label="MCF", color="#4CAF50", alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_ylabel("Churn-Free Ticks (%)")
    ax.set_title(
        "Churn-Free Tick Ratio by Algorithm\n"
        "Higher = more stable (fewer ticks where any adapter was moved)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color="gray", linestyle=":", alpha=0.3)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        p = out_dir / "mcf_churn_free_ratio.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {p}")
        plt.close(fig)
    else:
        plt.show()


def plot_churn_cdf(csv_dir: Path, save: bool, out_dir: Path):
    """Empirical CDF of per-tick churn values, one subplot per scenario."""
    import matplotlib.pyplot as plt
    import numpy as np

    available = []
    for name in SCENARIOS:
        if (csv_dir / f"{name}_churn.csv").exists():
            available.append(name)
    if not available:
        return

    n = len(available)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
    fig.suptitle(
        "Per-Tick Churn Distribution (Empirical CDF)\n"
        "Curves closer to top-left = lower churn",
        fontsize=13,
        fontweight="bold",
    )

    colors = {"hrw": "#2196F3", "random": "#F44336", "mcf": "#4CAF50"}

    for idx, name in enumerate(available):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        data = read_csv(csv_dir / f"{name}_churn.csv")
        meta_file = csv_dir / f"{name}_meta.csv"
        meta = read_meta(meta_file) if meta_file.exists() else {}

        for algo, key, color in [
            ("HRW", "hrw_churn", colors["hrw"]),
            ("Random", "random_churn", colors["random"]),
            ("MCF", "mcf_churn", colors["mcf"]),
        ]:
            vals = sorted(int(row.get(key, 0)) for row in data)
            n_pts = len(vals)
            ecdf_y = np.arange(1, n_pts + 1) / n_pts
            ax.step(vals, ecdf_y, where="post", label=algo, color=color, linewidth=2)

        ax.set_title(_short_label(name, meta), fontsize=10)
        ax.set_xlabel("Churn per tick")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(available), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()

    if save:
        p = out_dir / "mcf_churn_cdf.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {p}")
        plt.close(fig)
    else:
        plt.show()


def plot_efficiency_frontier(csv_dir: Path, save: bool, out_dir: Path):
    """Scatter: average slot utilization % vs total churn, one dot per (scenario, algo)."""
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"HRW": "#2196F3", "Random": "#F44336", "MCF": "#4CAF50"}
    markers = {"HRW": "o", "Random": "s", "MCF": "D"}

    # Collect data points: (util%, total_churn, algo, scenario_label)
    points: dict[str, list] = {"HRW": [], "Random": [], "MCF": []}

    for name in SCENARIOS:
        summary_file = csv_dir / f"{name}_summary.csv"
        load_file = csv_dir / f"{name}_load.csv"
        meta_file = csv_dir / f"{name}_meta.csv"
        if not summary_file.exists() or not load_file.exists():
            continue

        meta = read_meta(meta_file) if meta_file.exists() else {}
        total_slots = int(meta.get("total_slots", 32))

        # Compute average active LoRAs as proxy for slot utilization
        load_data = read_csv(load_file)
        avg_active = np.mean([int(r["active_loras"]) for r in load_data])
        util_pct = 100.0 * avg_active / total_slots

        summary = {}
        for row in read_csv(summary_file):
            summary[row["metric"]] = row
        total_churn = summary.get("total_churn", {})
        label = _short_label(name, meta)

        for algo in ["HRW", "Random", "MCF"]:
            churn = int(total_churn.get(algo.lower(), 0))
            points[algo].append((util_pct, churn, label))

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

    ax.set_xlabel("Avg Slot Utilization (%)", fontsize=12)
    ax.set_ylabel("Total Churn (loads + unloads)", fontsize=12)
    ax.set_title(
        "Churn Efficiency Frontier\n"
        "Bottom-right = ideal (high utilization, low churn)",
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
        p = out_dir / "mcf_efficiency_frontier.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {p}")
        plt.close(fig)
    else:
        plt.show()


def plot_placement_stability(csv_dir: Path, save: bool, out_dir: Path):
    """Heatmap of per-LoRA load stability for each algorithm.

    For each scenario that has per-LoRA load data, computes:
      stability(lora, algo) = fraction of ticks where the LoRA's load
      was unchanged from the previous tick (i.e. no rebalancing trigger).

    Plots a 3-column heatmap (HRW / Random / MCF) with LoRAs on the y-axis,
    ticks on the x-axis, and color = load value. Stable allocations show
    smooth color bands; unstable ones flicker.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    # Pick the most interesting scenarios for this viz
    target_scenarios = ["hot_lora_poisson", "daily", "spike", "mmpp"]
    chosen = None
    for name in target_scenarios:
        if (csv_dir / f"{name}_load.csv").exists() and (
            csv_dir / f"{name}_churn.csv"
        ).exists():
            chosen = name
            break
    if chosen is None:
        return

    load_file = csv_dir / f"{chosen}_load.csv"
    churn_file = csv_dir / f"{chosen}_churn.csv"
    meta_file = csv_dir / f"{chosen}_meta.csv"
    meta = read_meta(meta_file) if meta_file.exists() else {}

    load_data = read_csv(load_file)
    churn_data = read_csv(churn_file)
    if not load_data:
        return

    # Extract per-LoRA load matrix
    lora_cols = [
        k
        for k in load_data[0].keys()
        if k not in ("tick", "total_load", "active_loras")
    ]
    n_ticks = len(load_data)
    n_loras = len(lora_cols)

    # Build load matrix (loras x ticks)
    load_matrix = np.zeros((n_loras, n_ticks))
    for t, row in enumerate(load_data):
        for i, col in enumerate(lora_cols):
            load_matrix[i, t] = int(row.get(col, 0))

    # Sort LoRAs by total load (descending) to show Zipf structure
    total_loads = load_matrix.sum(axis=1)
    sort_idx = np.argsort(-total_loads)
    load_matrix = load_matrix[sort_idx]
    lora_names_sorted = [lora_cols[i] for i in sort_idx]

    # Only show top 40 LoRAs (most active) for readability
    max_show = min(40, n_loras)
    load_matrix = load_matrix[:max_show]
    lora_names_sorted = lora_names_sorted[:max_show]

    # Per-tick churn for each algorithm
    hrw_churn = np.array([int(r.get("hrw_churn", 0)) for r in churn_data])
    rand_churn = np.array([int(r.get("random_churn", 0)) for r in churn_data])
    mcf_churn = np.array([int(r.get("mcf_churn", 0)) for r in churn_data])

    # Compute per-LoRA stability score:
    # For load data, count ticks where the LoRA's load changed from prev tick
    # This approximates allocation instability triggers
    def stability_scores(matrix):
        """Return per-LoRA fraction of ticks with unchanged load."""
        diffs = np.diff(matrix, axis=1)
        changes = (diffs != 0).sum(axis=1)
        active = (matrix[:, 1:] > 0).sum(axis=1)  # only count active ticks
        return np.where(active > 0, 1.0 - changes / active, 1.0)

    load_stability = stability_scores(load_matrix)

    # ── Figure: 2 rows, 2 cols ─────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 2, figsize=(18, 10), gridspec_kw={"height_ratios": [2, 1]}
    )
    title = build_title(chosen, meta)
    fig.suptitle(
        f"Placement Stability Analysis — {title}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # Top-left: Load heatmap (the "input" — shows what the allocator sees)
    ax_load = axes[0, 0]
    vmax = max(load_matrix.max(), 1)
    cmap_load = plt.cm.YlOrRd
    im1 = ax_load.imshow(
        load_matrix,
        aspect="auto",
        cmap=cmap_load,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_load.set_title("Per-LoRA Load (input to allocator)", fontsize=11)
    ax_load.set_ylabel("LoRA (sorted by total load)")
    ax_load.set_xlabel("Tick")
    # Show every 5th LoRA name
    ytick_pos = list(range(0, max_show, max(1, max_show // 15)))
    ax_load.set_yticks(ytick_pos)
    ax_load.set_yticklabels([lora_names_sorted[i] for i in ytick_pos], fontsize=7)
    plt.colorbar(im1, ax=ax_load, label="Load (requests)", shrink=0.8)

    # Top-right: Churn heatmap (per-tick churn for all 3 algos, stacked)
    ax_churn = axes[0, 1]
    churn_matrix = np.stack([mcf_churn, hrw_churn, rand_churn])
    cmap_churn = mcolors.LinearSegmentedColormap.from_list(
        "churn_cmap", ["#FFFFFF", "#FFF9C4", "#FF9800", "#D32F2F"], N=256
    )
    im2 = ax_churn.imshow(
        churn_matrix,
        aspect="auto",
        cmap=cmap_churn,
        vmin=0,
        interpolation="nearest",
    )
    ax_churn.set_yticks([0, 1, 2])
    ax_churn.set_yticklabels(["MCF", "HRW", "Random"], fontsize=10, fontweight="bold")
    ax_churn.set_title("Per-Tick Churn by Algorithm (darker = more churn)", fontsize=11)
    ax_churn.set_xlabel("Tick")
    plt.colorbar(im2, ax=ax_churn, label="Churn (loads+unloads)", shrink=0.8)

    # Bottom-left: Per-LoRA stability bar chart
    ax_stab = axes[1, 0]
    x_pos = np.arange(max_show)
    ax_stab.bar(x_pos, 100 * load_stability, color="#4CAF50", alpha=0.7, width=0.8)
    ax_stab.set_ylabel("Load Stability (%)")
    ax_stab.set_xlabel("LoRA (sorted by total load, left = hottest)")
    ax_stab.set_title(
        "Per-LoRA Load Stability (% ticks with unchanged load)", fontsize=11
    )
    ax_stab.set_ylim(0, 105)
    ax_stab.axhline(y=50, color="gray", linestyle=":", alpha=0.4)
    ax_stab.set_xticks(list(range(0, max_show, max(1, max_show // 15))))
    ax_stab.set_xticklabels(
        [lora_names_sorted[i] for i in range(0, max_show, max(1, max_show // 15))],
        fontsize=7,
        rotation=45,
    )
    ax_stab.grid(axis="y", alpha=0.3)

    # Bottom-right: Churn comparison text summary + key insight
    ax_text = axes[1, 1]
    ax_text.axis("off")
    hrw_total = int(hrw_churn.sum())
    rand_total = int(rand_churn.sum())
    mcf_total = int(mcf_churn.sum())
    hrw_zero = int((hrw_churn == 0).sum())
    rand_zero = int((rand_churn == 0).sum())
    mcf_zero = int((mcf_churn == 0).sum())
    summary_text = (
        f"Scenario: {chosen}\n"
        f"{'─' * 45}\n"
        f"{'Metric':<28} {'MCF':>6} {'HRW':>6} {'Random':>8}\n"
        f"{'─' * 45}\n"
        f"{'Total churn':<28} {mcf_total:>6} {hrw_total:>6} {rand_total:>8}\n"
        f"{'Churn-free ticks':<28} {mcf_zero:>6} {hrw_zero:>6} {rand_zero:>8}\n"
        f"{'Churn-free %':<28} {100*mcf_zero/n_ticks:>5.0f}% {100*hrw_zero/n_ticks:>5.0f}% {100*rand_zero/n_ticks:>7.0f}%\n"
        f"{'Peak churn/tick':<28} {int(mcf_churn.max()):>6} {int(hrw_churn.max()):>6} {int(rand_churn.max()):>8}\n"
        f"{'─' * 45}\n"
    )
    if rand_total > 0:
        summary_text += (
            f"\nMCF vs Random: {100*(1-mcf_total/rand_total):.0f}% less churn\n"
            f"MCF vs HRW:    {100*(1-mcf_total/hrw_total):.0f}% less churn"
            if hrw_total > 0
            else ""
        )
    ax_text.text(
        0.05,
        0.95,
        summary_text,
        transform=ax_text.transAxes,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#F5F5F5", ec="#BDBDBD"),
    )

    plt.tight_layout()

    if save:
        p = out_dir / "mcf_placement_stability.png"
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
        help="Plot a single scenario (c20_low, c50_medium, c90_high, c_sine_wave, ...)",
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

    # Check matplotlib is available
    try:
        import matplotlib

        if args.save:
            matplotlib.use("Agg")  # Non-interactive backend for saving
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
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

        print("Plotting churn-free ratio...")
        plot_churn_free_ratio(csv_dir, args.save, out_dir)

        print("Plotting efficiency frontier...")
        plot_efficiency_frontier(csv_dir, args.save, out_dir)

        print("Plotting placement stability...")
        plot_placement_stability(csv_dir, args.save, out_dir)

    if args.save:
        print(f"\nAll plots saved to: {out_dir}")
    else:
        print("\nClose plot windows to continue.")


if __name__ == "__main__":
    main()
