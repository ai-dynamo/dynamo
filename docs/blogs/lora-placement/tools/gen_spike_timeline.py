#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate per-tick churn timeline for the Traffic Spikes scenario (Figure 6).

Produces a synthetic but representative timeline of 200 ticks showing how MCF,
HRW, and Random allocators react to traffic spikes. Aggregate totals are
calibrated to match the simulation results from the spec:
    - Random: ~35,752 total churn
    - HRW:    ~238 total churn
    - MCF:    ~103 total churn

Uses the Dynamo dark Plotly template (design_tokens.yaml + plotly_dynamo.py).

Usage:
    python3 gen_spike_timeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TICKS = 200
SPIKE_TICKS = [50, 130]  # tick indices where traffic spikes occur
RNG_SEED = 42

# Aggregate targets from simulation (Traffic Spikes scenario)
TARGET_RANDOM = 35752
TARGET_HRW = 238
TARGET_MCF = 103

# Colors
COLOR_MCF = "#76b900"  # Dynamo green
COLOR_HRW = "#0071c5"  # CPU blue
COLOR_RANDOM = "#8c8c8c"  # Medium gray
COLOR_SPIKE = "#fac200"  # Fluorite / amber


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _generate_random(rng: np.random.Generator) -> np.ndarray:
    """Random allocator: constant background churn ~178 ops/tick."""
    mean_per_tick = TARGET_RANDOM / N_TICKS  # ~178
    data = rng.poisson(lam=mean_per_tick, size=N_TICKS).astype(float)
    # Scale to hit target exactly
    data = data * (TARGET_RANDOM / data.sum())
    return data


def _generate_hrw(rng: np.random.Generator) -> np.ndarray:
    """HRW allocator: initial burst, then exponential-decay spikes.

    Rough budget:
      - Initial burst at t=1:        ~40 ops
      - Spike 1 (t~50): burst ~30, half-life ~8 ticks
      - Spike 2 (t~130): burst ~30, half-life ~8 ticks
      - Small scattered background noise
    Total target: ~238
    """
    data = np.zeros(N_TICKS)
    half_life = 8.0
    decay = np.log(2) / half_life

    # Initial placement burst (ticks 1-15)
    for t in range(1, 16):
        data[t] = 40.0 * np.exp(-decay * (t - 1))

    # Spike responses
    for spike_t in SPIKE_TICKS:
        burst = 30.0
        for dt in range(0, 30):
            t = spike_t + dt
            if t < N_TICKS:
                data[t] += burst * np.exp(-decay * dt)

    # Small background noise
    noise = rng.poisson(lam=0.05, size=N_TICKS).astype(float)
    data += noise

    # Scale to match target
    data = data * (TARGET_HRW / data.sum())
    return data


def _generate_mcf(rng: np.random.Generator) -> np.ndarray:
    """MCF allocator: sharp single-tick spikes, then immediately back to zero.

    Budget:
      - Initial burst at t=1:  ~32 ops
      - Spike at t=50:         ~20 ops
      - Spike at t=130:        ~15 ops
      - Tiny scattered residual
    Total target: ~103
    """
    data = np.zeros(N_TICKS)

    # Initial placement
    data[1] = 32.0

    # Sharp spike responses (MCF rebalances in one tick)
    data[50] = 20.0
    data[51] = 8.0
    data[130] = 15.0
    data[131] = 5.0

    # A few small scattered rebalances
    scatter_ticks = [10, 25, 75, 100, 160, 180]
    for t in scatter_ticks:
        data[t] += rng.uniform(1.0, 3.0)

    # Scale to match target
    data = data * (TARGET_MCF / data.sum())
    return data


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    random_data = _generate_random(rng)
    hrw_data = _generate_hrw(rng)
    mcf_data = _generate_mcf(rng)

    ticks = np.arange(N_TICKS)

    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    typo = tokens["typography"]

    fig = go.Figure()

    # MCF line
    fig.add_trace(
        go.Scatter(
            x=ticks,
            y=mcf_data,
            mode="lines",
            name="MCF (Dynamo)",
            line=dict(color=COLOR_MCF, width=2.5),
            hovertemplate=(
                "<b>MCF</b><br>" "Tick %{x}<br>" "Churn: %{y:.1f} ops" "<extra></extra>"
            ),
        )
    )

    # HRW line
    fig.add_trace(
        go.Scatter(
            x=ticks,
            y=hrw_data,
            mode="lines",
            name="HRW",
            line=dict(color=COLOR_HRW, width=2.5),
            hovertemplate=(
                "<b>HRW</b><br>" "Tick %{x}<br>" "Churn: %{y:.1f} ops" "<extra></extra>"
            ),
        )
    )

    # Vertical spike markers
    shapes = []
    annotations = []
    for spike_t in SPIKE_TICKS:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=spike_t,
                x1=spike_t,
                y0=0,
                y1=1,
                line=dict(color=COLOR_SPIKE, width=1.5, dash="dash"),
                layer="below",
            )
        )
        annotations.append(
            dict(
                x=spike_t,
                y=1.0,
                xref="x",
                yref="paper",
                text="<b>SPIKE</b>",
                showarrow=False,
                yanchor="bottom",
                yshift=4,
                font=dict(
                    family=typo["font_family"],
                    size=9,
                    color=COLOR_SPIKE,
                ),
            )
        )

    # Random annotation (off-chart, shown as text box in top-right)
    random_avg = TARGET_RANDOM / N_TICKS
    annotations.append(
        dict(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"<b>Random: ~{random_avg:.0f} ops/tick (off-chart)</b>",
            showarrow=False,
            xanchor="right",
            yanchor="top",
            font=dict(
                family=typo["font_family"],
                size=10,
                color=COLOR_RANDOM,
            ),
            bgcolor="#000000",
            bordercolor=COLOR_RANDOM,
            borderwidth=1,
            borderpad=4,
        )
    )

    # Faint horizontal band at the top to hint at Random's magnitude
    shapes.append(
        dict(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            x1=1,
            y0=0.92,
            y1=1.0,
            fillcolor="rgba(140, 140, 140, 0.08)",
            line=dict(width=0),
            layer="below",
        )
    )

    fig.update_layout(
        template=template,
        title=dict(text="PER-TICK CHURN  --  TRAFFIC SPIKE SCENARIO"),
        xaxis=dict(
            title="Tick",
            range=[0, N_TICKS],
            dtick=25,
        ),
        yaxis=dict(
            title="Churn (loads + unloads)",
            rangemode="tozero",
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        width=775,
        height=500,
        margin=dict(l=60, r=40, t=70, b=50),
        shapes=shapes,
        annotations=annotations,
    )

    # Write outputs
    images_dir = Path(__file__).resolve().parent.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig-6-spike-timeline"
    png = images_dir / f"{stem}.png"
    svg = images_dir / f"{stem}.svg"
    fig.write_image(str(png), scale=3)
    fig.write_image(str(svg))
    print(f"Wrote {png.name}")
    print(f"Wrote {svg.name}")
    print("\nSpike timeline chart generated.")


if __name__ == "__main__":
    main()
