#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Render the DynoSim hero figure in the unified Dynamo Dark aesthetic.

A Pareto-frontier scatter: a dense cloud of DynoSim-explored configurations
bounded above by the throughput/latency frontier, with a handful of
GPU-verified configurations sitting on that frontier. This is the digest
hero for "DynoSim: Simulating the Pareto Frontier".

Treatment follows the canonical Dynamo Dark tokens (design_tokens.yaml,
consumed via plotly_dynamo.py): pure-black ground, token accent colors,
Roboto Mono ticks, no rounded corners. The hero/display title uses the
Dynamo Dark display treatment — Helvetica Neue Light, title case — paired
with a muted Helvetica subtitle, distinct from the compact 18 px uppercase
chart title used for dense dashboards.

The explored-config cloud is a DETERMINISTIC, representative reproduction
(fixed RNG seed) of the frontier's shape and ranges, not the original
measured sweep: the source sweep data is not committed to this repo. Every
run reproduces the same figure byte-for-byte. The point cloud is
illustrative of "sweep the space"; no individual point asserts a measured
config result.

Usage:
    python3 gen_hero.py                 # -> ../dynosim-hero.png
    python3 gen_hero.py -o out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

# Pareto frontier shape: Tok/s/GPU falls as Tok/s/User rises.
# y = A * x**(-K) fitted to the corpus hero's endpoints
# (x=10 -> ~1450 Tok/s/GPU, x=110 -> ~70 Tok/s/GPU).
FRONTIER_A = 26640.0
FRONTIER_K = 1.264
X_MIN, X_MAX = 10.0, 118.0

# GPU-verified configs sit ON the frontier at these Tok/s/User points.
VERIFIED_X = [12.0, 22.0, 35.0, 48.0, 62.0, 78.0, 95.0, 108.0]

SEED = 42
N_EXPLORED = 900


def frontier(x: np.ndarray | float) -> np.ndarray | float:
    """Tok/s/GPU on the Pareto frontier for a given Tok/s/User."""
    return FRONTIER_A * np.power(x, -FRONTIER_K)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = str(Path(__file__).resolve().parent.parent / "dynosim-hero.png")
    parser.add_argument("-o", "--output", default=default_out, help="Output PNG path")
    parser.add_argument("--width", type=int, default=1600, help="Canvas width (px)")
    parser.add_argument("--height", type=int, default=1180, help="Canvas height (px)")
    args = parser.parse_args()

    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]
    font_sans = tokens["typography"]["font_family"]

    green = colors["accent"]["dynamo_green"]  # DynoSim explored configs
    blue = colors["accent"]["cpu_blue"]  # GPU-verified configs
    frontier_color = colors["text"]["secondary"]  # neutral reference curve
    text_primary = colors["text"]["primary"]
    text_muted = colors["text"]["muted"]

    # Hero/display title set: Helvetica Neue Light, title case (falls back to
    # the token sans stack where Helvetica Neue is unavailable).
    hero_font = "Helvetica Neue, Helvetica, Arial, sans-serif"

    rng = np.random.default_rng(SEED)

    # Explored cloud: sample x uniformly, then place each point at a fraction
    # of the frontier height so the cloud hugs the curve and thins downward.
    x_explored = rng.uniform(X_MIN, X_MAX, N_EXPLORED)
    frac = np.clip(1.0 - np.abs(rng.normal(0.0, 0.28, N_EXPLORED)), 0.06, 1.0)
    y_explored = frontier(x_explored) * frac

    # Frontier curve.
    x_line = np.linspace(X_MIN, 122.0, 240)
    y_line = frontier(x_line)

    # GPU-verified configs on the frontier.
    x_ver = np.array(VERIFIED_X)
    y_ver = frontier(x_ver)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Pareto Frontier",
            line=dict(color=frontier_color, width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_explored,
            y=y_explored,
            mode="markers",
            name="DynoSim Explored Configs",
            marker=dict(color=green, size=4.5, opacity=0.72, line=dict(width=0)),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_ver,
            y=y_ver,
            mode="markers",
            name="GPU Verified Configs",
            marker=dict(
                color=blue,
                size=13,
                symbol="diamond",
                line=dict(width=1.2, color=text_primary),
            ),
            hoverinfo="skip",
        )
    )

    annotations = [
        # Inline label for the frontier curve.
        dict(
            x=44,
            y=340,
            xref="x",
            yref="y",
            text="Pareto Frontier",
            showarrow=False,
            xanchor="left",
            font=dict(family=font_sans, size=15, color=text_primary),
        ),
        # One open-space punch line (Dynamo Dark: sans, muted, non-italic).
        dict(
            x=46,
            y=1120,
            xref="x",
            yref="y",
            text="Sweep the space in minutes, then deploy with confidence.",
            showarrow=False,
            xanchor="left",
            font=dict(family=font_sans, size=19, color=text_muted),
        ),
    ]

    fig.update_layout(
        template=template,
        title=dict(
            text="DynoSim: Simulating the Pareto Frontier",
            font=dict(family=hero_font, size=42, color=text_primary, weight=300),
            subtitle=dict(
                text=(
                    "Discrete-event simulation of the full Dynamo inference "
                    "stack — thousands of configs in minutes."
                ),
                font=dict(family=hero_font, size=20, color=text_muted, weight=300),
            ),
            x=0.03,
            xanchor="left",
            y=0.94,
            yanchor="top",
        ),
        xaxis=dict(
            title="Tok/s/User",
            range=[0, 125],
            tickmode="array",
            tickvals=[0, 20, 40, 60, 80, 100, 120],
        ),
        yaxis=dict(
            title="Tok/s/GPU",
            range=[0, 1550],
            tickmode="array",
            tickvals=[0, 200, 400, 600, 800, 1000, 1200, 1400],
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.11,
            yanchor="top",
            font=dict(family=font_sans, size=15, color=colors["text"]["secondary"]),
        ),
        width=args.width,
        height=args.height,
        margin=dict(l=95, r=55, t=180, b=120),
        annotations=annotations,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), scale=3)
    print(f"Wrote {out}  ({args.width}x{args.height} @3x)")


if __name__ == "__main__":
    main()
