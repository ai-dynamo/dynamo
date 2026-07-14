#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Example: architecture / data-flow diagram in the Dynamo Dark aesthetic.

A left-to-right pipeline of role-colored component boxes with orthogonal
green data-flow arrows and a squared **replay edge** (NV green, solid — the
primary data path) looping telemetry back to the front of the pipeline.

Demonstrates: role-based component colors from the token palette, computed
box geometry, arrowheads that land on exact box edges, and a squared
(right-angle) return path instead of a curved bezier.

Title uses the Dynamo Dark **display/hero** treatment: Helvetica Neue Light,
title case. The layout uses representative, hard-coded component names (no
external data). Renders deterministically.

Usage:
    python3 gen_fig_2_architecture.py        # -> images/fig-2-architecture.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

HERO_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

# (label, accent hex key, muted-fill hex key) per component, left to right.
STAGES = [
    ("Engine Cores", "dynamo_green", "green"),
    ("Router", "cpu_blue", "blue"),
    ("Planner", "amethyst", "purple"),
    ("Deployment", "emerald", "teal"),
]


def main() -> None:
    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]
    green = colors["accent"]["dynamo_green"]
    text_primary = colors["text"]["primary"]
    text_muted = colors["text"]["muted"]
    font_sans = tokens["typography"]["font_family"]

    fig = go.Figure()

    # Computed box geometry on a 0-100 canvas.
    n = len(STAGES)
    box_w, gap, y0, y1 = 19.0, 5.0, 44.0, 60.0
    left0 = 4.0
    y_mid = (y0 + y1) / 2
    xs = [left0 + i * (box_w + gap) for i in range(n)]

    shapes = []
    annotations = []
    for (label, accent_key, fill_key), x in zip(STAGES, xs):
        accent = colors["accent"][accent_key]
        fill = colors["fills"][fill_key]
        shapes.append(
            dict(
                type="rect",
                x0=x,
                x1=x + box_w,
                y0=y0,
                y1=y1,
                line=dict(color=accent, width=1.5),
                fillcolor=fill,
                layer="above",
            )
        )
        annotations.append(
            dict(
                x=x + box_w / 2,
                y=y_mid,
                text=label,
                showarrow=False,
                font=dict(family=font_sans, size=14, color=text_primary),
            )
        )

    # Forward data-flow arrows (green solid), edge to edge.
    for i in range(n - 1):
        annotations.append(
            dict(
                x=xs[i + 1],
                y=y_mid,
                ax=xs[i] + box_w,
                ay=y_mid,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.4,
                arrowwidth=2,
                arrowcolor=green,
                text="",
            )
        )

    # Squared replay edge (NV green, solid): last box -> down -> left -> up
    # into the first box. Drawn as one polyline, arrowhead added at the tip.
    x_start = xs[-1] + box_w / 2
    x_end = xs[0] + box_w / 2
    y_loop = 30.0
    fig.add_trace(
        go.Scatter(
            x=[x_start, x_start, x_end, x_end],
            y=[y0, y_loop, y_loop, y0 - 1.5],
            mode="lines",
            line=dict(color=green, width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    annotations.append(
        dict(
            x=x_end,
            y=y0,
            ax=x_end,
            ay=y0 - 1.5,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.4,
            arrowwidth=2,
            arrowcolor=green,
            text="",
        )
    )
    annotations.append(
        dict(
            x=(x_start + x_end) / 2,
            y=y_loop,
            yshift=-12,
            text="replay edge (KV events)",
            showarrow=False,
            font=dict(family=font_sans, size=12, color=text_muted),
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Dynamo Serving Stack: One Simulated Timeline",
            font=dict(family=HERO_FONT, size=40, color=text_primary, weight=300),
            subtitle=dict(
                text=(
                    "Engine cores, Router, Planner — one clock, one harness, "
                    "with KV events replayed to the front."
                ),
                font=dict(family=HERO_FONT, size=19, color=text_muted, weight=300),
            ),
            x=0.03,
            xanchor="left",
            y=0.93,
            yanchor="top",
        ),
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(range=[10, 72], visible=False),
        width=1600,
        height=780,
        margin=dict(l=40, r=40, t=150, b=30),
        shapes=shapes,
        annotations=annotations,
    )

    out = Path(__file__).parent / "images" / "fig-2-architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), scale=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
