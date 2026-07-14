#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Example: decision cascade with a horizontal delta bracket (Dynamo Dark).

A request flows through a routing decision cascade, ending in two outcome
chips — the baseline (Round Robin, grey) and the optimized path (KV Router,
green). A green **horizontal bracket** spans the two chips and carries the
delta label — a bracket is unambiguously a comparison; a floating number is
not.

Demonstrates: the two-chip + green-bracket delta pattern, muted grey
baseline vs green accent, and computed bracket geometry.

Title uses the Dynamo Dark display/hero treatment (Helvetica Neue Light,
title case). Values are representative, hard-coded (no external data).

Usage:
    python3 gen_fig_5_decision_cascade.py   # -> images/fig-5-decision-cascade.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

HERO_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

CASCADE = ["Request", "Prefix match?", "Worker warm?", "Route"]


def main() -> None:
    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]
    green = colors["accent"]["dynamo_green"]
    grey_fill = colors["chart_fills"][4]  # #555555 muted grey
    green_fill = colors["fills"]["green"]
    subtle = colors["border"]["subtle"]
    text_primary = colors["text"]["primary"]
    text_muted = colors["text"]["muted"]
    font_sans = tokens["typography"]["font_family"]
    font_mono = tokens["typography"]["font_family_mono"]

    fig = go.Figure()
    shapes = []
    annotations = []

    # Decision cascade across the top: small chips joined by arrows.
    n = len(CASCADE)
    cw, cgap, cy0, cy1 = 17.0, 6.0, 66.0, 78.0
    cxs = [6.0 + i * (cw + cgap) for i in range(n)]
    for label, x in zip(CASCADE, cxs):
        shapes.append(
            dict(
                type="rect",
                x0=x,
                x1=x + cw,
                y0=cy0,
                y1=cy1,
                line=dict(color=subtle, width=1),
                fillcolor=colors["fills"]["neutral"],
                layer="above",
            )
        )
        annotations.append(
            dict(
                x=x + cw / 2,
                y=(cy0 + cy1) / 2,
                text=label,
                showarrow=False,
                font=dict(family=font_sans, size=13, color=text_primary),
            )
        )
    for i in range(n - 1):
        annotations.append(
            dict(
                x=cxs[i + 1],
                y=(cy0 + cy1) / 2,
                ax=cxs[i] + cw,
                ay=(cy0 + cy1) / 2,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor=text_muted,
                text="",
            )
        )

    # Two outcome chips on a shared baseline.
    chip_w, oy0, oy1 = 30.0, 20.0, 40.0
    ax0, bx0 = 14.0, 56.0
    a_center = ax0 + chip_w / 2
    b_center = bx0 + chip_w / 2
    outcomes = [
        (ax0, "Round Robin", "0.38 prefix reuse", grey_fill, subtle),
        (bx0, "KV Router", "0.44 prefix reuse", green_fill, green),
    ]
    for x0, name, val, fill, stroke in outcomes:
        shapes.append(
            dict(
                type="rect",
                x0=x0,
                x1=x0 + chip_w,
                y0=oy0,
                y1=oy1,
                line=dict(color=stroke, width=1.5),
                fillcolor=fill,
                layer="above",
            )
        )
        annotations.append(
            dict(
                x=x0 + chip_w / 2,
                y=oy1 - 6,
                text=name,
                showarrow=False,
                font=dict(family=font_sans, size=15, color=text_primary),
            )
        )
        annotations.append(
            dict(
                x=x0 + chip_w / 2,
                y=oy0 + 6,
                text=val,
                showarrow=False,
                font=dict(family=font_mono, size=13, color=text_muted),
            )
        )

    # Green horizontal delta bracket spanning the two chip centers.
    y_spine = 48.0
    fig.add_trace(
        go.Scatter(
            x=[a_center, a_center, b_center, b_center],
            y=[oy1, y_spine, y_spine, oy1],
            mode="lines",
            line=dict(color=green, width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    annotations.append(
        dict(
            x=(a_center + b_center) / 2,
            y=y_spine,
            yshift=12,
            text="<b>+16% prefix reuse</b>",
            showarrow=False,
            font=dict(family=font_sans, size=15, color=green),
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Prefix-Aware Routing: The Decision Cascade",
            font=dict(family=HERO_FONT, size=40, color=text_primary, weight=300),
            subtitle=dict(
                text="Same cascade, two outcomes — KV Router lifts prefix reuse.",
                font=dict(family=HERO_FONT, size=19, color=text_muted, weight=300),
            ),
            x=0.03,
            xanchor="left",
            y=0.95,
            yanchor="top",
        ),
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(range=[14, 84], visible=False),
        width=1600,
        height=820,
        margin=dict(l=40, r=40, t=150, b=30),
        shapes=shapes,
        annotations=annotations,
    )

    out = Path(__file__).parent / "images" / "fig-5-decision-cascade.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), scale=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
