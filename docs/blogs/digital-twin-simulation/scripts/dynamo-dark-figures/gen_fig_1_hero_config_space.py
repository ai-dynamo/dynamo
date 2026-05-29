#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate fig-1 hero -- the configuration space scatter.

Tells the DynoSim story in one frame using the canonical
tok/s/user x tok/s/gpu Pareto plot every inference reader knows:

- ~2,500 dim NV-green dots: configs DynoSim explored (synthetic cloud,
  sitting below the Pareto ceiling because they are dominated)
- A thin NV-green dotted Pareto ceiling so "on the frontier" reads visually
- 8 fluorite diamonds: configs confirmed on real cluster GPUs, near-but-on
  the frontier (the configs you actually paid GPU time to measure)
- Title + single takeaway subtitle ("What GPUs run in hours, DynoSim runs
  in seconds."), and a rich legend that doubles as caption ("DynoSim · 3,000
  configs/min" / "GPU Measured · 8 configs/hr"). No chip strip -- earlier
  Before/With/Verified cards were dropped because the chip stripe colors
  collided with the legend colors (Alec feedback, MPDM 2026-05-26).

No "deployed config" marker: DynoSim sweeps the space cheaply, GPU runs
confirm you are on the frontier. Until silicon measures, DynoSim has not
gotten you anywhere.

Output:
    ../images/fig-1-hero-config-space.svg
    ../images/fig-1-hero-config-space.png  (scale=2)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly_dynamo import dynamo_template, load_tokens

HERE = Path(__file__).resolve().parent
TOKENS = load_tokens()

C = TOKENS["colors"]
TY = TOKENS["typography"]

BG = C["background"]["primary"]
TEXT_PRIMARY = C["text"]["primary"]
TEXT_SECONDARY = C["text"]["secondary"]
TEXT_MUTED = C["text"]["muted"]
NV_GREEN = C["accent"]["dynamo_green"]
# Bright blue for GPU markers -- the token cpu_blue (#0071c5) reads too
# dark on the rich-black canvas; bumping luminance so the diamonds anchor
# the eye as ground-truth measurement points.
GPU_BLUE = "#5ec5e5"
SANS = TY["font_family"]
MONO = TY["font_family_mono"]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def main() -> None:
    rng = np.random.default_rng(7)

    # --- Pareto ceiling curve.
    # tok/s/gpu vs tok/s/user trades off: low concurrency = high tok/s/user
    # but low tok/s/gpu (GPU under-utilized); high concurrency = high
    # tok/s/gpu (efficient) but low tok/s/user (each user gets less).
    # Frontier is the UPPER envelope: y = K / (1 + (x/x0)^a) declining
    # from upper-left to lower-right.
    def ceiling(xv: np.ndarray) -> np.ndarray:
        return 1850.0 / (1.0 + (xv / 18.0) ** 1.7)

    # --- DynoSim sim cloud (~2,500 configs).
    # Sample x uniformly across the range, then place points BELOW the
    # ceiling by an exponential amount so dominated configs cluster near
    # the frontier and tail off downward.
    n = 2500
    x = rng.uniform(8, 120, n)
    y_top = ceiling(x)
    below = rng.exponential(180, n)
    y = y_top - below
    keep = y > 80  # keep above the floor; drop off-canvas stragglers
    x, y = x[keep], y[keep]

    # --- Real-cluster runs: 8 fluorite diamonds, near-but-on the frontier
    # (the configs you actually paid GPU time to measure). Multiplicative
    # jitter so the diamonds stay a consistent visual distance below the
    # ceiling at any x -- prevents the rightmost points from clipping
    # against y=0 where the ceiling has collapsed to ~80 tok/s/gpu.
    rx = np.array([12, 22, 35, 48, 62, 78, 95, 108], dtype=float)
    ry = ceiling(rx) * (1.0 - rng.uniform(0.06, 0.18, len(rx)))

    fig = go.Figure()

    fx = np.linspace(8, 125, 400)
    fy = ceiling(fx)

    # --- (1) DynoSim sim cloud: small dots, slightly brighter than v34.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="DynoSim Explored Configs",
            legendrank=1,
            marker=dict(size=5, color=NV_GREEN, opacity=0.85, line=dict(width=0)),
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # --- (3) Pareto ceiling line: thick, solid, full opacity. The hard edge.
    # Drawn BEFORE the diamonds so the diamonds sit on top of the frontier
    # line where they overlap.
    fig.add_trace(
        go.Scatter(
            x=fx,
            y=fy,
            mode="lines",
            name="Pareto frontier",
            line=dict(color=TEXT_PRIMARY, width=1.2, dash="solid"),
            opacity=0.85,
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --- (4) Real-cluster runs: bold fluorite-filled diamonds with a
    # hairline off-white border. Drawn LAST so the diamonds sit on top of
    # the wash, the cloud, and the frontier line.
    fig.add_trace(
        go.Scatter(
            x=rx,
            y=ry,
            mode="markers",
            name="GPU Verified Configs",
            legendrank=2,
            marker=dict(
                size=16,
                color=GPU_BLUE,
                symbol="diamond",
                opacity=1.0,
                line=dict(width=0.4, color="rgba(255,255,255,0.85)"),
            ),
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # --- Editorial punchline: two clauses staggered diagonally so each
    # lands as its own beat. White italic serif against the dark canvas —
    # headline voice, sentence case, not chart chrome.
    EDITORIAL_FONT = dict(
        family="Iowan Old Style, Georgia, 'Times New Roman', serif",
        size=30,
        color=TEXT_PRIMARY,
        weight=300,
    )
    fig.add_annotation(
        x=30,
        y=1150,
        xref="x",
        yref="y",
        xanchor="left",
        yanchor="middle",
        text="<i>Sweep the configuration space in minutes.</i>",
        showarrow=False,
        font=EDITORIAL_FONT,
    )
    fig.add_annotation(
        x=60,
        y=890,
        xref="x",
        yref="y",
        xanchor="left",
        yanchor="middle",
        text="<i>Deploy your inference workload with confidence.</i>",
        showarrow=False,
        font=EDITORIAL_FONT,
    )

    # --- Title (paper-coord annotation, not layout.title). Pinned hard
    # into the top-left corner with minimal padding.
    # Positioning math (figure 920 px, t=150, plot_h=650, l=80):
    #   title_top    = 12 px from figure top
    #   paper_y      = 1 + (150 - 12) / 650 = 1.212
    #   title_left   = 30 px from figure left
    #   paper_x      = (30 - 80) / 1120 = -0.0446
    fig.add_annotation(
        x=-0.0446,
        y=1.212,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        text="DynoSim: Simulating the Pareto Frontier",
        showarrow=False,
        font=dict(
            family="Helvetica Neue, HelveticaNeue, sans-serif",
            size=58,
            color=TEXT_PRIMARY,
            weight=300,
        ),
    )

    # --- Subtitle: tight under the title, aligned on the same x.
    #   title_bottom = 12 + 58 = 70
    #   subtitle_top = 70 + 10 = 80
    #   paper_y      = 1 + (150 - 80) / 650 = 1.108
    fig.add_annotation(
        x=-0.0446,
        y=1.108,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        text=(
            "Discrete-event simulation of the full Dynamo inference stack — "
            "thousands of configs in minutes."
        ),
        showarrow=False,
        font=dict(
            family="Helvetica Neue, HelveticaNeue, sans-serif",
            size=24,
            color=TEXT_MUTED,
            weight=300,
        ),
    )

    # --- Pareto frontier label: sits directly above the curve, no leader
    # line. Roman (non-italic) display sans in white so it reads as a
    # clean callout label, not a quoted editorial aside.
    pareto_anchor_x = 45.0
    pareto_anchor_y = ceiling(np.array([pareto_anchor_x]))[0]  # on the curve
    fig.add_annotation(
        x=pareto_anchor_x,
        y=pareto_anchor_y,
        xref="x",
        yref="y",
        xanchor="left",
        yanchor="bottom",
        yshift=4,  # nudge baseline up a hair so descenders clear the curve
        text="Pareto Frontier",
        showarrow=False,
        font=dict(family=SANS, size=18, color=TEXT_PRIMARY, weight=400),
    )

    fig.update_layout(
        template=dynamo_template,
        # Title is rendered as a paper-coord annotation (not layout.title)
        # so it shares the exact same x reference as the subtitle and the
        # two land pixel-aligned on their left edges.
        showlegend=True,
        legend=dict(
            x=0.5,
            y=-0.13,
            xanchor="center",
            yanchor="top",
            orientation="h",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(family=SANS, size=18, color=TEXT_PRIMARY),
            itemsizing="constant",
            traceorder="normal",
        ),
        margin=dict(l=80, r=40, t=150, b=120),
        width=1240,
        height=920,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis=dict(
            title=dict(
                text="Tok/s/User →",
                font=dict(family=SANS, size=14, color=TEXT_MUTED),
                standoff=12,
            ),
            range=[0, 130],
            showgrid=True,
            gridcolor=rgba("#3a3a3a", 0.6),
            gridwidth=0.5,
            zeroline=False,
            tickfont=dict(family=MONO, size=13, color=TEXT_SECONDARY),
            ticks="",
            linewidth=0.5,
            showline=False,
        ),
        yaxis=dict(
            title=dict(
                text="↑ Tok/s/GPU",
                font=dict(family=SANS, size=14, color=TEXT_MUTED),
                standoff=12,
            ),
            range=[0, 1500],
            showgrid=True,
            gridcolor=rgba("#3a3a3a", 0.6),
            gridwidth=0.5,
            zeroline=False,
            tickfont=dict(family=MONO, size=13, color=TEXT_SECONDARY),
            ticks="",
            linewidth=0.5,
            showline=False,
        ),
    )

    out_svg = HERE.parent / "images" / "fig-1-hero-config-space.svg"
    out_png = HERE.parent / "images" / "fig-1-hero-config-space.png"
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_svg))
    fig.write_image(str(out_png), scale=2)
    print(f"wrote {out_svg}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
