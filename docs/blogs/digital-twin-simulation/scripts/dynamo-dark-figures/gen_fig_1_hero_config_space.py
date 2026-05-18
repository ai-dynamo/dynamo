#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate fig-1 hero -- the configuration space scatter.

Tells the BEFORE / WITH DYNOSIM / VERIFIED narrative in one frame using
the canonical tok/s/user x tok/s/gpu Pareto plot every inference reader
knows:

- ~2,500 dim NV-green dots: configs DynoSim explored (synthetic cloud,
  sitting below the Pareto ceiling because they are dominated)
- A thin NV-green dotted Pareto ceiling so "on the frontier" reads visually
- 8 fluorite diamonds: configs confirmed on real cluster GPUs, near-but-on
  the frontier (the configs you actually paid GPU time to measure)
- Three story blocks across the top, color-coded to the visual elements

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
EMERALD = C["accent"]["emerald"]
FLUORITE = C["accent"]["fluorite"]
# Local override: token amethyst (#5d1682) is too dim on black; brighter
# purple keeps card 3 visually in-step with cards 1 & 2.
AMETHYST = "#a960e8"
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

    # --- (1) Achievable region: shaded wash UNDER the Pareto ceiling.
    # Draws first so it sits behind everything. Soft NV-green fill so the
    # frontier reads as a literal ceiling with the achievable space glowing
    # beneath. fill="tozeroy" extends down to y=0.
    fx = np.linspace(8, 125, 400)
    fy = ceiling(fx)
    fig.add_trace(go.Scatter(
        x=fx, y=fy,
        mode="lines",
        name="achievable region",
        line=dict(color="rgba(0,0,0,0)", width=0),
        fill="tozeroy",
        fillcolor=rgba(NV_GREEN, 0.08),
        hoverinfo="skip", showlegend=False,
    ))

    # --- (2) DynoSim sim cloud: small dots, slightly brighter than v34.
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        name="DynoSim",
        legendrank=2,
        marker=dict(size=4, color=NV_GREEN, opacity=0.55, line=dict(width=0)),
        hoverinfo="skip", showlegend=True,
    ))

    # --- (3) Pareto ceiling line: thick, solid, full opacity. The hard edge.
    # Drawn BEFORE the diamonds so the diamonds sit on top of the frontier
    # line where they overlap.
    fig.add_trace(go.Scatter(
        x=fx, y=fy,
        mode="lines",
        name="Pareto frontier",
        line=dict(color=NV_GREEN, width=3, dash="solid"),
        opacity=0.95,
        hoverinfo="skip", showlegend=False,
    ))

    # --- (4) Real-cluster runs: bold fluorite-filled diamonds with a
    # hairline off-white border. Drawn LAST so the diamonds sit on top of
    # the wash, the cloud, and the frontier line.
    fig.add_trace(go.Scatter(
        x=rx, y=ry,
        mode="markers",
        name="GPU Measured",
        legendrank=1,
        marker=dict(
            size=18, color=FLUORITE, symbol="diamond",
            opacity=1.0,
            line=dict(width=0.75, color="rgba(255,255,255,0.85)"),
        ),
        hoverinfo="skip", showlegend=True,
    ))

    # --- Punch line in the open upper-right wedge above the Pareto ceiling.
    # Understated: muted text, regular weight, sans-serif. The diamonds and
    # cards carry the visual weight; this is the takeaway prose.
    fig.add_annotation(
        x=72, y=1080, xref="x", yref="y",
        xanchor="center", yanchor="middle",
        text="What GPUs run in hours, DynoSim runs in seconds.",
        showarrow=False,
        font=dict(family=SANS, size=18, color=TEXT_MUTED, weight=300),
    )

    # --- Three story-block cards (full figure width) ---
    # Plotly only supports xref="paper" (plot-area-relative) for shapes,
    # so we convert container coords -> paper coords by hand using the
    # known figure width / left margin.
    FIG_W = 1240
    L_MARGIN = 80
    R_MARGIN = 40
    PLOT_W = FIG_W - L_MARGIN - R_MARGIN
    def cx(container_x: float) -> float:
        """Container-x (0 = figure left, 1 = figure right) -> paper-x."""
        return (container_x * FIG_W - L_MARGIN) / PLOT_W
    def cw(container_width: float) -> float:
        """Container width -> paper width."""
        return container_width * FIG_W / PLOT_W

    CARD_HALF_C = 0.150  # in container coords
    CARD_HALF = cw(CARD_HALF_C)
    CARD_Y0, CARD_Y1 = 1.13, 1.35
    CARD_CENTERS_C = [0.18, 0.50, 0.82]  # container-x

    # Subtle full-width divider just below the cards.
    fig.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=cx(0.02), x1=cx(0.98),
        y0=1.10, y1=1.10,
        line=dict(color="#3a3a3a", width=0.5),
    )
    for cx_center, color in zip(CARD_CENTERS_C, (FLUORITE, NV_GREEN, AMETHYST)):
        x_paper = cx(cx_center)
        # Card body
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=x_paper - CARD_HALF, x1=x_paper + CARD_HALF,
            y0=CARD_Y0, y1=CARD_Y1,
            fillcolor=rgba(color, 0.07),
            line=dict(width=0),
            layer="below",
        )
        # Thin top accent bar to color-code the card without a full border.
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=x_paper - CARD_HALF, x1=x_paper + CARD_HALF,
            y0=CARD_Y1 - 0.006, y1=CARD_Y1,
            fillcolor=color,
            line=dict(width=0),
            layer="below",
        )

    # --- Three story blocks, left to right ---
    STORY = [
        (CARD_CENTERS_C[0], FLUORITE,
         "Before",
         "<b>8 configs on real GPUs in an hour</b>",
         "~5 min per config on the cluster"),
        (CARD_CENTERS_C[1], NV_GREEN,
         "With DynoSim",
         "<b>5 sec per config, single CPU thread</b>",
         "3,000 configs swept in parallel in minutes"),
        (CARD_CENTERS_C[2], AMETHYST,
         "Verified",
         "<b>1 GPU run to confirm</b>",
         "frontier swept in sim, confirmed on cluster"),
    ]
    for cx_center, color, kicker, headline, sub in STORY:
        x_paper = cx(cx_center)
        fig.add_annotation(
            x=x_paper, y=1.325, xref="paper", yref="paper",
            xanchor="center", yanchor="top",
            text=kicker,
            showarrow=False,
            font=dict(family=SANS, size=20, color=TEXT_PRIMARY),
        )
        fig.add_annotation(
            x=x_paper, y=1.255, xref="paper", yref="paper",
            xanchor="center", yanchor="top",
            text=headline,
            showarrow=False,
            font=dict(family=SANS, size=18, color=TEXT_PRIMARY),
        )
        fig.add_annotation(
            x=x_paper, y=1.170, xref="paper", yref="paper",
            xanchor="center", yanchor="top",
            text=sub,
            showarrow=False,
            font=dict(family=MONO, size=13, color=TEXT_MUTED),
        )

    fig.update_layout(
        template=dynamo_template,
        title=dict(
            text="DynoSim: Simulating the Final Frontier",
            x=0.02, xanchor="left",
            y=0.96, yanchor="top",
            font=dict(
                family="Helvetica Neue, HelveticaNeue, sans-serif",
                size=42, color=TEXT_PRIMARY,
                weight=300,
            ),
        ),
        showlegend=True,
        legend=dict(
            x=0.5, y=-0.16, xanchor="center", yanchor="top",
            orientation="h",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(family=SANS, size=14, color=TEXT_PRIMARY),
            itemsizing="constant",
            traceorder="normal",
        ),
        margin=dict(l=80, r=40, t=260, b=120),
        width=1240, height=820,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis=dict(
            title=dict(
                text="Tok/s/User →",
                font=dict(family=SANS, size=14, color=TEXT_MUTED),
                standoff=12,
            ),
            range=[0, 130],
            showgrid=True, gridcolor=rgba("#3a3a3a", 0.6), gridwidth=0.5,
            zeroline=False,
            tickfont=dict(family=MONO, size=13, color=TEXT_SECONDARY),
            ticks="", linewidth=0.5, showline=False,
        ),
        yaxis=dict(
            title=dict(
                text="↑ Tok/s/GPU",
                font=dict(family=SANS, size=14, color=TEXT_MUTED),
                standoff=12,
            ),
            range=[0, 1500],
            showgrid=True, gridcolor=rgba("#3a3a3a", 0.6), gridwidth=0.5,
            zeroline=False,
            tickfont=dict(family=MONO, size=13, color=TEXT_SECONDARY),
            ticks="", linewidth=0.5, showline=False,
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
