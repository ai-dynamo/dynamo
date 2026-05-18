#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate fig-5: planner cold-start SLA cliff (planner_exp_3 redraw).

Single panel: p90 TTFT (log y) vs engine startup_time (linear x).
A coral vertical dashed line marks the ~200 s SLA cliff; a muted
dashed line marks the TTFT SLA target at 1500 ms. Numbers source
from PR-9139 prose:
  - planner holds SLA cleanly up to ~180 s startup delay
  - cliff at ~200 s, monotonic degradation after
  - at 300 s, p90 TTFT = 242,000 ms (perpetually backlogged)

Values between the documented anchor points are approximated from
the engineer's rendered planner_exp_3.png in PR-9139.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly_dynamo import dynamo_template, load_tokens

HERE = Path(__file__).resolve().parent
TOKENS = load_tokens()

C = TOKENS["colors"]
TY = TOKENS["typography"]
TEXT_PRIMARY = C["text"]["primary"]
TEXT_SECONDARY = C["text"]["secondary"]
TEXT_MUTED = C["text"]["muted"]
BORDER_SUBTLE = C["border"]["subtle"]
NV_GREEN = C["accent"]["dynamo_green"]
CPU_BLUE = C["accent"]["cpu_blue"]
CORAL = C["accent"]["coral"]
SANS = TY["font_family"]
MONO = TY["font_family_mono"]

# (startup_time_s, p90_ttft_ms).
# Anchor points from prose: 0s≈1900ms (well under SLA), 180s≈3400ms
# (still passing), 300s=242,000ms (perpetually backlogged). The rest are
# approximated from planner_exp_3.png.
ROWS = [
    (0,     1900),
    (30,    2000),
    (60,    2100),
    (90,    2100),
    (120,   2500),
    (150,   3000),
    (180,   3400),
    (210,  25000),
    (240,  80000),
    (270, 120000),
    (300, 242000),
]
SLA_MS = 1500
CLIFF_S = 200


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def main() -> None:
    xs = [r[0] for r in ROWS]
    ys = [r[1] for r in ROWS]

    fig = go.Figure()

    # Light coral wash to the right of the cliff to reinforce the "SLA
    # broken" region without overpowering the data line.
    fig.add_vrect(
        x0=CLIFF_S, x1=320,
        fillcolor=rgba(CORAL, 0.10),
        line_width=0, layer="below",
    )

    # Vertical dashed cliff marker.
    fig.add_shape(
        type="line",
        x0=CLIFF_S, x1=CLIFF_S,
        y0=0, y1=1, yref="paper",
        line=dict(color=CORAL, width=1.4, dash="dash"),
    )

    # Cliff callout: Tufte block parked just inside the right half of the
    # plot near the top, with text aligned left.
    fig.add_annotation(
        xref="x domain", yref="y domain",
        x=0.72, y=0.93,
        xanchor="left", yanchor="top",
        align="left",
        text="<b>SLA Cliff ~200 s</b>",
        showarrow=False,
        bgcolor="rgba(20,20,20,0.65)",
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        borderpad=10,
        font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                  size=12, color=TEXT_PRIMARY, weight=300),
    )

    # SLA reference line.
    fig.add_hline(
        y=SLA_MS, line_dash="dash",
        line_color=TEXT_MUTED, line_width=1.0,
    )
    fig.add_annotation(
        x=10, y=SLA_MS,
        xref="x", yref="y",
        xanchor="left", yanchor="bottom",
        yshift=2,
        text="TTFT SLA 1500 ms",
        showarrow=False,
        font=dict(family=SANS, size=11, color=TEXT_MUTED),
    )

    # Main curve. CPU blue keeps planner figures visually consistent.
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line=dict(color=CPU_BLUE, width=2.2),
        marker=dict(color=CPU_BLUE, size=10,
                    line=dict(color=C["background"]["primary"], width=1)),
        hovertemplate="<b>%{x} s startup</b><br>p90 TTFT %{y:,.0f} ms<extra></extra>",
        showlegend=False,
        name="p90 TTFT",
    ))

    # Worst-case landmark callout: the 242 s p90 TTFT at startup=300 s.
    # Tufte block parked to the right of and below the cliff, where the
    # eye lands after tracing the steep ascent. Domain refs keep it in
    # frame regardless of axis range changes.
    fig.add_annotation(
        x=0.98, y=0.78,
        xref="x domain", yref="y domain",
        xanchor="right", yanchor="top",
        align="left",
        text="<b>242 s p90 TTFT</b><br>at 300 s startup",
        showarrow=False,
        bgcolor="rgba(20,20,20,0.65)",
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        borderpad=10,
        font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                  size=12, color=TEXT_PRIMARY, weight=300),
    )

    fig.update_layout(
        template=dynamo_template,
        title=dict(
            text="Cold-start: Planner holds SLA to ~180 s, then falls off a cliff",
            x=0.02, xanchor="left",
            y=0.95, yanchor="top",
            font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                      size=42, color=TEXT_PRIMARY, weight=300),
        ),
        showlegend=False,
        margin=dict(l=80, r=40, t=180, b=80),
        width=1240, height=560,
        shapes=[
            dict(  # cliff line (re-added because update_layout shapes=[] resets template)
                type="line",
                x0=CLIFF_S, x1=CLIFF_S,
                y0=0, y1=1, yref="paper",
                line=dict(color=CORAL, width=1.4, dash="dash"),
            ),
        ],
    )

    # Subtitle parked 5px below the title's bottom edge. Derived from
    # the standard formula in plotting.md:
    #   title_top    = (1 - 0.95) * 560 = 28
    #   title_bottom = 28 + 42 * 0.80   = 61.6
    #   subtitle_top = 61.6 + 5         = 66.6
    #   plot_h       = 560 - 180 - 80   = 300
    #   paper_y      = 1 + (180 - 66.6) / 300 = 1.378
    fig.add_annotation(
        x=-0.049, y=1.378,
        xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        text="Qwen3-32B / TP=2 / H200 — motivates predictive scaling and pre-warmed reserves.",
        showarrow=False,
        font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                  size=22, color=TEXT_MUTED, weight=300),
    )

    fig.update_xaxes(
        title=dict(text="Engine Startup Time (s)",
                   font=dict(family=SANS, size=11, color=TEXT_MUTED), standoff=8),
        showline=True, linecolor=BORDER_SUBTLE, linewidth=0.5, mirror=True,
        ticks="", showgrid=True, gridcolor=BORDER_SUBTLE, gridwidth=0.5,
        range=[-10, 320],
    )
    fig.update_yaxes(
        type="log",
        title=dict(text="p90 TTFT (ms, log)",
                   font=dict(family=SANS, size=11, color=TEXT_MUTED)),
        showline=True, linecolor=BORDER_SUBTLE, linewidth=0.5, mirror=True,
        ticks="", showgrid=True, gridcolor=BORDER_SUBTLE, gridwidth=0.5,
        tickvals=[1e3, 1e4, 1e5, 1e6],
        ticktext=["10³", "10⁴", "10⁵", "10⁶"],
        range=[3.0, 5.6],
    )

    out_svg = HERE.parent / "images" / "fig-6-planner-cold-start.svg"
    out_png = HERE.parent / "images" / "fig-6-planner-cold-start.png"
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_svg))
    fig.write_image(str(out_png), scale=2)
    print(f"wrote {out_svg}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
