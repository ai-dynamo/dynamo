#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate Planner Exp 2 (load_adjustment_interval sweep) figure.

Replaces the engineer's planner_exp_2.png in PR-9139 with a Dynamo Dark /
Tufte rendition. Both axes are log; the headline story is "5-10 s is the
sweet spot — below it the planner thrashes, above it it loses the signal".

Each data point keeps its scaling-event count annotation (1529 events at
1 s decays to 3 events at 300 s) — that's the secondary axis of the
story and gets shown on the figure rather than buried in the prose.

Data are visually transcribed from PR-9139's upstream planner_exp_2.png
since `planner_reports/blog_exp/planner_exp_2/results.json` is not in the
repo. Replace `ROWS` with the canonical JSON if/when it lands.
"""

from __future__ import annotations

import math
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

# (load_adjustment_interval_s, scaling_events, p90_ttft_ms)
# Visually transcribed from PR-9139 planner_exp_2.png.
ROWS = [
    (1,   1529,   4000),
    (2,    991,   3800),
    (5,    469,   4000),
    (10,   233,   4300),
    (20,   117,   9000),
    (30,    76,  15000),
    (60,    26,  46000),
    (120,    9,  88000),
    (300,    3, 250000),
]
SLA_MS = 1500
SWEET_LO_S = 5
SWEET_HI_S = 10


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def main() -> None:
    xs = [r[0] for r in ROWS]
    events = [r[1] for r in ROWS]
    ys = [r[2] for r in ROWS]

    fig = go.Figure()

    # Sweet-spot band 5-10 s (where the planner sees signal without thrashing).
    fig.add_vrect(
        x0=SWEET_LO_S, x1=SWEET_HI_S,
        fillcolor=rgba(NV_GREEN, 0.10),
        line_width=0, layer="below",
    )
    # Label the band itself as a Tufte block, anchored just above the
    # top of the band. x is the geometric midpoint of [5, 10] on the
    # log axis; using x domain coords avoids the kaleido
    # add_annotation-on-log-axis bug.
    SWEET_MID_X_DOMAIN = (
        (math.log10((SWEET_LO_S * SWEET_HI_S) ** 0.5) - math.log10(0.85))
        / (math.log10(360) - math.log10(0.85))
    )
    fig.add_annotation(
        x=SWEET_MID_X_DOMAIN, y=0.45,
        xref="x domain", yref="y domain",
        xanchor="center", yanchor="bottom",
        align="center",
        text=f"<b>Optimal Operating Band</b><br>({SWEET_LO_S}s – {SWEET_HI_S}s)",
        showarrow=False,
        bgcolor="rgba(20,20,20,0.65)",
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        borderpad=10,
        font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                  size=12, color=TEXT_PRIMARY, weight=300),
    )

    # SLA reference line at 1500 ms.
    fig.add_hline(
        y=SLA_MS,
        line=dict(color=TEXT_MUTED, width=1.0, dash="dash"),
        layer="below",
    )

    # Main p90 TTFT curve.
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines+markers",
        line=dict(color=CPU_BLUE, width=2.2),
        marker=dict(color=CPU_BLUE, size=10,
                    line=dict(color=C["background"]["primary"], width=1)),
        hovertemplate=("<b>interval %{x:.0f} s</b><br>"
                       "p90 TTFT %{y:,.0f} ms<extra></extra>"),
        showlegend=False,
        name="p90 TTFT",
    ))

    # Per-point scaling-event count labels. Rendered as a text-mode
    # scatter trace (not fig.add_annotation) because Plotly + kaleido
    # silently drops add_annotation text in data coords on log axes
    # during static export. y values are scaled up by ~1.35x so the
    # label sits with breathing room above the marker on the log axis
    # rather than kissing it.
    LABEL_Y_LIFT = 1.35
    fig.add_trace(go.Scatter(
        x=xs, y=[y * LABEL_Y_LIFT for y in ys],
        mode="text",
        text=[f"{n:,} events" for n in events],
        textposition="top center",
        textfont=dict(family=SANS, size=11, color=TEXT_SECONDARY, weight=300),
        cliponaxis=False,
        hoverinfo="skip",
        showlegend=False,
    ))

    # Tufte callout: explains WHY 5-10 s wins. Header intentionally
    # differs from the in-band label so the figure carries three
    # distinct messages (title = what, band label = where, callout = why).
    fig.add_annotation(
        x=0.98, y=0.04,
        xref="x domain", yref="y domain",
        xanchor="right", yanchor="bottom",
        align="left",
        text=("<b>Signal vs Thrashing</b><br>"
              "Faster than 5 s the Planner thrashes on 1.5k+ scaling events;<br>"
              "slower than 10 s it loses the signal and falls off the SLA cliff."),
        showarrow=False,
        bgcolor="rgba(20,20,20,0.65)",
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        borderpad=10,
        font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                  size=12, color=TEXT_PRIMARY, weight=300),
    )

    # SLA pill, top-left inside the panel.
    fig.add_annotation(
        x=0.018, y=0.10,
        xref="x domain", yref="y domain",
        xanchor="left", yanchor="bottom",
        text=f"TTFT SLA {SLA_MS:,} ms",
        showarrow=False,
        font=dict(family=SANS, size=11, color=TEXT_MUTED, weight=300),
    )

    fig.update_layout(
        template=dynamo_template,
        title=dict(
            text="Load-Adjustment Interval Sensitivity Sweep",
            x=0.02, xanchor="left",
            y=0.95, yanchor="top",
            font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                      size=42, color=TEXT_PRIMARY, weight=300),
        ),
        showlegend=False,
        margin=dict(l=80, r=40, t=180, b=80),
        width=1240, height=560,
    )

    # Subtitle. Position derived from the standard formula:
    #   title_top    = (1 - 0.95) * 560 = 28
    #   title_bottom = 28 + 42 * 1.00   = 70.0   # 1.00 = cap + descender
    #   subtitle_top = 70.0 + 2         = 72.0   # +2 px = snug
    #   paper_y      = 1 + (180 - 72) / 300 = 1.360
    fig.add_annotation(
        x=-0.049, y=1.360,
        xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        text="Qwen3-32B / TP=2 / H200 / vLLM — p90 TTFT is lowest at 5–10 s; shorter intervals add noise, longer miss load shifts.",
        showarrow=False,
        font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                  size=22, color=TEXT_MUTED, weight=300),
    )

    # X-axis: log, ticks at 1, 2, 5, 10, 20, 30, 60, 120, 300. X-axis
    # markers stay numeric because the per-point label is intuitive
    # (interval in seconds).
    fig.update_xaxes(
        type="log",
        title=dict(text="load_adjustment_interval (s, log)",
                   font=dict(family=SANS, size=11, color=TEXT_MUTED), standoff=8),
        showline=True, linecolor=BORDER_SUBTLE, linewidth=0.5, mirror=True,
        ticks="", showgrid=True, gridcolor=BORDER_SUBTLE, gridwidth=0.5,
        tickvals=[1, 2, 5, 10, 20, 30, 60, 120, 300],
        ticktext=["1", "2", "5", "10", "20", "30", "60", "120", "300"],
        range=[math.log10(0.85), math.log10(360)],
    )
    # Y-axis uses 10^k scientific-notation labels to match Hongkuan's
    # matplotlib renderings of Planner Exp 1 / Exp 3 in PR-9139. Major
    # ticks only — Plotly's default minor-tick labels (2, 5 between
    # decades) read as visual noise next to the other two figures.
    fig.update_yaxes(
        type="log",
        title=dict(text="p90 TTFT (ms, log)",
                   font=dict(family=SANS, size=11, color=TEXT_MUTED)),
        showline=True, linecolor=BORDER_SUBTLE, linewidth=0.5, mirror=True,
        ticks="", showgrid=True, gridcolor=BORDER_SUBTLE, gridwidth=0.5,
        tickvals=[1e3, 1e4, 1e5, 1e6],
        ticktext=["10³", "10⁴", "10⁵", "10⁶"],
        range=[math.log10(1000), math.log10(500000)],
    )

    out_svg = HERE.parent / "images" / "fig-5-planner-load-interval.svg"
    out_png = HERE.parent / "images" / "fig-5-planner-load-interval.png"
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_svg))
    fig.write_image(str(out_png), scale=2)
    print(f"wrote {out_svg}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
