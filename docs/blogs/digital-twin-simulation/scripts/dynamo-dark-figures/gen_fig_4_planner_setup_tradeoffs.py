#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate fig-4: planner setup tradeoffs (planner_exp_1 redraw, Dynamo Dark).

Simplified per blog-figure feedback (Alec, Hongkuan, May 18 2026):
- One line vs one point. Line = agg-static deployment Pareto (varying num
  replicas, from 4 to 16 GPU). Point = agg Planner-SLA, the headline.
- Disagg variants removed (agg dominates on this dataset).
- Other planner targets (throughput / latency) removed; only the SLA
  target carries the story.
- Two panels: p90 TTFT (log y) and p90 ITL (linear y).

Headline: a single Planner-SLA run lands at ~4 GPU-h (the leftmost
static config) but with TTFT and ITL comparable to a 10-GPU static run —
clear win on the cost-latency frontier.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
SANS = TY["font_family"]
MONO = TY["font_family_mono"]

# Static-deployment Pareto: (gpu_hours, p90_ttft_ms, p90_itl_ms, gpu_count).
# Approximated from the engineer's planner_exp_1.png in PR 9139.
AGG_STATIC = [
    (4.0,  5.0e6, 685, 4),
    (5.8,  2.0e6, 675, 6),
    (7.8,  1.0e6, 670, 8),
    (9.8,  9.0e3,  28, 10),
    (11.7, 6.5e3,  26, 12),
    (13.6, 5.0e3,  25, 14),
    (15.6, 4.0e3,  24, 16),
]

# Planner SLA-target run — the single point the figure is built around.
PLANNER_SLA = dict(gpu_hours=4.2, ttft=2.3e3, itl=25)


def _static_line(fig, row, col, y_index):
    """Draw the agg-static Pareto line on a subplot.

    y_index: 1 for TTFT (p90_ttft_ms), 2 for ITL (p90_itl_ms).
    """
    xs = [r[0] for r in AGG_STATIC]
    ys = [r[y_index] for r in AGG_STATIC]
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            name="Static deployment", legendgroup="static",
            showlegend=(row == 1 and col == 1),
            line=dict(color=CPU_BLUE, width=1.6),
            marker=dict(color=CPU_BLUE, size=8,
                        line=dict(color=C["background"]["primary"], width=1)),
            hovertemplate=("<b>Static deployment</b><br>"
                           "%{x:.1f} GPU-h<br>%{y:,.0f} ms<extra></extra>"),
        ),
        row=row, col=col,
    )

    # Direct GPU-count labels on each static point so the reader sees that
    # the line is a sweep across deployment sizes, not a continuous variable.
    LABELED = {4, 8, 12, 16}
    gpu_xs, gpu_ys, gpu_texts = [], [], []
    for r in AGG_STATIC:
        if r[3] not in LABELED:
            continue
        gpu_xs.append(r[0])
        gpu_ys.append(r[y_index])
        gpu_texts.append(f"{r[3]} GPU")
    fig.add_trace(
        go.Scatter(
            x=gpu_xs, y=gpu_ys, mode="text",
            text=gpu_texts, textposition="top right",
            textfont=dict(family=MONO, size=10, color=CPU_BLUE),
            showlegend=False, hoverinfo="skip",
        ),
        row=row, col=col,
    )


def _planner_point(fig, row, col, y_value):
    fig.add_trace(
        go.Scatter(
            x=[PLANNER_SLA["gpu_hours"]], y=[y_value],
            mode="markers",
            name="Planner (SLA target)", legendgroup="planner",
            showlegend=(row == 1 and col == 1),
            marker=dict(
                symbol="diamond", size=14,
                color=NV_GREEN,
                line=dict(color=C["background"]["primary"], width=1.5),
            ),
            hovertemplate=("<b>Planner (SLA target)</b><br>"
                           "%{x:.1f} GPU-h<br>%{y:,.0f} ms<extra></extra>"),
        ),
        row=row, col=col,
    )


def main() -> None:
    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.10,
        subplot_titles=("", ""),
    )

    _static_line(fig, row=1, col=1, y_index=1)
    _planner_point(fig, row=1, col=1, y_value=PLANNER_SLA["ttft"])
    _static_line(fig, row=1, col=2, y_index=2)
    _planner_point(fig, row=1, col=2, y_value=PLANNER_SLA["itl"])

    # Inline panel labels (above each panel, top-left, muted).
    for i, label in enumerate(("p90 TTFT", "p90 ITL")):
        idx = "" if i == 0 else str(i + 1)
        fig.add_annotation(
            x=0.0, y=1.04,
            xref=f"x{idx} domain", yref=f"y{idx} domain",
            xanchor="left", yanchor="bottom",
            text=label, showarrow=False,
            font=dict(family=SANS, size=11, color=TEXT_MUTED),
        )

    # Punch-line callout, anchored to the top-right of the TTFT panel.
    # The NV-green diamond at (4.2, 2.3e3) is its own visual anchor —
    # the callout sits above the static curve sweep, where there's air.
    fig.add_annotation(
        xref="x domain", yref="y domain",
        x=0.98, y=0.95,
        xanchor="right", yanchor="top",
        align="left",
        text="<b>Planner (SLA target)</b><br>"
             "~4 GPU-h budget,<br>"
             "~100x lower TTFT than<br>"
             "static at the same cost",
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
            text="Planner vs Static: Cost-Latency Frontier",
            x=0.02, xanchor="left",
            y=0.96, yanchor="top",
            font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                      size=42, color=TEXT_PRIMARY, weight=300),
        ),
        legend=dict(
            orientation="h",
            x=1.0, xanchor="right",
            y=-0.18, yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(family=SANS, size=11, color=TEXT_SECONDARY),
            itemsizing="constant",
            tracegroupgap=18,
        ),
        margin=dict(l=80, r=40, t=180, b=120),
        width=1240, height=560,
        shapes=[],
    )

    # Subtitle parked 2px below title bottom (snug, descender-safe).
    #   title_top    = (1 - 0.96) * 560 = 22.4
    #   title_bottom = 22.4 + 42 * 1.00 = 64.4
    #   subtitle_top = 64.4 + 2         = 66.4
    #   plot_h       = 560 - 180 - 120  = 260
    #   paper_y      = 1 + (180 - 66.4) / 260 = 1.437
    fig.add_annotation(
        x=-0.049, y=1.437,
        xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        text="Qwen3-32B / TP=2 / H200 — Planner matches a 10-GPU static deployment on TTFT and ITL at the cost of a 4-GPU one.",
        showarrow=False,
        font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                  size=22, color=TEXT_MUTED, weight=300),
    )

    fig.update_xaxes(
        title=dict(text="Cumulative GPU-Hours",
                   font=dict(family=SANS, size=11, color=TEXT_MUTED), standoff=8),
        showline=True, linecolor=BORDER_SUBTLE, linewidth=0.5, mirror=True,
        ticks="", showgrid=True, gridcolor=BORDER_SUBTLE, gridwidth=0.5,
        range=[2.5, 17],
    )
    fig.update_yaxes(
        showline=True, linecolor=BORDER_SUBTLE, linewidth=0.5, mirror=True,
        ticks="", showgrid=True, gridcolor=BORDER_SUBTLE, gridwidth=0.5,
    )
    fig.update_yaxes(type="log", title=dict(text="p90 TTFT (ms, log)",
                                            font=dict(family=SANS, size=11, color=TEXT_MUTED)),
                     tickvals=[1e3, 1e4, 1e5, 1e6, 1e7],
                     ticktext=["10³", "10⁴", "10⁵", "10⁶", "10⁷"],
                     range=[3.0, 7.0],
                     row=1, col=1)
    fig.update_yaxes(title=dict(text="p90 ITL (ms)",
                                font=dict(family=SANS, size=11, color=TEXT_MUTED)),
                     tickvals=[0, 200, 400, 600, 700],
                     ticktext=["0", "200", "400", "600", "700"],
                     range=[-50, 750],
                     row=1, col=2)

    out_svg = HERE.parent / "images" / "fig-4-planner-setup-tradeoffs.svg"
    out_png = HERE.parent / "images" / "fig-4-planner-setup-tradeoffs.png"
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_svg))
    fig.write_image(str(out_png), scale=2)
    print(f"wrote {out_svg}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
