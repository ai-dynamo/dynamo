#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate fig-4: planner setup tradeoffs (planner_exp_1 redraw, Dynamo Dark).

Two panels: p90 TTFT (log y) and p90 ITL (linear y) vs cumulative GPU-hours.
Static-deployment Pareto curves (agg in CPU blue, disagg in amber) overlay
six planner runs (3 targets x 2 modes) as open markers.

The headline: the agg planner-SLA point (NV green diamond) sits BELOW the
static Pareto on TTFT at roughly the same GPU-hours as a 4-GPU static run.

Data is approximated from the engineer's rendered PNG in PR 9139 so the
narrative beats match. Values are illustrative, not the source of truth.
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
AMBER = C["accent"]["amber"]
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
DISAGG_STATIC = [
    (3.9,  4.5e6, 25, 4),
    (5.9,  4.0e6, 25, 6),
    (8.6,  1.0e6, 25, 8),
    (9.6,  1.0e6, 25, 10),
    (11.8, 3.0e4, 25, 12),
    (13.8, 3.0e4, 25, 14),
]

# Planner runs: (mode, target, gpu_hours, ttft, itl).
PLANNER = [
    ("agg",    "throughput", 5.0, 4.5e4, 200),
    ("agg",    "latency",    5.0, 4.0e4, 195),
    ("agg",    "sla",        4.2, 2.3e3,  25),   # <- the headline
    ("disagg", "throughput", 6.5, 5.5e4, 200),
    ("disagg", "latency",    6.5, 3.8e4, 195),
    ("disagg", "sla",        8.5, 2.2e4, 175),
]

# Marker symbol per planner target.
TARGET_SYMBOL = {
    "throughput": "square-open",
    "latency":    "triangle-up-open",
    "sla":        "diamond-open",
}
MODE_COLOR = {"agg": CPU_BLUE, "disagg": AMBER}


def _ttft_panel(fig, row, col):
    # Static Pareto curves.
    for label, color, rows in (
        ("agg static",    CPU_BLUE, AGG_STATIC),
        ("disagg static", AMBER,    DISAGG_STATIC),
    ):
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=label, legendgroup=label,
                line=dict(color=color, width=1.6),
                marker=dict(color=color, size=8,
                            line=dict(color=C["background"]["primary"], width=1)),
                hovertemplate=f"<b>{label}</b><br>%{{x:.1f}} GPU-h<br>%{{y:,.0f}} ms<extra></extra>",
            ),
            row=row, col=col,
        )

    # GPU-count labels on the static curves so the reader can see that
    # each static-line marker is a separate deployment at a different
    # GPU budget (Hongkuan's PR-9139 feedback). Sparse-labeled (every
    # other point) to avoid label crowding at high TTFT where the
    # curves start; positioned to the side of the marker so the curve
    # itself stays readable. Agg labels go top-right, disagg top-left.
    AGG_LABELED = {4, 8, 12, 16}
    DISAGG_LABELED = {4, 8, 14}
    for label, color, rows, labeled, pos in (
        ("agg static",    CPU_BLUE, AGG_STATIC,    AGG_LABELED,    "top right"),
        ("disagg static", AMBER,    DISAGG_STATIC, DISAGG_LABELED, "bottom right"),
    ):
        gpu_xs, gpu_ys, gpu_texts = [], [], []
        for r in rows:
            if r[3] not in labeled:
                continue
            gpu_xs.append(r[0])
            gpu_ys.append(r[1])
            gpu_texts.append(f"{r[3]} GPU")
        fig.add_trace(
            go.Scatter(
                x=gpu_xs, y=gpu_ys, mode="text",
                text=gpu_texts, textposition=pos,
                textfont=dict(family=MONO, size=10, color=color),
                showlegend=False, hoverinfo="skip",
            ),
            row=row, col=col,
        )

    # Planner markers. Agg SLA is NV green (the punch line); others stay in
    # their mode color so the eye picks out the green diamond on TTFT.
    seen = set()
    for mode, target, x, y, _itl in PLANNER:
        color = NV_GREEN if (mode == "agg" and target == "sla") else MODE_COLOR[mode]
        name = f"{mode} Planner-{target}"
        showlegend = name not in seen
        seen.add(name)
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y], mode="markers",
                name=name, legendgroup=name, showlegend=showlegend,
                marker=dict(
                    symbol=TARGET_SYMBOL[target], size=11,
                    color=color, line=dict(color=color, width=1.6),
                ),
                hovertemplate=f"<b>{name}</b><br>%{{x:.1f}} GPU-h<br>%{{y:,.0f}} ms<extra></extra>",
            ),
            row=row, col=col,
        )


def _itl_panel(fig, row, col):
    for label, color, rows in (
        ("agg static",    CPU_BLUE, AGG_STATIC),
        ("disagg static", AMBER,    DISAGG_STATIC),
    ):
        xs = [r[0] for r in rows]
        ys = [r[2] for r in rows]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                showlegend=False,
                legendgroup=label,
                line=dict(color=color, width=1.6),
                marker=dict(color=color, size=8,
                            line=dict(color=C["background"]["primary"], width=1)),
                hovertemplate=f"<b>{label}</b><br>%{{x:.1f}} GPU-h<br>%{{y:.0f}} ms<extra></extra>",
            ),
            row=row, col=col,
        )

    for mode, target, x, _ttft, y in PLANNER:
        color = NV_GREEN if (mode == "agg" and target == "sla") else MODE_COLOR[mode]
        name = f"{mode} Planner-{target}"
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y], mode="markers",
                showlegend=False, legendgroup=name,
                marker=dict(
                    symbol=TARGET_SYMBOL[target], size=11,
                    color=color, line=dict(color=color, width=1.6),
                ),
                hovertemplate=f"<b>{name}</b><br>%{{x:.1f}} GPU-h<br>%{{y:.0f}} ms<extra></extra>",
            ),
            row=row, col=col,
        )


def main() -> None:
    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.10,
        subplot_titles=("", ""),
    )

    _ttft_panel(fig, row=1, col=1)
    _itl_panel(fig, row=1, col=2)

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

    # Punch-line callout, anchored to the top-right of the TTFT panel as
    # a Tufte block (subtle dark wash, hairline border, white display-sans,
    # left-aligned). The leader Scatter still binds to the subplot axes
    # and runs from the box down-left to the NV-green agg SLA diamond at
    # (4.2, 2.3e3).
    fig.add_trace(
        go.Scatter(
            x=[8.0, 4.6], y=[1.0e6, 3.0e3],
            mode="lines",
            line=dict(color=NV_GREEN, width=1.4),
            showlegend=False, hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_annotation(
        xref="x domain", yref="y domain",
        x=0.98, y=0.95,
        xanchor="right", yanchor="top",
        align="left",
        text="<b>SLA Planner</b><br>"
             "same GPU-hours,<br>"
             "~100x lower TTFT",
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
            text="Planner vs static: GPU-hours don't buy tail latency",
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
            tracegroupgap=12,
        ),
        margin=dict(l=80, r=40, t=180, b=120),
        width=1240, height=560,
        shapes=[],
    )

    # Subtitle parked 5px below the title's bottom edge. Derived from
    # the standard formula in plotting.md:
    #   title_top    = (1 - 0.96) * 560 = 22.4
    #   title_bottom = 22.4 + 42 * 0.80 = 56.0
    #   subtitle_top = 56.0 + 5         = 61.0
    #   plot_h       = 560 - 180 - 120  = 260
    #   paper_y      = 1 + (180 - 61) / 260 = 1.458
    fig.add_annotation(
        x=-0.049, y=1.458,
        xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        text="Qwen3-32B / TP=2 / H200 — agg SLA Planner sits below the static Pareto on TTFT.",
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
