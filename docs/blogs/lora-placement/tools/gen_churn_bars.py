#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate churn comparison bar charts for the LoRA placement blog.

Produces three figures from hardcoded simulation data:
  - Fig 4: Total Churn (grouped horizontal bars, log scale)
  - Fig 5: Churn-Free Tick Ratio (grouped horizontal bars, linear 0-100%)
  - Fig 7: MCF Churn Reduction vs HRW (single-series horizontal bars)

Uses the Dynamo dark Plotly template (design_tokens.yaml + plotly_dynamo.py).

Usage:
    python3 gen_churn_bars.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

# ---------------------------------------------------------------------------
# Data (Section 11 of the LoRA placement spec)
# ---------------------------------------------------------------------------

SCENARIOS = ["Zipf +\nPoisson", "Daily\nTraffic", "Traffic\nSpikes", "MMPP\n3-state"]

TOTAL_CHURN = {
    "MCF": [98, 136, 103, 88],
    "HRW": [242, 281, 238, 251],
    "Random": [35658, 33824, 35752, 35477],
}

CHURN_FREE_PCT = {
    "MCF": [95, 86, 90, 88],
    "HRW": [74, 61, 70, 66],
    "Random": [0, 0, 0, 0],
}

MCF_VS_HRW_PCT = [-60, -52, -57, -65]  # improvement %

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COLOR_MCF = "#76b900"  # Dynamo green
COLOR_HRW = "#0071c5"  # CPU blue
COLOR_RANDOM = "#8c8c8c"  # Medium gray


def _write(fig: go.Figure, stem: str) -> None:
    """Write SVG and PNG to ../images/ relative to this script."""
    images_dir = Path(__file__).resolve().parent.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    png = images_dir / f"{stem}.png"
    svg = images_dir / f"{stem}.svg"
    fig.write_image(str(png), scale=3)
    fig.write_image(str(svg))
    print(f"Wrote {png.name}")
    print(f"Wrote {svg.name}")


# ===================================================================
# Figure 4 -- Total Churn (grouped horizontal bars, log scale)
# ===================================================================


def fig4_total_churn(template: go.layout.Template, tokens: dict) -> None:
    typo = tokens["typography"]

    fig = go.Figure()

    # Bars are grouped; Plotly draws bottom-to-top for horizontal,
    # so add in reverse order so MCF is visually on top.
    for name, color in [
        ("Random", COLOR_RANDOM),
        ("HRW", COLOR_HRW),
        ("MCF", COLOR_MCF),
    ]:
        values = TOTAL_CHURN[name]
        fig.add_trace(
            go.Bar(
                y=SCENARIOS,
                x=values,
                name=name,
                orientation="h",
                marker=dict(color=color, opacity=0.92),
                text=[f"{v:,}" for v in values],
                textposition="outside",
                textfont=dict(
                    family=typo["font_family_mono"],
                    size=10,
                    color=color,
                ),
                hovertemplate=(
                    f"<b>{name}</b><br>" "%{y}: %{x:,} ops" "<extra></extra>"
                ),
            )
        )

    # Annotations: "MCF: -XX% vs HRW" next to MCF bars
    annotations = []
    for i, pct in enumerate(MCF_VS_HRW_PCT):
        mcf_val = TOTAL_CHURN["MCF"][i]
        annotations.append(
            dict(
                x=mcf_val,
                y=SCENARIOS[i],
                xref="x",
                yref="y",
                text=f"  <b>MCF: {pct:+d}% vs HRW</b>",
                showarrow=False,
                xanchor="left",
                xshift=55,
                font=dict(
                    family=typo["font_family"],
                    size=9,
                    color=COLOR_MCF,
                ),
                bgcolor="#000000",
                bordercolor=COLOR_MCF,
                borderwidth=1,
                borderpad=3,
            )
        )

    fig.update_layout(
        template=template,
        title=dict(text="TOTAL CHURN (LOADS + UNLOADS) OVER 200 TICKS"),
        barmode="group",
        bargroupgap=0.15,
        bargap=0.30,
        xaxis=dict(
            title="Total Churn (log scale)",
            type="log",
            range=[1.5, 4.62],  # ~30 to 40,000
            tickformat=",",
            tickvals=[100, 500, 1000, 5000, 10000, 40000],
            ticktext=["100", "500", "1K", "5K", "10K", "40K"],
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            categoryorder="array",
            categoryarray=SCENARIOS,
        ),
        legend=dict(
            x=0.98,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
            orientation="h",
        ),
        width=950,
        height=550,
        margin=dict(l=100, r=65, t=70, b=50),
        annotations=annotations,
    )

    _write(fig, "fig-4-total-churn")


# ===================================================================
# Figure 5 -- Churn-Free Tick Ratio (grouped horizontal bars, linear)
# ===================================================================


def fig5_churn_free_ratio(template: go.layout.Template, tokens: dict) -> None:
    typo = tokens["typography"]

    fig = go.Figure()

    for name, color in [
        ("Random", COLOR_RANDOM),
        ("HRW", COLOR_HRW),
        ("MCF", COLOR_MCF),
    ]:
        values = CHURN_FREE_PCT[name]
        fig.add_trace(
            go.Bar(
                y=SCENARIOS,
                x=values,
                name=name,
                orientation="h",
                marker=dict(color=color, opacity=0.92),
                text=[f"{v}%" for v in values],
                textposition="inside",
                insidetextanchor="end",
                textfont=dict(
                    family=typo["font_family_mono"],
                    size=11,
                    color="#ffffff",
                ),
                hovertemplate=(f"<b>{name}</b><br>" "%{y}: %{x}%" "<extra></extra>"),
            )
        )

    # Vertical dashed reference line at 50%
    border_subtle = tokens["colors"]["border"]["subtle"]
    shapes = [
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=50,
            x1=50,
            y0=0,
            y1=1,
            line=dict(color="#fac200", width=1.5, dash="dash"),
            layer="below",
        ),
    ]

    fig.update_layout(
        template=template,
        title=dict(text="CHURN-FREE TICK RATIO  (HIGHER IS BETTER)"),
        barmode="group",
        bargroupgap=0.15,
        bargap=0.30,
        xaxis=dict(
            title="Churn-Free Ticks (%)",
            range=[0, 105],
            ticksuffix="%",
            dtick=25,
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            categoryorder="array",
            categoryarray=SCENARIOS,
        ),
        legend=dict(
            x=0.98,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
            orientation="h",
        ),
        width=775,
        height=500,
        margin=dict(l=90, r=55, t=70, b=50),
        shapes=shapes,
    )

    # Add "50%" label on the dashed line
    fig.add_annotation(
        x=50,
        y=1.0,
        xref="x",
        yref="paper",
        text="50%",
        showarrow=False,
        yanchor="bottom",
        yshift=4,
        font=dict(
            family=typo["font_family_mono"],
            size=9,
            color="#fac200",
        ),
    )

    _write(fig, "fig-5-churn-free-ratio")


# ===================================================================
# Figure 7 -- MCF vs HRW Improvement (simple horizontal bars)
# ===================================================================


def fig7_churn_efficiency(template: go.layout.Template, tokens: dict) -> None:
    typo = tokens["typography"]

    avg_pct = sum(MCF_VS_HRW_PCT) / len(MCF_VS_HRW_PCT)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=SCENARIOS,
            x=MCF_VS_HRW_PCT,
            orientation="h",
            marker=dict(color=COLOR_MCF, opacity=0.92),
            text=[f"{v}%" for v in MCF_VS_HRW_PCT],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(
                family=typo["font_family_mono"],
                size=12,
                color="#ffffff",
            ),
            showlegend=False,
            hovertemplate=("%{y}: %{x}% churn reduction" "<extra></extra>"),
        )
    )

    # Vertical zero line shape
    shapes = [
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=0,
            x1=0,
            y0=0,
            y1=1,
            line=dict(color="#cdcdcd", width=1, dash="solid"),
            layer="above",
        ),
    ]

    # Average annotation
    annotations = [
        dict(
            x=avg_pct,
            y=1.0,
            xref="x",
            yref="paper",
            text=f"<b>Avg: {avg_pct:.0f}%</b>",
            showarrow=True,
            ax=0,
            ay=-30,
            arrowcolor=COLOR_MCF,
            arrowwidth=1.5,
            font=dict(
                family=typo["font_family"],
                size=11,
                color=COLOR_MCF,
            ),
            bgcolor="#000000",
            bordercolor=COLOR_MCF,
            borderwidth=1,
            borderpad=4,
        ),
        # Vertical dashed line at average
        dict(
            x=avg_pct,
            y=0,
            xref="x",
            yref="paper",
            text="",
            showarrow=False,
        ),
    ]

    # Dashed line at average
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=avg_pct,
            x1=avg_pct,
            y0=0,
            y1=1,
            line=dict(color=COLOR_MCF, width=1.5, dash="dash"),
            layer="below",
        ),
    )

    fig.update_layout(
        template=template,
        title=dict(text="MCF CHURN REDUCTION VS HRW"),
        xaxis=dict(
            title="Churn Reduction (%)",
            range=[-75, 5],
            ticksuffix="%",
            dtick=10,
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            categoryorder="array",
            categoryarray=SCENARIOS,
        ),
        width=650,
        height=400,
        margin=dict(l=90, r=40, t=70, b=50),
        shapes=shapes,
        annotations=annotations,
    )

    _write(fig, "fig-7-churn-efficiency")


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)

    fig4_total_churn(template, tokens)
    fig5_churn_free_ratio(template, tokens)
    fig7_churn_efficiency(template, tokens)
    print("\nAll churn bar charts generated.")


if __name__ == "__main__":
    main()
