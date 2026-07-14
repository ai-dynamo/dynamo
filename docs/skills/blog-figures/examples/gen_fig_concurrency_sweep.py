#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Example: dual-panel concurrency sweep + Pareto curve (Dynamo Dark).

Two panels sharing a green-vs-grey series encoding:
  - left: Mean TTFT (ms, log) vs concurrency, with per-point delta callouts;
  - right: throughput vs interactivity Pareto curve with concurrency labels
    and a "Prefix Reuse" inset.

Demonstrates: small multiples, log axis with round ticks, direct-labeled
delta callouts, a Pareto trade-off curve, and the green-accent-vs-grey
baseline convention.

Title uses the Dynamo Dark display/hero treatment (Helvetica Neue Light,
title case). Series values are representative, hard-coded (no external data).

Usage:
    python3 gen_fig_concurrency_sweep.py    # -> images/fig-concurrency-sweep.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

HERO_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

CONC = [64, 128, 256, 512]
TTFT_RR = [500, 800, 1800, 5100]  # Round Robin, ms
TTFT_KV = [240, 300, 490, 1450]  # KV Router, ms
# Right-panel Pareto: (interactivity Tok/s/User, throughput Tok/s/GPU) per c.
KV_XY = [(20, 200), (38, 172), (60, 138), (80, 105)]
RR_XY = [(25, 170), (40, 150), (60, 125), (80, 95)]


def main() -> None:
    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]
    green = colors["accent"]["dynamo_green"]
    grey = colors["text"]["medium"]  # #8c8c8c baseline series
    text_primary = colors["text"]["primary"]
    text_muted = colors["text"]["muted"]
    subtle = colors["border"]["subtle"]
    font_sans = tokens["typography"]["font_family"]
    font_mono = tokens["typography"]["font_family_mono"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Mean TTFT (ms, log)", "Throughput vs Interactivity"),
        horizontal_spacing=0.11,
    )

    # -- Left panel: TTFT vs concurrency (log y). --
    fig.add_trace(
        go.Scatter(
            x=CONC,
            y=TTFT_RR,
            mode="lines+markers",
            name="Round Robin",
            line=dict(color=grey, width=2.5),
            marker=dict(size=9, color=grey),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=CONC,
            y=TTFT_KV,
            mode="lines+markers",
            name="KV Router",
            line=dict(color=green, width=2.5),
            marker=dict(size=9, color=green, symbol="diamond"),
        ),
        row=1,
        col=1,
    )
    for c, rr, kv in zip(CONC, TTFT_RR, TTFT_KV):
        pct = round((kv / rr - 1) * 100)
        fig.add_annotation(
            x=math.log10(c),
            y=math.log10(kv),
            xref="x",
            yref="y",
            text=f"<b>{pct}%</b>",
            showarrow=False,
            yshift=-16,
            font=dict(family=font_mono, size=13, color=green),
        )

    # -- Right panel: throughput vs interactivity Pareto. --
    for xy, color, name, sym in [
        (RR_XY, grey, "Round Robin", "circle"),
        (KV_XY, green, "KV Router", "diamond"),
    ]:
        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2.5),
                marker=dict(size=9, color=color, symbol=sym),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        for (x, y), c in zip(xy, CONC[::-1]):
            fig.add_annotation(
                x=x,
                y=y,
                xref="x2",
                yref="y2",
                text=f"c={c}",
                showarrow=False,
                yshift=13,
                font=dict(family=font_mono, size=11, color=text_muted),
            )

    # "Prefix Reuse" inset on the right panel.
    fig.add_annotation(
        x=0.62,
        y=0.30,
        xref="paper",
        yref="paper",
        align="left",
        text=("<b>Prefix Reuse</b><br>Round Robin 0.38 · KV Router 0.44"),
        showarrow=False,
        bordercolor=subtle,
        borderwidth=1,
        borderpad=8,
        bgcolor=colors["background"]["surface"],
        font=dict(family=font_sans, size=13, color=text_primary),
    )

    fig.update_xaxes(
        title_text="Concurrency",
        tickmode="array",
        tickvals=CONC,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Mean TTFT (ms, Log)",
        type="log",
        tickmode="array",
        tickvals=[200, 500, 1000, 2000, 5000],
        ticktext=["200", "500", "1k", "2k", "5k"],
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Interactivity (Tok/s/User)", row=1, col=2)
    fig.update_yaxes(title_text="Throughput (Tok/s/GPU)", row=1, col=2)

    fig.update_layout(
        template=template,
        title=dict(
            text="Round-Robin vs KV Router: Concurrency Sweep and Pareto Curve",
            font=dict(family=HERO_FONT, size=38, color=text_primary, weight=300),
            subtitle=dict(
                text=(
                    "B200 / MiniMax-M2.5 / TP=4 / Mooncake trace — "
                    "KV Router cuts TTFT and lifts throughput."
                ),
                font=dict(family=HERO_FONT, size=18, color=text_muted, weight=300),
            ),
            x=0.03,
            xanchor="left",
            y=0.95,
            yanchor="top",
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.16,
            yanchor="top",
            font=dict(family=font_sans, size=14, color=colors["text"]["secondary"]),
        ),
        width=1600,
        height=760,
        margin=dict(l=70, r=40, t=170, b=110),
    )

    out = Path(__file__).parent / "images" / "fig-concurrency-sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), scale=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
