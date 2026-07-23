#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Example: compact horizontal-bar scoreboard (Dynamo Dark, compact title).

A dense backend-throughput comparison sorted by value: the winner takes the
single green accent, the rest fall back to the token grey ramp. In-bar mono
labels carry the numbers.

Demonstrates: the **compact / chart title** treatment (Arial, 18 px, weight
700, ALL-CAPS, 0.08em) — distinct from the display/hero Helvetica title —
plus selective accent, value sorting, and in-bar labels.

Values are representative, hard-coded (no external data).

Usage:
    python3 gen_fig_throughput_bars.py      # -> images/fig-throughput-bars.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

# (backend, throughput) — sorted descending so the winner is on top.
DATA = [
    ("Concurrent Positional Indexer", 170),
    ("Concurrent Radix Tree", 96),
    ("Radix Tree", 41),
    ("Inverted Index", 12),
    ("Naive Nested Map", 4),
]


def main() -> None:
    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]
    green = colors["accent"]["dynamo_green"]
    grey_ramp = [
        "#8c8c8c",
        "#555555",
        "#3a3a3a",
    ]  # text.medium, chart-fill grey, border.subtle
    text_primary = colors["text"]["primary"]
    font_sans = tokens["typography"]["font_family"]
    font_mono = tokens["typography"]["font_family_mono"]

    labels = [d[0] for d in DATA][::-1]
    values = [d[1] for d in DATA][::-1]
    # Winner (last, since reversed for horizontal bars) gets green; rest grey.
    bar_colors = []
    for i, _ in enumerate(values):
        if i == len(values) - 1:
            bar_colors.append(green)
        else:
            bar_colors.append(grey_ramp[min(len(values) - 2 - i, len(grey_ramp) - 1)])

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=bar_colors),
            text=[f"{v}M ops/s" for v in values],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(family=font_mono, size=13, color=text_primary),
            showlegend=False,
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            # Compact / chart title: Arial, weight 700, ALL-CAPS.
            text="INDEXER THROUGHPUT BY BACKEND  (HIGHER IS BETTER)",
            font=dict(family=font_sans, size=18, color=text_primary, weight=700),
            x=0.02,
            xanchor="left",
            y=0.96,
            yanchor="top",
        ),
        xaxis=dict(
            title="Achieved Throughput (M block ops/s)",
            tickmode="array",
            tickvals=[0, 50, 100, 150, 200],
        ),
        yaxis=dict(title=""),
        width=1100,
        height=520,
        margin=dict(l=250, r=40, t=70, b=60),
        bargap=0.45,
    )

    out = Path(__file__).parent / "images" / "fig-throughput-bars.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), scale=3)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
