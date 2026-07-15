#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""fig-4: cold-start phase breakdown with P2P RDMA + kernel artifacts.

A horizontal stacked-bar (Gantt-style) comparison of three cold-start paths,
restyled into the Dynamo Dark aesthetic. Each row is a per-phase time budget:

    Model Loading | Kernel Warmup, Graph Capture, KV Warmup | Others

The three phase colors repeat down the rows, so a bottom legend maps color to
phase. Dynamo green is held in reserve for the single winning row (the
RDMA + kernel-artifact cache path) — its badge and its 4.6x speed-up callout —
so the accent points straight at the takeaway.

Data are the measured phase durations from the ModelExpress cold-start blog
(seconds); the in-bar labels reproduce the reference values verbatim.

Compact / chart title treatment (Arial, 18 px, weight 700, uppercase).

Usage:
    python3 gen_fig_4_coldstart.py      # -> images/fig-4-coldstart-phases.{png,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

# ── Phase series: (display name, token accent key, in-bar text color) ──────────
# Model Loading -> CPU blue (network / weight transfer, cool)
# Kernel/Graph/KV warmup -> Fluorite gold (dominant compute phase, warm)
# Others -> Garnet (remaining overhead, hot)
SERIES = [
    ("Model Loading", "cpu_blue", "#ffffff"),
    ("Kernel Warmup, Graph Capture, KV Warmup", "fluorite", "#000000"),
    ("Others", "garnet", "#ffffff"),
]

# ── Rows, top -> bottom, durations in SECONDS (source: blog phase budget) ──────
# labels reproduce the reference verbatim; "" drops the label on tiny slivers.
ROWS = [
    dict(
        badge="BASELINE",
        kind="neutral",
        desc="Cold start from VAST, no P2P source",
        secs=[70, 349, 62],  # 1m10s + 5m49s + 1m2s = 8m1s
        labels=["1m 10s", "5m 49s", "1m 2s"],
        total="8m 1s",
        speedup=None,
    ),
    dict(
        badge="RDMA",
        kind="neutral",
        desc="P2P RDMA weights only",
        secs=[11, 350, 59],  # 11s + 5m50s + 59s = 7m0s
        labels=["", "5m 50s", "59s"],
        total="7m",
        speedup="1.1×",
    ),
    dict(
        badge="RDMA + CACHE",
        kind="winner",
        desc="P2P RDMA weights + kernel artifacts",
        secs=[9, 36, 59],  # 9s + 36s + 59s = 1m44s
        labels=["", "36s", "59s"],
        total="1m 44s",
        speedup="4.6×",
    ),
]

# ── Geometry ──────────────────────────────────────────────────────────────────
BAR_THICK = 0.46  # bar height in y-data units
HEADER_GAP = 0.20  # gap between bar top and its header line
DESC_X = 0.135  # fixed left column for row descriptions (paper x)
X_MAX = 495  # seconds; a hair past the 481 s baseline total


def main() -> None:
    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]
    accent = colors["accent"]
    green = accent["dynamo_green"]
    white = colors["text"]["primary"]
    secondary = colors["text"]["secondary"]
    muted = colors["text"]["muted"]
    elevated = colors["background"]["elevated"]
    subtle = colors["border"]["subtle"]
    font_sans = tokens["typography"]["font_family"]
    font_mono = tokens["typography"]["font_family_mono"]

    y_centers = [2.0, 1.0, 0.0]  # Baseline top, RDMA middle, RDMA+CACHE bottom

    fig = go.Figure()

    # One stacked trace per phase; x/text/color arrays index the three rows.
    for s_idx, (name, accent_key, text_color) in enumerate(SERIES):
        fig.add_trace(
            go.Bar(
                name=name,
                orientation="h",
                y=y_centers,
                x=[row["secs"][s_idx] for row in ROWS],
                width=BAR_THICK,
                marker=dict(color=accent[accent_key], line=dict(width=0)),
                text=[row["labels"][s_idx] for row in ROWS],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(family=font_mono, size=13, color=text_color),
                cliponaxis=False,
                hoverinfo="skip",
            )
        )

    # ── Per-row header line: badge + description (left), total + speed-up (right)
    annotations: list[dict] = []
    for row, yc in zip(ROWS, y_centers):
        header_y = yc + BAR_THICK / 2 + HEADER_GAP
        winner = row["kind"] == "winner"
        badge_bg = green if winner else elevated
        badge_fg = "#000000" if winner else white

        annotations.append(
            dict(
                x=0.0,
                y=header_y,
                xref="paper",
                yref="y",
                xanchor="left",
                yanchor="middle",
                text=f"<b>{row['badge']}</b>",
                showarrow=False,
                font=dict(family=font_sans, size=12, color=badge_fg, weight=700),
                bgcolor=badge_bg,
                borderpad=5,
            )
        )
        annotations.append(
            dict(
                x=DESC_X,
                y=header_y,
                xref="paper",
                yref="y",
                xanchor="left",
                yanchor="middle",
                text=row["desc"],
                showarrow=False,
                font=dict(family=font_sans, size=15, color=secondary),
            )
        )

        if row["speedup"] is None:
            right_text = f"<b>{row['total']}</b>"
        else:
            su_color = green if winner else muted
            right_text = (
                f"<b>{row['total']}</b>  "
                f"<span style='color:{su_color}'>{row['speedup']}</span>"
            )
        annotations.append(
            dict(
                x=1.0,
                y=header_y,
                xref="paper",
                yref="y",
                xanchor="right",
                yanchor="middle",
                text=right_text,
                showarrow=False,
                font=dict(family=font_mono, size=15, color=white),
            )
        )

    fig.update_layout(
        template=template,
        barmode="stack",
        title=dict(
            text="COLD START TIME WITH P2P RDMA AND KERNEL ARTIFACTS",
            font=dict(family=font_sans, size=18, color=white, weight=700),
            x=0.023,
            xanchor="left",
            y=0.965,
            yanchor="top",
        ),
        xaxis=dict(
            range=[0, X_MAX],
            tickmode="array",
            tickvals=list(range(0, 481, 60)),
            ticktext=[f"{m}m" for m in range(9)],
            showgrid=True,
            gridcolor=subtle,
            gridwidth=0.5,
            zeroline=False,
            ticks="outside",
            ticklen=4,
            tickcolor=subtle,
        ),
        yaxis=dict(range=[-0.55, 2.82], visible=False),
        legend=dict(
            orientation="h",
            traceorder="normal",
            x=0.5,
            xanchor="center",
            y=-0.16,
            yanchor="top",
            font=dict(family=font_sans, size=14, color=secondary),
        ),
        width=1280,
        height=600,
        margin=dict(l=30, r=30, t=96, b=104),
        annotations=annotations,
    )

    # Hairline separating the title from the chart body (matches the reference).
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0.0,
        x1=1.0,
        y0=1.075,
        y1=1.075,
        line=dict(color=subtle, width=1),
        layer="below",
    )

    out_dir = Path(__file__).parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig-4-coldstart-phases.png"
    svg = out_dir / "fig-4-coldstart-phases.svg"
    fig.write_image(str(png), scale=3)
    fig.write_image(str(svg))
    print(f"Wrote {png}")
    print(f"Wrote {svg}")


if __name__ == "__main__":
    main()
