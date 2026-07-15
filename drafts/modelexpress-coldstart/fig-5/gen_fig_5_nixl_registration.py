#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""fig-5 — NIXL registration-time scoreboard (Dynamo Dark, compact title).

Reference reproduced: a three-row horizontal-bar comparison of NIXL memory
registration time across three ModelExpress registration strategies. Restyled
into the Dynamo Dark compact-scoreboard treatment:

  - Compact / chart title (Arial, 18 px, weight 700, ALL-CAPS) carrying the
    takeaway, not the reference's category-name title.
  - Single green accent on the winning strategy (VMM arena); the slow baseline
    takes coral (the "loser" role), the middle strategy falls back to token grey.
    Two semantic accents + grey, per the palette discipline.
  - Direct-labelled rows (config name in the row's role color + env-var flag in
    mono) replace the reference's redundant legend.
  - In-bar mono value labels; a dedicated right-edge speedup column.
  - Subtle full-scale track behind each bar so the winner reads as a small
    fraction of the baseline.

Data is measured/stated (from the reference figure), not invented:

  strategy               env flag           time (s)   speedup vs baseline
  Per-tensor (default)   --                 8.16       1x
  Pool registration      MX_POOL_REG=1      1.14       7.1x
  VMM arena              MX_VMM_ARENA=1     0.79       10.3x

Usage:
    python3 gen_fig_5_nixl_registration.py   # -> images/fig-5-nixl-registration.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

# (label, env_flag, seconds) — reference order: slow baseline on top, winner at
# the bottom (descending by registration time, a domain-meaningful sort).
DATA = [
    ("Per-tensor registration", "per-tensor default", 8.16),
    ("Pool registration", "MX_POOL_REG=1", 1.14),
    ("VMM arena", "MX_VMM_ARENA=1", 0.79),
]

WIDTH, HEIGHT = 1280, 520
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 340, 160, 95, 78
X_MAX = 8.5  # a little headroom past the 8.16 s baseline
BAR_WIDTH = 0.52  # in y data units


def main() -> None:
    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]

    green = colors["accent"]["dynamo_green"]  # winner
    coral = colors["accent"]["coral"]  # slow baseline / loser
    grey = colors["text"]["medium"]  # middle strategy (neutral)
    white = colors["text"]["primary"]
    black = colors["brand"]["rich_black"]
    muted = colors["text"]["muted"]
    surface = colors["background"]["surface"]  # bar track
    font_sans = tokens["typography"]["font_family"]
    font_mono = tokens["typography"]["font_family_mono"]

    # Row role colors: baseline coral, pool grey, winner green.
    row_colors = [coral, grey, green]
    # In-bar label sits black on the green fill, white on the darker fills.
    text_colors = [white, white, black]
    # Speedup vs the per-tensor baseline (computed, not hard-coded).
    baseline = DATA[0][2]
    speedups = [baseline / d[2] for d in DATA]

    # y positions: top row (baseline) highest, winner at the bottom.
    n = len(DATA)
    ypos = list(range(n - 1, -1, -1))  # [2, 1, 0]

    xs = [d[2] for d in DATA]

    fig = go.Figure()

    # Full-scale track behind each bar (surface fill), so the tiny winner bar
    # reads as a fraction of the full baseline extent.
    for y in ypos:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=0,
            x1=X_MAX,
            y0=y - BAR_WIDTH / 2,
            y1=y + BAR_WIDTH / 2,
            fillcolor=surface,
            line=dict(width=0),
            layer="below",
        )

    fig.add_trace(
        go.Bar(
            x=xs,
            y=ypos,
            orientation="h",
            width=BAR_WIDTH,
            marker=dict(color=row_colors, line=dict(width=0)),
            text=[f"{d[2]:.2f}s" for d in DATA],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(family=font_mono, size=15, color=text_colors),
            cliponaxis=False,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Left column: config name (role color) + env-var flag (mono grey).
    for (label, flag, _), y, rc in zip(DATA, ypos, row_colors):
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0,
            y=y,
            xanchor="right",
            yanchor="bottom",
            xshift=-18,
            yshift=3,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(family=font_sans, size=16, color=rc),
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0,
            y=y,
            xanchor="right",
            yanchor="top",
            xshift=-18,
            yshift=-3,
            text=flag,
            showarrow=False,
            font=dict(family=font_mono, size=12, color=grey),
        )

    # Right column: speedup callouts, shared anchor x, biggest+green on winner.
    speedup_style = [
        (14, muted, 400),  # baseline "1x" — quiet, just anchors the column
        (19, grey, 700),  # pool
        (22, green, 700),  # winner — largest, green
    ]
    for sp, y, (size, color, weight) in zip(speedups, ypos, speedup_style):
        # Truncate to 1 decimal (matches the reference labels: 7.16 -> 7.1x,
        # 10.33 -> 10.3x); the baseline row shows a plain "1x".
        sp_trunc = math.floor(sp * 10) / 10
        txt = f"{sp_trunc:.1f}\u00d7" if sp > 1.05 else "1\u00d7"
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=1,
            y=y,
            xanchor="left",
            yanchor="middle",
            xshift=20,
            text=f"<b>{txt}</b>" if weight == 700 else txt,
            showarrow=False,
            font=dict(family=font_mono, size=size, color=color, weight=weight),
        )

    fig.update_layout(
        template=template,
        title=dict(
            # Compact / chart title: Arial, weight 700, ALL-CAPS, top-left.
            text="NIXL REGISTRATION TIME: VMM ARENA WINS BY 10.3\u00d7"
            "  (LOWER IS BETTER)",
            font=dict(family=font_sans, size=18, color=white, weight=700),
            x=0.02,
            xanchor="left",
            y=0.95,
            yanchor="top",
        ),
        xaxis=dict(
            title="Registration Time (s)",
            range=[0, X_MAX],
            tickmode="array",
            tickvals=[0, 2, 4, 6, 8],
            ticktext=["0", "2", "4", "6", "8"],
        ),
        yaxis=dict(
            range=[-0.6, n - 0.4],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        width=WIDTH,
        height=HEIGHT,
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
        bargap=0.0,
    )

    out_dir = Path(__file__).parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig-5-nixl-registration.png"
    svg = out_dir / "fig-5-nixl-registration.svg"
    fig.write_image(str(svg))
    fig.write_image(str(png), scale=3)
    print(f"Wrote {png}")
    print(f"Wrote {svg}")


if __name__ == "__main__":
    main()
