#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate fig-6 closer -- the tuning loop, end to end.

Five horizontal stages with cardinality on each (the funnel), plus a single
amethyst feedback arc from the deployed stage back into DynoSim grid search
(production telemetry refines the simulator's timing model).

Output:
    ../images/fig-6-tuning-loop.svg
    ../images/fig-6-tuning-loop.png  (scale=2)
"""

from __future__ import annotations

from pathlib import Path

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
BORDER_SUBTLE = C["border"]["subtle"]
NV_GREEN = C["accent"]["dynamo_green"]
FLUORITE = C["accent"]["fluorite"]
CPU_BLUE = C["accent"]["cpu_blue"]   # replaces emerald so we don't have two greens
# Local override: design-token amethyst (#5d1682) is too dim against black;
# use a brighter purple so the feedback loop has equal visual weight to
# the forward pipeline.
AMETHYST = "#a960e8"
SANS = TY["font_family"]
MONO = TY["font_family_mono"]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Stage layout: (label, cardinality, color, x_center, phase_tag).
# phase_tag is an optional temporal marker rendered as a small uppercase
# kicker above the box; None = no tag.
STAGES = [
    ("Configuration Space",   "~10⁹ feasible configs",      TEXT_MUTED,  100, None),
    ("DynoSim Wide Sweep",    "~10⁵ / hour · $0",           NV_GREEN,    300, None),
    ("Pareto Shortlist",      "~100 candidates",            FLUORITE,    500, None),
    ("Cluster A/B Verify",    "~10 / day · GPU $$",         CPU_BLUE,    700, "Initial Verification"),
    ("Deployed + Telemetry",  "1 config in production",     AMETHYST,    900, "Feedback Loop"),
]

BOX_W = 160
BOX_H = 100
Y_C = 60
ARROW_GAP = 12


def main() -> None:
    fig = go.Figure()

    # Stage boxes. Fill alpha bumped from 0.10 to 0.20 and borders thickened
    # so each box reads as a distinct, color-coded stop on the pipeline.
    for label, sub, color, xc, phase in STAGES:
        # Phase tag (small uppercase kicker) above the box, in the stage's
        # accent color. Only rendered if the stage has a phase_tag, so the
        # three temporal phases (Before, Initial Verification, Feedback
        # Loop) read clearly without cluttering the intermediate steps.
        if phase:
            # Kicker centered above the box with extra breathing room so the
            # uppercase label reads as a header rather than crowding the box.
            fig.add_annotation(
                x=xc, y=Y_C + BOX_H / 2 + 55,
                xref="x", yref="y",
                xanchor="center", yanchor="bottom",
                text=f"<b>{phase.upper()}</b>",
                showarrow=False,
                font=dict(
                    family=SANS, size=13, color=color,
                ),
            )
        fig.add_shape(
            type="rect",
            x0=xc - BOX_W / 2, x1=xc + BOX_W / 2,
            y0=Y_C - BOX_H / 2, y1=Y_C + BOX_H / 2,
            line=dict(color=color, width=2.0),
            fillcolor=rgba(color, 0.18) if color != TEXT_MUTED else rgba("#ffffff", 0.04),
            layer="above",
        )
        fig.add_annotation(
            x=xc, y=Y_C + 12,
            xref="x", yref="y",
            xanchor="center", yanchor="middle",
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(family=SANS, size=15, color=TEXT_PRIMARY),
        )
        fig.add_annotation(
            x=xc, y=Y_C - 14,
            xref="x", yref="y",
            xanchor="center", yanchor="middle",
            text=sub,
            showarrow=False,
            font=dict(family=MONO, size=12, color=color),
        )

    # Forward arrows (4 total). Slightly thicker line + larger triangle so
    # the pipeline flow reads as the primary path through the figure.
    arrow_x_targets, arrow_y_targets = [], []
    for i in range(len(STAGES) - 1):
        x_from = STAGES[i][3] + BOX_W / 2 + ARROW_GAP
        x_to = STAGES[i + 1][3] - BOX_W / 2 - ARROW_GAP
        fig.add_shape(
            type="line",
            x0=x_from, x1=x_to,
            y0=Y_C, y1=Y_C,
            line=dict(color=TEXT_SECONDARY, width=1.8),
            layer="above",
        )
        arrow_x_targets.append(x_to)
        arrow_y_targets.append(Y_C)
    fig.add_trace(go.Scatter(
        x=arrow_x_targets, y=arrow_y_targets,
        mode="markers",
        marker=dict(symbol="triangle-right", size=14,
                    color=TEXT_SECONDARY, line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))

    # Feedback loop: stage 5 -> stage 2 (calibration). Squared polyline
    # with two right angles (down from Deployed, left across the figure,
    # up into DynoSim Grid Search) -- mirrors the replay edge in fig-2 so
    # the post reads with consistent "wired return path" semantics.
    # Dashed amethyst, thicker than the forward arrows so the feedback
    # channel has its own visual weight while staying secondary.
    x_start = STAGES[4][3]                  # Deployed + Telemetry center
    x_end = STAGES[1][3]                    # DynoSim Grid Search center
    y_box_bottom = Y_C - BOX_H / 2          # 14
    y_loop = y_box_bottom - 50              # -36 -- depth of the loop
    arc_path = (
        f"M {x_start},{y_box_bottom} "
        f"L {x_start},{y_loop} "
        f"L {x_end},{y_loop} "
        f"L {x_end},{y_box_bottom}"
    )
    fig.add_shape(
        type="path",
        path=arc_path,
        line=dict(color=AMETHYST, width=2.5, dash="dash"),
        layer="above",
    )
    # Arrow tip sits visibly below DynoSim's box bottom so the arrowhead
    # is clearly "pointing up into the box" rather than buried inside it.
    fig.add_trace(go.Scatter(
        x=[x_end], y=[y_box_bottom - 8],
        mode="markers",
        marker=dict(symbol="triangle-up", size=14,
                    color=AMETHYST, line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))
    # "TELEMETRY" kicker above the dashed loop -- names what flows back from
    # production and lists concrete signals so the loop reads as a data path
    # rather than an abstract arrow.
    loop_mid_x = (x_start + x_end) / 2          # midpoint of the dashed segment
    fig.add_annotation(
        x=loop_mid_x, y=y_loop + 6,
        xref="x", yref="y",
        xanchor="center", yanchor="bottom",
        text="<b>TELEMETRY</b> · TTFT · TPOT · cache hit rate · queue depth",
        showarrow=False,
        font=dict(family=SANS, size=11, color=AMETHYST),
    )

    # Tufte editorial callout below the dashed feedback loop: semi-transparent
    # dark background, thin white border, white text. Centered on the dashed
    # loop midpoint (not the figure midpoint) so it visually anchors the loop.
    fig.add_annotation(
        x=loop_mid_x, y=y_loop - 20,
        xref="x", yref="y",
        xanchor="center", yanchor="top",
        text="<b>Calibration</b> · production telemetry refines DynoSim's timing model",
        showarrow=False,
        bgcolor="rgba(20,20,20,0.65)",
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        borderpad=10,
        font=dict(
            family="Helvetica Neue, HelveticaNeue, sans-serif",
            size=14, color=TEXT_PRIMARY, weight=300,
        ),
    )

    fig.update_layout(
        template=dynamo_template,
        title=dict(
            text="DynoSim in Prod",
            x=0.02, xanchor="left",
            y=0.95, yanchor="top",
            font=dict(
                family="Helvetica Neue, HelveticaNeue, sans-serif",
                size=42, color=TEXT_PRIMARY, weight=300,
            ),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=180, b=40),
        width=1240, height=520,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis=dict(
            range=[0, 1000],
            showgrid=False, zeroline=False,
            showticklabels=False, ticks="", showline=False,
        ),
        yaxis=dict(
            range=[-80, 165],
            showgrid=False, zeroline=False,
            showticklabels=False, ticks="", showline=False,
        ),
    )

    # Subtitle: 22pt, parked 2px below the title's descender bottom (snug).
    # x=-0.013 maps to the figure's x=0.02 mark to align with the title:
    #   (0.02*1240 - 40) / 1160 = -0.013
    # Uses font-size * 1.00 as the title height (cap + descender):
    #   title_top    = (1 - 0.95) * 520 = 26
    #   title_bottom = 26 + 42 * 1.00   = 68
    #   subtitle_top = 68 + 2           = 70   # +2 px = snug
    #   plot_h       = 520 - 180 - 40   = 300
    #   paper_y      = 1 + (180 - 70) / 300 = 1.367
    fig.add_annotation(
        x=-0.013, y=1.367,
        xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        text="Sweep in sim, verify on the cluster, calibrate from telemetry.",
        showarrow=False,
        font=dict(
            family="Helvetica Neue, HelveticaNeue, sans-serif",
            size=22, color=TEXT_MUTED, weight=300,
        ),
    )

    out_svg = HERE.parent / "images" / "fig-9-tuning-loop.svg"
    out_png = HERE.parent / "images" / "fig-9-tuning-loop.png"
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_svg))
    fig.write_image(str(out_png), scale=2)
    print(f"wrote {out_svg}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
