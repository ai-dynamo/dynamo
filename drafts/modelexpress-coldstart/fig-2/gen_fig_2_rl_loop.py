#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""fig-2 — the RL training loop, refit by ModelExpress (Dynamo Dark).

A four-stage reinforcement-learning loop laid out as a 2x2 ring with purely
orthogonal, clockwise connectors:

    Rollout ─[1]─▶ Reward ─[2]─▶ Trainer ─[3]─▶ Weight refit ─[4]─▶ (back to Rollout)

The source figure highlighted the Weight refit stage (ModelExpress) in blue
because that is the component the blog is about. In Dynamo Dark, green marks
the single hero item, so ModelExpress carries the green accent: a green
1.5 px border on its box and the green [4] "updated weights" edge that closes
the loop back into the rollout engines. The other three stages stay neutral
(#1a1a1a fill, #3a3a3a hairline) and the forward edges are medium grey.

Diagram rethink (vs. a pixel copy): the four boxes are placed on a computed
2x2 grid so every connector is a straight horizontal or vertical segment
landing on an exact box edge — no diagonals, no eyeballed endpoints. Backend
names are alphabetized per house convention (SGLang / TRT-LLM / vLLM).

Title uses the Dynamo Dark display/hero treatment (Helvetica Neue Light,
title case). Renders deterministically; no external data.

Usage:
    python3 gen_fig_2_rl_loop.py        # -> images/fig-2-rl-loop.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

HERO_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

# --- Computed grid geometry (0-100 x-axis, 0-100 y-axis; visible range set
#     below). Every coordinate derives from these named constants. ---
BOX_W = 30.0
BOX_H = 18.0
COL_L_X0 = 8.0  # left column left edge
COL_R_X0 = 62.0  # right column left edge
ROW_T_Y0 = 56.0  # top row bottom edge
ROW_B_Y0 = 16.0  # bottom row bottom edge

COL_L = (COL_L_X0, COL_L_X0 + BOX_W)  # (8, 38)
COL_R = (COL_R_X0, COL_R_X0 + BOX_W)  # (62, 92)
ROW_T = (ROW_T_Y0, ROW_T_Y0 + BOX_H)  # (56, 74)
ROW_B = (ROW_B_Y0, ROW_B_Y0 + BOX_H)  # (16, 34)


def _cx(col: tuple[float, float]) -> float:
    return (col[0] + col[1]) / 2


def _cy(row: tuple[float, float]) -> float:
    return (row[0] + row[1]) / 2


# (key, title, sub-line, column, row, is_hero)
STAGES = [
    ("rollout", "Rollout", "SGLang / TRT-LLM / vLLM", COL_L, ROW_T, False),
    ("reward", "Reward", "RM or rule-based", COL_R, ROW_T, False),
    ("trainer", "Trainer", "FSDP2 / DTensor / Megatron-Core", COL_R, ROW_B, False),
    ("weight_refit", "Weight refit", "ModelExpress", COL_L, ROW_B, True),
]


def main() -> None:
    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]
    green = colors["accent"]["dynamo_green"]
    surface = colors["background"]["surface"]
    border_subtle = colors["border"]["subtle"]
    text_primary = colors["text"]["primary"]
    text_secondary = colors["text"]["secondary"]
    text_medium = colors["text"]["medium"]
    text_muted = colors["text"]["muted"]
    font_sans = tokens["typography"]["font_family"]

    fig = go.Figure()
    shapes = []
    annotations = []

    # --- Stage boxes + two-line labels ---
    for _key, title, sub, col, row, is_hero in STAGES:
        x0, x1 = col
        y0, y1 = row
        cx, cy = _cx(col), _cy(row)
        shapes.append(
            dict(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line=dict(
                    color=green if is_hero else border_subtle,
                    width=1.5 if is_hero else 1,
                ),
                fillcolor=surface,
                layer="above",
            )
        )
        annotations.append(
            dict(
                x=cx,
                y=cy + 3.4,
                text=f"<b>{title}</b>",
                showarrow=False,
                font=dict(family=font_sans, size=16, color=text_primary),
            )
        )
        annotations.append(
            dict(
                x=cx,
                y=cy - 3.6,
                text=sub,
                showarrow=False,
                font=dict(family=font_sans, size=12.5, color=text_medium),
            )
        )

    # --- Ring geometry: box edge midpoints used as connector endpoints ---
    roll_c = (_cx(COL_L), _cy(ROW_T))  # (23, 65)
    rew_c = (_cx(COL_R), _cy(ROW_T))  # (77, 65)
    trn_c = (_cx(COL_R), _cy(ROW_B))  # (77, 25)
    wr_c = (_cx(COL_L), _cy(ROW_B))  # (23, 25)

    def arrow(x_tip, y_tip, x_tail, y_tail, color, width):
        annotations.append(
            dict(
                x=x_tip,
                y=y_tip,
                ax=x_tail,
                ay=y_tail,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.3,
                arrowwidth=width,
                arrowcolor=color,
                text="",
            )
        )

    # [1] Rollout -> Reward (top row, left to right)
    arrow(COL_R[0], roll_c[1], COL_L[1], roll_c[1], text_medium, 2)
    # [2] Reward -> Trainer (right column, top to bottom)
    arrow(rew_c[0], ROW_B[1], rew_c[0], ROW_T[0], text_medium, 2)
    # [3] Trainer -> Weight refit (bottom row, right to left)
    arrow(COL_L[1], trn_c[1], COL_R[0], trn_c[1], text_medium, 2)
    # [4] Weight refit -> Rollout (left column, bottom to top) — GREEN accent
    arrow(wr_c[0], ROW_T[0], wr_c[0], ROW_B[1], green, 2.5)

    # --- Edge labels ([4] bound to the green refit path) ---
    y_gap = (ROW_B[1] + ROW_T[0]) / 2  # vertical midline of the ring gap (45)
    edge_labels = [
        # (text, x, y, xanchor, yshift, color)
        ("[1] tokens + logprobs", 50, roll_c[1], "center", 14, text_secondary),
        ("[2] scalar rewards", rew_c[0] + 2, y_gap, "left", 0, text_secondary),
        ("[3] new weights \u03b8", 50, trn_c[1], "center", 14, text_secondary),
        ("[4] updated weights", wr_c[0] - 2, y_gap, "right", 0, green),
    ]
    for text, x, y, xanchor, yshift, color in edge_labels:
        annotations.append(
            dict(
                x=x,
                y=y,
                text=text,
                showarrow=False,
                xanchor=xanchor,
                yshift=yshift,
                font=dict(family=font_sans, size=13, color=color),
            )
        )

    fig.update_layout(
        template=template,
        title=dict(
            text="ModelExpress Closes the RL Training Loop",
            font=dict(family=HERO_FONT, size=40, color=text_primary, weight=300),
            subtitle=dict(
                text=(
                    "Rollout, reward, train, refit — ModelExpress refits weights "
                    "and closes the loop."
                ),
                font=dict(family=HERO_FONT, size=19, color=text_muted, weight=300),
            ),
            x=0.03,
            xanchor="left",
            y=0.93,
            yanchor="top",
        ),
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(range=[6, 84], visible=False),
        width=1600,
        height=820,
        margin=dict(l=40, r=40, t=150, b=40),
        shapes=shapes,
        annotations=annotations,
    )

    out = Path(__file__).parent / "images" / "fig-2-rl-loop.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), scale=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
