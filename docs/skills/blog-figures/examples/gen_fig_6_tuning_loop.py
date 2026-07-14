#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Example: closed-loop tuning pipeline with phase tags + a feedback loop.

A horizontal pipeline — Sweep, Verify, Deploy, Telemetry — with selective
**phase tags** above the inflection stages (INITIAL VERIFICATION over the
Verify box in CPU blue; FEEDBACK LOOP over the Telemetry box in amethyst)
and a squared, dashed **calibration loop** returning telemetry to the sweep.

Demonstrates: selective phase markers bound to their stage's accent color,
a squared (right-angle) dashed return path matched to the target's accent,
and y-range padding so the phase tags do not clip.

Title uses the Dynamo Dark display/hero treatment (Helvetica Neue Light,
title case). Layout is representative, hard-coded (no external data).

Usage:
    python3 gen_fig_6_tuning_loop.py        # -> images/fig-6-tuning-loop.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

HERO_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

# (label, accent key, muted-fill key)
STAGES = [
    ("Sweep", "dynamo_green", "green"),
    ("Verify\n(Cluster A/B)", "cpu_blue", "blue"),
    ("Deploy", "emerald", "teal"),
    ("Telemetry", "amethyst", "purple"),
]


def main() -> None:
    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]
    green = colors["accent"]["dynamo_green"]
    cpu_blue = colors["accent"]["cpu_blue"]
    amethyst = colors["accent"]["amethyst"]
    text_primary = colors["text"]["primary"]
    text_muted = colors["text"]["muted"]
    font_sans = tokens["typography"]["font_family"]

    fig = go.Figure()
    shapes = []
    annotations = []

    n = len(STAGES)
    box_w, gap, y0, y1 = 19.0, 5.0, 46.0, 62.0
    xs = [4.0 + i * (box_w + gap) for i in range(n)]
    y_mid = (y0 + y1) / 2

    for (label, accent_key, fill_key), x in zip(STAGES, xs):
        accent = colors["accent"][accent_key]
        shapes.append(
            dict(
                type="rect",
                x0=x,
                x1=x + box_w,
                y0=y0,
                y1=y1,
                line=dict(color=accent, width=1.5),
                fillcolor=colors["fills"][fill_key],
                layer="above",
            )
        )
        annotations.append(
            dict(
                x=x + box_w / 2,
                y=y_mid,
                text=label.replace("\n", "<br>"),
                showarrow=False,
                font=dict(family=font_sans, size=14, color=text_primary),
            )
        )

    # Forward arrows (green solid).
    for i in range(n - 1):
        annotations.append(
            dict(
                x=xs[i + 1],
                y=y_mid,
                ax=xs[i] + box_w,
                ay=y_mid,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.4,
                arrowwidth=2,
                arrowcolor=green,
                text="",
            )
        )

    # Selective phase tags above the inflection stages.
    phase_tags = [(1, "INITIAL VERIFICATION", cpu_blue), (3, "FEEDBACK LOOP", amethyst)]
    for idx, tag, color in phase_tags:
        annotations.append(
            dict(
                x=xs[idx] + box_w / 2,
                y=y1 + 8,
                text=f"<b>{tag}</b>",
                showarrow=False,
                font=dict(family=font_sans, size=12, color=color),
            )
        )

    # Squared, dashed calibration loop (amethyst): Telemetry -> Sweep.
    x_start = xs[-1] + box_w / 2
    x_end = xs[0] + box_w / 2
    y_loop = 30.0
    fig.add_trace(
        go.Scatter(
            x=[x_start, x_start, x_end, x_end],
            y=[y0, y_loop, y_loop, y0 - 1.5],
            mode="lines",
            line=dict(color=amethyst, width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    annotations.append(
        dict(
            x=x_end,
            y=y0,
            ax=x_end,
            ay=y0 - 1.5,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.4,
            arrowwidth=2,
            arrowcolor=amethyst,
            text="",
        )
    )
    annotations.append(
        dict(
            x=(x_start + x_end) / 2,
            y=y_loop,
            yshift=-12,
            text="calibrate from telemetry",
            showarrow=False,
            font=dict(family=font_sans, size=12, color=amethyst),
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="Sweep, Verify, Calibrate: The Tuning Loop",
            font=dict(family=HERO_FONT, size=40, color=text_primary, weight=300),
            subtitle=dict(
                text="Sweep in sim, verify on the cluster, calibrate from telemetry.",
                font=dict(family=HERO_FONT, size=19, color=text_muted, weight=300),
            ),
            x=0.03,
            xanchor="left",
            y=0.94,
            yanchor="top",
        ),
        # y-range padded above the boxes so phase tags at y1+8 do not clip.
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(range=[10, 78], visible=False),
        width=1600,
        height=800,
        margin=dict(l=40, r=40, t=150, b=30),
        shapes=shapes,
        annotations=annotations,
    )

    out = Path(__file__).parent / "images" / "fig-6-tuning-loop.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), scale=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
