#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Figure 2 - TTFT for stable / varying / stripped prompt prefixes.

Strip chart with deltas. The story is the transition: a varying per-session
header at position zero defeats prefix caching (911 ms, coral). Stripping it
on the frontend brings TTFT back to the stable-prefix baseline (169 ms vs
168 ms). A curved arrow from 911 -> 169 carries the 5x callout, so the eye
moves from problem to fix in one motion.

Workload: 52K-token prompt against
``nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`` on a single B200 in
aggregated serving mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

# Stable and Stripped are 1 ms apart; jitter onto separate y-rows so each
# gets clean label space. Varying sits on the center row.
# Top row: the two production states (Stripped = the fix, Varying = the
# failure mode). Bottom row: Stable, an idealized-but-impractical baseline.
POINTS = [
    {
        "label": "Stripped Prefix",
        "ttft_ms": 169,
        "y": 0.45,
        "color_role": "dynamo_green",
        "sublabel": "--strip-anthropic-preamble",
        "sublabel_mono": True,
    },
    {
        "label": "Varying Prefix",
        "ttft_ms": 911,
        "y": 0.45,
        "color_role": "coral",
        "sublabel": "per-session billing header",
        "sublabel_mono": False,
    },
    {
        "label": "Stable Prefix",
        "ttft_ms": 168,
        "y": -0.45,
        "color_role": "neutral",
        "sublabel": "loses per-session metering",
        "sublabel_mono": False,
    },
]

# Editorial pair: Geist for body / headlines, Geist Mono for numerals + code.
# Designed together by Vercel; reads more typeset and less UI than Inter.
SANS = "Geist, 'Helvetica Neue', Helvetica, Arial, system-ui, sans-serif"
MONO = "'Geist Mono', 'SF Mono', Menlo, Consolas, monospace"


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig-2-ttft-prefix-stability.png"
    svg_path = out_dir / "fig-2-ttft-prefix-stability.svg"

    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    border_subtle = tokens["colors"]["border"]["subtle"]
    arrow_color = "#e6e6e6"
    headline_color = "#ffffff"
    deck_color = "#9a9a9a"

    color_map = {
        "coral": tokens["colors"]["accent"]["coral"],
        "dynamo_green": tokens["colors"]["accent"]["dynamo_green"],
        "neutral": tokens["colors"]["text"]["medium"],
    }

    width, height = 1024, 460
    margin_l, margin_r, margin_t, margin_b = 60, 60, 110, 90

    x_axis_max = 1000

    fig = go.Figure()

    # Stable baseline reference: a faint vertical line at 168 ms extending
    # the full plot height. "What good looks like."
    stable_x = next(p["ttft_ms"] for p in POINTS if p["label"] == "Stable Prefix")
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=stable_x,
        x1=stable_x,
        y0=0,
        y1=1,
        line=dict(color=border_subtle, width=1, dash="dot"),
        layer="below",
    )

    # Markers, drawn smallest-quietest -> loudest so the hero sits on top.
    for p in POINTS:
        color = color_map[p["color_role"]]
        is_hero = p["color_role"] == "dynamo_green"
        is_anomaly = p["color_role"] == "coral"

        # Outer halo for hero (green) and anomaly (coral) only.
        if is_hero or is_anomaly:
            fig.add_trace(
                go.Scatter(
                    x=[p["ttft_ms"]],
                    y=[p["y"]],
                    mode="markers",
                    marker=dict(
                        size=34 if is_hero else 28,
                        color=color,
                        opacity=0.14,
                        line=dict(width=0),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[p["ttft_ms"]],
                y=[p["y"]],
                mode="markers",
                name=p["label"],
                marker=dict(
                    size=18 if is_hero else (14 if is_anomaly else 10),
                    color=color,
                    line=dict(
                        color="#000000",
                        width=2 if is_hero else 0,
                    ),
                ),
                hovertemplate=(
                    f"<b>{p['label']}</b><br>" f"TTFT: {p['ttft_ms']} ms<extra></extra>"
                ),
                showlegend=False,
            )
        )

    annotations: list[dict] = []

    # Headline + deck above the strip. Sentence case, finding-led.
    annotations.append(
        dict(
            x=0,
            xref="paper",
            xanchor="left",
            y=1.0,
            yref="paper",
            yanchor="bottom",
            yshift=58,
            text="<b>Stripping the per-session header restores prefix caching</b>",
            showarrow=False,
            font=dict(family=SANS, size=22, color=headline_color),
        )
    )
    annotations.append(
        dict(
            x=0,
            xref="paper",
            xanchor="left",
            y=1.0,
            yref="paper",
            yanchor="bottom",
            yshift=28,
            text=(
                "Time to first token, 52K-token prompt - "
                "<span style='font-family:" + MONO + "'>"
                "Nemotron-3-Super-120B-A12B-NVFP4</span> on one B200"
            ),
            showarrow=False,
            font=dict(family=SANS, size=13, color=deck_color),
        )
    )

    # Per-point label stacks. Title case for names, mono for the ms value.
    for p in POINTS:
        color = color_map[p["color_role"]]
        if p["label"] == "Stripped Prefix":
            # Top row, hero. Labels stack above the marker.
            name_yshift, ms_yshift, sub_yshift = 46, 26, -22
            sub_anchor = "top"
        elif p["label"] == "Stable Prefix":
            # Bottom row, dimmed baseline. Labels stack below the marker.
            name_yshift, ms_yshift, sub_yshift = -46, -26, 22
            sub_anchor = "bottom"
        else:  # Varying Prefix, center row
            name_yshift, ms_yshift, sub_yshift = 46, 26, -22
            sub_anchor = "top"

        annotations.append(
            dict(
                x=p["ttft_ms"],
                y=p["y"],
                xref="x",
                yref="y",
                text=f"<b>{p['label']}</b>",
                showarrow=False,
                xanchor="center",
                yanchor="bottom" if name_yshift > 0 else "top",
                yshift=name_yshift,
                font=dict(family=SANS, size=12, color="#ffffff"),
            )
        )
        annotations.append(
            dict(
                x=p["ttft_ms"],
                y=p["y"],
                xref="x",
                yref="y",
                text=f"<b>{p['ttft_ms']} ms</b>",
                showarrow=False,
                xanchor="center",
                yanchor="bottom" if ms_yshift > 0 else "top",
                yshift=ms_yshift,
                font=dict(family=MONO, size=15, color=color),
            )
        )
        sub_family = MONO if p.get("sublabel_mono") else SANS
        annotations.append(
            dict(
                x=p["ttft_ms"],
                y=p["y"],
                xref="x",
                yref="y",
                text=p["sublabel"],
                showarrow=False,
                xanchor="center",
                yanchor=sub_anchor,
                yshift=sub_yshift,
                font=dict(family=sub_family, size=10, color=color),
            )
        )

    # Horizontal arrow from Varying (911) -> Stripped (169) along the top
    # row. Carries a single-line "5x faster" callout above.
    arrow_y = 0.45
    annotations.append(
        dict(
            x=185,
            y=arrow_y,
            ax=895,
            ay=arrow_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.4,
            arrowwidth=2.2,
            arrowcolor=arrow_color,
            standoff=14,
            startstandoff=14,
        )
    )
    annotations.append(
        dict(
            x=540,
            y=arrow_y,
            xref="x",
            yref="y",
            text=("<span style='font-family:" + MONO + "'><b>5×</b></span>" "  faster"),
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            yshift=8,
            font=dict(family=SANS, size=13, color=headline_color),
        )
    )

    fig.update_layout(
        template=template,
        title=None,
        xaxis=dict(
            title=dict(
                text="TTFT (ms)",
                font=dict(family=SANS, size=11, color=deck_color),
            ),
            range=[-30, x_axis_max + 30],
            showgrid=True,
            gridcolor=border_subtle,
            zeroline=False,
            tickvals=[0, 200, 400, 600, 800, 1000],
            tickfont=dict(family=MONO, size=11, color="#cccccc"),
        ),
        yaxis=dict(
            title="",
            range=[-1, 1],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False,
        ),
        width=width,
        height=height,
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        annotations=annotations,
        font=dict(family=SANS),
    )

    fig.write_image(str(png_path), scale=3)
    fig.write_image(str(svg_path))
    print(f"Wrote {png_path.name}  ({width}x{height})")
    print(f"Wrote {svg_path.name}  ({width}x{height})")


if __name__ == "__main__":
    main()
