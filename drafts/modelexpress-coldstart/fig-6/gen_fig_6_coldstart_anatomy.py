#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""fig-6: Cold Start Anatomy -- P2P weights, cold JIT caches (Dynamo Dark).

A hero-width single-row stacked timeline that breaks the observed 420 s cold
start of a DeepSeek-V4-Pro / vLLM deployment into its eight phases, anchored by
a large "420s" KPI. ModelExpress delivers weights peer-to-peer (Model load via
MX P2P is only 11 s / 2.6 %, the single green accent), so the cost shifts almost
entirely to the cold JIT-cache warmup cluster (profiling + DeepGEMM compile +
CUDA graph capture = 350 s / 83 %, rendered in the gold warmup family so it
reads the same as the warmup phase in fig-4).

Restyle of the reference light-theme breakdown bar into Dynamo Dark:
    - pure-black ground, token palette only, border-radius 0
    - display / hero title treatment (Helvetica Neue Light, title case)
    - single green accent on the ModelExpress weight-load phase (the hero
      data-plane weight load); the JIT-cache warmup cluster carries the gold
      "warmup" role, matching fig-4; setup phases recede to grey
    - real seconds x-axis so the minute ticks align exactly to the bar

Data (durations in seconds) is the source of truth; percentages are derived.

Usage:
    python3 gen_fig_6_coldstart_anatomy.py   # -> images/fig-6-coldstart-anatomy.{png,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

# Hero / display title set: Helvetica Neue Light, title case (falls back to the
# token sans stack where Helvetica Neue is unavailable).
HERO_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

TOTAL_SEC = 420  # observed total, start -> Application startup complete (7m)

# Semantic roles decide color:
#   grey  -> process / infra setup (recedes)
#   green -> the ModelExpress win (single accent, hero data-plane weight load)
#   gold  -> the cold-JIT-cache warmup cluster (peak = brightest gold), matching
#            fig-4's warmup phase; coral stays reserved for a baseline/slow path,
#            which this single optimized run does not contain
# Phases in time order. `place`: "in" = label inside the bar; "above" = leader
# line + marker + duration above the bar (used for the narrow phases).
SEGMENTS = [
    {
        "name": "Python & vLLM imports & others",
        "dur": 27,
        "role": "setup",
        "place": "above",
    },
    {
        "name": "Engine config & core spawn",
        "dur": 14,
        "role": "setup",
        "place": "above",
    },
    {
        "name": "Worker spawn & distributed init",
        "dur": 17,
        "role": "setup",
        "place": "above",
    },
    {"name": "Model load via MX P2P", "dur": 11, "role": "win", "place": "above"},
    {
        "name": "Memory profiling (JIT wave)",
        "dur": 100,
        "role": "cost",
        "place": "in",
        "inbar": "Profiling + JIT",
    },
    {
        "name": "JIT warmup (DeepGEMM compile)",
        "dur": 142,
        "role": "peak",
        "place": "in",
        "inbar": "DeepGEMM warmup",
    },
    {
        "name": "CUDA graph capture",
        "dur": 108,
        "role": "cost",
        "place": "in",
        "inbar": "Graph capture",
    },
    {"name": "API server ready", "dur": 1, "role": "setup", "place": "above"},
]

# Two-column legend order matches the reference (reading down each column).
LEGEND_LEFT = [
    "Python & vLLM imports & others",
    "Worker spawn & distributed init",
    "Memory profiling (JIT wave)",
    "CUDA graph capture",
]
LEGEND_RIGHT = [
    "Engine config & core spawn",
    "Model load via MX P2P",
    "JIT warmup (DeepGEMM compile)",
    "API server ready",
]

# ---- geometry (data coords: x = seconds [0,420]; y = paper-equivalent [0,1]) ----
BAR_Y0, BAR_Y1 = 0.500, 0.650
LABEL_Y_HIGH, LABEL_Y_LOW = 0.760, 0.705
TICK_Y0, TICK_Y1 = 0.470, 0.500
TICK_LABEL_Y = 0.435


def main() -> None:
    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]
    font_sans = tokens["typography"]["font_family"]
    font_mono = tokens["typography"]["font_family_mono"]

    green = colors["accent"]["dynamo_green"]  # ModelExpress win (accent)
    warmup = colors["accent"]["fluorite"]  # peak warmup phase (gold, matches fig-4)
    warmup_muted = colors["chart_fills"][2]  # #9a7800 -- flanking warmup (muted gold)
    grey = colors["text"]["medium"]  # #8c8c8c -- setup (neutral)
    subtle = colors["border"]["subtle"]  # #3a3a3a -- separators / ticks
    black = colors["background"]["primary"]  # #000000 -- hairline gaps + in-bar text
    text_primary = colors["text"]["primary"]
    text_secondary = colors["text"]["secondary"]
    text_muted = colors["text"]["muted"]

    role_color = {
        "setup": grey,
        "win": green,
        "cost": warmup_muted,
        "peak": warmup,
    }

    def pct(dur: int) -> float:
        return round(dur / TOTAL_SEC * 100, 1)

    seg_color = {s["name"]: role_color[s["role"]] for s in SEGMENTS}
    seg_dur = {s["name"]: s["dur"] for s in SEGMENTS}

    fig = go.Figure()
    shapes: list[dict] = []
    annotations: list[dict] = []

    # ---- header divider (hairline under the subtitle) ----
    shapes.append(
        dict(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.0,
            x1=1.0,
            y0=1.005,
            y1=1.005,
            line=dict(color=subtle, width=1),
            layer="above",
        )
    )

    # ---- big KPI number + caption ----
    annotations.append(
        dict(
            xref="paper",
            yref="paper",
            x=0.0,
            y=0.955,
            xanchor="left",
            yanchor="top",
            text="420s",
            showarrow=False,
            font=dict(family=font_mono, size=64, color=text_primary, weight=500),
        )
    )
    annotations.append(
        dict(
            xref="paper",
            yref="paper",
            x=0.108,
            y=0.905,
            xanchor="left",
            yanchor="middle",
            text="<b>7m</b> · observed total, start → Application startup complete",
            showarrow=False,
            font=dict(family=font_mono, size=15, color=text_secondary),
        )
    )

    # ---- stacked timeline bar (fills, no borders) ----
    cursor = 0
    for s in SEGMENTS:
        x0, x1 = cursor, cursor + s["dur"]
        cursor = x1
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=x0,
                x1=x1,
                y0=BAR_Y0,
                y1=BAR_Y1,
                fillcolor=seg_color[s["name"]],
                line=dict(width=0),
                layer="above",
            )
        )
        # in-bar label for the wide phases. The wide phases are the gold warmup
        # cluster, so the label sits black-on-gold (AA on both gold shades).
        if s["place"] == "in":
            in_label_color = black if s["role"] in ("cost", "peak") else text_primary
            annotations.append(
                dict(
                    xref="x",
                    yref="y",
                    x=(x0 + x1) / 2,
                    y=(BAR_Y0 + BAR_Y1) / 2,
                    xanchor="center",
                    yanchor="middle",
                    text=f"{s['inbar']} · {s['dur']}s",
                    showarrow=False,
                    font=dict(
                        family=font_mono, size=15, color=in_label_color, weight=500
                    ),
                )
            )

    # ---- hairline separators at each internal boundary ----
    cursor = 0
    for s in SEGMENTS[:-1]:
        cursor += s["dur"]
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=cursor,
                x1=cursor,
                y0=BAR_Y0,
                y1=BAR_Y1,
                line=dict(color=black, width=1.5),
                layer="above",
            )
        )

    # ---- above-bar labels for the narrow phases (leader line + marker + text) ----
    marker_x, marker_y, marker_c = [], [], []
    stagger = [LABEL_Y_HIGH, LABEL_Y_LOW]
    cursor = 0
    above_i = 0
    for s in SEGMENTS:
        x0, x1 = cursor, cursor + s["dur"]
        cursor = x1
        if s["place"] != "above":
            continue
        cx = (x0 + x1) / 2
        ly = stagger[above_i % 2]
        above_i += 1
        c = seg_color[s["name"]]
        # leader line from bar top up to just below the marker
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=cx,
                x1=cx,
                y0=BAR_Y1,
                y1=ly - 0.016,
                line=dict(color=subtle, width=1),
                layer="above",
            )
        )
        marker_x.append(cx)
        marker_y.append(ly)
        marker_c.append(c)
        # Above-bar text sits right of the marker. The narrow ModelExpress win is
        # the single green accent, so instead of a bare "11s" we bind its phase
        # name AND value into one callout anchored to the green marker + leader
        # line -- making "Model load via MX P2P = 11s" unmistakable at a glance.
        # Mirrors the in-bar "name · Xs" pattern. The name uses the light text
        # token (AA on black); the value stays green to echo the segment.
        if s["role"] == "win":
            label_text = f'{s["name"]} · <span style="color:{green}">{s["dur"]}s</span>'
        else:
            label_text = f"{s['dur']}s"
        annotations.append(
            dict(
                xref="x",
                yref="y",
                x=cx,
                y=ly,
                xshift=10,
                xanchor="left",
                yanchor="middle",
                text=label_text,
                showarrow=False,
                font=dict(family=font_mono, size=14, color=text_secondary, weight=500),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=marker_x,
            y=marker_y,
            mode="markers",
            marker=dict(symbol="square", size=11, color=marker_c, line=dict(width=0)),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # ---- x-axis: minute ticks aligned to the bar ----
    ticks = [0, 60, 120, 180, 240, 300, 360, 420]
    tick_txt = ["0s", "1m", "2m", "3m", "4m", "5m", "6m", "420s"]
    # subtle baseline under the bar
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="y",
            x0=0,
            x1=TOTAL_SEC,
            y0=BAR_Y0,
            y1=BAR_Y0,
            line=dict(color=subtle, width=1),
            layer="above",
        )
    )
    for t, txt in zip(ticks, tick_txt):
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=t,
                x1=t,
                y0=TICK_Y0,
                y1=TICK_Y1,
                line=dict(color=subtle, width=1),
                layer="above",
            )
        )
        anchor = "left" if t == 0 else "right" if t == TOTAL_SEC else "center"
        annotations.append(
            dict(
                xref="x",
                yref="y",
                x=t,
                y=TICK_LABEL_Y,
                xanchor=anchor,
                yanchor="top",
                text=txt,
                showarrow=False,
                font=dict(family=font_mono, size=13, color=grey),
            )
        )

    # ---- legend table (two columns, swatch + name + right-aligned dur + pct) ----
    sw_w, sw_h = 13 / 1504, 13 / 617  # visually square swatch in paper units
    cols = [
        {
            "names": LEGEND_LEFT,
            "sw_x": 0.0,
            "name_x": 0.020,
            "dur_x": 0.400,
            "pct_x": 0.460,
        },
        {
            "names": LEGEND_RIGHT,
            "sw_x": 0.510,
            "name_x": 0.530,
            "dur_x": 0.910,
            "pct_x": 0.970,
        },
    ]
    row_ys = [0.300, 0.213, 0.126, 0.039]
    for col in cols:
        for name, ry in zip(col["names"], row_ys):
            shapes.append(
                dict(
                    type="rect",
                    xref="paper",
                    yref="paper",
                    x0=col["sw_x"],
                    x1=col["sw_x"] + sw_w,
                    y0=ry - sw_h / 2,
                    y1=ry + sw_h / 2,
                    fillcolor=seg_color[name],
                    line=dict(width=0),
                    layer="above",
                )
            )
            annotations.append(
                dict(
                    xref="paper",
                    yref="paper",
                    x=col["name_x"],
                    y=ry,
                    xanchor="left",
                    yanchor="middle",
                    text=name,
                    showarrow=False,
                    font=dict(family=font_sans, size=15, color=text_secondary),
                )
            )
            annotations.append(
                dict(
                    xref="paper",
                    yref="paper",
                    x=col["dur_x"],
                    y=ry,
                    xanchor="right",
                    yanchor="middle",
                    text=f"{seg_dur[name]}s",
                    showarrow=False,
                    font=dict(
                        family=font_mono, size=15, color=text_primary, weight=500
                    ),
                )
            )
            annotations.append(
                dict(
                    xref="paper",
                    yref="paper",
                    x=col["pct_x"],
                    y=ry,
                    xanchor="right",
                    yanchor="middle",
                    text=f"{pct(seg_dur[name])}%",
                    showarrow=False,
                    font=dict(family=font_mono, size=14, color=text_muted),
                )
            )

    fig.update_layout(
        template=template,
        title=dict(
            text="Cold Start Anatomy: P2P Weights, Cold JIT Caches",
            font=dict(family=HERO_FONT, size=40, color=text_primary, weight=300),
            subtitle=dict(
                text=(
                    "DeepSeek-V4-Pro · vLLM 0.23.0 · TP8 + EP + fp8 KV cache "
                    "on one 8-GPU node · load_format=modelexpress — weights "
                    "arrive peer-to-peer, but every JIT kernel cache starts empty."
                ),
                font=dict(family=HERO_FONT, size=19, color=text_muted, weight=300),
            ),
            x=0.03,
            xanchor="left",
            y=0.95,
            yanchor="top",
        ),
        xaxis=dict(range=[0, TOTAL_SEC], visible=False, fixedrange=True),
        yaxis=dict(range=[0, 1], visible=False, fixedrange=True),
        width=1600,
        height=840,
        margin=dict(l=48, r=48, t=176, b=44),
        shapes=shapes,
        annotations=annotations,
        showlegend=False,
    )

    out_dir = Path(__file__).parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig-6-coldstart-anatomy.png"
    svg = out_dir / "fig-6-coldstart-anatomy.svg"
    fig.write_image(str(png), scale=2)
    fig.write_image(str(svg))
    print(f"Wrote {png}")
    print(f"Wrote {svg}")


if __name__ == "__main__":
    main()
