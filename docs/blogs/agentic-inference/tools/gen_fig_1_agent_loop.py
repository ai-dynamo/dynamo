#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Figure 1 - The agent loop, side-by-side sequence diagram.

Two halves, same shape. Standard server on the left (muted) shows three
turns where the harness rebuilds the full prompt every time and re-parses
text replies. Dynamo on the right (green) shows the same three turns
with the prefix cached server-side and structured tool_call_dispatch
events instead of raw text.

The figure makes the *repetition* visible: the agent loop is hundreds of
turns per session, and the wire format determines whether each turn is
fresh work or a continuation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import load_tokens

SANS = "Geist, 'Helvetica Neue', Helvetica, Arial, system-ui, sans-serif"
MONO = "'Geist Mono', 'SF Mono', Menlo, Consolas, monospace"


# ---------------------------------------------------------------------------
# Layout (data axis is normalized 0..1 on both x and y)
# ---------------------------------------------------------------------------
# Two halves split at x = 0.50.
#   Left  (Standard): harness lifeline at x=0.10, server lifeline at x=0.42
#   Right (Dynamo) : harness lifeline at x=0.58, server lifeline at x=0.90
LIFELINES = {
    "std_harness": 0.10,
    "std_server": 0.42,
    "dyn_harness": 0.58,
    "dyn_server": 0.90,
}

# Vertical extent for the sequence body. Title sits above 0.93;
# captions sit below 0.10.
LIFELINE_TOP = 0.78
LIFELINE_BOTTOM = 0.16

# Two turns, evenly spaced. For each turn we draw a request (top) and
# a response (bottom). Title says "two turns of the agent loop"; the
# `...repeats...` continuation marker carries the rest.
TURN_Y = [
    {"req": 0.65, "resp": 0.55},
    {"req": 0.42, "resp": 0.32},
]


def _arrow(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    width: float = 2.0,
    opacity: float = 1.0,
) -> dict:
    return dict(
        x=x1,
        y=y1,
        ax=x0,
        ay=y0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=width,
        arrowcolor=color,
        opacity=opacity,
        text="",
        standoff=2,
        startstandoff=2,
    )


def _label(
    cx: float,
    cy: float,
    text: str,
    color: str,
    family: str = SANS,
    size: int = 11,
    xanchor: str = "center",
    yanchor: str = "middle",
    italic: bool = False,
    bold: bool = False,
) -> dict:
    if italic:
        text = f"<i>{text}</i>"
    if bold:
        text = f"<b>{text}</b>"
    return dict(
        x=cx,
        y=cy,
        xref="x",
        yref="y",
        text=text,
        showarrow=False,
        xanchor=xanchor,
        yanchor=yanchor,
        font=dict(family=family, size=size, color=color),
    )


def _lifeline(x: float, color: str) -> dict:
    return dict(
        type="line",
        xref="x",
        yref="y",
        x0=x,
        x1=x,
        y0=LIFELINE_BOTTOM,
        y1=LIFELINE_TOP,
        line=dict(color=color, width=1, dash="dot"),
        layer="below",
    )


def _node_rect(
    cx: float,
    cy: float,
    w: float,
    h: float,
    fill: str,
    stroke: str,
    stroke_w: float = 1.5,
) -> dict:
    return dict(
        type="rect",
        xref="x",
        yref="y",
        x0=cx - w / 2,
        x1=cx + w / 2,
        y0=cy - h / 2,
        y1=cy + h / 2,
        fillcolor=fill,
        line=dict(color=stroke, width=stroke_w),
        layer="above",
    )


def _draw_turn(
    annotations: list,
    shapes: list,
    *,
    harness_x: float,
    server_x: float,
    req_y: float,
    resp_y: float,
    request_text: str,
    request_text_mono: bool,
    request_accent: str | None,
    accent_color: str,
    response_text: str,
    response_text_mono: bool,
    response_accent: str | None,
    base_color: str,
    arrow_color: str,
    arrow_width: float,
) -> None:
    """Draw one turn (a request arrow + a response arrow with labels)."""

    # Request arrow: harness -> server
    annotations.append(
        _arrow(
            harness_x + 0.005,
            req_y,
            server_x - 0.005,
            req_y,
            color=arrow_color,
            width=arrow_width,
        )
    )
    # Request label sits ABOVE the arrow
    label_x = (harness_x + server_x) / 2
    if request_accent and request_accent in request_text:
        # Split into pre / accent / post for inline color
        pre, _, rest = request_text.partition(request_accent)
        post = rest
        rich = (
            f"<span style='color:{base_color}'>{pre}</span>"
            f"<span style='color:{accent_color}'><b>{request_accent}</b></span>"
            f"<span style='color:{base_color}'>{post}</span>"
        )
        annotations.append(
            _label(
                label_x,
                req_y + 0.022,
                rich,
                color=base_color,
                size=12,
                family=MONO if request_text_mono else SANS,
                yanchor="bottom",
            )
        )
    else:
        annotations.append(
            _label(
                label_x,
                req_y + 0.022,
                request_text,
                color=base_color,
                size=12,
                family=MONO if request_text_mono else SANS,
                yanchor="bottom",
            )
        )

    # Response arrow: server -> harness
    annotations.append(
        _arrow(
            server_x - 0.005,
            resp_y,
            harness_x + 0.005,
            resp_y,
            color=arrow_color,
            width=arrow_width,
        )
    )
    if response_accent and response_accent in response_text:
        pre, _, rest = response_text.partition(response_accent)
        post = rest
        rich = (
            f"<span style='color:{base_color}'>{pre}</span>"
            f"<span style='color:{accent_color}'><b>{response_accent}</b></span>"
            f"<span style='color:{base_color}'>{post}</span>"
        )
        annotations.append(
            _label(
                label_x,
                resp_y + 0.022,
                rich,
                color=base_color,
                size=12,
                family=MONO if response_text_mono else SANS,
                yanchor="bottom",
            )
        )
    else:
        annotations.append(
            _label(
                label_x,
                resp_y + 0.022,
                response_text,
                color=base_color,
                size=12,
                family=MONO if response_text_mono else SANS,
                yanchor="bottom",
            )
        )


def main() -> None:
    tokens = load_tokens()
    bg = tokens["colors"]["background"]["primary"]
    dynamo_green = tokens["colors"]["accent"]["dynamo_green"]
    coral = tokens["colors"]["accent"]["coral"]
    text_primary = tokens["colors"]["text"]["primary"]
    text_secondary = tokens["colors"]["text"]["secondary"]
    text_muted = tokens["colors"]["text"]["muted"]

    # Tones.
    std_lifeline_color = "#5d5d5d"
    std_arrow_color = "#909090"
    std_label_color = text_secondary
    std_node_fill = "#181818"
    std_node_stroke = "#5d5d5d"

    dyn_lifeline_color = "#3a5a00"
    dyn_arrow_color = dynamo_green
    dyn_label_color = text_primary
    dyn_node_fill = "#181818"
    dyn_node_stroke = dynamo_green
    dyn_accent = dynamo_green

    shapes: list[dict] = []
    annotations: list[dict] = []

    # -----------------------------------------------------------------------
    # Lifelines (drawn first, sit below everything)
    # -----------------------------------------------------------------------
    shapes.append(_lifeline(LIFELINES["std_harness"], std_lifeline_color))
    shapes.append(_lifeline(LIFELINES["std_server"], std_lifeline_color))
    shapes.append(_lifeline(LIFELINES["dyn_harness"], dyn_lifeline_color))
    shapes.append(_lifeline(LIFELINES["dyn_server"], dyn_lifeline_color))

    # Vertical separator between the two halves. Stops below the side
    # title banners so each half visually owns its title.
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="y",
            x0=0.50,
            x1=0.50,
            y0=0.06,
            y1=0.84,
            line=dict(color="#2a2a2a", width=1),
            layer="below",
        )
    )

    # -----------------------------------------------------------------------
    # Side titles -- large bold banner above each half. This is the primary
    # "which side is which" signal. A thin underline anchors each title as
    # a section banner rather than a stray label.
    # -----------------------------------------------------------------------
    SIDE_TITLE_Y = 0.875
    SIDE_RULE_Y = 0.852
    std_center = (LIFELINES["std_harness"] + LIFELINES["std_server"]) / 2
    dyn_center = (LIFELINES["dyn_harness"] + LIFELINES["dyn_server"]) / 2

    annotations.append(
        _label(
            std_center,
            SIDE_TITLE_Y,
            "Standard Server",
            color=text_secondary,
            size=18,
            bold=True,
        )
    )
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="y",
            x0=LIFELINES["std_harness"] - 0.04,
            x1=LIFELINES["std_server"] + 0.04,
            y0=SIDE_RULE_Y,
            y1=SIDE_RULE_Y,
            line=dict(color="#3a3a3a", width=1),
            layer="below",
        )
    )

    annotations.append(
        _label(
            dyn_center,
            SIDE_TITLE_Y,
            "Dynamo",
            color=dynamo_green,
            size=18,
            bold=True,
        )
    )
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="y",
            x0=LIFELINES["dyn_harness"] - 0.04,
            x1=LIFELINES["dyn_server"] + 0.04,
            y0=SIDE_RULE_Y,
            y1=SIDE_RULE_Y,
            line=dict(color=dynamo_green, width=1),
            layer="below",
        )
    )

    # -----------------------------------------------------------------------
    # Lifeline header boxes (Harness / Server) -- one pair per side
    # -----------------------------------------------------------------------
    NODE_W, NODE_H = 0.13, 0.045
    HEADER_Y = LIFELINE_TOP + 0.015

    # Standard headers
    shapes.append(
        _node_rect(
            LIFELINES["std_harness"],
            HEADER_Y,
            NODE_W,
            NODE_H,
            std_node_fill,
            std_node_stroke,
        )
    )
    annotations.append(
        _label(
            LIFELINES["std_harness"],
            HEADER_Y,
            "Harness",
            color=text_secondary,
            size=13,
            bold=True,
        )
    )
    shapes.append(
        _node_rect(
            LIFELINES["std_server"],
            HEADER_Y,
            NODE_W,
            NODE_H,
            std_node_fill,
            std_node_stroke,
        )
    )
    annotations.append(
        _label(
            LIFELINES["std_server"],
            HEADER_Y,
            "Inference Server",
            color=text_secondary,
            size=13,
            bold=True,
        )
    )

    # Dynamo headers
    shapes.append(
        _node_rect(
            LIFELINES["dyn_harness"],
            HEADER_Y,
            NODE_W,
            NODE_H,
            dyn_node_fill,
            std_node_stroke,
        )
    )
    annotations.append(
        _label(
            LIFELINES["dyn_harness"],
            HEADER_Y,
            "Harness",
            color=text_primary,
            size=13,
            bold=True,
        )
    )
    shapes.append(
        _node_rect(
            LIFELINES["dyn_server"],
            HEADER_Y,
            NODE_W,
            NODE_H,
            dyn_node_fill,
            dyn_node_stroke,
            stroke_w=2.0,
        )
    )
    annotations.append(
        _label(
            LIFELINES["dyn_server"],
            HEADER_Y,
            "Dynamo",
            color=text_primary,
            size=13,
            bold=True,
        )
    )

    # -----------------------------------------------------------------------
    # Turn dividers (faint horizontal lines between turns) and turn labels
    # -----------------------------------------------------------------------
    for i, turn in enumerate(TURN_Y, start=1):
        # Faint divider above each turn (except above turn 1)
        if i > 1:
            divider_y = (TURN_Y[i - 2]["resp"] + turn["req"]) / 2
            shapes.append(
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=0.02,
                    x1=0.98,
                    y0=divider_y,
                    y1=divider_y,
                    line=dict(color="#1a1a1a", width=1),
                    layer="below",
                )
            )
        # Turn label in the left margin
        annotations.append(
            _label(
                0.005,
                (turn["req"] + turn["resp"]) / 2,
                f"Turn {i}",
                color=text_muted,
                size=11,
                bold=True,
                xanchor="left",
                yanchor="middle",
            )
        )

    # -----------------------------------------------------------------------
    # Turn 1 (cold start -- both sides do the same thing)
    # -----------------------------------------------------------------------
    _draw_turn(
        annotations,
        shapes,
        harness_x=LIFELINES["std_harness"],
        server_x=LIFELINES["std_server"],
        req_y=TURN_Y[0]["req"],
        resp_y=TURN_Y[0]["resp"],
        request_text="prompt #1  [52K tok]",
        request_text_mono=True,
        request_accent=None,
        accent_color=coral,
        response_text="text + <tool_call> markup",
        response_text_mono=True,
        response_accent=None,
        base_color=std_label_color,
        arrow_color=std_arrow_color,
        arrow_width=2.0,
    )
    _draw_turn(
        annotations,
        shapes,
        harness_x=LIFELINES["dyn_harness"],
        server_x=LIFELINES["dyn_server"],
        req_y=TURN_Y[0]["req"],
        resp_y=TURN_Y[0]["resp"],
        request_text="prompt #1  [52K tok, stable prefix]",
        request_text_mono=True,
        request_accent="stable prefix",
        accent_color=dyn_accent,
        response_text="event: tool_call_dispatch",
        response_text_mono=True,
        response_accent="tool_call_dispatch",
        base_color=dyn_label_color,
        arrow_color=dyn_arrow_color,
        arrow_width=2.5,
    )

    # -----------------------------------------------------------------------
    # Turn 2 (the divergence appears)
    # -----------------------------------------------------------------------
    _draw_turn(
        annotations,
        shapes,
        harness_x=LIFELINES["std_harness"],
        server_x=LIFELINES["std_server"],
        req_y=TURN_Y[1]["req"],
        resp_y=TURN_Y[1]["resp"],
        request_text="prompt #2  [53K tok, REBUILT]",
        request_text_mono=True,
        request_accent="REBUILT",
        accent_color=coral,
        response_text="text + <tool_call> markup  (re-parse)",
        response_text_mono=False,
        response_accent="re-parse",
        base_color=std_label_color,
        arrow_color=std_arrow_color,
        arrow_width=2.0,
    )
    _draw_turn(
        annotations,
        shapes,
        harness_x=LIFELINES["dyn_harness"],
        server_x=LIFELINES["dyn_server"],
        req_y=TURN_Y[1]["req"],
        resp_y=TURN_Y[1]["resp"],
        request_text="+1K tok delta  [prefix cached]",
        request_text_mono=True,
        request_accent="cached",
        accent_color=dyn_accent,
        response_text="event: tool_call_dispatch",
        response_text_mono=True,
        response_accent="tool_call_dispatch",
        base_color=dyn_label_color,
        arrow_color=dyn_arrow_color,
        arrow_width=2.5,
    )

    # -----------------------------------------------------------------------
    # Continuation marker -- "..." below the last turn to suggest the
    # loop runs on for hundreds of turns per session.
    # -----------------------------------------------------------------------
    annotations.append(
        _label(
            (LIFELINES["std_harness"] + LIFELINES["std_server"]) / 2,
            TURN_Y[-1]["resp"] - 0.06,
            "...repeats for hundreds of turns per session...",
            color=text_muted,
            size=11,
            italic=True,
        )
    )
    annotations.append(
        _label(
            (LIFELINES["dyn_harness"] + LIFELINES["dyn_server"]) / 2,
            TURN_Y[-1]["resp"] - 0.06,
            "...repeats with cache compounding the win...",
            color=text_muted,
            size=11,
            italic=True,
        )
    )

    # -----------------------------------------------------------------------
    # Section captions at bottom of each side. Parallel structure: both
    # regular weight, same size; color is the only delta (coral = pain,
    # green = gain).
    # -----------------------------------------------------------------------
    annotations.append(
        _label(
            std_center,
            0.085,
            "every turn re-sends the full prompt and re-parses the reply",
            color=coral,
            size=12,
        )
    )
    annotations.append(
        _label(
            dyn_center,
            0.085,
            "prefix stays warm; tool calls arrive parsed",
            color=dynamo_green,
            size=12,
        )
    )

    # -----------------------------------------------------------------------
    # Title block (top-left)
    # -----------------------------------------------------------------------
    annotations.append(
        _label(
            0.005,
            0.985,
            "Two turns of the agent loop, side by side",
            color=text_primary,
            size=22,
            bold=True,
            xanchor="left",
            yanchor="top",
        )
    )
    annotations.append(
        _label(
            0.005,
            0.935,
            "Same harness, same model. The wire format determines whether each turn is fresh work or a continuation.",
            color=text_secondary,
            size=13,
            xanchor="left",
            yanchor="top",
        )
    )

    # -----------------------------------------------------------------------
    # Compose figure
    # -----------------------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="markers", marker=dict(opacity=0))
    )

    fig.update_layout(
        width=1280,
        height=720,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(visible=False, range=[0, 1], fixedrange=True),
        yaxis=dict(visible=False, range=[0, 1], fixedrange=True),
    )

    images_dir = Path(__file__).resolve().parents[1] / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    png_path = images_dir / "fig-1-agent-loop.png"
    svg_path = images_dir / "fig-1-agent-loop.svg"
    fig.write_image(str(png_path), scale=2)
    fig.write_image(str(svg_path))
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
