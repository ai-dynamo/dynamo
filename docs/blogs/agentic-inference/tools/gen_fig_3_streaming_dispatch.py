#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Figure 3 - Tool dispatch timing on a single turn (two-lane version).

Two stacked token-stream lanes for the same model turn:

  * Top lane (Standard server, coral): the harness only sees a tool
    call after parsing the buffer at end-of-stream, so dispatch lands
    at the right edge of the lane.
  * Bottom lane (Dynamo, green): a structured tool_call_dispatch SSE
    event fires the moment the server-side parser closes the call,
    mid-stream. The rest of the stream keeps flowing, so the tool
    runs in parallel.

A diagonal white connector spans the two dispatch markers (top-right
to bottom-mid) and is labeled with Δt - the gap that Dynamo earns and
the standard server wastes.
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
LANE_LEFT = 0.10
LANE_RIGHT = 0.95

# Standard lane on top, Dynamo lane on bottom.
LANE_Y = {
    "standard": 0.66,
    "dynamo": 0.32,
}
LANE_HEIGHT = 0.060  # half-height of each lane bar

NUM_TICKS = 36

# Where the tool-call PAYLOAD finishes streaming (close-brace of the JSON).
# Same x-position on both lanes - it's the same model output.
TOOL_CALL_END_X = 0.50

# Dynamo dispatch fires the moment the server-side parser closes the call.
DYN_DISPATCH_X = TOOL_CALL_END_X

# Standard dispatch fires after end-of-stream + the harness-side parse pass.
STD_DISPATCH_X = LANE_RIGHT


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


def _arrow(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    width: float = 2.0,
    opacity: float = 1.0,
    arrowhead: int = 3,
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
        arrowhead=arrowhead,
        arrowsize=1.2,
        arrowwidth=width,
        arrowcolor=color,
        opacity=opacity,
        text="",
        standoff=2,
        startstandoff=2,
    )


def _lane_bar(
    cy: float,
    x0: float,
    x1: float,
    fill: str,
    stroke: str,
    stroke_w: float = 1.0,
) -> dict:
    return dict(
        type="rect",
        xref="x",
        yref="y",
        x0=x0,
        x1=x1,
        y0=cy - LANE_HEIGHT,
        y1=cy + LANE_HEIGHT,
        fillcolor=fill,
        line=dict(color=stroke, width=stroke_w),
        layer="below",
    )


def _tick(cy: float, x: float, color: str, opacity: float = 0.6) -> dict:
    return dict(
        type="line",
        xref="x",
        yref="y",
        x0=x,
        x1=x,
        y0=cy - LANE_HEIGHT * 0.55,
        y1=cy + LANE_HEIGHT * 0.55,
        line=dict(color=color, width=1),
        opacity=opacity,
        layer="above",
    )


def _vbar(
    cy_top: float,
    cy_bot: float,
    x: float,
    color: str,
    width: float = 2.5,
    dash: str | None = None,
) -> dict:
    line_kwargs: dict = dict(color=color, width=width)
    if dash:
        line_kwargs["dash"] = dash
    return dict(
        type="line",
        xref="x",
        yref="y",
        x0=x,
        x1=x,
        y0=cy_bot,
        y1=cy_top,
        line=line_kwargs,
        layer="above",
    )


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _draw_lane(
    shapes: list,
    annotations: list,
    *,
    cy: float,
    bar_fill: str,
    bar_stroke: str,
    tick_color: str,
    side_label: str,
    side_color: str,
    side_underline_color: str,
) -> None:
    """Draw the lane bar, ticks, payload band, and section banner above."""
    # Section banner above lane (matches Fig 1's side-title style).
    # Left-anchored so the banner stays clear of mid-lane dispatch markers.
    title_y = cy + LANE_HEIGHT + 0.082
    rule_y = cy + LANE_HEIGHT + 0.058
    annotations.append(
        _label(
            LANE_LEFT,
            title_y,
            side_label,
            color=side_color,
            size=18,
            bold=True,
            xanchor="left",
            yanchor="bottom",
        )
    )
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="y",
            x0=LANE_LEFT - 0.02,
            x1=LANE_RIGHT + 0.02,
            y0=rule_y,
            y1=rule_y,
            line=dict(color=side_underline_color, width=1),
            layer="below",
        )
    )

    # Lane bar
    shapes.append(_lane_bar(cy, LANE_LEFT, LANE_RIGHT, bar_fill, bar_stroke))

    # Token ticks
    for i in range(NUM_TICKS):
        tx = LANE_LEFT + (LANE_RIGHT - LANE_LEFT) * (i + 0.5) / NUM_TICKS
        shapes.append(_tick(cy, tx, tick_color, opacity=0.55))

    # Tool-call payload band (neutral gray)
    payload_x0 = LANE_LEFT + (TOOL_CALL_END_X - LANE_LEFT) * 0.45
    payload_x1 = TOOL_CALL_END_X
    payload_fill = f"rgba{(*hex_to_rgb('#cccccc'), 0.10)}"
    payload_stroke = "#666666"
    shapes.append(
        dict(
            type="rect",
            xref="x",
            yref="y",
            x0=payload_x0,
            x1=payload_x1,
            y0=cy - LANE_HEIGHT,
            y1=cy + LANE_HEIGHT,
            fillcolor=payload_fill,
            line=dict(color=payload_stroke, width=1, dash="dot"),
            layer="above",
        )
    )


def _dispatch_marker(
    shapes: list,
    annotations: list,
    *,
    x: float,
    cy: float,
    color: str,
    label: str,
    label_mono: bool,
    label_above: bool,
    label_xanchor: str = "center",
    label_xshift: float = 0.0,
    line_extend_top: float = 0.04,
    line_extend_bot: float = 0.04,
) -> None:
    """Vertical dispatch line + dot + a single name label above OR below.

    ``line_extend_top`` / ``line_extend_bot`` control how far the colored
    dispatch line extends past the lane edge in each direction. Default
    is symmetric. Shorten when an external bracket / banner is in the way.
    """
    shapes.append(
        _vbar(
            cy + LANE_HEIGHT + line_extend_top,
            cy - LANE_HEIGHT - line_extend_bot,
            x,
            color=color,
            width=2.5,
        )
    )
    shapes.append(
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=x - 0.007,
            x1=x + 0.007,
            y0=cy - 0.018,
            y1=cy + 0.018,
            fillcolor=color,
            line=dict(color=color, width=0),
            layer="above",
        )
    )
    if label_above:
        annotations.append(
            _label(
                x + label_xshift,
                cy + LANE_HEIGHT + 0.05,
                label,
                color=color,
                size=13,
                bold=True,
                xanchor=label_xanchor,
                yanchor="bottom",
                family=MONO if label_mono else SANS,
            )
        )
    else:
        annotations.append(
            _label(
                x + label_xshift,
                cy - LANE_HEIGHT - 0.05,
                label,
                color=color,
                size=13,
                bold=True,
                xanchor=label_xanchor,
                yanchor="top",
                family=MONO if label_mono else SANS,
            )
        )


def main() -> None:
    tokens = load_tokens()
    bg = tokens["colors"]["background"]["primary"]
    dynamo_green = tokens["colors"]["accent"]["dynamo_green"]
    coral = tokens["colors"]["accent"]["coral"]
    text_primary = tokens["colors"]["text"]["primary"]
    text_secondary = tokens["colors"]["text"]["secondary"]

    bar_fill = "#181818"
    bar_stroke = "#3a3a3a"
    tick_color = "#5d5d5d"

    shapes: list[dict] = []
    annotations: list[dict] = []

    # -----------------------------------------------------------------------
    # Standard lane (top)
    # -----------------------------------------------------------------------
    _draw_lane(
        shapes,
        annotations,
        cy=LANE_Y["standard"],
        bar_fill=bar_fill,
        bar_stroke=bar_stroke,
        tick_color=tick_color,
        side_label="Standard Server",
        side_color=text_secondary,
        side_underline_color="#3a3a3a",
    )
    _dispatch_marker(
        shapes,
        annotations,
        x=STD_DISPATCH_X,
        cy=LANE_Y["standard"],
        color=coral,
        label="dispatch tool",
        label_mono=False,
        label_above=False,
        label_xanchor="right",
        label_xshift=-0.010,
        line_extend_bot=0.005,  # short - dashed bracket carries downward
    )
    # Payload caption above the Standard payload band
    payload_mid_x = LANE_LEFT + (TOOL_CALL_END_X - LANE_LEFT) * 0.725
    annotations.append(
        _label(
            payload_mid_x,
            LANE_Y["standard"] + LANE_HEIGHT + 0.008,
            "tool-call payload",
            color="#bdbdbd",
            size=11,
            bold=True,
            yanchor="bottom",
            family=MONO,
        )
    )

    # -----------------------------------------------------------------------
    # Dynamo lane (bottom)
    # -----------------------------------------------------------------------
    _draw_lane(
        shapes,
        annotations,
        cy=LANE_Y["dynamo"],
        bar_fill=bar_fill,
        bar_stroke=bar_stroke,
        tick_color=tick_color,
        side_label="Dynamo",
        side_color=dynamo_green,
        side_underline_color=dynamo_green,
    )
    # Parallel-work band: faint green tint on the Dynamo lane from the
    # dispatch point to end-of-stream. This is what Dynamo buys you.
    parallel_fill = f"rgba{(*hex_to_rgb(dynamo_green), 0.12)}"
    parallel_stroke = f"rgba{(*hex_to_rgb(dynamo_green), 0.45)}"
    shapes.append(
        dict(
            type="rect",
            xref="x",
            yref="y",
            x0=DYN_DISPATCH_X,
            x1=STD_DISPATCH_X,
            y0=LANE_Y["dynamo"] - LANE_HEIGHT,
            y1=LANE_Y["dynamo"] + LANE_HEIGHT,
            fillcolor=parallel_fill,
            line=dict(color=parallel_stroke, width=1, dash="dot"),
            layer="above",
        )
    )
    annotations.append(
        _label(
            (DYN_DISPATCH_X + STD_DISPATCH_X) / 2,
            LANE_Y["dynamo"],
            "tool + stream run in parallel",
            color="#9ed667",
            size=12,
            bold=True,
            italic=True,
            yanchor="middle",
            family=SANS,
        )
    )
    _dispatch_marker(
        shapes,
        annotations,
        x=DYN_DISPATCH_X,
        cy=LANE_Y["dynamo"],
        color=dynamo_green,
        label="event: tool_call_dispatch",
        label_mono=True,
        label_above=False,
        line_extend_top=0.005,  # short - dashed bracket carries upward
    )
    # Payload caption above the Dynamo payload band
    annotations.append(
        _label(
            payload_mid_x,
            LANE_Y["dynamo"] + LANE_HEIGHT + 0.008,
            "tool-call payload",
            color="#bdbdbd",
            size=11,
            bold=True,
            yanchor="bottom",
            family=MONO,
        )
    )

    # -----------------------------------------------------------------------
    # Δt connector: a short horizontal white line in the gap between lanes,
    # with two vertical drops up to the Standard dispatch and down to the
    # Dynamo dispatch. Reads as a measured span linking the two markers.
    # -----------------------------------------------------------------------
    std_anchor_y = LANE_Y["standard"] - LANE_HEIGHT - 0.005
    dyn_anchor_y = LANE_Y["dynamo"] + LANE_HEIGHT + 0.005
    # Bridge sits closer to the Standard lane so the Dynamo section
    # banner has clear air above its lane.
    bridge_y = (std_anchor_y + dyn_anchor_y) / 2

    # Dashed verticals: short stubs near the bridge that hint at the
    # vertical drop without redundantly tracing the colored dispatch lines.
    gap = 0.012  # gap between dashed stub and the horizontal Δt span
    stub = 0.055  # stub length above/below the bridge

    # Standard dashed vertical (above bridge)
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="y",
            x0=STD_DISPATCH_X,
            x1=STD_DISPATCH_X,
            y0=bridge_y + gap,
            y1=bridge_y + gap + stub,
            line=dict(color="#cdcdcd", width=1.2, dash="dash"),
            layer="above",
        )
    )
    # Dynamo dashed vertical (below bridge)
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="y",
            x0=DYN_DISPATCH_X,
            x1=DYN_DISPATCH_X,
            y0=bridge_y - gap,
            y1=bridge_y - gap - stub,
            line=dict(color="#cdcdcd", width=1.2, dash="dash"),
            layer="above",
        )
    )

    # Horizontal Δt span from Dynamo x to Standard x. Sits free of the
    # dashed verticals (gap on each side keeps them separated).
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="y",
            x0=DYN_DISPATCH_X,
            x1=STD_DISPATCH_X,
            y0=bridge_y,
            y1=bridge_y,
            line=dict(color="#ffffff", width=1.5),
            layer="above",
        )
    )
    # Δt label sits on the bridge with a bg fill so the line passes
    # behind it cleanly without crowding the text.
    delta_mid_x = (STD_DISPATCH_X + DYN_DISPATCH_X) / 2
    annotations.append(
        dict(
            x=delta_mid_x,
            y=bridge_y,
            xref="x",
            yref="y",
            text="<b>\u0394t</b>",
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            font=dict(family=MONO, size=18, color=text_primary),
            bgcolor=bg,
            borderpad=4,
        )
    )

    # (Standard subnote removed: deck already names the contrast and the
    # parallel-work band on the Dynamo lane carries the visual message.)

    # -----------------------------------------------------------------------
    # Time axis under the bottom lane
    # -----------------------------------------------------------------------
    axis_y = 0.13
    axis_color = "#bdbdbd"
    annotations.append(
        _arrow(
            LANE_LEFT,
            axis_y,
            LANE_RIGHT + 0.005,
            axis_y,
            color=axis_color,
            width=2.0,
        )
    )
    # Endpoint labels only - the arrow head on the right and the natural
    # axis origin on the left orient the reader without competing ticks.
    for tx, lab, lab_xanchor in [
        (LANE_LEFT, "t = 0", "left"),
        (LANE_RIGHT, "end of stream", "right"),
    ]:
        annotations.append(
            _label(
                tx,
                axis_y - 0.022,
                lab,
                color=text_secondary,
                size=12,
                family=MONO,
                xanchor=lab_xanchor,
                yanchor="top",
            )
        )
    # Drop the redundant 'time' label - axis arrow + t=0 / end of stream
    # ticks already orient the reader.

    # -----------------------------------------------------------------------
    # Title block
    # -----------------------------------------------------------------------
    annotations.append(
        _label(
            0.005,
            0.985,
            "Tool dispatch timing on a single turn",
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
            0.940,
            "\u0394t = time saved per tool call. The standard server waits for end of stream; "
            "Dynamo dispatches the moment the call is parsed.",
            color=text_secondary,
            size=13,
            xanchor="left",
            yanchor="top",
        )
    )

    # -----------------------------------------------------------------------
    # Compose
    # -----------------------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="markers", marker=dict(opacity=0))
    )
    fig.update_layout(
        width=1280,
        height=760,
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
    png_path = images_dir / "fig-3-streaming-dispatch-timeline.png"
    svg_path = images_dir / "fig-3-streaming-dispatch-timeline.svg"
    fig.write_image(str(png_path), scale=2)
    fig.write_image(str(svg_path))
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
