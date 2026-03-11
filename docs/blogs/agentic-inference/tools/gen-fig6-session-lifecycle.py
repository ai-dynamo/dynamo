#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate Figure 7 -- "Cache Fate: Session Lifecycle" for the agentic-inference blog.

Dual-track timeline showing a Claude Code session lifecycle and its cache
implications.  Top track maps agent activity (lead agent, explore subagent,
team of four); bottom track shows the corresponding cache state as blocks
that transition from live to dead (ghost outlines).

Output: ../images/fig-7-session-lifecycle.svg

PNG variant (2x resolution):
    rsvg-convert -z 2 -o fig-7-session-lifecycle.png ../images/fig-7-session-lifecycle.svg
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Design tokens (aligned with design_tokens.yaml)
# ---------------------------------------------------------------------------
BG = "#0a0a0a"
GREEN = "#76b900"
GREEN_FILL = "#2a4a10"
CORAL = "#b04040"
RED_FILL = "#2a1010"
MUTED = "#767676"
TEXT_LIGHT = "#e0e0e0"
TEXT_MID = "#cdcdcd"
GHOST_STROKE = "#3a3a3a"
FONT = "Arial, Helvetica, sans-serif"

W, H = 1000, 550

# Time-axis geometry
CHART_LEFT = 160
CHART_RIGHT = 970
CHART_WIDTH = CHART_RIGHT - CHART_LEFT
T_MAX = 60

# Swim-lane geometry (y-center of each lane)
BAR_H = 28
LANE_Y = {1: 105, 2: 165, 3: 225}

# Cache-block geometry
BLK = 12
BLK_GAP = 2
CACHE_ROW_Y = {1: 310, 2: 355, 3: 390}


def tx(t: float) -> float:
    """Map a time value (0-60) to an x-pixel coordinate."""
    return CHART_LEFT + t * CHART_WIDTH / T_MAX


# ---------------------------------------------------------------------------
# SVG primitives (matching gen_priority_dispatch.py conventions)
# ---------------------------------------------------------------------------


def _el(
    tag: str,
    parent: ET.Element | None = None,
    text: str | None = None,
    **attrs: str,
) -> ET.Element:
    elem = (
        ET.SubElement(parent, tag, **attrs)
        if parent is not None
        else ET.Element(tag, **attrs)
    )
    if text is not None:
        elem.text = text
    return elem


def _text(
    parent: ET.Element,
    x: float,
    y: float,
    label: str,
    *,
    fill: str = MUTED,
    size: float = 12,
    anchor: str = "middle",
    weight: str = "normal",
    style: str = "normal",
    spacing: str | None = None,
    dy: str = "0",
) -> ET.Element:
    attrs: dict[str, str] = {
        "x": str(x),
        "y": str(y),
        "fill": fill,
        "font-size": str(size),
        "font-family": FONT,
        "text-anchor": anchor,
        "font-weight": weight,
        "font-style": style,
        "dy": dy,
    }
    if spacing:
        attrs["letter-spacing"] = spacing
    return _el("text", parent, label, **attrs)


def _rect(
    parent: ET.Element,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = BG,
    stroke: str = "none",
    sw: float = 1,
    rx: float = 0,
    dash: str | None = None,
) -> ET.Element:
    attrs: dict[str, str] = {
        "x": str(x),
        "y": str(y),
        "width": str(w),
        "height": str(h),
        "fill": fill,
        "stroke": stroke,
        "rx": str(rx),
        "ry": str(rx),
    }
    if sw != 1:
        attrs["stroke-width"] = str(sw)
    if dash:
        attrs["stroke-dasharray"] = dash
    return _el("rect", parent, **attrs)


def _line(
    parent: ET.Element,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = GHOST_STROKE,
    sw: float = 1,
    dash: str | None = None,
) -> ET.Element:
    attrs: dict[str, str] = {
        "x1": str(x1),
        "y1": str(y1),
        "x2": str(x2),
        "y2": str(y2),
        "stroke": stroke,
    }
    if sw != 1:
        attrs["stroke-width"] = str(sw)
    if dash:
        attrs["stroke-dasharray"] = dash
    return _el("line", parent, **attrs)


# ---------------------------------------------------------------------------
# Composite drawing helpers
# ---------------------------------------------------------------------------


def _bar(
    parent: ET.Element,
    t_start: float,
    t_end: float,
    lane: int,
    *,
    fill: str,
    stroke: str,
    sw: float = 1.5,
    label: str | None = None,
    label_above: str | None = None,
    label_color: str = TEXT_LIGHT,
    label_size: float = 11,
    above_color: str = CORAL,
    above_size: float = 10,
) -> None:
    """Draw a horizontal activity bar inside a swim lane."""
    x0, x1 = tx(t_start), tx(t_end)
    y_top = LANE_Y[lane] - BAR_H / 2
    _rect(parent, x0, y_top, x1 - x0, BAR_H, fill=fill, stroke=stroke, sw=sw, rx=4)
    if label:
        _text(
            parent,
            (x0 + x1) / 2,
            LANE_Y[lane] + 4,
            label,
            fill=label_color,
            size=label_size,
        )
    if label_above:
        _text(
            parent,
            (x0 + x1) / 2,
            y_top - 6,
            label_above,
            fill=above_color,
            size=above_size,
        )


def _cache_blocks(
    parent: ET.Element,
    t_start: float,
    count: int,
    row_y: float,
    rows: int = 1,
    *,
    ghost: bool = False,
) -> None:
    """Draw a cluster of cache-block squares starting at *t_start*."""
    x0 = tx(t_start)
    cols = count // rows
    for r in range(rows):
        for c in range(cols):
            bx = x0 + c * (BLK + BLK_GAP)
            by = row_y + r * (BLK + BLK_GAP)
            if ghost:
                _rect(
                    parent,
                    bx,
                    by,
                    BLK,
                    BLK,
                    fill=BG,
                    stroke=GHOST_STROKE,
                    rx=2,
                    dash="3,3",
                )
            else:
                _rect(parent, bx, by, BLK, BLK, fill=GREEN_FILL, stroke=GREEN, rx=2)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_title(svg: ET.Element) -> None:
    _text(
        svg,
        W / 2,
        28,
        "CACHE FATE \u2014 SESSION LIFECYCLE",
        fill=TEXT_LIGHT,
        size=16,
        weight="bold",
        spacing="0.08em",
    )


def _build_legend(svg: ET.Element) -> None:
    lx, ly = 830, 50
    gap = 22

    _rect(svg, lx, ly, 12, 12, fill=GREEN_FILL, stroke=GREEN, rx=2)
    _text(svg, lx + 18, ly + 11, "Active", fill=TEXT_MID, size=10, anchor="start")

    _rect(svg, lx, ly + gap, 12, 12, fill=RED_FILL, stroke=CORAL, rx=2)
    _text(
        svg,
        lx + 18,
        ly + gap + 11,
        "Critical event",
        fill=TEXT_MID,
        size=10,
        anchor="start",
    )

    _rect(svg, lx, ly + 2 * gap, 12, 12, fill=BG, stroke=GHOST_STROKE, rx=2, dash="3,3")
    _text(
        svg,
        lx + 18,
        ly + 2 * gap + 11,
        "Dead cache",
        fill=TEXT_MID,
        size=10,
        anchor="start",
    )


def _build_top_track(svg: ET.Element) -> None:
    _text(
        svg,
        CHART_LEFT,
        58,
        "AGENT ACTIVITY",
        fill=MUTED,
        size=11,
        spacing="0.08em",
        anchor="start",
    )

    # Lane labels (right-aligned to the left of the chart area)
    for lane, name in ((1, "Lead Agent"), (2, "Explore Subagent"), (3, "Team of 4")):
        _text(
            svg,
            CHART_LEFT - 14,
            LANE_Y[lane] + 5,
            name,
            fill=TEXT_MID,
            size=13,
            anchor="end",
        )

    # --- Lane 1: Lead Agent ---
    _bar(
        svg,
        0,
        35,
        1,
        fill=GREEN_FILL,
        stroke=GREEN,
        label="Cold start \u2192 Turns 2\u201310",
    )
    _bar(
        svg,
        35,
        38,
        1,
        fill=RED_FILL,
        stroke=CORAL,
        sw=1,
        label_above="Summarize 175K\u219240K",
    )
    _bar(
        svg,
        38,
        60,
        1,
        fill=GREEN_FILL,
        stroke=GREEN,
        label="Turns 11\u201320, new prefix",
    )

    # --- Lane 2: Explore Subagent ---
    _bar(
        svg,
        10,
        18,
        2,
        fill=GREEN_FILL,
        stroke=GREEN,
        label="Spawn \u2192 Turns 2\u20133",
    )
    _bar(svg, 18, 19, 2, fill=RED_FILL, stroke=CORAL, sw=1, label_above="End")

    # --- Lane 3: Team of 4 ---
    _bar(svg, 22, 34, 3, fill=GREEN_FILL, stroke=GREEN, label="Teammates A\u2013D")
    _bar(svg, 34, 35, 3, fill=RED_FILL, stroke=CORAL, sw=1, label_above="End")

    # Dashed vertical lines at termination events
    _line(svg, tx(35), 75, tx(35), 430, stroke=GHOST_STROKE, dash="4,4")
    _line(svg, tx(19), 148, tx(19), 380, stroke=GHOST_STROKE, dash="4,4")


def _build_bottom_track(svg: ET.Element) -> None:
    _text(
        svg,
        CHART_LEFT,
        298,
        "CACHE STATE",
        fill=MUTED,
        size=11,
        spacing="0.08em",
        anchor="start",
    )

    # Lead Agent cache: live -> ghost -> new live
    _cache_blocks(svg, 5, 12, CACHE_ROW_Y[1], rows=2)
    _cache_blocks(svg, 36, 12, CACHE_ROW_Y[1], rows=2, ghost=True)
    _cache_blocks(svg, 45, 8, CACHE_ROW_Y[1], rows=2)

    # Explore Subagent cache: live -> ghost
    _cache_blocks(svg, 11, 4, CACHE_ROW_Y[2])
    _cache_blocks(svg, 20, 4, CACHE_ROW_Y[2], ghost=True)

    # Team of 4 cache: live -> ghost
    _cache_blocks(svg, 23, 6, CACHE_ROW_Y[3])
    _cache_blocks(svg, 36, 6, CACHE_ROW_Y[3], ghost=True)

    # Annotation
    _text(
        svg,
        tx(25),
        438,
        "Cache holds dead blocks \u2014 no lifecycle signal",
        fill=MUTED,
        size=11,
        style="italic",
    )


def _build_x_axis(svg: ET.Element) -> None:
    axis_y = 458
    _line(svg, CHART_LEFT, axis_y, CHART_RIGHT, axis_y, stroke=GHOST_STROKE)

    for t in range(0, T_MAX + 1, 10):
        x = tx(t)
        _line(svg, x, axis_y, x, axis_y + 6, stroke=GHOST_STROKE)
        _text(svg, x, axis_y + 18, str(t), fill=MUTED, size=10)

    _text(
        svg,
        (CHART_LEFT + CHART_RIGHT) / 2,
        axis_y + 35,
        "Time (relative)",
        fill=MUTED,
        size=11,
    )


def apply_outer_frame(svg: ET.Element) -> None:
    """1 px NVIDIA-green border matching flash-indexer convention."""
    _rect(svg, 0, 0, W, H, fill="none", stroke=GREEN, rx=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate() -> str:
    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        viewBox=f"0 0 {W} {H}",
        width=str(W),
        height=str(H),
    )
    _rect(svg, 0, 0, W, H, fill=BG, stroke="none")

    _build_title(svg)
    _build_legend(svg)
    _build_top_track(svg)
    _build_bottom_track(svg)
    _build_x_axis(svg)
    apply_outer_frame(svg)

    out = (
        Path(__file__).resolve().parent
        / ".."
        / "images"
        / "fig-7-session-lifecycle.svg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode", xml_declaration=True)
    return str(out.resolve())


if __name__ == "__main__":
    path = generate()
    print(f"wrote {path}")
