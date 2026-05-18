#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Hand-craft fig-2 architecture map as SVG so we get pixel-perfect central
axis alignment that D2 + elk can't guarantee when the replay-loop edge is
on the right side.

Layout (vertical, centered):
    Load Driver  (top, centered on the axis)
         |
       events
         v
    Multi Engine Simulation  (centered)
       [ Single Engine Sim x 4 ]  [ Router ]  [ Planner ]
         |
       metrics
         v
    Trace Collector  (bottom, centered on the axis)

The replay edge exits Trace Collector's right side, goes up the right
side of Replay Harness, and re-enters Load Driver's right side. Squared
right-angle path, NV-green stroke.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- Canvas geometry ---
W = 1240
TITLE_BAND_H = 160
DIAG_H = 700
H = TITLE_BAND_H + DIAG_H
AXIS_X = W / 2

# --- Palette ---
BLACK = "#000000"
TEXT_PRIMARY = "#ffffff"
TEXT_MUTED = "#767676"
NV_GREEN = "#76b900"
AMBER = "#c08050"
AMBER_FILL = "#201810"
EMERALD = "#3a7a70"
EMERALD_FILL = "#142025"
FLUORITE = "#fac200"
AMETHYST = "#9560cc"
AMETHYST_FILL = "#1a1428"
NEUTRAL_STROKE = "#5d5d5d"
NEUTRAL_FILL = "#0a0a0a"
SES_STROKE = "#636363"
SES_FILL = "#1a1a1a"


def rect(
    x: float, y: float, w: float, h: float,
    fill: str, stroke: str, stroke_w: float = 2,
    rx: float = 0,
    stroke_dasharray: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{stroke_dasharray}"' if stroke_dasharray else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}" '
        f'rx="{rx}" ry="{rx}"{dash_attr}/>'
    )


def text(
    x: float, y: float, content: str,
    family: str = "Geist, Inter, Helvetica Neue, sans-serif",
    size: int = 18, color: str = TEXT_PRIMARY,
    weight: str = "bold", anchor: str = "middle",
) -> str:
    return (
        f'<text x="{x}" y="{y}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{color}" text-anchor="{anchor}" '
        f'dominant-baseline="middle">{content}</text>'
    )


def cylinder(
    cx: float, cy: float, w: float, h: float,
    fill: str, stroke: str, stroke_w: float = 2,
) -> str:
    """Database cylinder: rect body with elliptical top + bottom."""
    half_w = w / 2
    ellipse_h = 12  # vertical radius of top/bottom ellipses
    top_y = cy - h / 2 + ellipse_h
    bot_y = cy + h / 2 - ellipse_h
    return (
        # Body rect (no top/bottom stroke -- ellipses give them)
        f'<rect x="{cx - half_w}" y="{top_y}" width="{w}" height="{bot_y - top_y}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
        # Top ellipse
        f'<ellipse cx="{cx}" cy="{top_y}" rx="{half_w}" ry="{ellipse_h}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
        # Bottom ellipse (only bottom half visible, so paint full)
        f'<ellipse cx="{cx}" cy="{bot_y}" rx="{half_w}" ry="{ellipse_h}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
    )


def queue_shape(
    cx: float, cy: float, w: float, h: float,
    fill: str, stroke: str, stroke_w: float = 2,
) -> str:
    """Queue: rect with a vertical line on the right marking the queued slot."""
    half_w = w / 2
    half_h = h / 2
    notch_x = cx + half_w - 14
    return (
        f'<rect x="{cx - half_w}" y="{cy - half_h}" width="{w}" height="{h}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
        f'<line x1="{notch_x}" y1="{cy - half_h}" x2="{notch_x}" y2="{cy + half_h}" '
        f'stroke="{stroke}" stroke-width="{stroke_w}"/>'
    )


def hexagon(
    cx: float, cy: float, r: float,
    fill: str, stroke: str, stroke_w: float = 2,
) -> str:
    """Flat-topped hexagon centered at (cx, cy) with circumradius r."""
    import math
    pts = []
    for i in range(6):
        angle = math.radians(60 * i)
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (
        f'<polygon points="{pts_str}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
    )


def arrow_down(x1: float, y1: float, x2: float, y2: float, color: str, label: str | None = None) -> str:
    """Vertical arrow from (x1, y1) to (x2, y2) with arrowhead at end."""
    pieces = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2 - 6}" '
        f'stroke="{color}" stroke-width="2"/>',
        # Arrowhead triangle
        f'<polygon points="{x2 - 6},{y2 - 8} {x2 + 6},{y2 - 8} {x2},{y2}" '
        f'fill="{color}"/>',
    ]
    if label:
        mid_y = (y1 + y2) / 2
        pieces.append(text(x2 + 10, mid_y, label, size=13, color=color, anchor="start"))
    return "".join(pieces)


def replay_loop(
    src_x: float, src_y: float, dst_x: float, dst_y: float,
    margin_right: float, color: str = NV_GREEN, label: str = "replay",
) -> str:
    """Squared loop from (src_x, src_y) on right side of Trace Collector,
    around the right margin, to (dst_x, dst_y) on right side of Load Driver."""
    bend_x = margin_right
    pieces = [
        # Out right from src
        f'<line x1="{src_x}" y1="{src_y}" x2="{bend_x}" y2="{src_y}" '
        f'stroke="{color}" stroke-width="3"/>',
        # Up the right side
        f'<line x1="{bend_x}" y1="{src_y}" x2="{bend_x}" y2="{dst_y}" '
        f'stroke="{color}" stroke-width="3"/>',
        # Back left into dst, with arrowhead at end
        f'<line x1="{bend_x}" y1="{dst_y}" x2="{dst_x + 8}" y2="{dst_y}" '
        f'stroke="{color}" stroke-width="3"/>',
        f'<polygon points="{dst_x + 10},{dst_y - 6} {dst_x + 10},{dst_y + 6} {dst_x},{dst_y}" '
        f'fill="{color}"/>',
        # Label on the vertical leg
        text(bend_x + 14, (src_y + dst_y) / 2, label, size=14,
             color=color, anchor="start", weight="bold"),
    ]
    return "".join(pieces)


def build_svg() -> str:
    parts: list[str] = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" style="background:{BLACK};">',
        # Background
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="{BLACK}"/>',
        # Title + subtitle. y values match the snug +2 px formula used in the
        # Plotly figures: title_top = 0.04 * H, title_bottom = title_top +
        # font_size, subtitle_top = title_bottom + 2 px. SVG dominant-baseline
        # is "middle" so y = top + font_size/2.
        text(
            0.02 * W, (0.04 * H) + 42 / 2, "Anatomy of a Digital Twin",
            family="Helvetica Neue, HelveticaNeue, sans-serif",
            size=42, weight="300", anchor="start",
        ),
        text(
            0.02 * W, (0.04 * H) + 42 + 2 + 22 / 2,
            "Engine cores, Router, Planner — one simulated clock, one harness.",
            family="Helvetica Neue, HelveticaNeue, sans-serif",
            size=22, weight="300", color=TEXT_MUTED, anchor="start",
        ),
    ]

    # Y offsets within the diagram band.
    Y_BASE = TITLE_BAND_H

    # Replay Harness outer frame. HARNESS_X=80 matches the plot-frame inset
    # used by the Plotly figures (margin l=80), so fig-2's green frame lines
    # up horizontally with the rest of the figure set.
    HARNESS_X = 80
    HARNESS_Y = Y_BASE + 30
    HARNESS_W = W - 2 * HARNESS_X
    HARNESS_H = DIAG_H - 60
    parts.append(rect(HARNESS_X, HARNESS_Y, HARNESS_W, HARNESS_H,
                      fill=BLACK, stroke=NV_GREEN, stroke_w=2))
    parts.append(text(
        HARNESS_X + HARNESS_W / 2, HARNESS_Y + 24, "REPLAY HARNESS",
        size=22, color=TEXT_PRIMARY, weight="bold",
    ))

    # Load Driver -- top center on the axis.
    DRIVER_W, DRIVER_H = 220, 64
    DRIVER_CX = AXIS_X
    DRIVER_CY = HARNESS_Y + 90
    parts.append(queue_shape(DRIVER_CX, DRIVER_CY, DRIVER_W, DRIVER_H,
                             fill=NEUTRAL_FILL, stroke=NEUTRAL_STROKE))
    parts.append(text(DRIVER_CX - 8, DRIVER_CY, "Load Driver", size=18))

    # MES container -- centered.
    MES_W, MES_H = 880, 220
    MES_X = AXIS_X - MES_W / 2
    MES_Y = DRIVER_CY + DRIVER_H / 2 + 70
    parts.append(rect(MES_X, MES_Y, MES_W, MES_H,
                      fill=NEUTRAL_FILL, stroke=NEUTRAL_STROKE, stroke_w=2))
    parts.append(text(MES_X + MES_W / 2, MES_Y + 22, "MULTI ENGINE SIMULATION",
                      size=20, weight="bold"))

    # Inside MES: SES (stacked) -- Router (hexagon) -- Planner -- KVBM.
    # KVBM is rendered with a dashed stroke to signal "optional component"
    # (Yongming's feedback on PR-9139); emerald palette signals "storage/cache".
    # Load Driver and Trace Collector are I/O harness, not data components, so
    # they share the neutral palette to keep visual emphasis on the engine loop.
    INNER_Y = MES_Y + MES_H / 2 + 20
    SES_CX = MES_X + MES_W * 0.18
    ROUTER_CX = MES_X + MES_W * 0.42
    PLANNER_CX = MES_X + MES_W * 0.62
    KVBM_CX = MES_X + MES_W * 0.84

    # Single Engine Sim x 4 -- main rect plus 2 offset shadows behind.
    SES_W, SES_H = 200, 90
    for dx, dy, alpha in [(14, -14, 1), (7, -7, 1)]:
        parts.append(rect(
            SES_CX - SES_W / 2 + dx, INNER_Y - SES_H / 2 + dy,
            SES_W, SES_H,
            fill=SES_FILL, stroke=SES_STROKE, stroke_w=2,
        ))
    parts.append(rect(
        SES_CX - SES_W / 2, INNER_Y - SES_H / 2, SES_W, SES_H,
        fill=SES_FILL, stroke=SES_STROKE, stroke_w=2,
    ))
    parts.append(text(SES_CX, INNER_Y, "Single Engine Sim  x 4", size=16))

    # Router hexagon.
    parts.append(hexagon(ROUTER_CX, INNER_Y, 46, fill="#151515",
                         stroke=FLUORITE, stroke_w=2))
    parts.append(text(ROUTER_CX, INNER_Y, "Router", size=17))

    # Planner rect.
    PLANNER_W, PLANNER_H = 130, 76
    parts.append(rect(
        PLANNER_CX - PLANNER_W / 2, INNER_Y - PLANNER_H / 2,
        PLANNER_W, PLANNER_H,
        fill=AMETHYST_FILL, stroke=AMETHYST, stroke_w=2,
    ))
    parts.append(text(PLANNER_CX, INNER_Y, "Planner", size=17))

    # KVBM dashed rect -- optional component, same emerald family as
    # Trace Collector. Slightly muted text for the "(optional)" subtitle.
    KVBM_W, KVBM_H = 130, 76
    parts.append(rect(
        KVBM_CX - KVBM_W / 2, INNER_Y - KVBM_H / 2,
        KVBM_W, KVBM_H,
        fill=EMERALD_FILL, stroke=EMERALD, stroke_w=2,
        stroke_dasharray="6,4",
    ))
    parts.append(text(KVBM_CX, INNER_Y - 8, "KVBM", size=17))
    parts.append(text(KVBM_CX, INNER_Y + 14, "(optional)", size=12,
                      color=TEXT_MUTED, weight="normal"))

    # Trace Collector -- bottom center on the axis (cylinder).
    COLL_CY = MES_Y + MES_H + 110
    COLL_W, COLL_H = 220, 80
    parts.append(cylinder(AXIS_X, COLL_CY, COLL_W, COLL_H,
                          fill=NEUTRAL_FILL, stroke=NEUTRAL_STROKE))
    parts.append(text(AXIS_X, COLL_CY, "Trace Collector", size=18))

    # Vertical edges down the central axis.
    parts.append(arrow_down(
        DRIVER_CX, DRIVER_CY + DRIVER_H / 2,
        DRIVER_CX, MES_Y - 4,
        color=AMBER, label="events",
    ))
    parts.append(arrow_down(
        AXIS_X, MES_Y + MES_H,
        AXIS_X, COLL_CY - COLL_H / 2 - 12,
        color=TEXT_MUTED, label="metrics",
    ))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    images_dir = HERE.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_svg = images_dir / "fig-2-architecture-map.svg"
    out_png = images_dir / "fig-2-architecture-map.png"
    out_svg.write_text(build_svg(), encoding="utf-8")
    subprocess.run(
        ["rsvg-convert", "-z", "2", str(out_svg), "-o", str(out_png)],
        check=True,
    )
    print(f"wrote {out_svg}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
