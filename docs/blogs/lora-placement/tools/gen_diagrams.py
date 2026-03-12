#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate architecture SVG diagrams for the LoRA placement blog.

Produces three figures as SVG (and optionally PNG via cairosvg):
  - Fig 1: Architecture Overview   (fig-1-architecture-overview.svg)
  - Fig 2: Control Loop            (fig-2-control-loop.svg)
  - Fig 3: MCF Bipartite Graph     (fig-3-mcf-bipartite.svg)

Uses the Dynamo dark theme colors from design_tokens.yaml.

Usage:
    python3 gen_diagrams.py
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TOOLS_DIR = Path(__file__).resolve().parent
IMAGES_DIR = TOOLS_DIR.parent / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Design Tokens (from design_tokens.yaml, inlined for zero-dependency usage)
# ---------------------------------------------------------------------------

# Backgrounds
BG_PRIMARY = "#000000"
BG_CANVAS = "#0a0a0a"
BG_SURFACE = "#1a1a1a"
BG_SURFACE_ALT = "#2a2a2a"
BG_ELEVATED = "#3a3a3a"

# Muted fills
FILL_GREEN = "#3a5a00"
FILL_BLUE = "#0f1e30"
FILL_PURPLE = "#1a1428"
FILL_TEAL = "#142025"
FILL_WARM = "#201810"
FILL_RED = "#2a1010"
FILL_NEUTRAL = "#1a1a1a"

# Accents
GREEN = "#76b900"
BLUE = "#0071c5"
FLUORITE = "#fac200"
EMERALD = "#008564"
AMETHYST = "#5d1682"
AMBER = "#c08050"
CORAL = "#b04040"
OLIVE = "#909040"

# Text
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#cdcdcd"
TEXT_MUTED = "#767676"
TEXT_MEDIUM = "#8c8c8c"

# Borders
BORDER_SUBTLE = "#3a3a3a"
BORDER_CONTAINER = "#5d5d5d"

# Typography
FONT_FAMILY = "Arial, Helvetica, sans-serif"
FONT_MONO = "Roboto Mono, SF Mono, Menlo, Consolas, Liberation Mono, monospace"

# SPDX comment to inject into SVG files
SPDX_COMMENT = (
    "SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION "
    "& AFFILIATES. All rights reserved.\n"
    "SPDX-License-Identifier: Apache-2.0"
)

SVG_NS = "http://www.w3.org/2000/svg"


# ---------------------------------------------------------------------------
# SVG Helpers
# ---------------------------------------------------------------------------


def make_svg_root(width: int, height: int) -> ET.Element:
    """Create the root <svg> element with namespace and viewBox."""
    svg = ET.Element("svg")
    svg.set("xmlns", SVG_NS)
    svg.set("version", "1.1")
    svg.set("width", str(width))
    svg.set("height", str(height))
    svg.set("viewBox", f"0 0 {width} {height}")
    return svg


def add_defs(svg: ET.Element) -> ET.Element:
    """Add <defs> with reusable arrow markers."""
    defs = ET.SubElement(svg, "defs")

    # Solid arrow markers for different colors
    for name, color in [
        ("arrow-white", TEXT_PRIMARY),
        ("arrow-green", GREEN),
        ("arrow-blue", BLUE),
        ("arrow-emerald", EMERALD),
        ("arrow-amber", AMBER),
        ("arrow-coral", CORAL),
        ("arrow-fluorite", FLUORITE),
        ("arrow-muted", TEXT_MUTED),
        ("arrow-secondary", TEXT_SECONDARY),
        ("arrow-subtle", BORDER_SUBTLE),
    ]:
        marker = ET.SubElement(defs, "marker")
        marker.set("id", name)
        marker.set("viewBox", "0 0 10 7")
        marker.set("refX", "10")
        marker.set("refY", "3.5")
        marker.set("markerWidth", "10")
        marker.set("markerHeight", "7")
        marker.set("orient", "auto")
        polygon = ET.SubElement(marker, "polygon")
        polygon.set("points", "0,0 10,3.5 0,7")
        polygon.set("fill", color)

    return defs


def rect(
    parent: ET.Element,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = BG_SURFACE,
    stroke: str = BORDER_SUBTLE,
    stroke_width: float = 1,
    rx: float = 6,
    opacity: float = 1.0,
    dash: str | None = None,
) -> ET.Element:
    """Add a rounded rectangle."""
    r = ET.SubElement(parent, "rect")
    r.set("x", f"{x}")
    r.set("y", f"{y}")
    r.set("width", f"{w}")
    r.set("height", f"{h}")
    r.set("rx", f"{rx}")
    r.set("ry", f"{rx}")
    r.set("fill", fill)
    r.set("stroke", stroke)
    r.set("stroke-width", f"{stroke_width}")
    if opacity < 1.0:
        r.set("opacity", f"{opacity}")
    if dash:
        r.set("stroke-dasharray", dash)
    return r


def text(
    parent: ET.Element,
    x: float,
    y: float,
    content: str,
    *,
    fill: str = TEXT_PRIMARY,
    size: float = 12,
    weight: str = "normal",
    anchor: str = "middle",
    font: str | None = None,
) -> ET.Element:
    """Add a text element (single line)."""
    t = ET.SubElement(parent, "text")
    t.set("x", f"{x}")
    t.set("y", f"{y}")
    t.set("fill", fill)
    t.set("font-size", f"{size}")
    t.set("font-weight", weight)
    t.set("font-family", font or FONT_FAMILY)
    t.set("text-anchor", anchor)
    t.text = content
    return t


def multiline_text(
    parent: ET.Element,
    x: float,
    y: float,
    lines: list[str],
    *,
    fill: str = TEXT_PRIMARY,
    size: float = 12,
    weight: str = "normal",
    anchor: str = "middle",
    line_height: float = 1.3,
    font: str | None = None,
) -> None:
    """Add a multi-line text block using <tspan> elements."""
    t = ET.SubElement(parent, "text")
    t.set("x", f"{x}")
    t.set("y", f"{y}")
    t.set("fill", fill)
    t.set("font-size", f"{size}")
    t.set("font-weight", weight)
    t.set("font-family", font or FONT_FAMILY)
    t.set("text-anchor", anchor)
    for i, line in enumerate(lines):
        tspan = ET.SubElement(t, "tspan")
        tspan.set("x", f"{x}")
        if i == 0:
            tspan.set("dy", "0")
        else:
            tspan.set("dy", f"{size * line_height}")
        tspan.text = line


def line(
    parent: ET.Element,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = TEXT_MUTED,
    width: float = 1.5,
    dash: str | None = None,
    marker_end: str | None = None,
) -> ET.Element:
    """Add a line element."""
    ln = ET.SubElement(parent, "line")
    ln.set("x1", f"{x1}")
    ln.set("y1", f"{y1}")
    ln.set("x2", f"{x2}")
    ln.set("y2", f"{y2}")
    ln.set("stroke", stroke)
    ln.set("stroke-width", f"{width}")
    if dash:
        ln.set("stroke-dasharray", dash)
    if marker_end:
        ln.set("marker-end", f"url(#{marker_end})")
    return ln


def polyline(
    parent: ET.Element,
    points: list[tuple[float, float]],
    *,
    stroke: str = TEXT_MUTED,
    width: float = 1.5,
    dash: str | None = None,
    marker_end: str | None = None,
    fill: str = "none",
) -> ET.Element:
    """Add a polyline (for bent arrows)."""
    pl = ET.SubElement(parent, "polyline")
    pts = " ".join(f"{x},{y}" for x, y in points)
    pl.set("points", pts)
    pl.set("stroke", stroke)
    pl.set("stroke-width", f"{width}")
    pl.set("fill", fill)
    if dash:
        pl.set("stroke-dasharray", dash)
    if marker_end:
        pl.set("marker-end", f"url(#{marker_end})")
    return pl


def path_d(
    parent: ET.Element,
    d: str,
    *,
    stroke: str = TEXT_MUTED,
    width: float = 1.5,
    dash: str | None = None,
    marker_end: str | None = None,
    fill: str = "none",
) -> ET.Element:
    """Add a <path> element with a raw d attribute."""
    p = ET.SubElement(parent, "path")
    p.set("d", d)
    p.set("stroke", stroke)
    p.set("stroke-width", f"{width}")
    p.set("fill", fill)
    if dash:
        p.set("stroke-dasharray", dash)
    if marker_end:
        p.set("marker-end", f"url(#{marker_end})")
    return p


def ellipse(
    parent: ET.Element,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    *,
    fill: str = BG_SURFACE,
    stroke: str = BORDER_SUBTLE,
    stroke_width: float = 1,
) -> ET.Element:
    """Add an ellipse."""
    e = ET.SubElement(parent, "ellipse")
    e.set("cx", f"{cx}")
    e.set("cy", f"{cy}")
    e.set("rx", f"{rx}")
    e.set("ry", f"{ry}")
    e.set("fill", fill)
    e.set("stroke", stroke)
    e.set("stroke-width", f"{stroke_width}")
    return e


def circle(
    parent: ET.Element,
    cx: float,
    cy: float,
    r: float,
    *,
    fill: str = BG_SURFACE,
    stroke: str = BORDER_SUBTLE,
    stroke_width: float = 1.5,
) -> ET.Element:
    """Add a circle."""
    c = ET.SubElement(parent, "circle")
    c.set("cx", f"{cx}")
    c.set("cy", f"{cy}")
    c.set("r", f"{r}")
    c.set("fill", fill)
    c.set("stroke", stroke)
    c.set("stroke-width", f"{stroke_width}")
    return c


def component_box(
    parent: ET.Element,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    subtitle: str | None = None,
    *,
    border_color: str = GREEN,
    fill_color: str = FILL_GREEN,
    text_color: str = TEXT_PRIMARY,
    subtitle_color: str = TEXT_SECONDARY,
    rx: float = 5,
) -> None:
    """Draw a component box with a colored left accent strip and label."""
    # Main box
    rect(parent, x, y, w, h, fill=fill_color, stroke=border_color, rx=rx)
    # Left accent strip
    rect(parent, x, y, 4, h, fill=border_color, stroke="none", rx=0)
    # Rounded left corners
    rect(parent, x, y, 4, min(h, rx * 2), fill=border_color, stroke="none", rx=0)

    # Label
    ty = y + h / 2 - (6 if subtitle else 0)
    text(parent, x + w / 2 + 2, ty, label, fill=text_color, size=12, weight="bold")

    if subtitle:
        text(
            parent,
            x + w / 2 + 2,
            ty + 16,
            subtitle,
            fill=subtitle_color,
            size=10,
        )


def container_box(
    parent: ET.Element,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    border_color: str = BORDER_CONTAINER,
    fill_color: str = "#111111",
    label_color: str = TEXT_SECONDARY,
    rx: float = 8,
    font_size: float = 11,
) -> None:
    """Draw a labeled container with a title at the top-left."""
    rect(parent, x, y, w, h, fill=fill_color, stroke=border_color, rx=rx)
    text(
        parent,
        x + 14,
        y + 18,
        label,
        fill=label_color,
        size=font_size,
        weight="600",
        anchor="start",
    )


def write_svg(svg: ET.Element, filepath: Path) -> None:
    """Write SVG element tree to file with SPDX comment."""
    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")

    xml_str = ET.tostring(svg, encoding="unicode", xml_declaration=True)
    # Inject SPDX comment after XML declaration
    decl_end = xml_str.index("?>") + 2
    xml_with_comment = (
        xml_str[:decl_end] + f"\n<!-- {SPDX_COMMENT} -->\n" + xml_str[decl_end:]
    )

    filepath.write_text(xml_with_comment, encoding="utf-8")
    print(f"  Wrote {filepath}")


def try_png(svg_path: Path) -> None:
    """Try to convert SVG to PNG using cairosvg. Skip silently if unavailable."""
    try:
        import cairosvg  # type: ignore[import-untyped]

        png_path = svg_path.with_suffix(".png")
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=2)
        print(f"  Wrote {png_path}")
    except ImportError:
        pass  # cairosvg not available; handled in main


# ===========================================================================
# Figure 1: Architecture Overview
# ===========================================================================


def gen_fig1_architecture() -> None:
    """Generate fig-1-architecture-overview.svg."""
    W, H = 1050, 700
    svg = make_svg_root(W, H)
    add_defs(svg)

    # Background
    rect(svg, 0, 0, W, H, fill=BG_CANVAS, stroke="none", rx=0)

    # -----------------------------------------------------------------------
    # Client Request oval at top
    # -----------------------------------------------------------------------
    client_cx, client_cy = W / 2, 35
    ellipse(
        svg,
        client_cx,
        client_cy,
        90,
        22,
        fill=FILL_NEUTRAL,
        stroke=TEXT_MEDIUM,
        stroke_width=1.5,
    )
    text(
        svg,
        client_cx,
        client_cy + 1,
        "Client Request",
        fill=TEXT_PRIMARY,
        size=12,
        weight="bold",
    )
    text(
        svg,
        client_cx,
        client_cy + 14,
        "POST /v1/chat",
        fill=TEXT_MUTED,
        size=9,
    )

    # Arrow from client to router process
    line(
        svg,
        client_cx,
        client_cy + 22,
        client_cx,
        88,
        stroke=FLUORITE,
        width=1.5,
        marker_end="arrow-fluorite",
    )

    # -----------------------------------------------------------------------
    # Router Process (outer container)
    # -----------------------------------------------------------------------
    rp_x, rp_y, rp_w, rp_h = 100, 90, 780, 440
    container_box(
        svg,
        rp_x,
        rp_y,
        rp_w,
        rp_h,
        "ROUTER PROCESS",
        border_color=BORDER_CONTAINER,
        fill_color="#0e0e0e",
        label_color=TEXT_SECONDARY,
        font_size=13,
    )

    # -- Control Plane container --
    cp_x, cp_y, cp_w, cp_h = 120, 120, 740, 110
    container_box(
        svg,
        cp_x,
        cp_y,
        cp_w,
        cp_h,
        "Control Plane -- Periodic",
        border_color=EMERALD,
        fill_color="#0c1a10",
        label_color=EMERALD,
        font_size=11,
    )

    # Control Plane components
    cp_box_y = cp_y + 32
    cp_box_h = 60
    cp_box_w = 160

    # LoraController
    component_box(
        svg,
        cp_x + 15,
        cp_box_y,
        cp_box_w,
        cp_box_h,
        "LoraController",
        "Periodic Recompute",
        border_color=GREEN,
        fill_color=FILL_GREEN,
    )

    # LoadEstimator
    component_box(
        svg,
        cp_x + 195,
        cp_box_y,
        cp_box_w,
        cp_box_h,
        "LoadEstimator",
        "Windowed Rate",
        border_color=EMERALD,
        fill_color=FILL_TEAL,
    )

    # LoraAllocator
    component_box(
        svg,
        cp_x + 375,
        cp_box_y,
        cp_box_w,
        cp_box_h,
        "LoraAllocator",
        "Slot-Aware HRW",
        border_color=GREEN,
        fill_color=FILL_GREEN,
    )

    # McfPlacementSolver
    component_box(
        svg,
        cp_x + 555,
        cp_box_y,
        cp_box_w + 10,
        cp_box_h,
        "McfPlacementSolver",
        "Min-Cost Flow",
        border_color=GREEN,
        fill_color=FILL_GREEN,
    )

    # -- Shared State container --
    ss_x, ss_y, ss_w, ss_h = 120, 248, 740, 100
    container_box(
        svg,
        ss_x,
        ss_y,
        ss_w,
        ss_h,
        "Shared State -- Thread-safe (DashMap)",
        border_color=CORAL,
        fill_color="#180c0c",
        label_color=CORAL,
        font_size=11,
    )

    ss_box_y = ss_y + 32
    ss_box_h = 52
    ss_box_w = 200

    # LoraRoutingTable
    component_box(
        svg,
        ss_x + 80,
        ss_box_y,
        ss_box_w,
        ss_box_h,
        "LoraRoutingTable",
        "LoRA -> ReplicaConfig",
        border_color=EMERALD,
        fill_color=FILL_TEAL,
    )

    # LoraStateTracker
    component_box(
        svg,
        ss_x + 380,
        ss_box_y,
        ss_box_w,
        ss_box_h,
        "LoraStateTracker",
        "loaded_locations, capacity",
        border_color=EMERALD,
        fill_color=FILL_TEAL,
    )

    # -- Data Plane container --
    dp_x, dp_y, dp_w, dp_h = 120, 366, 740, 100
    container_box(
        svg,
        dp_x,
        dp_y,
        dp_w,
        dp_h,
        "Data Plane -- Per-Request",
        border_color=BLUE,
        fill_color="#0a0e18",
        label_color=BLUE,
        font_size=11,
    )

    dp_box_y = dp_y + 32
    dp_box_h = 52
    dp_box_w = 200

    # LoraFilter
    component_box(
        svg,
        dp_x + 80,
        dp_box_y,
        dp_box_w,
        dp_box_h,
        "LoraFilter",
        "Active / Inactive / Fallback",
        border_color=BLUE,
        fill_color=FILL_BLUE,
    )

    # KvRouter
    component_box(
        svg,
        dp_x + 380,
        dp_box_y,
        dp_box_w,
        dp_box_h,
        "KvRouter",
        "KV-Cache-Aware Selection",
        border_color=BLUE,
        fill_color=FILL_BLUE,
    )

    # -----------------------------------------------------------------------
    # Internal arrows (control flow -- dashed)
    # -----------------------------------------------------------------------
    # Controller -> LoadEstimator
    line(
        svg,
        cp_x + 15 + cp_box_w,
        cp_box_y + cp_box_h / 2,
        cp_x + 195,
        cp_box_y + cp_box_h / 2,
        stroke=BLUE,
        width=1,
        dash="5,3",
        marker_end="arrow-blue",
    )

    # Controller -> Allocator
    line(
        svg,
        cp_x + 15 + cp_box_w,
        cp_box_y + cp_box_h / 2 - 8,
        cp_x + 375,
        cp_box_y + cp_box_h / 2 - 8,
        stroke=BLUE,
        width=1,
        dash="5,3",
        marker_end="arrow-blue",
    )

    # Controller -> MCF
    line(
        svg,
        cp_x + 15 + cp_box_w,
        cp_box_y + cp_box_h / 2 + 8,
        cp_x + 555,
        cp_box_y + cp_box_h / 2 + 8,
        stroke=BLUE,
        width=1,
        dash="5,3",
        marker_end="arrow-blue",
    )

    # Controller -> RoutingTable (Write Plan)
    ctrl_bot_x = cp_x + 15 + cp_box_w / 2
    ctrl_bot_y = cp_box_y + cp_box_h
    rt_top_x = ss_x + 80 + ss_box_w / 2
    rt_top_y = ss_box_y
    line(
        svg,
        ctrl_bot_x,
        ctrl_bot_y,
        rt_top_x,
        rt_top_y,
        stroke=BLUE,
        width=1,
        dash="5,3",
        marker_end="arrow-blue",
    )
    text(
        svg,
        (ctrl_bot_x + rt_top_x) / 2 - 30,
        (ctrl_bot_y + rt_top_y) / 2 - 2,
        "Write Plan",
        fill=BLUE,
        size=9,
        anchor="end",
    )

    # Controller -> StateTracker (Read Topology)
    st_top_x = ss_x + 380 + ss_box_w / 2
    line(
        svg,
        ctrl_bot_x + 40,
        ctrl_bot_y,
        st_top_x,
        rt_top_y,
        stroke=BLUE,
        width=1,
        dash="5,3",
        marker_end="arrow-blue",
    )
    text(
        svg,
        (ctrl_bot_x + 40 + st_top_x) / 2 + 30,
        (ctrl_bot_y + rt_top_y) / 2 - 2,
        "Read Topology",
        fill=BLUE,
        size=9,
        anchor="start",
    )

    # LoraFilter -> RoutingTable (Lookup)
    filter_top_x = dp_x + 80 + dp_box_w / 2
    filter_top_y = dp_box_y
    rt_bot_x = ss_x + 80 + ss_box_w / 2
    rt_bot_y = ss_box_y + ss_box_h
    line(
        svg,
        filter_top_x - 20,
        filter_top_y,
        rt_bot_x,
        rt_bot_y,
        stroke=AMETHYST,
        width=1,
        marker_end="arrow-muted",
    )
    text(
        svg,
        (filter_top_x - 20 + rt_bot_x) / 2 - 20,
        (filter_top_y + rt_bot_y) / 2 - 2,
        "Lookup",
        fill=AMETHYST,
        size=9,
        anchor="end",
    )

    # LoraFilter -> StateTracker (Check Loaded)
    line(
        svg,
        filter_top_x + 20,
        filter_top_y,
        st_top_x,
        rt_bot_y,
        stroke=AMETHYST,
        width=1,
        marker_end="arrow-muted",
    )
    text(
        svg,
        (filter_top_x + 20 + st_top_x) / 2 + 20,
        (filter_top_y + rt_bot_y) / 2 - 2,
        "Check Loaded",
        fill=AMETHYST,
        size=9,
        anchor="start",
    )

    # LoraFilter -> KvRouter (data flow)
    line(
        svg,
        dp_x + 80 + dp_box_w,
        dp_box_y + dp_box_h / 2,
        dp_x + 380,
        dp_box_y + dp_box_h / 2,
        stroke=FLUORITE,
        width=1.5,
        marker_end="arrow-fluorite",
    )
    text(
        svg,
        dp_x + 80 + dp_box_w + (380 - 80 - dp_box_w) / 2,
        dp_box_y + dp_box_h / 2 - 8,
        "Filtered Candidates",
        fill=FLUORITE,
        size=9,
    )

    # -----------------------------------------------------------------------
    # Worker Cluster
    # -----------------------------------------------------------------------
    wc_x, wc_y, wc_w, wc_h = 100, 555, 580, 110
    container_box(
        svg,
        wc_x,
        wc_y,
        wc_w,
        wc_h,
        "WORKER CLUSTER",
        border_color=AMBER,
        fill_color="#141008",
        label_color=AMBER,
        font_size=12,
    )

    workers = [
        ("Worker 1", "K=4 slots", wc_x + 30),
        ("Worker 2", "K=4 slots", wc_x + 210),
        ("Worker 3", "K=6 slots", wc_x + 390),
    ]
    w_box_w, w_box_h = 155, 55
    w_box_y = wc_y + 35
    for wname, wsub, wx in workers:
        component_box(
            svg,
            wx,
            w_box_y,
            w_box_w,
            w_box_h,
            wname,
            wsub,
            border_color=AMBER,
            fill_color=FILL_WARM,
        )

    # KvRouter -> Workers arrows
    kv_bot_x = dp_x + 380 + dp_box_w / 2
    kv_bot_y = dp_box_y + dp_box_h

    for i, (wname, wsub, wx) in enumerate(workers):
        target_x = wx + w_box_w / 2
        target_y = w_box_y
        # Route through the router process bottom edge
        mid_y = rp_y + rp_h + 10
        path_d(
            svg,
            f"M {kv_bot_x + (i - 1) * 15},{kv_bot_y} "
            f"L {kv_bot_x + (i - 1) * 15},{mid_y} "
            f"L {target_x},{mid_y} "
            f"L {target_x},{target_y}",
            stroke=FLUORITE,
            width=1.2,
            marker_end="arrow-fluorite",
        )

    text(
        svg,
        kv_bot_x + 50,
        rp_y + rp_h + 8,
        "Route",
        fill=FLUORITE,
        size=9,
        anchor="start",
    )

    # Arrow from client request to LoraFilter (data flow into router)
    # (already drawn client -> router top, now connect internally)
    line(
        svg,
        client_cx,
        88,
        filter_top_x - 20,
        dp_box_y,
        stroke=FLUORITE,
        width=1.5,
        dash="2,4",
        marker_end="arrow-fluorite",
    )
    text(
        svg,
        client_cx + 20,
        (88 + dp_box_y) / 2,
        "Request",
        fill=FLUORITE,
        size=9,
        anchor="start",
    )

    # -----------------------------------------------------------------------
    # External: Discovery MDC (right side)
    # -----------------------------------------------------------------------
    disc_x, disc_y, disc_w, disc_h = 910, 250, 120, 70
    component_box(
        svg,
        disc_x,
        disc_y,
        disc_w,
        disc_h,
        "Discovery",
        "MDC Events",
        border_color=EMERALD,
        fill_color=FILL_TEAL,
    )

    # Workers -> Discovery (events, dashed)
    for i, (wname, wsub, wx) in enumerate(workers):
        src_x = wx + w_box_w
        src_y = w_box_y + w_box_h / 2
        line(
            svg,
            src_x,
            src_y,
            disc_x,
            disc_y + disc_h,
            stroke=AMBER,
            width=1,
            dash="4,3",
            marker_end="arrow-amber",
        )

    text(
        svg,
        disc_x - 10,
        disc_y + disc_h + 20,
        "Events",
        fill=AMBER,
        size=9,
        anchor="end",
    )

    # Discovery -> StateTracker (MDC Updates, dashed)
    line(
        svg,
        disc_x,
        disc_y + disc_h / 2,
        ss_x + 380 + ss_box_w,
        ss_box_y + ss_box_h / 2,
        stroke=AMBER,
        width=1,
        dash="4,3",
        marker_end="arrow-amber",
    )
    text(
        svg,
        disc_x - 10,
        disc_y + disc_h / 2 - 8,
        "MDC Updates",
        fill=AMBER,
        size=9,
        anchor="end",
    )

    # -----------------------------------------------------------------------
    # External: Prometheus (right side, top)
    # -----------------------------------------------------------------------
    prom_x, prom_y, prom_w, prom_h = 910, 130, 120, 70
    component_box(
        svg,
        prom_x,
        prom_y,
        prom_w,
        prom_h,
        "Prometheus",
        "Metrics Export",
        border_color=OLIVE,
        fill_color="#1a1a10",
    )

    # Controller -> Prometheus (dashed)
    line(
        svg,
        cp_x + cp_w,
        cp_box_y + cp_box_h / 2,
        prom_x,
        prom_y + prom_h / 2,
        stroke=OLIVE,
        width=1,
        dash="4,3",
        marker_end="arrow-muted",
    )
    text(
        svg,
        prom_x - 10,
        prom_y + prom_h / 2 - 8,
        "Export",
        fill=OLIVE,
        size=9,
        anchor="end",
    )

    # -----------------------------------------------------------------------
    # Legend (bottom right)
    # -----------------------------------------------------------------------
    legend_x, legend_y = 720, 565
    rect(svg, legend_x, legend_y, 310, 105, fill="#0e0e0e", stroke=BORDER_SUBTLE, rx=6)
    text(
        svg,
        legend_x + 155,
        legend_y + 18,
        "Legend",
        fill=TEXT_SECONDARY,
        size=11,
        weight="bold",
    )

    legend_items = [
        (FLUORITE, "solid", "Data Flow (request path)"),
        (BLUE, "dashed", "Control Flow (periodic)"),
        (AMETHYST, "solid", "State Lookup"),
        (AMBER, "dashed", "Events / MDC"),
    ]
    for i, (color, style, label) in enumerate(legend_items):
        ly = legend_y + 34 + i * 18
        d = "4,3" if style == "dashed" else None
        line(
            svg,
            legend_x + 20,
            ly,
            legend_x + 55,
            ly,
            stroke=color,
            width=1.5,
            dash=d,
            marker_end=f"arrow-{color.replace('#', '').lower()}"
            if color == FLUORITE
            else None,
        )
        text(
            svg,
            legend_x + 65,
            ly + 4,
            label,
            fill=TEXT_SECONDARY,
            size=10,
            anchor="start",
        )

    # Write
    out = IMAGES_DIR / "fig-1-architecture-overview.svg"
    write_svg(svg, out)
    try_png(out)


# ===========================================================================
# Figure 2: Control Loop
# ===========================================================================


def gen_fig2_control_loop() -> None:
    """Generate fig-2-control-loop.svg."""
    W, H = 950, 350
    svg = make_svg_root(W, H)
    add_defs(svg)

    # Background
    rect(svg, 0, 0, W, H, fill=BG_CANVAS, stroke="none", rx=0)

    # -----------------------------------------------------------------------
    # Main pipeline boxes (horizontal)
    # -----------------------------------------------------------------------
    box_w, box_h = 108, 64
    pipeline_y = 80
    gap = 12

    # Pipeline stages: name, subtitle, border_color, fill_color, x
    stages = [
        ("Requests", "Incoming LoRA\ntraffic", BLUE, FILL_BLUE, 30),
        (
            "LoadEstimator",
            "Windowed Rate\n+ In-flight",
            EMERALD,
            FILL_TEAL,
            30 + (box_w + gap),
        ),
        (
            "LoraController",
            "Periodic\nRecompute",
            GREEN,
            FILL_GREEN,
            30 + 2 * (box_w + gap),
        ),
        (
            "RoutingTable",
            "LoRA ->\nReplicaConfig",
            EMERALD,
            FILL_TEAL,
            30 + 3 * (box_w + gap),
        ),
        (
            "LoraFilter",
            "Candidate Set\nFiltering",
            BLUE,
            FILL_BLUE,
            30 + 4 * (box_w + gap),
        ),
        (
            "KvRouter",
            "KV-Cache-Aware\nSelection",
            BLUE,
            FILL_BLUE,
            30 + 5 * (box_w + gap),
        ),
        (
            "Worker",
            "Serve Request\nRAII Guard",
            AMBER,
            FILL_WARM,
            30 + 6 * (box_w + gap),
        ),
    ]

    for name, subtitle, border, fill, x in stages:
        # Box
        rect(svg, x, pipeline_y, box_w, box_h, fill=fill, stroke=border, rx=5)
        # Left accent strip
        rect(svg, x, pipeline_y, 4, box_h, fill=border, stroke="none", rx=0)

        # Text
        text(
            svg,
            x + box_w / 2 + 2,
            pipeline_y + 20,
            name,
            fill=TEXT_PRIMARY,
            size=11,
            weight="bold",
        )
        sub_lines = subtitle.split("\n")
        for si, sline in enumerate(sub_lines):
            text(
                svg,
                x + box_w / 2 + 2,
                pipeline_y + 36 + si * 13,
                sline,
                fill=TEXT_SECONDARY,
                size=9,
            )

    # Forward arrows between pipeline stages
    for i in range(len(stages) - 1):
        x1 = stages[i][4] + box_w
        x2 = stages[i + 1][4]
        mid_y = pipeline_y + box_h / 2
        line(
            svg,
            x1,
            mid_y,
            x2,
            mid_y,
            stroke=FLUORITE,
            width=1.5,
            marker_end="arrow-fluorite",
        )

    # Arrow labels on key connections
    arrow_labels = [
        (0, "Arrive"),
        (1, "Rate Signal"),
        (2, "Replica Plan"),
        (3, "Candidate Set"),
        (4, "Narrowed"),
        (5, "Selected"),
    ]
    for idx, label in arrow_labels:
        ax = stages[idx][4] + box_w + gap / 2
        text(svg, ax, pipeline_y - 6, label, fill=FLUORITE, size=8)

    # -----------------------------------------------------------------------
    # Feedback loop (RAII Guard Drop) -- curved arrow underneath
    # -----------------------------------------------------------------------
    worker_x = stages[-1][4]
    worker_cx = worker_x + box_w / 2
    worker_bot = pipeline_y + box_h

    requests_x = stages[0][4]
    requests_cx = requests_x + box_w / 2

    feedback_y = pipeline_y + box_h + 80

    # Path: Worker bottom -> down -> left -> up -> Requests bottom
    path_d(
        svg,
        f"M {worker_cx},{worker_bot} "
        f"L {worker_cx},{feedback_y} "
        f"L {requests_cx},{feedback_y} "
        f"L {requests_cx},{worker_bot}",
        stroke=CORAL,
        width=1.5,
        dash="6,4",
        marker_end="arrow-coral",
    )
    text(
        svg,
        (worker_cx + requests_cx) / 2,
        feedback_y + 16,
        "RAII Guard Drop (feedback -- load signal returns)",
        fill=CORAL,
        size=10,
        weight="bold",
    )

    # -----------------------------------------------------------------------
    # Side branch: LoraStateTracker (below controller, separate)
    # -----------------------------------------------------------------------
    tracker_x = stages[2][4] + (stages[3][4] - stages[2][4]) / 2 - box_w / 2
    tracker_y = 10
    tracker_w = box_w + 20
    tracker_h = 52

    component_box(
        svg,
        tracker_x,
        tracker_y,
        tracker_w,
        tracker_h,
        "LoraStateTracker",
        "loaded_locations, capacity",
        border_color=EMERALD,
        fill_color=FILL_TEAL,
    )

    # Controller -> StateTracker (dashed, upward)
    ctrl_x = stages[2][4]
    ctrl_cx = ctrl_x + box_w / 2
    line(
        svg,
        ctrl_cx,
        pipeline_y,
        tracker_x + tracker_w / 2,
        tracker_y + tracker_h,
        stroke=BLUE,
        width=1,
        dash="5,3",
        marker_end="arrow-blue",
    )
    text(
        svg,
        ctrl_cx - 12,
        pipeline_y - 12,
        "Topology",
        fill=BLUE,
        size=9,
        anchor="end",
    )

    # Worker -> StateTracker (MDC events, dashed, upward arc)
    path_d(
        svg,
        f"M {worker_cx},{pipeline_y} "
        f"C {worker_cx},{pipeline_y - 50} "
        f"{tracker_x + tracker_w},{tracker_y - 30} "
        f"{tracker_x + tracker_w},{tracker_y + tracker_h / 2}",
        stroke=AMBER,
        width=1,
        dash="4,3",
        marker_end="arrow-amber",
    )
    text(
        svg,
        worker_cx - 40,
        pipeline_y - 38,
        "MDC Events",
        fill=AMBER,
        size=9,
    )

    # -----------------------------------------------------------------------
    # Legend (bottom right)
    # -----------------------------------------------------------------------
    legend_x, legend_y = 670, 255
    rect(svg, legend_x, legend_y, 260, 85, fill="#0e0e0e", stroke=BORDER_SUBTLE, rx=6)
    text(
        svg,
        legend_x + 130,
        legend_y + 16,
        "Legend",
        fill=TEXT_SECONDARY,
        size=10,
        weight="bold",
    )

    legend_items = [
        (FLUORITE, None, "Data Flow (forward path)"),
        (CORAL, "6,4", "Feedback (RAII guard)"),
        (BLUE, "5,3", "Control Flow"),
        (AMBER, "4,3", "MDC Events"),
    ]
    for i, (color, dash, label) in enumerate(legend_items):
        ly = legend_y + 30 + i * 14
        line(
            svg,
            legend_x + 15,
            ly,
            legend_x + 45,
            ly,
            stroke=color,
            width=1.5,
            dash=dash,
        )
        text(
            svg,
            legend_x + 55,
            ly + 4,
            label,
            fill=TEXT_SECONDARY,
            size=9,
            anchor="start",
        )

    out = IMAGES_DIR / "fig-2-control-loop.svg"
    write_svg(svg, out)
    try_png(out)


# ===========================================================================
# Figure 3: MCF Bipartite Graph
# ===========================================================================


def gen_fig3_mcf_bipartite() -> None:
    """Generate fig-3-mcf-bipartite.svg."""
    W, H = 900, 450
    svg = make_svg_root(W, H)
    add_defs(svg)

    # Background
    rect(svg, 0, 0, W, H, fill=BG_CANVAS, stroke="none", rx=0)

    # Column X positions
    src_cx = 70
    lora_cx = 260
    worker_cx = 560
    snk_cx = 750
    overflow_cx = 560

    # Y positions for nodes
    row_ys = [90, 175, 260]

    node_w, node_h = 130, 48
    circle_r = 24

    # -----------------------------------------------------------------------
    # Source node
    # -----------------------------------------------------------------------
    circle(svg, src_cx, row_ys[1], circle_r, fill=FILL_NEUTRAL, stroke=TEXT_MEDIUM)
    text(svg, src_cx, row_ys[1] + 5, "SRC", fill=TEXT_PRIMARY, size=12, weight="bold")

    # -----------------------------------------------------------------------
    # Sink node
    # -----------------------------------------------------------------------
    circle(svg, snk_cx, row_ys[1], circle_r, fill=FILL_NEUTRAL, stroke=TEXT_MEDIUM)
    text(svg, snk_cx, row_ys[1] + 5, "SNK", fill=TEXT_PRIMARY, size=12, weight="bold")

    # -----------------------------------------------------------------------
    # LoRA nodes container
    # -----------------------------------------------------------------------
    lora_cont_x = lora_cx - node_w / 2 - 20
    lora_cont_y = row_ys[0] - node_h / 2 - 30
    lora_cont_w = node_w + 40
    lora_cont_h = (row_ys[2] - row_ys[0]) + node_h + 50
    container_box(
        svg,
        lora_cont_x,
        lora_cont_y,
        lora_cont_w,
        lora_cont_h,
        "Adapters",
        border_color=EMERALD,
        fill_color="#0c1810",
        label_color=EMERALD,
        font_size=11,
    )

    loras = [
        ("LoRA-A", "r = 3", row_ys[0]),
        ("LoRA-B", "r = 2", row_ys[1]),
        ("LoRA-C", "r = 1", row_ys[2]),
    ]
    for name, sub, y in loras:
        bx = lora_cx - node_w / 2
        by = y - node_h / 2
        rect(svg, bx, by, node_w, node_h, fill=FILL_TEAL, stroke=EMERALD, rx=5)
        rect(svg, bx, by, 4, node_h, fill=EMERALD, stroke="none", rx=0)
        text(svg, lora_cx + 2, y - 4, name, fill=TEXT_PRIMARY, size=12, weight="bold")
        text(svg, lora_cx + 2, y + 12, sub, fill=TEXT_SECONDARY, size=10)

    # -----------------------------------------------------------------------
    # Worker nodes container
    # -----------------------------------------------------------------------
    worker_cont_x = worker_cx - node_w / 2 - 20
    worker_cont_y = row_ys[0] - node_h / 2 - 30
    worker_cont_w = node_w + 40
    worker_cont_h = (row_ys[2] - row_ys[0]) + node_h + 50
    container_box(
        svg,
        worker_cont_x,
        worker_cont_y,
        worker_cont_w,
        worker_cont_h,
        "Workers",
        border_color=AMBER,
        fill_color="#141008",
        label_color=AMBER,
        font_size=11,
    )

    workers_list = [
        ("Worker 1", "K = 4 slots", row_ys[0]),
        ("Worker 2", "K = 4 slots", row_ys[1]),
        ("Worker 3", "K = 6 slots", row_ys[2]),
    ]
    for name, sub, y in workers_list:
        bx = worker_cx - node_w / 2
        by = y - node_h / 2
        rect(svg, bx, by, node_w, node_h, fill=FILL_WARM, stroke=AMBER, rx=5)
        rect(svg, bx, by, 4, node_h, fill=AMBER, stroke="none", rx=0)
        text(svg, worker_cx + 2, y - 4, name, fill=TEXT_PRIMARY, size=12, weight="bold")
        text(svg, worker_cx + 2, y + 12, sub, fill=TEXT_SECONDARY, size=10)

    # -----------------------------------------------------------------------
    # Overflow node (below workers)
    # -----------------------------------------------------------------------
    overflow_y = 340
    ov_w, ov_h = 110, 40
    rect(
        svg,
        overflow_cx - ov_w / 2,
        overflow_y - ov_h / 2,
        ov_w,
        ov_h,
        fill=FILL_RED,
        stroke=CORAL,
        rx=5,
        dash="5,3",
    )
    rect(
        svg,
        overflow_cx - ov_w / 2,
        overflow_y - ov_h / 2,
        4,
        ov_h,
        fill=CORAL,
        stroke="none",
        rx=0,
    )
    text(
        svg,
        overflow_cx + 2,
        overflow_y + 4,
        "Overflow",
        fill=TEXT_PRIMARY,
        size=11,
        weight="bold",
    )

    # -----------------------------------------------------------------------
    # Edges: SRC -> LoRAs
    # -----------------------------------------------------------------------
    src_caps = ["cap=3", "cap=2", "cap=1"]
    for i, (name, sub, y) in enumerate(loras):
        lx = lora_cx - node_w / 2
        line(
            svg,
            src_cx + circle_r,
            row_ys[1] + (i - 1) * 10,
            lx,
            y,
            stroke=TEXT_MEDIUM,
            width=1.2,
            marker_end="arrow-secondary",
        )
        # Edge label
        mid_x = (src_cx + circle_r + lx) / 2
        mid_y = (row_ys[1] + (i - 1) * 10 + y) / 2
        text(
            svg,
            mid_x,
            mid_y - 6,
            src_caps[i],
            fill=TEXT_MUTED,
            size=9,
        )

    # -----------------------------------------------------------------------
    # Edges: LoRAs -> Workers (sparse bipartite)
    # -----------------------------------------------------------------------
    # (lora_idx, worker_idx, cost_label)
    bipartite_edges = [
        (0, 0, "cost=c11"),  # L1 -> W1
        (0, 1, "cost=c12"),  # L1 -> W2
        (1, 0, "cost=c21"),  # L2 -> W1
        (1, 2, "cost=c23"),  # L2 -> W3
        (2, 1, "cost=c32"),  # L3 -> W2
        (2, 2, "cost=c33"),  # L3 -> W3
    ]

    for li, wi, cost in bipartite_edges:
        ly = loras[li][2]
        wy = workers_list[wi][2]
        lx_right = lora_cx + node_w / 2
        wx_left = worker_cx - node_w / 2

        # Offset slightly to avoid overlap
        y_offset = (li - wi) * 3

        line(
            svg,
            lx_right,
            ly + y_offset,
            wx_left,
            wy + y_offset,
            stroke=EMERALD,
            width=1.2,
            marker_end="arrow-emerald",
        )
        # Cost label at midpoint
        mid_x = (lx_right + wx_left) / 2
        mid_y = (ly + y_offset + wy + y_offset) / 2
        text(
            svg,
            mid_x,
            mid_y - 7,
            cost,
            fill=TEXT_MUTED,
            size=8,
            font=FONT_MONO,
        )

    # -----------------------------------------------------------------------
    # Edge: L3 -> Overflow (dashed, coral)
    # -----------------------------------------------------------------------
    l3_y = loras[2][2]
    l3_right = lora_cx + node_w / 2
    ov_left = overflow_cx - ov_w / 2
    path_d(
        svg,
        f"M {l3_right},{l3_y + 10} "
        f"L {(l3_right + ov_left) / 2},{overflow_y - 5} "
        f"L {ov_left},{overflow_y}",
        stroke=CORAL,
        width=1.2,
        dash="5,3",
        marker_end="arrow-coral",
    )
    text(
        svg,
        (l3_right + ov_left) / 2 - 20,
        overflow_y - 16,
        "cap=1, cost=inf",
        fill=CORAL,
        size=8,
        font=FONT_MONO,
    )

    # -----------------------------------------------------------------------
    # Edges: Workers -> SNK
    # -----------------------------------------------------------------------
    snk_caps = ["cap=4", "cap=4", "cap=6"]
    for i, (name, sub, y) in enumerate(workers_list):
        wx_right = worker_cx + node_w / 2
        line(
            svg,
            wx_right,
            y,
            snk_cx - circle_r,
            row_ys[1] + (i - 1) * 10,
            stroke=TEXT_MEDIUM,
            width=1.2,
            marker_end="arrow-secondary",
        )
        mid_x = (wx_right + snk_cx - circle_r) / 2
        mid_y = (y + row_ys[1] + (i - 1) * 10) / 2
        text(
            svg,
            mid_x,
            mid_y - 6,
            snk_caps[i],
            fill=TEXT_MUTED,
            size=9,
        )

    # -----------------------------------------------------------------------
    # Edge: Overflow -> SNK (dashed, coral)
    # -----------------------------------------------------------------------
    ov_right = overflow_cx + ov_w / 2
    path_d(
        svg,
        f"M {ov_right},{overflow_y} "
        f"L {(ov_right + snk_cx) / 2},{overflow_y - 20} "
        f"L {snk_cx - circle_r},{row_ys[1] + 20}",
        stroke=CORAL,
        width=1.2,
        dash="5,3",
        marker_end="arrow-coral",
    )
    text(
        svg,
        (ov_right + snk_cx) / 2 + 10,
        overflow_y - 26,
        "cap=excess",
        fill=CORAL,
        size=8,
        font=FONT_MONO,
    )

    # -----------------------------------------------------------------------
    # Cost function legend at bottom
    # -----------------------------------------------------------------------
    legend_y = H - 45
    rect(svg, 120, legend_y - 12, 660, 40, fill="#0e0e0e", stroke=BORDER_SUBTLE, rx=6)
    text(
        svg,
        W / 2,
        legend_y + 8,
        "Cost Function:",
        fill=TEXT_SECONDARY,
        size=11,
        weight="bold",
    )
    text(
        svg,
        W / 2,
        legend_y + 24,
        "cost(l, s) = \u03b1 \u00b7 rank  +  \u03b3 \u00b7 w \u00b7 1[new]  \u2212  \u03b2 \u00b7 w \u00b7 1[keep]",
        fill=FLUORITE,
        size=12,
        weight="bold",
        font=FONT_MONO,
    )

    out = IMAGES_DIR / "fig-3-mcf-bipartite.svg"
    write_svg(svg, out)
    try_png(out)


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    print("Generating LoRA Placement architecture diagrams...")
    print(f"  Output directory: {IMAGES_DIR}")

    gen_fig1_architecture()
    gen_fig2_control_loop()
    gen_fig3_mcf_bipartite()

    # Check for cairosvg
    try:
        import cairosvg  # noqa: F401

        print("\nPNG versions generated via cairosvg (2x scale).")
    except ImportError:
        print(
            "\nNote: cairosvg not installed -- PNG generation skipped."
            "\n  Install with: pip install cairosvg"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
