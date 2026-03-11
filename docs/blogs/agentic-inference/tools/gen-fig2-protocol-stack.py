#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate Figure 2 -- protocol stack comparison for the agentic-inference blog.

Two-panel side-by-side comparison showing protocol stacks:
  * "Today"       — MCP and A2A exist; inference has no protocol (gap).
  * "With Dynamo" — nvext fills the gap with inference infrastructure.

Output: ../images/fig-2-protocol-stack.svg

PNG variant (2x resolution):
    rsvg-convert -z 2 -o fig-2-protocol-stack.png ../images/fig-2-protocol-stack.svg
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG = "#0a0a0a"
SURFACE = "#1a1a1a"
BORDER = "#636363"
GREEN = "#76b900"
GREEN_DIM = "#2a4a10"
GREEN_BORDER = "#4a8c00"
HARNESS_FILL = "#1a1428"
HARNESS_STROKE = "#7650a0"
TARGET_R_FILL = "#151515"
MUTED = "#767676"
TEXT_LIGHT = "#e0e0e0"
FONT = "Arial, Helvetica, sans-serif"

W, H = 960, 420

# Layout geometry
LABEL_X = 70
LEFT_X = 80
PANEL_W = 410
PANEL_GAP = 40
RIGHT_X = LEFT_X + PANEL_W + PANEL_GAP

HEADER_Y = 38
BAND_TOP = 58
BAND_H = 60
PROTO_H = 70
BAND_GAP = 6
BAND_HEIGHTS = [BAND_H, BAND_H, PROTO_H, BAND_H]

COL_GAP = 6
COL_W = (PANEL_W - 2 * COL_GAP) / 3

LAYERS = ["APPLICATION", "HARNESS", "PROTOCOL", "TARGETS"]


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

    lines = label.split("\n")
    if len(lines) == 1:
        return _el("text", parent, label, **attrs)

    t = _el("text", parent, **attrs)
    for i, line in enumerate(lines):
        tspan_attrs: dict[str, str] = {"x": str(x), "dy": "1.2em" if i > 0 else "0"}
        _el("tspan", t, line, **tspan_attrs)
    return t


def _rect(
    parent: ET.Element,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = SURFACE,
    stroke: str = BORDER,
    sw: float = 1,
    rx: float = 0,
    dash: str | None = None,
    filt: str | None = None,
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
    if filt:
        attrs["filter"] = f"url(#{filt})"
    return _el("rect", parent, **attrs)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _band_y(i: int) -> float:
    """Y coordinate for the top of band *i* (0-indexed)."""
    y = BAND_TOP
    for j in range(i):
        y += BAND_HEIGHTS[j] + BAND_GAP
    return y


def _col_x(panel_x: float, col: int) -> float:
    return panel_x + col * (COL_W + COL_GAP)


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------


def _add_glow_filter(svg: ET.Element) -> None:
    """SVG filter: soft green glow behind the nvext cell."""
    defs = _el("defs", svg)
    filt = _el(
        "filter",
        defs,
        id="glow",
        x="-20%",
        y="-20%",
        width="140%",
        height="140%",
    )
    blur = _el("feGaussianBlur", filt, stdDeviation="3", result="blur")
    blur.set("in", "SourceGraphic")
    merge = _el("feMerge", filt)
    n1 = _el("feMergeNode", merge)
    n1.set("in", "blur")
    n2 = _el("feMergeNode", merge)
    n2.set("in", "SourceGraphic")


def _draw_layer_labels(svg: ET.Element) -> None:
    for i, name in enumerate(LAYERS):
        by = _band_y(i)
        bh = BAND_HEIGHTS[i]
        _text(
            svg,
            LABEL_X,
            by + bh / 2,
            name,
            fill=MUTED,
            size=9,
            anchor="end",
            weight="600",
            spacing="0.1em",
            dy="0.35em",
        )


def _draw_cell(
    svg: ET.Element,
    x: float,
    y: float,
    w: float,
    h: float,
    main: str,
    *,
    sub: str | None = None,
    fill: str = SURFACE,
    stroke: str = BORDER,
    sw: float = 1,
    dash: str | None = None,
    filt: str | None = None,
    main_fill: str = TEXT_LIGHT,
    sub_fill: str = MUTED,
) -> None:
    """Draw a labelled band or column cell."""
    _rect(svg, x, y, w, h, fill=fill, stroke=stroke, sw=sw, dash=dash, filt=filt)

    if sub:
        _text(
            svg,
            x + w / 2,
            y + h / 2 - 8,
            main,
            fill=main_fill,
            size=15,
            weight="bold",
            dy="0.35em",
        )
        _text(
            svg,
            x + w / 2,
            y + h / 2 + 10,
            sub,
            fill=sub_fill,
            size=11,
            dy="0.35em",
        )
    else:
        _text(
            svg,
            x + w / 2,
            y + h / 2,
            main,
            fill=main_fill,
            size=15,
            weight="bold",
            dy="0.35em",
        )


def _build_panel(
    svg: ET.Element, px: float, title: str, *, dynamo: bool = False
) -> None:
    title_fill = GREEN if dynamo else MUTED
    _text(
        svg,
        px + PANEL_W / 2,
        HEADER_Y,
        title,
        fill=title_fill,
        size=15,
        weight="bold",
        spacing="0.05em",
    )

    # Band 0 — Agent Logic
    by0 = _band_y(0)
    _draw_cell(svg, px, by0, PANEL_W, BAND_H, "Agent Logic")

    # Band 1 — Harness
    by1 = _band_y(1)
    _draw_cell(
        svg,
        px,
        by1,
        PANEL_W,
        BAND_H,
        "Harness",
        sub="Claude Code \u00b7 Codex \u00b7 Custom",
        fill=HARNESS_FILL,
        stroke=HARNESS_STROKE,
    )

    # Band 2 — Protocol (3 columns)
    by2 = _band_y(2)
    ph = PROTO_H

    _draw_cell(
        svg,
        _col_x(px, 0),
        by2,
        COL_W,
        ph,
        "MCP",
        sub="Tools & Data",
    )
    _draw_cell(
        svg,
        _col_x(px, 1),
        by2,
        COL_W,
        ph,
        "A2A",
        sub="Other Agents",
    )

    cx2 = _col_x(px, 2)
    if dynamo:
        _draw_cell(
            svg,
            cx2,
            by2,
            COL_W,
            ph,
            "nvext",
            sub="Inference Infrastructure",
            fill=GREEN_DIM,
            stroke=GREEN,
            sw=2,
            filt="glow",
            main_fill=GREEN,
            sub_fill=GREEN,
        )
    else:
        _rect(svg, cx2, by2, COL_W, ph, fill="none", stroke=BORDER, dash="6,4")
        _text(
            svg,
            cx2 + COL_W / 2,
            by2 + ph / 2,
            "???",
            fill=BORDER,
            size=15,
            weight="bold",
            dy="0.35em",
        )

    # Band 3 — Targets (full width)
    by3 = _band_y(3)
    if dynamo:
        _draw_cell(
            svg,
            px,
            by3,
            PANEL_W,
            BAND_H,
            "GPU Cluster \u00b7 SGLang \u00b7 vLLM \u00b7 TRT-LLM",
            fill=TARGET_R_FILL,
            stroke=GREEN_BORDER,
        )
    else:
        _draw_cell(svg, px, by3, PANEL_W, BAND_H, "External Systems")

    # Annotation below the stack, aligned to the third column
    annot_y = by3 + BAND_H + 24
    annot_x = cx2 + COL_W / 2
    if dynamo:
        _text(
            svg,
            annot_x,
            annot_y,
            "Dynamo operates here",
            fill=GREEN,
            size=11,
            style="italic",
        )
    else:
        _text(
            svg,
            annot_x,
            annot_y,
            "No standard exists",
            fill=MUTED,
            size=11,
            style="italic",
        )


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

    _add_glow_filter(svg)
    _draw_layer_labels(svg)
    _build_panel(svg, LEFT_X, "Today")
    _build_panel(svg, RIGHT_X, "With Dynamo", dynamo=True)

    out = Path(__file__).resolve().parent / ".." / "images" / "fig-2-protocol-stack.svg"
    out.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode", xml_declaration=True)
    return str(out.resolve())


if __name__ == "__main__":
    path = generate()
    print(f"wrote {path}")
