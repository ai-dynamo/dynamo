#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate Figure 4 -- "Two Gates" request-flow diagram.

Three requests from an agentic coding session flow left-to-right through two
gate lines, reordering at each:

  * Gate 1 (Router) -- *latency_sensitivity* reorders dispatch priority.
  * Gate 2 (Engine) -- *priority* steers batch scheduling and cache eviction.

Output: ../images/fig-4-two-gates.svg
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# ── Design tokens ──────────────────────────────────────────────────────────
BG = "#0a0a0a"
GREEN = "#76b900"
GREEN_F = "#2a4a10"
AMBER_S = "#c4a035"
AMBER_F = "#201810"
GRAY_S = "#636363"
GRAY_F = "#1a1a1a"
CORAL = "#b04040"
HDR = "#e0e0e0"
MUTED = "#767676"
FONT = "Arial, Helvetica, sans-serif"
MONO = "'Courier New', Courier, monospace"

W, H = 960, 380

# ── Layout constants ───────────────────────────────────────────────────────
BOX_X, BOX_W, BOX_H, BOX_GAP = 30, 220, 52, 15
BOX_Y0 = 115
GATE1_X, GATE2_X = 330, 640
OUT_X, OUT_W, OUT_H = 665, 270, 38

# Requests in arrival order (top → bottom):
#   (label, stroke, fill, text_fill, ls, priority, outcome_text, outcome_fill)
REQUESTS = [
    (
        "Background · Lint Check",
        GRAY_S,
        GRAY_F,
        "#a0a0a0",
        0.2,
        1,
        "batch: last · cache: evict first",
        CORAL,
    ),
    (
        "Subagent · Code Search",
        AMBER_S,
        AMBER_F,
        AMBER_S,
        0.7,
        5,
        "batch: middle · cache: moderate",
        AMBER_S,
    ),
    (
        "Lead Agent · Developer Response",
        GREEN,
        GREEN_F,
        GREEN,
        0.9,
        10,
        "batch: first · cache: pinned",
        GREEN,
    ),
]

# After Gate 1 reorder (descending latency_sensitivity): indices into REQUESTS
GATE1_ORDER = [2, 1, 0]


# ── SVG helpers ────────────────────────────────────────────────────────────


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
    family: str = FONT,
) -> ET.Element:
    attrs: dict[str, str] = {
        "x": str(x),
        "y": str(y),
        "fill": fill,
        "font-size": str(size),
        "font-family": family,
        "text-anchor": anchor,
        "font-weight": weight,
        "font-style": style,
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
    fill: str = GRAY_F,
    stroke: str = GRAY_S,
    sw: float = 1,
    rx: float = 0,
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
    return _el("rect", parent, **attrs)


def _cy(pos: int) -> float:
    """Vertical centre of the *pos*-th lane slot."""
    return BOX_Y0 + pos * (BOX_H + BOX_GAP) + BOX_H / 2


# ── Build ──────────────────────────────────────────────────────────────────


def generate() -> str:
    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        viewBox=f"0 0 {W} {H}",
        width=str(W),
        height=str(H),
    )
    _rect(svg, 0, 0, W, H, fill=BG, stroke="none")

    # Column headers
    for cx, label in [
        (BOX_X + BOX_W / 2, "ARRIVAL ORDER"),
        ((GATE1_X + GATE2_X) / 2, "DISPATCH ORDER"),
        (OUT_X + OUT_W / 2, "ENGINE TREATMENT"),
    ]:
        _text(svg, cx, 36, label, fill=MUTED, size=9, weight="bold", spacing="0.1em")

    # Gate headers
    for gx, num, name, field, q in [
        (
            GATE1_X,
            "1",
            "ROUTER",
            "latency_sensitivity",
            "How soon does this reach a worker?",
        ),
        (GATE2_X, "2", "ENGINE", "priority", "How is it treated once there?"),
    ]:
        _text(svg, gx, 55, f"GATE {num}: {name}", fill=HDR, size=13, weight="bold")
        _text(svg, gx, 71, field, fill=GREEN, size=11, style="italic", family=MONO)
        _text(svg, gx, 85, q, fill=MUTED, size=10)

    # Gate dashed lines
    for gx in (GATE1_X, GATE2_X):
        _el(
            "line",
            svg,
            **{
                "x1": str(gx),
                "y1": "92",
                "x2": str(gx),
                "y2": "340",
                "stroke": GRAY_S,
                "stroke-dasharray": "6,4",
            },
        )

    # Request boxes (left, arrival order)
    for i, (label, stk, fl, tf, ls, p, *_rest) in enumerate(REQUESTS):
        by = BOX_Y0 + i * (BOX_H + BOX_GAP)
        _rect(svg, BOX_X, by, BOX_W, BOX_H, fill=fl, stroke=stk)
        _text(svg, BOX_X + BOX_W / 2, by + 22, label, fill=tf, size=13, weight="bold")
        _text(
            svg,
            BOX_X + BOX_W / 2,
            by + 40,
            f"ls={ls}  p={p}",
            fill=tf,
            size=10,
            style="italic",
        )

    # Lane lines -- draw crossing lanes first, straight lane last (on top)
    for arrival_idx in [2, 0, 1]:
        _, stk, *_rest = REQUESTS[arrival_idx]
        post_pos = GATE1_ORDER.index(arrival_idx)
        sy = _cy(arrival_idx)
        ey = _cy(post_pos)
        sx = BOX_X + BOX_W
        ex = OUT_X

        if arrival_idx == post_pos:
            d = f"M {sx},{sy} L {ex},{ey}"
        else:
            c1x = sx + (GATE1_X - sx) * 0.45
            c2x = sx + (GATE1_X - sx) * 0.55
            d = (
                f"M {sx},{sy} "
                f"C {c1x},{sy} {c2x},{ey} {GATE1_X},{ey} "
                f"L {ex},{ey}"
            )

        _el(
            "path",
            svg,
            **{
                "d": d,
                "stroke": stk,
                "fill": "none",
                "stroke-width": "2",
                "opacity": "0.8",
            },
        )

    # Outcome boxes (right, in dispatch order)
    for post_pos, arr_idx in enumerate(GATE1_ORDER):
        _, stk, fl, _, _, _, outcome, ofill = REQUESTS[arr_idx]
        oy = _cy(post_pos) - OUT_H / 2
        _rect(svg, OUT_X, oy, OUT_W, OUT_H, fill=fl, stroke=stk)
        _text(
            svg,
            OUT_X + OUT_W / 2,
            oy + OUT_H / 2 + 4,
            outcome,
            fill=ofill,
            size=11,
            weight="600",
        )

    # Write SVG
    out = Path(__file__).resolve().parent / ".." / "images" / "fig-4-two-gates.svg"
    out.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode", xml_declaration=True)
    return str(out.resolve())


if __name__ == "__main__":
    path = generate()
    print(f"wrote {path}")
